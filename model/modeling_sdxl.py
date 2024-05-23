import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import (
    StableDiffusionXLPipeline,
)
from diffusers.utils.torch_utils import randn_tensor

# from sdxl_training.util import normalize_neg_one_to_one, unnormalize_zero_to_one
from diffusers.image_processor import VaeImageProcessor
from diffusers import DDPMScheduler


class SDXL(nn.Module):
    def __init__(self, model, config=None, variant=None):
        super().__init__()

        pipe = StableDiffusionXLPipeline.from_pretrained(model, use_safetensors=True)
        self.unet = pipe.unet
        self.vae = pipe.vae
        self.text_encoder = pipe.text_encoder
        self.text_encoder_2 = pipe.text_encoder_2
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        self.scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
        self.config = config
        self._freeze_encoders()

    def forward(self, images, tokens=None, prediction_type="epsilon"):
        # encode images and texts
        latents = self._encode_image(images)
        text_embeds, pooled_text_embeds = self._encode_prompt(tokens)
        text_embeds, pooled_text_embeds = self._drop_text_emb(
            text_embeds
        ), self._drop_text_emb(pooled_text_embeds)

        # additional input conditions for SDXL
        b, c, h, w = latents.shape
        add_text_embeds = pooled_text_embeds
        add_time_ids = self._get_add_time_ids(
            (h * self.vae_scale_factor, w * self.vae_scale_factor)
        )
        add_time_ids = add_time_ids.to("cuda").repeat(b, 1)
        added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}

        # get noise
        noise, input_latents, timesteps = self._get_noise(latents, self.scheduler)

        # unet forward
        output = self.unet(
            input_latents, timesteps, text_embeds, added_cond_kwargs=added_cond_kwargs
        ).sample

        if prediction_type == "epsilon":
            target = noise
        elif prediction_type == "velocity":
            target = self.scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise NotImplementedError
        loss = F.mse_loss(output, target)

        return [loss]

    def _freeze_encoders(self):
        for encoder_module in [self.vae, self.text_encoder, self.text_encoder_2]:
            for param in encoder_module.parameters():
                param.requires_grad = False

    @torch.no_grad()
    def _get_noise(self, x_start, scheduler):
        b = x_start.shape[0]
        noise = torch.randn(x_start.shape).cuda()
        timesteps = (
            torch.randint(0, scheduler.config["num_train_timesteps"], (b,))
            .long()
            .cuda()
        )
        noisy_images = scheduler.add_noise(x_start, noise, timesteps)
        return noise, noisy_images, timesteps

    @torch.no_grad()
    def _encode_image(self, images):
        return (
            self.vae.encode(images).latent_dist.sample()
            * self.vae.config.scaling_factor
        )

    @torch.no_grad()
    def _encode_prompt(
        self, tokens, num_images_per_prompt=1, do_classifier_free_guidance=False
    ):
        device = "cuda"

        tokens = (
            [tokens[:, 0], tokens[:, 1]]
            if self.text_encoder is not None
            else [self.text_encoder_2]
        )
        text_encoders = (
            [self.text_encoder, self.text_encoder_2]
            if self.text_encoder is not None
            else [self.text_encoder_2]
        )

        prompt_embeds_list = []
        for text_input_ids, text_encoder in zip(tokens, text_encoders):
            prompt_embeds = text_encoder(
                text_input_ids.to(device), output_hidden_states=True
            )

            pooled_prompt_embeds = prompt_embeds[0]  # Pooled output of the CLIP encoder
            prompt_embeds = prompt_embeds.hidden_states[
                -2
            ]  # Hidden states (features) of the CLIP encoder
            prompt_embeds_list.append(prompt_embeds)

        prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)

        bs_embed, seq_len, _ = prompt_embeds.shape

        pooled_prompt_embeds = pooled_prompt_embeds.repeat(
            1, num_images_per_prompt
        ).view(bs_embed * num_images_per_prompt, -1)

        return prompt_embeds, pooled_prompt_embeds

    @torch.no_grad()
    def _get_add_time_ids(
        self,
        original_size,
        crops_coords_top_left=(0, 0),
        target_size=None,
        dtype=None,
        text_encoder_projection_dim=None,
    ):
        """
        Get additional conditions including
        (1) image resoultion, c_size = (h_original, w_original): original_size
        (2) Cropping coordinates, (c_top, c_left) : crops_coords_top_left
        It will be added into the timestep embedding.

        checkpoit Sec 2.2 of the SDXL original paper:
        https://arxiv.org/pdf/2307.01952.pdf
        """
        if target_size is None:
            target_size = original_size
        if text_encoder_projection_dim is None:
            text_encoder_projection_dim = self.text_encoder_2.config.projection_dim

        add_time_ids = list(original_size + crops_coords_top_left + target_size)

        passed_add_embed_dim = (
            self.unet.config.addition_time_embed_dim * len(add_time_ids)
            + text_encoder_projection_dim
        )
        expected_add_embed_dim = self.unet.add_embedding.linear_1.in_features

        if expected_add_embed_dim != passed_add_embed_dim:
            raise ValueError(
                f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created. The model has an incorrect config. Please check `unet.config.time_embedding_type` and `text_encoder_2.config.projection_dim`."
            )

        add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
        return add_time_ids.cuda()

    @torch.no_grad()
    def _drop_text_emb(self, text_emb, drop_prob=0.1):
        """

        Args:
            text_emb (Tensor: NxLxC):
            p (float [0, 1]):
        """

        # (MAF) NOTE: make sure this zeroing-out by random index yields pull tensor or not.
        # (MAF) WARNING: currently, below would perform pull-tensor, leading to significant(?) degradation.
        # drop = torch.rand((text_emb.shape[0])).cuda() < p
        # text_emb[drop] = 0.0

        probs = torch.ones((text_emb.shape[0])) * (1 - drop_prob)
        masks = torch.bernoulli(probs)
        while len(masks.shape) < len(text_emb.shape):
            masks = masks.unsqueeze(-1)
        return masks.cuda() * text_emb

    def save_pretrained(self, save_path):
        pipeline = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            vae=self.vae,
            unet=self.unet,
        )
        pipeline.save_pretrained(save_path)
