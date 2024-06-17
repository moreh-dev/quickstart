import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers import StableDiffusionXLPipeline
from diffusers.image_processor import VaeImageProcessor
from diffusers.models.unet_2d_condition import UNet2DConditionModel
from diffusers import DDPMScheduler

# from sdxl_training.util import normalize_neg_one_to_one, unnormalize_zero_to_one
from diffusers.utils import  convert_state_dict_to_diffusers
from peft.utils import get_peft_model_state_dict
from contextlib import nullcontext

VAE_DOWNSCALE_FACTOR = 8

class SDXL(nn.Module):
    def __init__(self, model, config=None, variant=None, ignore_vae = False, dtype = torch.float32, train_text_encoder = None, prediction_type='epsilon'):
        super().__init__()
        self.dtype=dtype
        self.pipe = StableDiffusionXLPipeline.from_pretrained(model, use_safetensors=True)
        self.unet = self.pipe.unet.to(dtype = self.dtype)
        self.vae = self.pipe.vae.to(dtype = torch.float32)
        self.text_encoder = self.pipe.text_encoder
        self.text_encoder_2 = self.pipe.text_encoder_2
        self.image_processor = VaeImageProcessor(vae_scale_factor=VAE_DOWNSCALE_FACTOR)
        self.scheduler = DDPMScheduler.from_config(self.pipe.scheduler.config)
        if prediction_type is not None:
            self.scheduler.register_to_config(prediction_type=prediction_type)
        self.config = config
        self.train_text_encoder = train_text_encoder
        if config is not None:
            self.unet = UNet2DConditionModel.from_config(config)
        self.ignore_vae = ignore_vae
        self.snr_gamma = 5.0
        self._freeze_encoders()
        if ignore_vae:
            del self.vae


    def forward(self, images, tokens=None, snr_gamma=5):
        # encode images and texts
        ctx = torch.no_grad() if self.train_text_encoder is None else nullcontext()
        if not self.ignore_vae:
            latents = self._encode_image(VaeImageProcessor.normalize(images)).to(self.dtype)
        else:
            b, _, h, w = images.shape
            latents = torch.rand((b, 4, h//VAE_DOWNSCALE_FACTOR, w//VAE_DOWNSCALE_FACTOR), dtype=self.dtype).cuda()

        with ctx:
            text_embeds, pooled_text_embeds = self._encode_prompt(tokens)
            text_embeds, pooled_text_embeds = self._drop_text_emb(text_embeds), self._drop_text_emb(pooled_text_embeds)

        # additional input conditions for SDXL
        b, c, h, w = latents.shape
        add_text_embeds = pooled_text_embeds
        add_time_ids = self._get_add_time_ids((h * VAE_DOWNSCALE_FACTOR, w * VAE_DOWNSCALE_FACTOR))
        add_time_ids = add_time_ids.to('cuda').repeat(b, 1)
        added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}

        # get noise
        noise, input_latents, timesteps = self._get_noise(latents, self.scheduler)

        # unet forward
        output = self.unet(input_latents, timesteps, text_embeds, added_cond_kwargs=added_cond_kwargs).sample

        snr = self._compute_snr(timesteps).detach()

        mse_loss_weights = (torch.stack([snr, self.snr_gamma * torch.ones_like(timesteps)], dim=1).min(dim=1)[0] /
                            snr)
        mse_loss_weights[snr == 0] = 1.0

        if self.scheduler.config.prediction_type == 'epsilon':
            target = noise
        elif self.scheduler.config.prediction_type == "v_prediction":
            target = self.scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise NotImplementedError
        
        loss = F.mse_loss(output.float(), target.float(), reduction='none')
        loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
        loss = loss.mean()
        return [loss]
    
    def _compute_snr(self, timesteps):
        alphas_cumprod = self.scheduler.alphas_cumprod
        sqrt_alphas_cumprod = alphas_cumprod**0.5
        sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod)**0.5

        sqrt_alphas_cumprod = sqrt_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
        while len(sqrt_alphas_cumprod.shape) < len(timesteps.shape):
            sqrt_alphas_cumprod = sqrt_alphas_cumprod[..., None]
        alpha = sqrt_alphas_cumprod.expand(timesteps.shape)

        sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
        while len(sqrt_one_minus_alphas_cumprod.shape) < len(timesteps.shape):
            sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod[..., None]
        sigma = sqrt_one_minus_alphas_cumprod.expand(timesteps.shape)

        # Compute SNR.
        snr = (alpha / sigma)**2
        return snr

    def _freeze_encoders(self):
        for encoder_module in [self.vae, self.text_encoder, self.text_encoder_2]:
            for param in encoder_module.parameters():
                param.requires_grad = False

    @torch.no_grad()
    def _get_noise(self, x_start, scheduler):
        b = x_start.shape[0]
        noise = torch.randn(x_start.shape,dtype =self.dtype).cuda()
        timesteps = torch.randint(0, scheduler.config['num_train_timesteps'], (b, )).long().cuda()
        noisy_images = scheduler.add_noise(x_start, noise, timesteps)
        return noise, noisy_images, timesteps

    @torch.no_grad()
    def _encode_image(self, images):
        return self.vae.encode(images).latent_dist.sample() * self.vae.config.scaling_factor

    def _encode_prompt(self, tokens, num_images_per_prompt=1, do_classifier_free_guidance=False):
        device = 'cuda'

        tokens = ([tokens[:, 0], tokens[:, 1]] if self.text_encoder is not None else [self.text_encoder_2])
        text_encoders = ([self.text_encoder, self.text_encoder_2]
                         if self.text_encoder is not None else [self.text_encoder_2])

        prompt_embeds_list = []
        for text_input_ids, text_encoder in zip(tokens, text_encoders):
            prompt_embeds = text_encoder(text_input_ids.to(device), output_hidden_states=True)

            pooled_prompt_embeds = prompt_embeds[0]  # Pooled output of the CLIP encoder
            prompt_embeds = prompt_embeds.hidden_states[-2]  # Hidden states (features) of the CLIP encoder
            prompt_embeds_list.append(prompt_embeds)

        prompt_embeds = torch.concat(prompt_embeds_list, dim=-1).to(self.dtype)

        bs_embed, seq_len, _ = prompt_embeds.shape

        pooled_prompt_embeds = pooled_prompt_embeds.repeat(1, num_images_per_prompt).view(
            bs_embed * num_images_per_prompt, -1)

        return prompt_embeds, pooled_prompt_embeds

    @torch.no_grad()
    def _get_add_time_ids(self,
                          original_size,
                          crops_coords_top_left=(0, 0),
                          target_size=None,
                          dtype=None,
                          text_encoder_projection_dim=None):
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

        passed_add_embed_dim = (self.unet.config.addition_time_embed_dim * len(add_time_ids) +
                                text_encoder_projection_dim)
        expected_add_embed_dim = self.unet.add_embedding.linear_1.in_features

        if expected_add_embed_dim != passed_add_embed_dim:
            raise ValueError(
                f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created. The model has an incorrect config. Please check `unet.config.time_embedding_type` and `text_encoder_2.config.projection_dim`."
            )

        add_time_ids = torch.tensor([add_time_ids], dtype=self.dtype)
        return add_time_ids.cuda()

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
            masks = masks.unsqueeze(-1).to(self.dtype)
        return masks.cuda() * text_emb

    def save_pretrained(self, save_path, is_lora, train_text_encoder = None):
        if is_lora:
            unet_lora_state_dict = convert_state_dict_to_diffusers(get_peft_model_state_dict(self.unet))
            if train_text_encoder:
                text_encoder_lora_layers = convert_state_dict_to_diffusers(get_peft_model_state_dict(self.text_encoder))
                text_encoder_2_lora_layers = convert_state_dict_to_diffusers(get_peft_model_state_dict(self.text_encoder_2))
            else:
                text_encoder_lora_layers = None
                text_encoder_2_lora_layers = None
            StableDiffusionXLPipeline.save_lora_weights(
                save_directory=save_path,
                unet_lora_layers=unet_lora_state_dict,
                text_encoder_lora_layers=text_encoder_lora_layers,
                text_encoder_2_lora_layers=text_encoder_2_lora_layers,
            )

        else:
            self.pipe.save_pretrained(save_path)
