import torch
from loguru import logger
import os
from diffusers import DDIMPipeline
from hub_model.model.modeling_diffusion_pipelines import DPMSolverPipeline


class BaseConditionalImageGenerator():

    def __init__(self, model, condition_iter, text_encoder=None):
        self.model = model
        self.condition_iter = condition_iter
        self.text_encoder = text_encoder
        if self.text_encoder is not None:
            self.text_encoder.cuda()

    def generate_images(self, output_path='./output', img_names=None, **kwargs):
        os.makedirs(output_path, exist_ok=True)

        if img_names is not None:
            # img_names should be batch-iterable.
            batch_loader = zip(img_names, self.condition_iter)
        else:
            batch_loader = enumerate(self.condition_iter, 0)

        try:
            idx = 0
            for name_batch, batch in batch_loader:
                pil_images = self.sample(batch, **kwargs)

                # Default img name is the generated order.
                if isinstance(name_batch, int):
                    name_batch = range(idx, idx + len(pil_images))

                for name, img in zip(name_batch, pil_images):
                    img = self.postprocess_image(img, **kwargs)
                    img.save(output_path + f"/{name}.png")
                    idx += 1

        except StopIteration:
            logger.debug("Input batch iterable stop its iteration. "
                         "Please check the dataloader is valid or "
                         "length of given img_names is matched with dataloader")
            raise StopIteration

    @torch.no_grad()
    def sample(self, batch, **kwargs):
        raise NotImplementedError

    def postprocess_image(self, img, **kwargs):
        return img

    def get_text_emb(self, raw_text):
        raise NotImplementedError


class BaseUnconditionalImageGenerator():

    def __init__(self, model, bsz=64):
        self.model = model
        self.bsz = bsz

    def generate_images(self, output_path='./output', total_img=50000, verbose=0, offset=0, **kwargs):
        os.makedirs(output_path, exist_ok=True)
        n_imgs = 0
        while n_imgs <= total_img:
            if verbose:
                logger.info("Start generate image")
            pil_images = self.sample(**kwargs)
            if verbose:
                logger.info("Generation end for a batch")
            for img in pil_images:
                img = self.postprocess_image(img, **kwargs)
                img.save(output_path + f"/{offset + n_imgs}.png")
                n_imgs += 1
                if n_imgs >= total_img:
                    break

    @torch.no_grad()
    def sample(self, batch, **kwargs):
        raise NotImplementedError

    def postprocess_image(self, img, **kwargs):
        return img


class ImagenConditionalImageGenerator(BaseConditionalImageGenerator):

    def __init__(self, model, condition_iter, tokenizer, bsz, cond_scale=1.0, text_encoder=None):
        super().__init__(model, condition_iter, text_encoder=text_encoder)
        self.bsz = bsz
        self.cond_scale = cond_scale
        self.tokenizer = tokenizer

    def sample(self, batch, **kwargs):
        _, text_tokens = batch
        if len(text_tokens.shape) == 2:
            assert self.text_encoder is not None, "Given text input is text-tokens. In this case, 'text_encoder' should be passed"
            attn_mask = (text_tokens != self.tokenizer.pad_token_id)
            with torch.no_grad():
                text_emb = self.text_encoder.encode_text(token_ids=text_tokens, attn_mask=attn_mask)
        elif len(text_tokens.shape) == 3:
            text_emb = text_tokens
            attn_mask = (text_emb.sum(-1) != self.tokenizer.pad_token_id)
        else:
            raise IndexError("text in the batch is neither encoded text (bsz, seq_len, dim)"
                             " nor tokenized text (bsz, seq_len)")
        text_emb = text_emb.cuda().float()

        pil_img = self.model.sample(text_embeds=text_emb, batch_size=self.bsz, return_pil_images=True)
        return pil_img


class UnconditionalDiffusionModelGenerator(BaseUnconditionalImageGenerator):

    def __init__(self, model, bsz=64, scheduler_type='dpmsolver', steps=10, **kwargs):
        super().__init__(model, bsz)
        if scheduler_type.lower() == 'ddim':
            self.pipeline = DDIMPipeline(self.model.unet, self.model.noise_scheduler)
        elif scheduler_type.lower() == 'dpmsolver':
            self.pipeline = DPMSolverPipeline(self.model.unet, self.model.noise_scheduler)
        # NOTE: torch.minimum in MAF does not allow constant input. Thus, below casting is needed.
        self.pipeline.scheduler.config.clip_sample_range = torch.Tensor(
            [self.pipeline.scheduler.config.clip_sample_range]).cuda()
        self.steps = steps
        self.pipeline.to('cuda')

    @torch.no_grad()
    def sample(self, **kwargs):
        logger.debug("")
        image = self.pipeline(batch_size=self.bsz, num_inference_steps=self.steps).images
        return image
