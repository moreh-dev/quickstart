import argparse
import time
import torch
import os

from torch.optim import AdamW
from loguru import logger
from diffusers.models.lora import LoRALinearLayer
from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import (
    StableDiffusionXLPipeline,
)
from torch.utils.data import DataLoader
import datasets
from diffusers.optimization import get_scheduler
from albumentations.pytorch.transforms import ToTensorV2
from diffusers import StableDiffusionXLPipeline
import albumentations as A
import numpy as np
import copy
from model.modeling_sdxl import SDXL


def parse_args():
    parser = argparse.ArgumentParser(description="SDXL Training Script")
    parser.add_argument(
        "--model-name-or-path",
        type=str,
        default="stabilityai/stable-diffusion-xl-base-1.0",
        help="Pretrained model name or path",
    )
    parser.add_argument("--lr", type=float, default=1e-5, help="learning rate")

    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument("--accum-step", type=int, default=1, help="grad accum step")
    parser.add_argument(
        "--epochs", type=int, default=1, help="number of training epoch"
    )
    parser.add_argument("--batch-size", type=int, default=16, help="train batch size")
    parser.add_argument(
        "--num-workers", type=int, default=4, help="number of data loader workers"
    )
    parser.add_argument("--log-interval", type=int, default=1, help="logging interval")
    parser.add_argument("--dataset-path", type=str, default="lambdalabs/naruto-blip-captions")
    parser.add_argument("--save-dir", type=str, default="sdxl-finetuned")
    parser.add_argument(
        "--save-bf-model", action="store_true", help="whether to save bfloat model"
    )
    parser.add_argument(
        "--unet-config",
        type=str,
        default=None,
        help="unet configuration. if not specified, just use the SDXL-version UNet.",
    )
    parser.add_argument("--use-custom-dataset", action="store_true")
    # LoRA
    parser.add_argument("--lora", action="store_true", help="enable LoRA")
    parser.add_argument("--rank", type=int, default=32, help="LoRA rank")

    return parser.parse_args()


def rectangle_img_to_square(img):
    if type(img) == np.ndarray:
        height, width, _ = img.shape

        if height != width:
            size = min(height, width)

            x_start = (width - size) // 2
            x_end = x_start + size
            y_start = (height - size) // 2
            y_end = y_start + size

            square_img = img[y_start:y_end, x_start:x_end]
            return square_img
        else:
            return img

    # If image is a subclass of PIL.ImageFile
    else:
        width, height = img.size

        if height != width:
            size = min(height, width)

            left = (width - size) / 2
            top = (height - size) / 2
            right = (width + size) / 2
            bottom = (height + size) / 2

            # Crop the center of the image
            square_img = img.crop((left, top, right, bottom))
            return square_img
        else:
            return img


class TextImageSDXLCollator:
    def __init__(self, model, image_size=1024):
        image_size = (
            image_size
            if type(image_size) in [tuple, list]
            else (image_size, image_size)
        )
        height, width = image_size
        pipeline = StableDiffusionXLPipeline.from_pretrained(
            model, use_safetensors=True
        )
        self.tokenizer = copy.deepcopy(pipeline.tokenizer)
        self.tokenizer_2 = copy.deepcopy(pipeline.tokenizer_2)
        self.transform = A.Compose(
            [
                A.Lambda(lambda img, **kwargs: rectangle_img_to_square(img)),
                A.Resize(height=height, width=width),
                ToTensorV2(),
            ]
        )
        del pipeline

    def __call__(self, batch):
        """
        Args:
            batch (Tuple(Tuple[Tensor, str]))
        """
        images = []
        concated_tokens = []

        for data in batch:
            image = data["image"]
            caption = data["text"]
            image = np.array(image.convert("RGB"))
            token_pair = []
            for tokenizer in [self.tokenizer, self.tokenizer_2]:
                tokens = tokenizer(
                    caption,
                    padding="max_length",
                    max_length=tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                )
                token_input_ids = tokens.input_ids
                untruncated_ids = tokenizer(
                    caption, padding="longest", return_tensors="pt"
                ).input_ids

                if untruncated_ids.shape[-1] >= token_input_ids.shape[
                    -1
                ] and not torch.equal(token_input_ids, untruncated_ids):
                    pass
                    # (MAF) NOTE: Below tensor indexing (from original StableDiffusionXLPipeline) raises ValueError: step must be greater than zero.
                    # not sure CUDA system raises same error.
                    # removed_text = tokenizer.batch_decode(untruncated_ids[:tokenizer.model_max_length - 1:-1])
                    # logger.warning(
                    #     "The following part of your input was truncated because CLIP can only handle sequences up to"
                    #     f" {tokenizer.model_max_length} tokens: {removed_text}")
                token_pair.append(token_input_ids)

            concated_tokens.append(torch.vstack(token_pair))
            image: torch.Tensor = self.transform(image=image)["image"]
            image = image.type(torch.float) / 255
            images.append(image)

        newbatch = []
        for i in range(len(concated_tokens)):
            newbatch.append((images[i], concated_tokens[i]))

        return torch.utils.data.dataloader.default_collate(newbatch)


def create_dataloader(hf_dataset, batch_size, model, num_workers):
    dataset = datasets.load_dataset(hf_dataset)
    collator = TextImageSDXLCollator(model, image_size=1024)
    dataloader = DataLoader(
        dataset=dataset["train"],
        batch_size=batch_size,
        collate_fn=collator,
        num_workers=num_workers,
        drop_last=True,
    )
    return dataloader


def main(args):
    torch.moreh.option.enable_advanced_parallelization()

    os.makedirs(args.save_dir, exist_ok=True)

    model = SDXL(args.model_name_or_path).cuda()

    if args.lora:
        model.vae.requires_grad_(False)
        model.text_encoder.requires_grad_(False)
        model.text_encoder_2.requires_grad_(False)
        model.unet.requires_grad_(False)

        unet_lora_parameters = []
        for attn_processor_name, attn_processor in model.unet.attn_processors.items():
            attn_module = model.unet
            for n in attn_processor_name.split(".")[:-1]:
                attn_module = getattr(attn_module, n)

            attn_module.to_q.set_lora_layer(
                LoRALinearLayer(
                    in_features=attn_module.to_q.in_features,
                    out_features=attn_module.to_q.out_features,
                    rank=args.rank,
                )
            )
            attn_module.to_k.set_lora_layer(
                LoRALinearLayer(
                    in_features=attn_module.to_k.in_features,
                    out_features=attn_module.to_k.out_features,
                    rank=args.rank,
                )
            )
            attn_module.to_v.set_lora_layer(
                LoRALinearLayer(
                    in_features=attn_module.to_v.in_features,
                    out_features=attn_module.to_v.out_features,
                    rank=args.rank,
                )
            )
            attn_module.to_out[0].set_lora_layer(
                LoRALinearLayer(
                    in_features=attn_module.to_out[0].in_features,
                    out_features=attn_module.to_out[0].out_features,
                    rank=args.rank,
                )
            )
            unet_lora_parameters.extend(attn_module.to_q.lora_layer.parameters())
            unet_lora_parameters.extend(attn_module.to_k.lora_layer.parameters())
            unet_lora_parameters.extend(attn_module.to_v.lora_layer.parameters())
            unet_lora_parameters.extend(attn_module.to_out[0].lora_layer.parameters())

        optim = AdamW(unet_lora_parameters, lr=args.lr, weight_decay=1e-2)
    else:
        optim = AdamW(model.parameters(), lr=args.lr)

    model = model.cuda()

    train_data_loader = create_dataloader(
        args.dataset_path,
        batch_size=args.batch_size // args.accum_step,
        num_workers=args.num_workers,
        model=args.model_name_or_path,
    )

    total_steps = 1
    start_time = time.perf_counter()
    total_step_per_epoch = len(train_data_loader)
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optim,
        num_warmup_steps=20,
        num_training_steps=args.epochs * total_step_per_epoch,
    )
    for epoch in range(1, args.epochs + 1):
        for nbatch, batch in enumerate(train_data_loader, 1):
            input_images, text_tokens = batch
            outputs = model(
                input_images.cuda(),
                tokens=text_tokens.cuda(),
                prediction_type="epsilon",
            )

            loss = outputs[0] / args.accum_step
            loss.backward()

            if total_steps % args.accum_step == 0:
                optim.step()
                optim.zero_grad(set_to_none=True)
                lr_scheduler.step()

            if total_steps % args.log_interval == 0:
                log_loss = loss.item()
                duration = time.perf_counter() - start_time
                throughput = (args.batch_size * args.log_interval) / duration
                start_time = time.perf_counter()
                logger.info(
                    f"Epoch: {epoch} | "
                    f"Step : [{nbatch // args.accum_step}/{total_step_per_epoch // args.accum_step}] | "
                    f"Loss: {log_loss:.6f} | "
                    f"duration: {duration:.2f} | "
                    f"throughput: {throughput:.2f} imgs/sec"
                )

            total_steps += 1

    if (total_steps - 1) % args.log_interval != 0:
        log_loss = loss.item()
        duration = time.perf_counter() - start_time
        throughput = (args.batch_size * (total_steps % args.log_interval)) / duration
        start_time = time.perf_counter()
        logger.info(
            f"Epoch: {epoch} | "
            f"Step : [{nbatch // args.accum_step}/{total_step_per_epoch // args.accum_step}] | "
            f"Loss: {log_loss:.6f} | "
            f"duration: {duration:.2f} | "
            f"throughput: {throughput:.2f} imgs/sec"
        )

    if args.save_bf_model:
        model = model.bfloat16()

    logger.info(f"save model to {args.save_dir}")
    model.save_pretrained(args.save_dir, is_lora=args.lora)
    logger.info("model save finished")


if __name__ == "__main__":
    args = parse_args()
    main(args)
