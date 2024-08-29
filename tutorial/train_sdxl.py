import argparse
import time
import torch
import os

from torch.optim import AdamW
from loguru import logger
from torch.utils.data import DataLoader
import datasets
from diffusers.optimization import get_scheduler
from albumentations.pytorch.transforms import ToTensorV2
from transformers import AutoTokenizer
import albumentations as A
import numpy as np
import sys, os
from peft import LoraConfig, set_peft_model_state_dict
from model.modeling_sdxl import SDXL


def parse_args():
    parser = argparse.ArgumentParser(description="SDXL FineTuning")
    parser.add_argument(
        "--model-name-or-path",
        type=str,
        default="stabilityai/stable-diffusion-xl-base-1.0",
        help="Pretrained model name or path",
    )
    parser.add_argument("--lr", type=float, default=1e-5, help="learning rate")

    parser.add_argument(
        "--lr-scheduler",
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
    parser.add_argument(
        "--dataset-path", type=str, default="lambdalabs/naruto-blip-captions"
    )
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
    parser.add_argument(
        "--train-text-encoder", action="store_true", help="Adapt LoRA to text encoders"
    )
    parser.add_argument("--rank", type=int, default=32, help="LoRA rank")
    parser.add_argument(
        "--prediction_type",
        type=str,
        default=None,
        help="The prediction_type that shall be used for training. Choose between 'epsilon' or 'v_prediction' or leave `None`. If left to `None` the default prediction type of the scheduler: `noise_scheduler.config.prediction_type` is chosen.",
    )
    parser.add_argument(
        "--validation-prompt",
        type=str,
        default=None,
        help="A prompt that is used during validation to verify that the model is learning.",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=1024,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=82,
    )

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
        self.tokenizer = AutoTokenizer.from_pretrained(model, subfolder="tokenizer")
        self.tokenizer_2 = AutoTokenizer.from_pretrained(model, subfolder="tokenizer_2")
        self.transform = A.Compose(
            [
                A.Lambda(lambda img, **kwargs: rectangle_img_to_square(img)),
                A.Resize(height=height, width=width),
                ToTensorV2(),
            ]
        )

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


def create_dataloader(hf_dataset, batch_size, model, num_workers, img_size):
    dataset = datasets.load_dataset(hf_dataset)
    collator = TextImageSDXLCollator(model, image_size=img_size)
    dataloader = DataLoader(
        dataset=dataset["train"],
        batch_size=batch_size,
        collate_fn=collator,
        num_workers=num_workers,
        drop_last=True,
    )
    return dataloader


def main(args):

    try:
        from moreh.driver.common.config import set_backend_config

        set_backend_config("miopen_mode", 3)
        torch.moreh.option.enable_advanced_parallelization()
        is_moreh = True
    except:
        from accelerate import Accelerator

        accelerator = Accelerator(mixed_precision="bf16")
        is_moreh = False

    os.makedirs(args.save_dir, exist_ok=True)

    model = SDXL(
        args.model_name_or_path,
        train_text_encoder=args.train_text_encoder,
        prediction_type=args.prediction_type,
    ).cuda()

    if args.lora:
        model.vae.requires_grad_(False)
        model.text_encoder.requires_grad_(False)
        model.text_encoder_2.requires_grad_(False)
        model.unet.requires_grad_(False)
        unet_lora_config = LoraConfig(
            r=args.rank,
            lora_alpha=args.rank,
            init_lora_weights="gaussian",
            target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        )
        model.unet.add_adapter(unet_lora_config)

        params_to_optimize = list(
            filter(lambda p: p.requires_grad, model.unet.parameters())
        )

        if args.train_text_encoder:
            # ensure that dtype is float32, even if rest of the model that isn't trained is loaded in fp16
            text_lora_config = LoraConfig(
                r=args.rank,
                lora_alpha=args.rank,
                init_lora_weights="gaussian",
                target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
            )
            model.text_encoder.add_adapter(text_lora_config)
            model.text_encoder_2.add_adapter(text_lora_config)
            params_to_optimize = (
                params_to_optimize
                + list(
                    filter(lambda p: p.requires_grad, model.text_encoder.parameters())
                )
                + list(
                    filter(lambda p: p.requires_grad, model.text_encoder_2.parameters())
                )
            )
        optim = AdamW(params_to_optimize, lr=args.lr, weight_decay=1e-2)
    else:
        optim = AdamW(model.parameters(), lr=args.lr)

    model = model.cuda()

    train_data_loader = create_dataloader(
        args.dataset_path,
        batch_size=args.batch_size // args.accum_step,
        num_workers=args.num_workers,
        model=args.model_name_or_path,
        img_size=args.img_size,
    )

    total_steps = 1
    start_time = time.time()

    if not is_moreh:
        model, optim, train_data_loader = accelerator.prepare(
            model, optim, train_data_loader
        )
    total_step_per_epoch = len(train_data_loader)
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optim,
        num_warmup_steps=20,
        num_training_steps=args.epochs * total_step_per_epoch,
    )
    for epoch in range(1, args.epochs + 1):
        model.unet.train()
        if args.lora and args.train_text_encoder:
            model.text_encoder.train()
            model.text_encoder_2.train()
        for nbatch, batch in enumerate(train_data_loader, 1):
            input_images, text_tokens = batch
            outputs = model(input_images.cuda(), tokens=text_tokens.cuda())

            loss = outputs[0] / args.accum_step
            if is_moreh:
                loss.backward()
            else:
                accelerator.backward(loss)

            if total_steps % args.accum_step == 0:
                optim.step()
                optim.zero_grad(set_to_none=True)
                lr_scheduler.step()

            if total_steps % args.log_interval == 0:
                log_loss = loss.item()
                duration = time.time() - start_time
                throughput = (args.batch_size * args.log_interval) / duration
                start_time = time.time()
                logger.info(
                    f"Epoch: {epoch} | "
                    f"Step : [{nbatch // args.accum_step}/{total_step_per_epoch // args.accum_step}] | "
                    f"Loss: {log_loss:.6f} | "
                    f"duration: {duration:.2f} | "
                    f"throughput: {throughput:.2f} imgs/sec"
                )

            total_steps += 1

        if args.validation_prompt is not None:
            with torch.no_grad():
                model.unet.eval()
                if args.lora and args.train_text_encoder:
                    model.text_encoder.eval()
                    model.text_encoder_2.eval()
                generator = torch.Generator().manual_seed(args.seed)
                img = model.pipe(
                    args.validation_prompt, num_inference_steps=25, generator=generator
                )
                img.images[0].save(
                    os.path.join(args.save_dir, f"sdxl_validation_{epoch}.png")
                )

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
    model.save_pretrained(
        args.save_dir, is_lora=args.lora, train_text_encoder=args.train_text_encoder
    )
    logger.info("model save finished")


if __name__ == "__main__":
    args = parse_args()
    main(args)
