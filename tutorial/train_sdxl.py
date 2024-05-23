from model.modeling_sdxl import SDXL
from torch.optim import AdamW
from loguru import logger
from peft import LoraConfig
import diffusers
import time
import torch
import argparse
from diffusers.optimization import get_scheduler
from torch.utils.data import DataLoader
import datasets

from albumentations.pytorch.transforms import ToTensorV2
from diffusers import StableDiffusionXLPipeline
import albumentations as A
import numpy as np
import copy


def parse_arguments():
    parser = argparse.ArgumentParser(description="SDXL Training Script")
    parser.add_argument(
        "--model-name-or-path",
        type=str,
        default="stabilityai/stable-diffusion-xl-base-1.0",
        help="Pretrained model name or path",
    )
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument("--epochs", type=int, default=5, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument(
        "--num-workers", type=int, default=0, help="Number of data loader workers"
    )
    parser.add_argument("--log-interval", type=int, default=1, help="Logging interval")
    parser.add_argument("--dataset-name-or-path", type=str, default=None)
    # LoRA
    parser.add_argument("--lora", action="store_true", help="enable LoRA")
    parser.add_argument("--rank", type=int, default=4, help="LoRA rank")
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
    model = SDXL(args.model_name_or_path).cuda()
    if args.lora:
        unet_lora_config = LoraConfig(
            r=args.rank,
            lora_alpha=args.rank,
            init_lora_weights="gaussian",
            target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        )
        model.unet.add_adapter(unet_lora_config)
        lora_layers = filter(lambda p: p.requires_grad, model.unet.parameters())
        optim = AdamW(lora_layers, lr=args.lr)
    else:
        optim = AdamW(model.parameters(), lr=args.lr)

    train_data_loader = create_dataloader(
        args.dataset_name_or_path, batch_size=args.batch_size, num_workers=args.num_workers, model=args.model_name_or_path
    )

    one_iter_len = len(train_data_loader)
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optim,
        num_warmup_steps=20,
        num_training_steps=args.epochs * one_iter_len,
    )

    start_time = time.perf_counter()
    for e in range(args.epochs):
        for nbatch, batch in enumerate(train_data_loader, 1):
            input_images, text_tokens = batch
            # amp = torch.autocast(device_type='cuda', dtype=torch.float16)
            # with amp:
            outputs = model(
                input_images.cuda(),
                tokens=text_tokens.cuda(),
                prediction_type="epsilon",
            )

            loss = outputs[0]
            loss.backward()

            if nbatch % args.log_interval == 0:
                loss_out = loss.item()
                duration = time.perf_counter() - start_time
                throughput = (args.batch_size * args.log_interval) / duration
                start_time = time.perf_counter()
                logger.info(
                    f"[EPOCH : {e}] [STEP : {nbatch}]  Loss: {loss_out:.6f}; duration: {duration:.2f}; throughput: {throughput:.2f} imgs/sec;"
                )

            optim.step()
            lr_scheduler.step()
            optim.zero_grad()

    model.save_pretrained("./sdxl-finetuned")


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
