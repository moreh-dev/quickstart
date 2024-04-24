import copy
import time
import torch

from loguru import logger
from argparse import ArgumentParser

from transformers import AdamW, AutoTokenizer, AutoModelForCausalLM


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--model-name-or-path",
        type=str,
        default="Qwen/Qwen1.5-7B",
        help="Hugging Face Model",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=2,
        help="Epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="train batch size",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=2048,
        help="max input token length",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-6,
        help="learning rate",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=1,
        help="log interval",
    )
    parser.add_argument(
        "--ignore-index",
        type=int,
        default=-100,
        help="pad token ignore idx",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="./qwen_code_generation",
        help="model save dir",
    )
    parser.add_argument(
        "--dataset-name-or-path",
        type=str,
        default="./python_code_instructions_18k_alpaca.pt",
        help="dataset name or path",
    )

    args = parser.parse_args()

    return args


def main(args):

    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)

    # Apply Advanced Parallelization
    torch.moreh.option.enable_advanced_parallelization()

    # Prepare the model for training on Accelerator
    model.train()
    model.cuda()

    # Apply preprocess function
    dataset = torch.load(args.dataset_name_or_path)

    # Create a DataLoader for the training set
    train_dataloader = torch.utils.data.DataLoader(
        dataset["train"],
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )
    
    # Mask pad tokens for training
    def mask_pads(input_ids, attention_mask, ignore_index=args.ignore_index):
        idx_mask = attention_mask
        labels = copy.deepcopy(input_ids)
        labels[~idx_mask.bool()] = ignore_index
        return labels

    # Define AdamW optimizer
    optim = AdamW(model.parameters(), lr=args.lr)

    # Calculate total training steps
    total_step = len(train_dataloader) * args.epochs
    print(f"total_step: {total_step}")

    # Start training
    for epoch in range(args.epochs):
        for i, batch in enumerate(train_dataloader, start=1):
            start_time = time.perf_counter()
            input_ids = batch["input_ids"]
            attn_mask = batch["attention_mask"]
            labels = mask_pads(input_ids, attn_mask)
            outputs = model(
                input_ids.cuda(),
                attention_mask=attn_mask.cuda(),
                labels=labels.cuda(),
                use_cache=False,
            )
            loss = outputs[0]
            loss.backward()

            optim.step()
            model.zero_grad(set_to_none=True)

            duration = time.perf_counter() - start_time
            throughput = (args.batch_size * args.block_size) / duration
            if i % args.log_interval == 0:
                logger.info(f"[Step {i+(epoch*len(train_dataloader))}/{total_step}] | Loss: {loss.item()} | Duration: {duration:.2f} | Throughput: {throughput:.2f} tokens/sec")

    # Save trained model
    print("Training Done")
    print("Saving Model...")
    model.save_pretrained(args.save_dir)
    print(f"Model saved in {args.save_dir}")

if __name__ == "__main__":
    args = parse_args()
    main(args)
