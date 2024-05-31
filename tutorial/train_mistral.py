import copy
import time
import torch

from loguru import logger
from argparse import ArgumentParser

from transformers import AdamW, AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--model-name-or-path",
        type=str,
        default="mistralai/Mistral-7B-v0.1",
        help="Hugging Face Model",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="train batch size",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=1024,
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
        default=10,
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
        default="./mistral_code_generation",
        help="model save dir",
    )
    parser.add_argument(
        "--dataset-name-or-path",
        type=str,
        default="iamtarun/python_code_instructions_18k_alpaca",
        help="dataset name or path",
    )

    args = parser.parse_args()

    return args


def main(args):
    torch.moreh.option.enable_advanced_parallelization()
    # Load model and tokenizer
    print(f"Loading {args.model_name_or_path} Tokenizer...")
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    tokenizer.pad_token_id = tokenizer.eos_token_id

    # Prepare the model for training on Accelerator
    model.train()
    model.cuda()

    print(f"Downloading {args.dataset_name_or_path} dataset...")
    dataset = load_dataset(args.dataset_name_or_path).with_format("torch")
    def create_prompt(prompt):
        full_prompt = f"{prompt['prompt']}</s>"
        return full_prompt

    # Tokenize and prepare the input prompt
    def preprocess(prompt):
        tokenized = tokenizer(
            create_prompt(prompt),
            padding="max_length",
            truncation=True,
            max_length=args.block_size,
        )

        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
        }
    print("Preprocessing dataset...")
    # Preprocess dataset
    dataset = dataset.map(preprocess, num_proc=16, load_from_cache_file=True)

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

    # Start training
    for epoch in range(args.epochs):
        for i, batch in enumerate(train_dataloader, 1):
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
    tokenizer.save_pretrained(args.save_dir)
    print(f"Model saved in {args.save_dir}")

if __name__ == "__main__":
    args = parse_args()
    main(args)
