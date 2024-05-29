import copy
import torch

from datasets import load_dataset
from argparse import ArgumentParser
from transformers import AutoTokenizer


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--model-name-or-path",
        type=str,
        default="Qwen/Qwen1.5-7B",
        help="model name or path",
    )
    parser.add_argument(
        "--dataset-name-or-path",
        type=str,
        default="iamtarun/python_code_instructions_18k_alpaca",
        help="dataset name or path",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=1024,
        help="max input token length",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default="./qwen_dataset.pt",
        help="dataset save path",
    )

    return parser.parse_args()
    

def main(args):

    # Load tokenizer
    print(f"Loading {args.model_name_or_path} Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        padding_side="left",
    )

    # Use eos token id as pad token id
    tokenizer.pad_token_id = tokenizer.eos_token_id

    print(f"Downloading {args.dataset_name_or_path} dataset...")
    # Load dataset and set its format to PyTorch tensors
    dataset = load_dataset(args.dataset_name_or_path).with_format("torch")

    # Construct a formatted prompt
    def create_prompt(prompt):
        full_prompt = f"{prompt['prompt']}<|endoftext|>"
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

    print("Saving datset into torch format...")
    torch.save(dataset, args.save_path)
    print(f"Dataset saved as {args.save_path}")

        
if __name__ == "__main__":
    args = parse_args()
    main(args)
