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
        default="./llama-2-13b-hf",
        help="model name or path",
    )
    parser.add_argument(
        "--dataset-name-or-path", 
        type=str,
        default="cnn_dailymail",
        help="dataset name or path",
    )
    parser.add_argument(
        "--block-size", 
        type=int,
        default=2048,
        help="max input token length",
    )
    parser.add_argument(
        "--save-path", 
        type=str,
        default="./llama2_dataset.pt",
        help="dataset save path",
    )

    return parser.parse_args()

def main(args):

    # Load tokenizer
    print(f"Loading {args.model_name_or_path} Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    # Set pad token
    tokenizer.pad_token_id = 0

    # Load dataset and set its format to PyTorch tensors
    print(f"Downloading {args.dataset_name_or_path} dataset...")
    if args.dataset_name_or_path == "cnn_dailymail":
        dataset = load_dataset(args.dataset_name_or_path, "3.0.0").with_format("torch")
    else:
        dataset = load_dataset(args.dataset_name_or_path).with_format("torch")

    # Construct a formatted prompt
    def create_prompt(prompt):
        full_prompt = f"[SUMMARIZE] {prompt['article']} [/SUMMARIZE]\n{prompt['highlights']}</s>"
        return full_prompt

    # Tokenize and prepare the input prompt
    def preprocess(prompt):
        input_ids = tokenizer(
            create_prompt(prompt),
            return_attention_mask=False,
            return_token_type_ids=False,
            padding="max_length",
            truncation=True,
            max_length=args.block_size,
        )["input_ids"]

        return {"input_ids": input_ids}


    print("Preprocessing dataset...")
    # Preprocess dataset
    dataset = dataset.map(preprocess, num_proc=16, load_from_cache_file=True)

    print("Saving datset into torch format...")
    torch.save(dataset, args.save_path)
    print(f"Dataset saved as {args.save_path}")

        
if __name__ == "__main__":
    args = parse_args()
    main(args)
