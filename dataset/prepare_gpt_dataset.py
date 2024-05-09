from datasets import load_dataset
from transformers import AdamW, AutoTokenizer
import torch
import argparse

def dataset_preprocess(args):

    dataset = load_dataset(args.dataset, split = "train").with_format("torch")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    # Construct a formatted prompt

    def create_prompt(prompt):
        full_prompt = f"[INST] {prompt['instruction']} [/INST]\n{prompt['output']}</s>"
        return full_prompt
    tokenizer.pad_token_id = tokenizer.unk_token_id


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

    # Apply preprocess function
    _dataset = dataset.map(preprocess)
    torch.save(_dataset, args.save_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=str, default="mlabonne/Evol-Instruct-Python-26k"
    )
    parser.add_argument(
        "--model", type=str, default="cerebras/Cerebras-GPT-13B"
    )
    parser.add_argument(
        "--block-size", default=1024, type=int
    )
    parser.add_argument(
        "--save-path", type=str, default='./gpt_dataset.pt'
    )
    args = parser.parse_args()
    dataset_preprocess(args)
