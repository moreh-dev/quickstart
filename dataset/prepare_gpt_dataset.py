from datasets import load_dataset
from transformers import AdamW, AutoTokenizer
import torch
from argparse import ArgumentParser



def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--dataset-name-or-path",
        type=str,
        help="dataset name or path",
        default='mlabonne/Evol-Instruct-Python-26k'
    )
    parser.add_argument(
        "--model-name-or-path",
        type=str,
        help="model name or path",
        default='cerebras/Cerebras-GPT-13B'
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=1024,
        help="max input token length"
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default='./gpt_dataset.pt'
    )
    args = parser.parse_args()
    return args

def main(args):
    dataset = load_dataset(args.dataset_name_or_path, split = "train").with_format("torch")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
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
    _dataset = dataset.map(preprocess, num_proc=16, load_from_cache_file=True)
    torch.save(_dataset, args.save_path) 



if __name__ == "__main__":
    args = parse_args()
    main(args)
