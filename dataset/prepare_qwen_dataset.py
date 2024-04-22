import copy
import torch

from datasets import load_dataset
from transformers import AutoTokenizer


# Model name
MODEL_NAME = "Qwen/Qwen1.5-7B"

# Dataset name
DATA_NAME = "iamtarun/python_code_instructions_18k_alpaca"

# Max input length
MAX_LENGTH = 1024


def main():

    # Load tokenizer
    print("Loading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        padding_side="left",
        #trust_remote_code=True,
    )

    # Use eos token id as pad token id
    tokenizer.pad_token_id = tokenizer.eos_token_id

    print("Downloading dataset...")
    # Load dataset and set its format to PyTorch tensors
    dataset = load_dataset(DATA_NAME).with_format("torch")

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
            max_length=MAX_LENGTH,
        )

        return {
            "input_ids": tokenized["input_ids"], 
            "attention_mask": tokenized["attention_mask"],
        }

    print("Preprocessing dataset...")
    # Preprocess dataset
    dataset = dataset.map(preprocess)

    print("Saving datset into torch format...")
    torch.save(dataset, 'python_code_instructions_18k_alpaca.pt')
    print("Dataset saved as ./python_code_instructions_18k_alpaca.pt")

        
if __name__ == "__main__":
    main()
