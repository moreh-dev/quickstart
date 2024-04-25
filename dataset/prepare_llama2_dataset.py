import copy
import torch

from datasets import load_dataset
from transformers import AutoTokenizer


def main():

    # Load tokenizer
    print("Loading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("./llama-2-13b-hf")

    # Set pad token
    tokenizer.pad_token_id = 0

    # Load dataset and set its format to PyTorch tensors
    print("Downloading dataset...")
    dataset = load_dataset("cnn_dailymail", '3.0.0').with_format("torch")

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
            max_length=2048,
        )["input_ids"]

        return {"input_ids": input_ids}


    print("Preprocessing dataset...")
    # Preprocess dataset
    dataset = dataset.map(preprocess)

    print("Saving datset into torch format...")
    torch.save(dataset, "./cnn_dailymail.pt")
    print("Dataset saved as ./cnn_dailymail.pt")

        
if __name__ == "__main__":
    main()
