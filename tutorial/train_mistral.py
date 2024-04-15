import copy
import torch

from loguru import logger
from datasets import load_dataset
from argparse import ArgumentParser

from transformers import AdamW, AutoTokenizer
from model.modeling_mistral import MistralForCausalLM


# Model Name
MODEL_NAME = "mistralai/Mistral-7B-v0.1"

# Batch Size
BATCH_SIZE = 16

# Train Epoch
EPOCH = 3

# Max Input Token Length
MAX_LENGTH = 2048

# Learning Rate
LR = 0.00001

# Log Interval
LOG_INTERVAL = 1

# Path to Save Model
SAVE_MODEL_DIR = "./code_generation"


def main():

    # Apply Advanced Parallelization
    torch.moreh.option.enable_advanced_parallelization()

    # Load base model and tokenizer
    model = MistralForCausalLM.from_pretrained(MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # Prepare the model for training on Accelerator
    model.cuda()
    model.train()

    # Use unknown token id as pad token id
    tokenizer.pad_token_id = tokenizer.unk_token_id

    # Load MBPP dataset and set its format to PyTorch tensors
    dataset = load_dataset("mbpp").with_format("torch")

    # Construct a formatted prompt
    def create_prompt(prompt):
        full_prompt = f"[INST] {prompt['text']} [/INST]\n{prompt['code']}</s>"
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

    # Apply preprocess function
    dataset = dataset.map(preprocess)

    # Create a DataLoader for the training set
    train_dataloader = torch.utils.data.DataLoader(
        dataset["train"],
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
    )
    
    # Mask pad tokens for training
    def mask_pads(input_ids, attention_mask, ignore_index = -100):
        idx_mask = attention_mask
        labels = copy.deepcopy(input_ids)
        labels[~idx_mask.bool()] = ignore_index
        return labels

    # Define AdamW optimizer
    optim = AdamW(model.parameters(), lr=LR)

    # Calculate total training steps
    total_step = len(train_dataloader) * EPOCH

    # Start training
    for epoch in range(EPOCH):
        for i, batch in enumerate(train_dataloader, 1):
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
            if i % LOG_INTERVAL == 0:
                logger.info(f"[Step {i+(epoch*len(train_dataloader))}/{total_step}] Loss: {loss.item()}")

    # Save trained model
    print("Training Done")
    print("Saving Model...")
    model.save_pretrained(SAVE_MODEL_DIR)
    print(f"Model saved in {SAVE_MODEL_DIR}")

    #test_text = "[INST] Write a python function to find the volume of a triangular prism. [/INST]"
    #test_ids = tokenizer(test_text, return_tensors="pt").input_ids
    #model.eval()
    #ouptuts = model.generate(test_ids, max_new_tokens=1024)
    #decoded_outputs = tokenizer.decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
    #
    #print(f"Test Input:\n{test_text}\n\n")
    #print(f"Decoded Model Ouput:\n{decoded_outputs}")
    
if __name__ == "__main__":
    args = parse_args()
    main(args)
