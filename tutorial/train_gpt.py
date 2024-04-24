
import os 
import time
import copy
import argparse

from loguru import logger
from transformers import AutoModelForCausalLM, AdamW, AutoTokenizer
import torch
import transformers
from model.modeling_gpt import GPTModel



def train(args):

    # Apply Advanced Parallelization
    torch.moreh.option.enable_advanced_parallelization()
    # Load base model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    
    # Prepare the model for training on Accelerator
    model.cuda()
    model.train()

    # Use unknown token id as pad token id
    tokenizer.pad_token_id = tokenizer.unk_token_id

    # Load MBPP dataset and set its format to PyTorch tensors
    dataset = torch.load(args.dataset)

    # Create a DataLoader for the training set
    train_dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )
    token_per_iter = args.batch_size * args.block_size
    
    # Mask pad tokens for training
    def mask_pads(input_ids, attention_mask, ignore_index = -100):
        idx_mask = attention_mask
        labels = copy.deepcopy(input_ids)
        labels[~idx_mask.bool()] = ignore_index
        return labels

    # Define AdamW optimizer
    optim = AdamW(model.parameters(), lr=args.lr)

    # Calculate total training steps
    total_step = len(train_dataloader) * args.epochs
    with open("gpt_log.log", "w") as f:
        f.write("step,loss\n")

    # Start training
    for epoch in range(args.epochs):
        for i, batch in enumerate(train_dataloader):

            start = time.time()
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
            end = time.time()
            if i % args.log_interval == 0:
                loss_scalar = loss.item()
                logger.info(f"[Step {i+(epoch*len(train_dataloader))}/{total_step}] Loss: {loss_scalar} Throughput: {token_per_iter/(end-start):.2f} tokens/sec" )
                with open("gpt_log", "a") as f:
                    f.write(f"{i+epoch*len(train_dataloader)},{loss_scalar}\n")
        print("Saving Model...")
        model.save_pretrained(args.model_save_path)


    # Save trained model
    print("Training Done")
    print("Saving Model...")
    model.save_pretrained(args.model_save_path)
    print(f"Model saved in {args.model_save_path}")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type = str, default = "cerebras/Cerebras-GPT-13B")
    parser.add_argument("--batch-size", type = int, default = 64)
    parser.add_argument("--block-size", type = int, default = 2048)
    parser.add_argument("--lr", type=float, default=0.00001)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--dataset", type =str, default="./gpt_dataset.pt")
    parser.add_argument("--model-save-path", type =str, default="./gpt_checkpoint")
    parser.add_argument("--log-interval", type =int, default=10)
    args = parser.parse_args()
    train(args)