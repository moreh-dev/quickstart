import copy
import torch

from loguru import logger
from argparse import ArgumentParser
from transformers import AdamW, AutoModelForCausalLM
import sys, os
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(sys.path[0]), 'model')))
from modeling_baichuan import BaichuanForCausalLM

# Compose pad token mask
def create_mask(input_ids, tokenizer):
    pad_token_ids = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    return (input_ids != pad_token_ids).long()

# Mask pad tokens
def mask_pads(inputs, tokenizer, ignore_index = -100):
    idx_mask = create_mask(inputs, tokenizer)
    labels = copy.deepcopy(inputs)
    labels[~idx_mask.bool()] = ignore_index
    return labels

# Construct a formatted prompt
def create_prompt(prompt):
    full_prompt = f"[INST] {prompt['instruction']} \n Category is {prompt['category']} \n Intent is {prompt['intent']} [/INST]\n{prompt['response']}"
    return full_prompt


# Arguments    
def parse_args():
    parser = ArgumentParser(description="Baichuan2 FineTuning")
    parser.add_argument(
        "--model-name-or-path",
        type=str,
        help="model name or path",
        default='baichuan-inc/Baichuan-13B-Base'
    )
    parser.add_argument(
        "--dataset-name-or-path",
        type=str,
        default='./baichuan_dataset.pt',
    )
    parser.add_argument(
        "--epochs", 
        type=int, 
        default=3, 
        help="num training epochs"
    )
    parser.add_argument(
        "--batch-size", 
        type=int, 
        default=1024, 
        help="train bacth size"
    )
    parser.add_argument(
        "--block-size", 
        type=int, 
        default=1024, 
        help="max input token length"
    )
    parser.add_argument(
        "--lr", 
        type=float, 
        default=0.00005, 
        help="learning rate"
    )
    parser.add_argument(
        "--log-interval", 
        type=int, 
        default=10, 
        help="log interval"
    )
    parser.add_argument(
        "--save-model-dir", 
        type=str, 
        default="./baichuan_code_generation", 
        help="path to save model"
    )
    args = parser.parse_args()
    return args


def main(args):
    
    # Apply Advanced Parallelization
    torch.moreh.option.enable_advanced_parallelization()
    
    # Load base model and tokenizer
    model = BaichuanForCausalLM.from_pretrained(args.model_name_or_path, trust_remote_code=True)

    # Prepare the model for training on Accelerator
    model.cuda()
    model.train()

    dataset = torch.load(args.dataset_name_or_path)
    # Create a DataLoader for the training set
    train_dataloader = torch.utils.data.DataLoader(
        dataset["train"],
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )

    def mask_pads(input_ids, attention_mask, ignore_index = -100):
        idx_mask = attention_mask
        labels = copy.deepcopy(input_ids)
        labels[~idx_mask.bool()] = ignore_index
        return labels

    # Define AdamW optimizer
    optim = AdamW(model.parameters(), lr=args.lr)

    # Calculate total training steps
    total_step = len(train_dataloader) * args.epochs
    token_per_step = args.block_size * args.batch_size

    logger.add("file.log", format="{time} {level} {message}", level="INFO")
    # Strat training
    for epoch in range(args.epochs):
        for step, batch in enumerate(train_dataloader, start=1):
            input_ids = batch["input_ids"]
            attn_mask = batch["attention_mask"]
            labels = mask_pads(input_ids, attn_mask)
            start_time = time.time()
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
            end_time = time.time()
            
            if step % args.log_interval == 0:
                logger.info(f"[Step {step+(epoch*len(train_dataloader))}/{total_step}] Loss: {loss.item()}, Throughput : {token_per_step / (end_time - start_time)}tokens/sec")
            else:
                logger.info(f"[Step {step+(epoch*len(train_dataloader))}/{total_step}] Throughput : {token_per_step / (end_time - start_time)}tokens/sec")
    

    print("Training Done")
    print("Saving Model...")
    # Save trained model
    model = model.to("cpu")
    model.save_pretrained(args.save_model_dir)
    print(f"Model saved in {args.save_model_dir}")



if __name__ == '__main__':
    args = parse_args()
    main(args)


