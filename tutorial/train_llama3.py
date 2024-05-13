import copy
import time
import torch

from loguru import logger
from datasets import load_dataset
from argparse import ArgumentParser
from transformers import AdamW, LlamaForCausalLM, LlamaTokenizer, AutoTokenizer

# Compose pad token mask
def create_mask(input_ids, tokenizer):
    pad_token_ids = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    return (input_ids != pad_token_ids).long()

# Mask pad tokens
def mask_pads(inputs, tokenizer, ignore_index=-100):
    idx_mask = create_mask(inputs, tokenizer)
    labels = copy.deepcopy(inputs)
    labels[~idx_mask.bool()] = ignore_index
    return labels

# Arguments    
def parse_args():
    parser = ArgumentParser(description="LLaMA2 FineTuning")
    parser.add_argument(
        "--model-name-or-path",
        type=str,
        default="./llama3-8b",
        help="model name or path",
    )
    parser.add_argument(
        "--epochs", 
        type=int, 
        default=1, 
        help="num training epochs"
    )
    parser.add_argument(
        "--batch-size", 
        type=int, 
        default=32, 
        help="train bacth size"
    )
    parser.add_argument(
        "--block-size", 
        type=int, 
        default=1024, 
        help="max input token length"
    )
    parser.add_argument(
        "--dataset-name-or-path", 
        type=str, 
        default="./llama3_dataset.pt", 
        help="dataset name or path"
    )
    parser.add_argument(
        "--lr", 
        type=float, 
        default=0.00001, 
        help="learning rate"
    )
    parser.add_argument(
        "--log-interval", 
        type=int, 
        default=2, 
        help="log interval"
    )
    parser.add_argument(
        "--save-model-dir", 
        type=str, 
        default="./llama2_summarization", 
        help="path to save model"
    )
    args = parser.parse_args()


    return args


def main(args):
    
    # Load base model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = LlamaForCausalLM.from_pretrained(args.model_name_or_path)

    # Apply Advanced Parallelization
    torch.moreh.option.enable_advanced_parallelization()
    
    # Set pad token
    tokenizer.pad_token_id = 0
    
    # Prepare the model for training on Accelerator
    model.cuda()
    model.train()

    # Load dataset
    dataset = torch.load(args.dataset_name_or_path)

    # Create a DataLoader for the training set
    train_dataloader = torch.utils.data.DataLoader(
        dataset["train"],
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )

    # Define AdamW optimizer
    optim = AdamW(model.parameters(), lr=args.lr)

    # Calculate total training steps
    total_step = len(train_dataloader) * args.epochs

    # Start training
    for epoch in range(args.epochs):
        for step, batch in enumerate(train_dataloader, start=1):
            #breakpoint()
            start_time = time.perf_counter()
            input_ids = batch["input_ids"]
            inputs, labels = input_ids, mask_pads(input_ids, tokenizer)
            attn_mask = create_mask(inputs, tokenizer)
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
            if step % args.log_interval == 0:
                logger.info(f"[Step {step+(epoch*len(train_dataloader))}/{total_step}] | Loss: {loss.item()} | Duration: {duration:.2f} | Throughput: {throughput:.2f} tokens/sec")
    print("Training Done")
    print("Saving Model...")
    model.save_pretrained(args.save_model_dir)
    print(f"Model saved in {args.save_model_dir}")

if __name__ == "__main__":

    args = parse_args()
    main(args)
