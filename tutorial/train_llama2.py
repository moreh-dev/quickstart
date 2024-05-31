import copy
import time
import torch

from loguru import logger
from datasets import load_dataset
from argparse import ArgumentParser
from transformers import AdamW, AutoModelForCausalLM, AutoTokenizer


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
        default="meta-llama/Llama-2-13b-hf",
        help="model name or path",
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
        default=256, 
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
        default="cnn_dailymail", 
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
        default=10, 
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
    torch.moreh.option.enable_advanced_parallelization() 
    # Load base model and tokenizer
    print(f"Load {args.model_name_or_path} model checkpoint and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)
    model.cuda()
    model.train()

    # Apply Advanced Parallelization
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
    tokenizer.save_pretrained(args.save_model_dir)
    print(f"Model saved in {args.save_model_dir}")

if __name__ == "__main__":

    args = parse_args()
    main(args)
