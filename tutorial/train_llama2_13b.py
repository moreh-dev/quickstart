import copy
import torch

from loguru import logger
from datasets import load_dataset
from argparse import ArgumentParser
from transformers import AdamW, LlamaForCausalLM, LlamaTokenizer


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
    full_prompt = f"[SUMMARIZE] {prompt['article']} [/SUMMARIZE]\n{prompt['highlights']}"
    return full_prompt


# Arguments    
def parse_args():
    parser = ArgumentParser(description="LLaMA2 FineTuning")
    parser.add_argument(
        "--model-name-or-path",
        type=str,
        help="model name or path",
    )
    parser.add_argument(
        "--num-train-epochs", 
        type=int, 
        default=1, 
        help="num training epochs"
    )
    parser.add_argument(
        "--batch-size", 
        type=int, 
        default=64, 
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
        default="./outputs", 
        help="path to save model"
    )
    args = parser.parse_args()


    return args


def main(args):
    
    # Apply Advanced Parallelization
    torch.moreh.option.enable_advanced_parallelization()
    
    # Load base model and tokenizer
    tokenizer = LlamaTokenizer.from_pretrained(args.model_name_or_path)
    model = LlamaForCausalLM.from_pretrained(args.model_name_or_path)

    # Set pad token
    tokenizer.pad_token_id = 0
    
    # Prepare the model for training on Accelerator
    model.cuda()
    model.train()

    # Load CNN/Daily Mail dataset and set its format to PyTorch tensors
    dataset = load_dataset("cnn_dailymail", '3.0.0').with_format("torch")
    
    
    # Tokenize and prepare the input prompt
    def preprocess(prompt):
        input_ids = tokenizer(
            create_prompt(prompt),
            return_attention_mask=False,
            return_token_type_ids=False,
            padding="max_length",
            truncation=True,
            max_length=args.block_size,
        )['input_ids']
        return {"input_ids": input_ids}
    
    # Apply preprocess function
    dataset = dataset.map(preprocess)

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
    total_step = len(train_dataloader) * args.num_train_epochs

    # Strat training
    for epoch in range(args.num_train_epochs):
        for step, batch in enumerate(train_dataloader, start=1):
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
            if step % args.log_interval == 0:
                logger.info(f"[Step {step+(epoch*len(train_dataloader))}/{total_step}] Loss: {loss.item()}")
    
    print("Training Done")
    print("Saving Model...")
    # Save trained model
    model = model.to("cpu")
    model.save_pretrained(args.save_model_dir)
    print(f"Model saved in {args.save_model_dir}")

if __name__ == "__main__":

    args = parse_args()
    main(args)
