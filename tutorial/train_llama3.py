import torch

from loguru import logger
from datasets import load_dataset
from argparse import ArgumentParser
from transformers import AdamW, AutoModelForCausalLM, AutoTokenizer
import time
import copy
from peft import get_peft_model, LoraConfig

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

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

# Arguments    
def parse_args():
    parser = ArgumentParser(description="LLaMA3 FineTuning")
    parser.add_argument(
        "--model-name-or-path",
        type=str,
        default="meta-llama/Meta-Llama-3-8B",
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
        "--eval-step",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--save-model-dir", 
        type=str, 
        default="./llama3_summarization", 
        help="path to save model"
    )
    parser.add_argument(
        "--use-lora",
        action="store_true",
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--lora-dropout",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--lora-r",
        type=int,
        default=16,
    )
    args = parser.parse_args()


    return args

def eval(model, eval_dataloader, tokenizer):
    with torch.no_grad():
        logger.info("[START EPOCH EVAL]")
        model.eval()
        ev_st = time.time()
        eval_loss = torch.tensor([0], device='cuda')
        total_correct = torch.tensor([0], device='cuda')
        for e_step, e_batch in enumerate(eval_dataloader, start=1):
            e_input_ids = e_batch["input_ids"]
            e_inputs, e_labels = e_input_ids, mask_pads(e_input_ids, tokenizer)
            e_attn_mask = create_mask(e_inputs, tokenizer)

            if e_step % 10 == 0:
                logger.info(f"EVAL STEP: {e_step} / {len(eval_dataloader)}")
            e_outputs = model(
                e_inputs.cuda(),
                attention_mask=e_attn_mask.cuda(),
                labels=e_labels.cuda(),
                use_cache=False,
            )
            eval_loss += e_outputs[0]
        logger.info(f"EVAL STEP: {e_step} / {len(eval_dataloader)}")
        logger.info(f"Eval Loss: {eval_loss.item()/len(eval_dataloader)} | ELAPSED EVAL TIME: {(time.time() - ev_st)} sec")

def main(args):
    torch.moreh.option.enable_advanced_parallelization()
    # Load base model and tokenizer
    print(f"Load {args.model_name_or_path} model checkpoint and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)

    # Set pad token
    tokenizer.pad_token_id = 0
    
    if args.use_lora:
        config = LoraConfig(
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            r=args.lora_r,
            target_modules=["q_proj", "v_proj"],
            bias="none",
            task_type="CAUSAL_LM",
            )
        model = get_peft_model(model, config)
    print_trainable_parameters(model)
    # Prepare the model for training on Accelerator
    model.cuda()
    model.train()

    # Load dataset
    print(f"Downloading {args.dataset_name_or_path} dataset...")
    if args.dataset_name_or_path == "cnn_dailymail":
        dataset = load_dataset(args.dataset_name_or_path, "3.0.0").with_format("torch")
    else:
        dataset = load_dataset(args.dataset_name_or_path).with_format("torch")
        if "validation" not in dataset:
            dataset["train"] = load_dataset(args.dataset_name_or_path,  split="train[:80%]").with_format("torch")
            dataset["validation"] = load_dataset(args.dataset_name_or_path,  split="train[80%:]").with_format("torch")

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
    # Create a DataLoader for the validation set
    eval_dataloader = torch.utils.data.DataLoader(
        dataset["validation"],
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
        st = time.time()
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

            # Logging
            if step == 1:
                loss.item()
                logger.info(f"Model load and warmup done. Duration: {(time.time() - st):.2f}")
                st = time.time()
                continue
            if step % args.log_interval == 0:
                if step == args.log_interval: step_interval = args.log_interval - 1
                else : step_interval = args.log_interval
                logger.info(f"[Step {step+(epoch*len(train_dataloader))}/{total_step}] | Loss: {loss.item()} | Duration: {(time.time() - st):.2f} | {((step_interval * args.batch_size)/(time.time() - st)):.2f} | Throughput: {((step_interval * args.batch_size * args.block_size)/(time.time() - st)):.2f} tokens/sec")
                st = time.time()

            if step % args.eval_step == 0:
                # Evaluation
                eval(model, eval_dataloader, tokenizer)
                model.train()
                st = time.time()

        # Evaluation
        eval(model, eval_dataloader, tokenizer)
        model.train()
        st = time.time()


    print("Training Done")
    print("Saving Model...")
    model.save_pretrained(args.save_model_dir)
    tokenizer.save_pretrained(args.save_model_dir)
    print(f"Model saved in {args.save_model_dir}")

if __name__ == "__main__":

    args = parse_args()
    main(args)
