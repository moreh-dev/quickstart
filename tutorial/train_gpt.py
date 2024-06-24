import copy
import time
import torch

from loguru import logger
from datasets import load_dataset
from argparse import ArgumentParser
from transformers import AdamW, AutoModelForCausalLM, AutoTokenizer

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
    parser = ArgumentParser(description="GPT FineTuning")
    parser.add_argument(
        "--model-name-or-path",
        type=str,
        default="cerebras/Cerebras-GPT-13B",
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
        default=16,
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
        default="mlabonne/Evol-Instruct-Python-26k",
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
        default="./gpt_checkpoint",
        help="path to save model"
    )
    parser.add_argument(
        "--lora", 
        action="store_true"
    )
    args = parser.parse_args()


    return args

def main(args):
    torch.moreh.option.enable_advanced_parallelization()
    # Load base model and tokenizer
    print(f"Load {args.model_name_or_path} model checkpoint and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    tokenizer.pad_token_id = tokenizer.unk_token_id
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, torch_dtype = torch.bfloat16)

    if args.lora:
        from peft import get_peft_model, LoraConfig
        config = LoraConfig(
            lora_alpha=16,
            lora_dropout=0.1,
            r=64,
            target_modules=["c_proj", "c_fc"],
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, config)

    print_trainable_parameters(model) 
    model.cuda()
    model.train()

    dataset = load_dataset(args.dataset_name_or_path).with_format("torch")
    # Tokenize and prepare the input prompt
    def create_prompt(prompt):
        full_prompt = f"[INST] {prompt['instruction']} [/INST]\n{prompt['output']}</s>"
        return full_prompt

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

    for epoch in range(args.epochs):
        st = time.time()
        cnt = 0
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

            cnt += 1
            if step == 1:
                logger.info(
                    f"Model prepare and warmup Done. [Step {step+(epoch*len(train_dataloader))}/{total_step}] | Loss: {loss.item()} | Duration: {(time.time() - st):.2f}"
                )
                st = time.time()
                cnt = 0
                continue
            if step % args.log_interval == 0:
                logger.info(
                    f"[Step {step+(epoch*len(train_dataloader))}/{total_step}] | Loss: {loss.item()} |"
                )
                logger.info(
                    f"Duration: {(time.time() - st):.2f} | Throughput: {((cnt * args.batch_size * args.block_size)/(time.time() - st)):.2f} tokens/sec"
                )
                st = time.time()
                cnt = 0

    print("Training Done")
    print("Saving Model...")
    model.save_pretrained(args.save_model_dir)
    tokenizer.save_pretrained(args.save_model_dir)
    print(f"Model saved in {args.save_model_dir}")

if __name__ == "__main__":
    args = parse_args()
    main(args)
