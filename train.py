from datasets import load_dataset
from trl import SFTConfig, SFTTrainer
from transformers import AutoTokenizer, AutoModelForCausalLM
from model.modeling_baichuan import BaichuanForCausalLM
from peft import LoraConfig, get_peft_model
from train_utils import TrainCallback
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType, FullStateDictConfig
from accelerate.logging import get_logger
from accelerate import Accelerator
from tqdm.auto import tqdm
import datasets
import transformers
import time
import copy
import torch
import argparse

def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lora", action="store_true")
    parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3-8B")
    parser.add_argument("--dataset", type=str, default="bitext/Bitext-customer-support-llm-chatbot-training-dataset")
    parser.add_argument("--train-batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--sequence-length", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--output-dir", type=str, default="llama-finetuned")
    parser.add_argument("--num-epochs", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--log-interval", type=int, default=5)
    args = parser.parse_args()
    return args

def main(args):
    torch.moreh.option.enable_advanced_parallelization()

    accelerator = Accelerator()
    logger = get_logger(__name__)
    logger.info(accelerator.state, main_process_only=True)
    logger.warning(accelerator.state, main_process_only=True)

    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    model = BaichuanForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float32,
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model, 
        trust_remote_code=True,
        padding_side="right",
    )

    tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset(args.dataset).with_format("torch")
    dataset["train"] = load_dataset(args.dataset, split="train[5%:]")
    dataset["validation"] = load_dataset(args.dataset, split="train[:5%]")

    def preprocess(prompt): 
        if tokenizer.chat_template is not None:
            chat = [
                {"role": "user", "content": f"{prompt['instruction']}"},
                {"role": "assistant", "content": f"{prompt['response']}"},
                ]
            chat = tokenizer.apply_chat_template(chat, tokenize=False)
        else:
            chat = f"##INSTRUCTION {prompt['instruction']}\n\n ##RESPONSE {prompt['response']}"
        result = tokenizer(chat, truncation=True, max_length=args.sequence_length, padding="max_length")
        result['labels'] = copy.deepcopy(result['input_ids'])
        # result['position_ids'] = torch.arange(0, len(result['labels']))
        return result

    dataset = dataset.map(preprocess, num_proc=1)
    dataset = dataset.remove_columns(['flags', 'instruction', 'category', 'intent', 'response'])

    # SFTConfig
    trainer_config = SFTConfig(
        num_train_epochs=args.num_epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        output_dir=args.output_dir,
        max_seq_length=args.sequence_length,
        optim='adamw_torch',
        lr_scheduler_type="cosine",
        learning_rate=args.lr,
        warmup_steps=50,
        bf16=True,
        do_eval=True,
        eval_strategy="epoch",
        logging_steps=args.log_interval,
        logging_first_step=True,
        report_to='none',
        logging_nan_inf_filter=False,
        max_grad_norm = 0
    )

    peft_config = LoraConfig(
        r=64,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
    )

    
    if args.lora:
        model = get_peft_model(model, peft_config)

    total_train_steps = (len(dataset["train"]) // (args.train_batch_size)) * args.num_epochs
    
    trainer = SFTTrainer(
        model,
        args=trainer_config,
        train_dataset=dataset['train'],
        eval_dataset=dataset['validation'],
        callbacks=[TrainCallback(total_steps=total_train_steps)]
    )

    trainer.train()
    if accelerator.is_local_main_process:
        print("Skip to save model")

if __name__ == "__main__":
    args = arg_parse()
    main(args)
