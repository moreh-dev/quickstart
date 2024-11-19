import argparse

from accelerate.logging import get_logger
from accelerate import Accelerator
import datasets
from datasets import load_dataset
import transformers
import torch
from trl import SFTConfig, SFTTrainer
from loguru import logger

from train_utils import load_model, TrainCallback, Preprocessor

def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3-8B")
    parser.add_argument("--dataset", type=str, default="bitext/Bitext-customer-support-llm-chatbot-training-dataset")
    parser.add_argument("--train-batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--sequence-length", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--output-dir", type=str, default="llama-finetuned")
    parser.add_argument("--num-epochs", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--lora", action="store_true")
    parser.add_argument("--lora-r", type=int, default=64)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.1)
    args = parser.parse_args()
    return args

def main(args):
    try:
        import moreh
        torch.moreh.option.enable_advanced_parallelization()
    except ImportError:
        logger.warning("Cannot use Moreh Driver")
        
    accelerator = Accelerator()
    logger = get_logger(__name__)
    logger.info(accelerator.state, main_process_only=True)
    logger.warning(accelerator.state, main_process_only=True)

    datasets.utils.logging.set_verbosity_warning()
    transformers.utils.logging.set_verbosity_info()

    model, tokenizer = load_model(args)

    dataset = load_dataset(args.dataset).with_format("torch")
    dataset["train"] = load_dataset(args.dataset, split="train[5%:]")
    dataset["validation"] = load_dataset(args.dataset, split="train[:5%]")

    preprocess = Preprocessor(model, tokenizer, args.sequence_length)

    dataset = dataset.map(preprocess, num_proc=1)
    dataset = dataset.remove_columns(['flags', 'instruction', 'category', 'intent', 'response'])
    total_train_steps = (len(dataset["train"]) // (args.train_batch_size)) * args.num_epochs

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
        do_eval=True,
        eval_strategy="epoch",
        logging_steps=args.log_interval,
        logging_first_step=True,
        report_to='none',
        logging_nan_inf_filter=False,
        max_grad_norm = 0
    )
    
    trainer = SFTTrainer(
        model,
        args=trainer_config,
        train_dataset=dataset['train'],
        eval_dataset=dataset['validation'],
        callbacks=[TrainCallback(total_steps=total_train_steps)]
    )

    trainer.train()

if __name__ == "__main__":
    args = arg_parse()
    main(args)
