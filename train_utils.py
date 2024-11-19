import copy
import time

from loguru import logger
from peft import get_peft_model
from peft import LoraConfig
import torch
from transformers import AutoConfig
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from transformers import TrainerCallback

from model.modeling_baichuan import BaichuanForCausalLM


class Preprocessor:

    def __init__(self, model, tokenizer, seq_length):
        self.tokenizer = tokenizer
        self.model_name = model.config.architectures[0].lower()
        self.seq_length = seq_length

    def __call__(self, prompt):
        if self.tokenizer.chat_template is not None:
            chat = [
                {
                    "role": "user",
                    "content": f"{prompt['instruction']}"
                },
                {
                    "role": "assistant",
                    "content": f"{prompt['response']}"
                },
            ]
            chat = self.tokenizer.apply_chat_template(chat, tokenize=False)
        else:
            chat = f"##INSTRUCTION {prompt['instruction']}\n\n##RESPONSE {prompt['response']}"
        result = self.tokenizer(chat,
                                truncation=True,
                                max_length=self.seq_length,
                                padding="max_length")
        result['labels'] = copy.deepcopy(result['input_ids'])
        if 'baichuan' not in self.model_name:
            result['position_ids'] = torch.arange(0, len(result['labels']))
        return result


class TrainCallback(TrainerCallback):

    def __init__(self, total_steps):
        self.total_train_steps = total_steps
        self.warmup_checker = False

    def on_train_begin(self, args, state, control, **kwargs):
        self.start = time.time()
        self.accum = 0

    def on_step_begin(self, args, state, control, **kwargs):
        self.accum += 1

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not self.warmup_checker:
            self.warmup_duration = time.time() - self.start
            if state.is_local_process_zero:
                logger.info(
                    f"[Step {state.global_step}] Model loading duration : {self.warmup_duration:.2f} sec"
                )
            self.start = time.time()
            self.accum = 0
            self.warmup_checker = True
        else:
            duration = time.time() - self.start
            tps = (args.max_seq_length * args.per_device_train_batch_size *
                   self.accum) / duration
            if 'loss' in logs:
                loss = logs['loss']
                lr = logs['learning_rate']
                if state.is_local_process_zero:
                    logger.info(
                        f"[Step {state.global_step}] | TPS: {tps:.2f} tokens/sec | Loss: {loss:.4f} | LR : {lr:.8f} | Duration for 1 Step: {duration / self.accum:.2f} sec"
                    )
            self.accum = 0
            self.start = time.time()

    def on_evaluate(self, args, state, control, **kwargs):
        self.start = time.time()
        self.accum = 0


def load_model(args):
    config = AutoConfig.from_pretrained(args.model)

    if "baichuan" in config.architectures[0].lower():
        model = BaichuanForCausalLM.from_pretrained(
            args.model,
            trust_remote_code=True,
        )
        if args.lora:
            peft_config = LoraConfig(
                task_type="CAUSAL_LM",
                target_modules=["W_pack"],
                inference_mode=False,
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
            )
            model = get_peft_model(model, peft_config)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            trust_remote_code=True,
        )
        if args.lora:
            peft_config = LoraConfig(
                task_type="CAUSAL_LM",
                target_modules=["q_proj", "v_proj"],
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                bias="none",
            )

            model = get_peft_model(model, peft_config)

    tokenizer = AutoTokenizer.from_pretrained(args.model,
                                              trust_remote_code=True,
                                              padding_side="right")

    tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer
