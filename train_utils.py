import copy
import time
import sys
from tqdm import tqdm

from loguru import logger
from peft import get_peft_model
from peft import LoraConfig
import torch
from transformers import AutoConfig
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from transformers import TrainerCallback


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
        result['position_ids'] = torch.arange(0, len(result['labels']))
        return result


class TrainCallback(TrainerCallback):
    def __init__(self, total_steps, max_seq_length, max_str_len: int = 100):
        self.training_bar = None
        self.prediction_bar = None
        self.max_str_len = max_str_len

        self.total_train_steps = total_steps
        self.warmup_checker = False
        self.max_seq_length = max_seq_length

    def on_train_begin(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            self.training_bar = tqdm(total=state.max_steps, dynamic_ncols=True, leave=False)
        self.current_step = 0
        self.start = time.time()
        self.accum = 0

    def on_step_begin(self, args, state, control, **kwargs):
        self.accum += 1
    
    def on_step_end(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            if state.global_step == 1:
                pass
            elif state.global_step == 2:
                self.training_bar.update(state.global_step - self.current_step + 1)
            else:
                self.training_bar.update(state.global_step - self.current_step)
            self.current_step = state.global_step

    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.is_world_process_zero and self.training_bar is not None:
            if not self.warmup_checker:
                self.warmup_duration = time.time() - self.start
                if state.is_local_process_zero:  
                    logs["grad_norm"] = round(logs["grad_norm"], 2)
                    logs["learning_rate"] = round(logs["learning_rate"], 2)
                    self.start = time.time()
                    self.accum = 0
                    self.warmup_checker = True
                    shallow_logs = {"warmup_duration": round(self.warmup_duration, 2)}
                    self.training_bar.write(str(shallow_logs))
            else:
                if state.is_local_process_zero:
                    duration = time.time() - self.start
                    tps = (self.max_seq_length * args.per_device_train_batch_size *
                        self.accum) / duration
                    if not control.should_training_stop:
                        logs["step"] = state.global_step
                        logs["tps"] = round(tps, 2)
                        logs["duration"] = round(duration, 2)
                        logs["max_seq_len"] = self.max_seq_length
                        logs["grad_norm"] = round(logs["grad_norm"], 2)
                        logs["learning_rate"] = round(logs["learning_rate"], 2)
                    self.accum = 0
                    self.start = time.time()

                    shallow_logs = {}
                    for k, v in logs.items():
                        if isinstance(v, str) and len(v) > self.max_str_len:
                            shallow_logs[k] = (
                                f"[String too long to display, length: {len(v)} > {self.max_str_len}. "
                                "Consider increasing `max_str_len` if needed.]"
                            )
                        else:
                            shallow_logs[k] = v
                    _ = shallow_logs.pop("total_flos", None)
                    # round numbers so that it looks better in console
                    if "epoch" in shallow_logs:
                        shallow_logs["epoch"] = round(shallow_logs["epoch"], 2)
                    self.training_bar.write(str(shallow_logs))

    def on_evaluate(self, args, state, control, **kwargs):
        self.start = time.time()
        self.accum = 0
        if state.is_world_process_zero:
            if self.prediction_bar is not None:
                self.prediction_bar.close()
            self.prediction_bar = None


def load_model(args):
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
