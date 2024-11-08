from transformers import TrainerCallback
from accelerate.logging import get_logger
from tqdm.auto import tqdm
import time


class TrainCallback(TrainerCallback):
    def __init__(self, total_steps):
        self.step_st = None
        self.warm_up_st = time.time()
        self.warm_up_ed = None
        self.eval_st = None
        self.eval_ed = None
        self.step_tps = 0
        self.total_train_steps = total_steps
        self.warmup_checker = False

    def on_train_begin(self, args, state, control, **kwargs):
        self.start = time.time()
        self.duration_st = time.time()
        self.accum = 0

    def on_step_begin(self, args, state, control, **kwargs):
        self.accum += 1

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not self.warmup_checker:
            self.warmup_duration = time.time() - self.start
            if state.is_local_process_zero:
                print(f"[Step {state.global_step}] Model loading duration : {self.warmup_duration:.2f} sec")
            self.start = time.time()
            self.accum = 0
            self.warmup_checker = True
        else:
            duration = time.time() - self.start
            tps = (args.max_seq_length * args.per_device_train_batch_size * self.accum) / duration
            if 'loss' in logs:
                loss = logs['loss']
                lr = logs['learning_rate']
                if state.is_local_process_zero:
                    print(f"[Step {state.global_step}] | TPS: {tps:.2f} tokens/sec | Loss: {loss:.6f} | LR : {lr:.8f} | Duration for 1 Step: {duration / self.accum:.2f} sec")
            self.accum = 0
            self.start = time.time()

