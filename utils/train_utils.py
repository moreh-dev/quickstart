import time
from typing import Union
import torch
from utils import save_checkpoint, log_training, logging_helper
from loguru import logger
from typing import Union
from namespace import ModelNames, OptimizerNames
from utils.batch_utils import BatchProcessor


class TrainingArgs:
    ARGS_DEFAULTS = {
        "epochs": 1,
        "train_log_interval": 100,
        "grad_accumulation_steps": 1,
        "checkpoint_iter": 0,
        "clip_grad_norm": False,
        "max_grad_norm": 1.0,
        "train_batch_size": 4,
        "loss_reduction": "mean",
        "checkpoint_step_interval": 10000,
        "checkpoint_epoch_interval": 0,
        "checkpoint_path": "./checkpoint",
        "skip_eval": False,
        "eval_start": 0,
        "eval_interval": 1,
        "corruption_rate": 0.15,
        "num_extra_ids": 100,
    }

    def __init__(self, parsed_args):
        for arg_name, default_value in TrainingArgs.ARGS_DEFAULTS.items():
            setattr(self, arg_name, getattr(parsed_args, arg_name, default_value))


class MorehTrainer:

    def __init__(self, model, optim, lr_sched, model_type: str, optim_name: Union[str, OptimizerNames] = "adamw"):
        self.model = model
        self.optim = optim
        self.lr_sched = lr_sched
        self.model_type = ModelNames(model_type)
        self._optim_name = optim_name

    def model_forward(self, inputs, labels, attention_mask, *args, **kwargs):

        if self.model_type in [ModelNames.LLAMA2, ModelNames.QWEN, ModelNames.GPT, ModelNames.MISTRAL]:
            outputs = self.model(inputs, labels=labels, use_cache=False, attention_mask=attention_mask)
        elif self.model_type in [ModelNames.LLAVA]:
            outputs = self.model(input_ids=inputs,
                                 attention_mask=attention_mask,
                                 pixel_values=kwargs['pixel_values'],
                                 labels=labels)
        elif self.model_type in [ModelNames.DIFFUSION]:
            outputs = self.model(inputs)
        else:
            raise NameError(f"not a valid model name {self.model_type}")

        return outputs

    @property
    def optim_name(self):
        return self._optim_name

    @optim_name.setter
    def optim_name(self, name: Union[str, OptimizerNames]):
        self._optim_name = OptimizerNames(name)

    def train_epochs(self,
                     parsed_args,
                     start_epoch,
                     tokenizer,
                     train_data_loader,
                     val_data_loader,
                     evaluate_func,
                     total_step=0,
                     **kwargs):

        training_args = TrainingArgs(parsed_args)

        extra_id_start = len(tokenizer)
        num_iterations = len(train_data_loader)
        log_this_step = logging_helper(num_iterations, training_args.train_log_interval)
        steps_per_epoch = len(train_data_loader) // training_args.grad_accumulation_steps

        for epoch in range(start_epoch, training_args.epochs + 1):
            self.model.train()
            logger.info(f'Epoch {epoch}/{training_args.epochs} start')

            num_batches = 0
            num_tokens = 0
            epoch_start = time.perf_counter()
            log_interval_time = time.perf_counter()

            for nbatch, batch in enumerate(train_data_loader, 1):
                if training_args.checkpoint_iter > 0 and \
                        nbatch <= training_args.checkpoint_iter * training_args.grad_accumulation_steps and \
                        epoch == start_epoch:
                    continue

                processed_batch = BatchProcessor.get_processed_batch(self.model_type, parsed_args, batch, tokenizer,
                                                                     extra_id_start)
                inputs, labels, attention_mask, batch_kwargs = processed_batch
                kwargs = {**kwargs, **batch_kwargs}

                num_batches += inputs.size(0)
                num_tokens += inputs.numel()

                if nbatch % training_args.grad_accumulation_steps == 0:
                    iters = nbatch // training_args.grad_accumulation_steps
                    iter_left = steps_per_epoch - iters
                    do_log, processed_iters = log_this_step(iters)
                else:
                    do_log = False

                outputs = self.model_forward(inputs, labels, attention_mask, **kwargs)

                loss = outputs[0]

                labels = None
                batch_kwargs = None

                if training_args.grad_accumulation_steps > 1:
                    loss /= training_args.grad_accumulation_steps

                if not training_args.loss_reduction == 'none' or do_log:
                    loss.backward()
                else:
                    backward_grad = torch.ones_like(loss, device='cuda')
                    loss.backward(backward_grad)
                    backward_grad = None
                    loss = None

                num_batches += inputs.size(0)
                num_tokens += inputs.numel()
                inputs = None
                attention_mask = None

                if nbatch % training_args.grad_accumulation_steps == 0:
                    total_step += 1

                    if training_args.clip_grad_norm:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), training_args.max_grad_norm)

                    if self.optim_name == OptimizerNames.SOPHIA:
                        self.optim.step(bs=training_args.train_batch_size)
                    else:
                        self.optim.step()
                    self.model.zero_grad(set_to_none=True)

                    if self.lr_sched is not None:
                        self.lr_sched.step()

                    if do_log:
                        current_lr = self.optim.param_groups[0]['lr']
                        num_losses = outputs.masked_time_indices.sum() if hasattr(outputs, 'masked_time_indices') else 1
                        loss /= num_losses
                        num_batches, log_interval_time = \
                            log_training(parsed_args, epoch, iter_left, iters, log_interval_time, loss, num_batches,
                                         processed_iters, steps_per_epoch, current_lr)

                        if total_step % training_args.checkpoint_step_interval == 0:
                            save_checkpoint(epoch, iters, training_args.checkpoint_path, self.model.cpu(), self.optim,
                                            self.lr_sched, tokenizer)
                            self.model = self.model.cuda()

                outputs = None

            logger.info('TRAIN_EPOCH_END | '
                        f'Epoch: {epoch} | '
                        f'Duration: {time.perf_counter() - epoch_start:.3f} s')

            if training_args.checkpoint_epoch_interval != 0 and epoch % training_args.checkpoint_epoch_interval == 0:
                save_checkpoint(epoch, 0, training_args.checkpoint_path, self.model, self.optim, self.lr_sched,
                                tokenizer)

                self.model = self.model.cuda()

            if (not parsed_args.skip_eval) and (
                (epoch >= parsed_args.eval_start and epoch % parsed_args.eval_interval == 0)
                    or epoch == parsed_args.epochs):
                logger.info('Evaluation start')

                evaluate_func(self.model,
                              tokenizer,
                              self.model_type,
                              val_data_loader,
                              parsed_args,
                              epoch,
                              metric_names=parsed_args.eval_metrics)

                self.model.train()
