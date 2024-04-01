import copy
import os
import random
import time
import warnings
from functools import wraps
from typing import Tuple

import numpy
import torch
from loguru import logger

from transformers import PreTrainedTokenizerBase, get_linear_schedule_with_warmup


def _check_type(obj, types, msg):
    if not isinstance(types, list):
        if not isinstance(obj, types):
            raise TypeError(msg)
    else:
        if not any([isinstance(obj, t) for t in types]):
            raise TypeError(msg)


def save_checkpoint(epoch, nbatch, checkpoint_path, model, optimizer, lr_scheduler, tokenizer):
    _check_type(epoch, int, 'save_checkpoint(): argument \'epoch\' must be an integer')
    _check_type(model, [torch.nn.Module], 'save_checkpoint(): argument \'model\' must be an instance of '
                'torch.nn.Module')
    _check_type(tokenizer, [PreTrainedTokenizerBase], 'save_checkpoint(): argument \'model\' must be an instance of '
                'torch.nn.Module')
    _check_type(optimizer, torch.optim.Optimizer, 'save_checkpoint(): argument \'optimizer\' must be an instance '
                'of torch.optim.Optimizer')
    if torch.__version__ >= '2.0.0':
        lr_scheduler_cls = torch.optim.lr_scheduler.LRScheduler
    else:
        lr_scheduler_cls = torch.optim.lr_scheduler._LRScheduler
    _check_type(
        lr_scheduler, lr_scheduler_cls, 'save_checkpoint(): argument \'lr_scheduler\' must be an '
        'instance of torch.optim.lr_scheduler._LRScheduler')

    save_dir = os.path.join(checkpoint_path, str(epoch) + "_" + str(nbatch))

    logger.info(f'Save checkpoint at {save_dir}')

    os.makedirs(save_dir, exist_ok=True)
    model_save_path = os.path.join(save_dir, 'model')
    optimizer_save_path = os.path.join(save_dir, 'optimizer.pt')
    lr_scheduler_save_path = os.path.join(save_dir, 'lr_scheduler.pt')
    checkpoint_info_save_path = os.path.join(save_dir, 'checkpoint_info.pt')

    def _save(obj, path, key):
        torch.save({key: obj.state_dict()}, path)

    # as modified model cannot be loaded to hf
    # use save_pretrained method
    # use tokenizer save for further usage on transformers
    save_model(model_save_path, model, tokenizer)
    _save(optimizer, optimizer_save_path, 'optimizer')

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore",
                                message='Please also save or load the state of the optimizer when '
                                'saving or loading the scheduler.')
        _save(lr_scheduler, lr_scheduler_save_path, 'lr_scheduler')
    torch.save({
        'epoch': epoch,
    }, checkpoint_info_save_path)
    logger.info('Save complete!')


def load_checkpoint(saved_checkpoint_path, model, optimizer, lr_scheduler, tokenizer, args):
    _check_type(model, torch.nn.Module, 'load_checkpoint(): argument \'model\' must be an instance of '
                'torch.nn.Module')
    _check_type(optimizer, torch.optim.Optimizer, 'load_checkpoint(): argument \'optimizer\' must be an instance '
                'of torch.optim.Optimizer')
    _check_type(
        lr_scheduler, torch.optim.lr_scheduler._LRScheduler, 'load_checkpoint(): argument \'lr_scheduler\' must be an '
        'instance of torch.optim.lr_scheduler._LRScheduler')

    logger.info(f'Load checkpoint from {saved_checkpoint_path}')

    saved_model_path = os.path.join(saved_checkpoint_path, 'model')
    saved_optimizer_path = os.path.join(saved_checkpoint_path, 'optimizer.pt')
    saved_lr_scheduler_path = os.path.join(saved_checkpoint_path, 'lr_scheduler.pt')
    saved_checkpoint_info_path = os.path.join(saved_checkpoint_path, 'checkpoint_info.pt')

    def _load(obj, path, key):
        obj.load_state_dict(torch.load(path, map_location=torch.device('cpu'))[key])

    epoch = torch.load(saved_checkpoint_info_path)['epoch']
    model, tokenizer = load_model(saved_model_path, model, tokenizer, args)
    _load(optimizer, saved_optimizer_path, 'optimizer')

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore",
                                message='Please also save or load the state of the optimizer when '
                                'saving or loading the scheduler.')
        _load(lr_scheduler, saved_lr_scheduler_path, 'lr_scheduler')

    logger.info('Load complete!')

    return epoch, model, optimizer, lr_scheduler, tokenizer


def load_pretrained(model: torch.nn.Module, path, tokenizer, args):
    state_dict = torch.load(os.path.join(path, 'pytorch_model.bin'), map_location='cpu')

    from hub_model.model import init_model
    model, _ = init_model(args, tokenizer)

    model.load_state_dict(state_dict)


def save_pretrained(model: torch.nn.Module, path):
    os.makedirs(path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(path, 'pytorch_model.bin'))
    torch.save(model.config, os.path.join(path, 'config.json'))


def save_model(_save_path, model, tokenizer):
    _check_type(model, torch.nn.Module, 'save_model(): argument \'model\' must be an instance of '
                'torch.nn.Module')

    logger.info(f'Save model and tokenizer at {_save_path}')
    tokenizer.save_pretrained(_save_path)
    if hasattr(model, 'save_pretrained'):
        model.save_pretrained(_save_path)
    else:
        save_pretrained(model, _save_path)
    logger.info('Save complete!')


def load_model(_init_path, model, tokenizer, args):
    _check_type(model, torch.nn.Module, 'load_model(): argument \'model\' must be an instance of '
                'torch.nn.Module')

    logger.info(f'Load model\'s params and tokenizer from {_init_path}')

    tokenizer = tokenizer.from_pretrained(_init_path, trust_remote_code=True)

    if hasattr(model, 'from_pretrained'):
        torch_dtype = torch.bfloat16 if args.moreh.bfloat16 else 'auto'
        state_dict = model.from_pretrained(_init_path, torch_dtype=torch_dtype, trust_remote_code=True).state_dict()
        model.load_state_dict(state_dict)
    else:
        load_pretrained(model, _init_path, tokenizer, args)

    logger.info('Load complete!')

    return model, tokenizer


def optimizer_to(optim, device):
    for param in optim.state.values():
        # Not sure if there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)


def logging_helper(steps_per_epoch, log_interval):
    steps_per_epoch = steps_per_epoch
    log_interval = log_interval
    last_step = 0

    def _log_this_step(step):
        nonlocal steps_per_epoch, log_interval, last_step
        assert step > 0, "step must start from 1."
        last_step = 0 if last_step > step else last_step

        steps_left = steps_per_epoch - step

        do_log = step % log_interval == 0 \
                 or steps_left == 0 \
                 or step == int(log_interval * 0.1)

        if do_log:
            if step == int(log_interval * 0.1):
                processed_steps = int(log_interval * 0.1)
                last_step = step
            elif step % log_interval == 0:
                processed_steps = step - last_step
                last_step = step
            else:
                processed_steps = (step - last_step) % log_interval
        else:
            processed_steps = None

        return do_log, processed_steps

    return _log_this_step


def lr_scheduler(optimizer, num_warmup_steps, num_training_steps):
    return get_linear_schedule_with_warmup(optimizer,
                                           num_warmup_steps=num_warmup_steps,
                                           num_training_steps=num_training_steps)


def print_number_of_parameters(model):
    trainable_params = 0
    params = 0
    for p in model.parameters():
        params += p.numel()
        if p.requires_grad:
            trainable_params += p.numel()

    logger.info(f"NUMBER OF TRAINABLE PARAMETERS: {trainable_params}")
    logger.info(f"NUMBER OF TOTAL PARAMETERS: {params}")


def log_training(args, epoch, iter_left, iters, log_interval_time, loss, num_batches, processed_iters, steps_per_epoch,
                 current_lr, **kwargs):

    loss_item = loss.item() * args.grad_accumulation_steps
    duration = time.perf_counter() - log_interval_time
    iter_per_sec = processed_iters / duration
    batch_per_sec = num_batches / duration
    sequences = iters * args.train_batch_size
    total_sequences = steps_per_epoch * args.train_batch_size

    logger.info('TRAIN_STEP | '
                f'Epoch: {epoch:>3} | '
                f'Sequences: {sequences:>9}/{total_sequences:>9} | '
                f'Loss: {loss_item:>8.3f} | '
                f'Throughput: {batch_per_sec:>6.1f} batches/s | '
                f'Duration: {duration:>4.3f} s | '
                f'Estimated Time Remaining: {iter_left / iter_per_sec:.0f} s | '
                f'Current lr: {current_lr:.10f}')

    num_batches = 0
    log_interval_time = time.perf_counter()
    return num_batches, log_interval_time


def identity(t, *args, **kwargs):
    return t


def first(arr, d=None):
    if len(arr) == 0:
        return d
    return arr[0]


def maybe(fn):

    @wraps(fn)
    def inner(x):
        if not (x is not None):
            return x
        return fn(x)

    return inner


def once(fn):
    called = False

    @wraps(fn)
    def inner(x):
        nonlocal called
        if called:
            return
        called = True
        return fn(x)

    return inner


print_once = once(print)


def default(val, d):
    if val is not None:
        return val
    return d() if callable(d) else d


def cast_tuple(val, length=None):
    if isinstance(val, list):
        val = tuple(val)

    output = val if isinstance(val, tuple) else ((val, ) * default(length, 1))

    if length is not None:
        assert len(output) == length

    return output


def compact(input_dict):
    return {key: value for key, value in input_dict.items() if value is not None}


def maybe_transform_dict_key(input_dict, key, fn):
    if key not in input_dict:
        return input_dict

    copied_dict = input_dict.copy()
    copied_dict[key] = fn(copied_dict[key])
    return copied_dict


def module_device(module):
    return next(module.parameters()).device


def create_attn_mask(input_ids, tokenizer):
    pad_token_ids = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    return (input_ids != pad_token_ids).long()


def mask_pads(inputs, tokenizer, ignore_index=-100):
    idx_mask = create_attn_mask(inputs, tokenizer)
    labels = copy.deepcopy(inputs)
    labels[~idx_mask.bool()] = ignore_index
    return labels


def set_random_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    numpy.random.seed(seed)


def log_evaluation(iter_left, iters, log_interval_time, loss, num_batches, processed_iters, **kwargs):
    loss_item = loss.item()
    duration = time.perf_counter() - log_interval_time
    iters_per_sec = processed_iters / duration
    batches_per_sec = num_batches / duration

    logger.info('EVAL_STEP | '
                f'Iteration: {iters}/{iter_left + iters} | '
                f'Throughput: {batches_per_sec:.2f} batches/s | '
                f'Evaluation Loss: {loss_item:.3f} | '
                f'Estimated time left: {iter_left / iters_per_sec:.3f} s')
    num_batches = 0

    return time.perf_counter()


def add_extra_ids_to_tokenizer(tokenizer, extra_ids_num):
    sentinels = [f'<mask_{i}>' for i in range(extra_ids_num)]
    tokenizer.add_tokens(sentinels)
    return tokenizer
