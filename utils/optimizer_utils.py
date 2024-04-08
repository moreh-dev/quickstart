import torch
from transformers import Adafactor, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from torch.optim import Adam, AdamW
from model.lr_scheduler import InverseSquareRootLRScheduler, LinearWarmupPolyDecayScheduler, StaticScheduler
from namespace import SchedulerNames, OptimizerNames
from loguru import logger

try:
    moreh_ops = torch.ops.moreh
    MorehAdafactor = moreh_ops.Adafactor
except:
    MorehAdafactor = Adafactor


def get_optimizer(model, args):
    optimizer_kwargs = {}
    if args.optimizer_epsilon is not None:
        optimizer_kwargs['eps'] = args.optimizer_epsilon
    if args.weight_decay is not None:
        optimizer_kwargs['weight_decay'] = args.weight_decay
    if args.optimizer_betas is not None:
        optimizer_kwargs['betas'] = args.optimizer_betas

    if args.optimizer == OptimizerNames.ADAM:
        lr = 1 / (args.warmup_steps**0.5) if args.lr is None else args.lr
        optim = Adam(model.parameters(), lr=lr, **optimizer_kwargs)
    elif args.optimizer == OptimizerNames.ADAMW:
        lr = 6e-4 if args.lr is None else args.lr
        optim = AdamW(model.parameters(), lr=lr, **optimizer_kwargs)
    elif args.optimizer == OptimizerNames.ADAFACTOR:
        lr = 3e-4 if args.lr is None else args.lr
        optim = Adafactor(model.parameters(), lr=lr, scale_parameter=False, relative_step=False, **optimizer_kwargs)

    elif args.optimizer == OptimizerNames.MOREH_ADAFACTOR:
        lr = 3e-4 if args.lr is None else args.lr
        optim = MorehAdafactor(model.parameters(),
                               lr=lr,
                               scale_parameter=False,
                               relative_step=False,
                               **optimizer_kwargs)
    else:
        raise ValueError(f"optimizer type {args.optimizer} is not added in script")

    return optim


def get_lr_scheduler(optim, args, steps_per_epoch):

    if args.lr_scheduler == SchedulerNames.InverseSquareRoot:
        return InverseSquareRootLRScheduler(optim, args.warmup_steps)
    elif args.lr_scheduler == SchedulerNames.LinearWarmup:
        return get_linear_schedule_with_warmup(optim,
                                               steps_per_epoch * args.epochs * 0.25,
                                               num_training_steps=int(steps_per_epoch * args.epochs))
    elif args.lr_scheduler == SchedulerNames.LinearWarmupPolyDecay:
        return LinearWarmupPolyDecayScheduler(optim,
                                              start_warmup_steps=0,
                                              warmup_steps=args.warmup_steps,
                                              total_steps=int(steps_per_epoch * args.epochs),
                                              end_learning_rate=args.lr * 0.1,
                                              degree=1.0)
    elif args.lr_scheduler == SchedulerNames.CosineWarmup:
        return get_cosine_schedule_with_warmup(optim, args.warmup_steps, steps_per_epoch)
    elif args.lr_scheduler == SchedulerNames.Static:
        logger.warning("using static lr scheduler. lr will not be changed and other args will be discarded")
        return StaticScheduler(optim, args.warmup_steps, int(steps_per_epoch * args.epochs))
