import torch
from args import parse_args
from dataset_factory import get_dataloader_kwargs, get_dataloaders
from model import init_model
from namespace import ModelNames

from utils import (
    load_checkpoint,
    optimizer_to,
    print_number_of_parameters,
    save_model,
    set_random_seed,
)

from utils.eval_utils import evaluate
from utils.optimizer_utils import get_lr_scheduler, get_optimizer
from utils.tokenizer_utils import get_tokenizer
from utils.train_utils import MorehTrainer
from loguru import logger

torch.multiprocessing.set_sharing_strategy('file_system')


def determine_system():
    global MOREH_SYSTEM
    MOREH_SYSTEM = ()
    if hasattr(torch, 'moreh'):
        MOREH_SYSTEM = (True)


def train(args):
    ## set envs
    determine_system()
    set_random_seed(args.seed)
    if MOREH_SYSTEM:
        if args.use_advanced_parallelization:
            if args.use_tensor_parallel:
                logger.warning("tensor parallel is set but not used")
            logger.warning("AUTOPARALLEL START")
            torch.moreh.option.enable_advanced_parallelization(mixed_precision=args.bfloat16)

        else:
            torch.moreh.option.parallelizer_init(
                mixed_precision=args.bfloat16,
                distribute_parameter=args.distribute_parameter,
                pipeline_parallel=args.use_pipeline,
                tensor_parallel=args.use_tensor_parallel,
                num_micro_batches=args.num_micro_batches,
                tensor_parallel_group_size=args.tensor_parallel_group_size,
                initialized_on_accelerator=args.gpu_init,
            )

    ## tokenizer & model
    tokenizer = get_tokenizer(args.tokenizer_name_or_path,
                              args.fixed_vocab,
                              args.update_tokenizer,
                              model_max_length=args.tokenizer_max_length,
                              use_fast=args.use_fast_tokenizer,
                              num_extra_ids=args.num_extra_ids)
    model, config = init_model(args, tokenizer)

    if args.update_tokenizer:
        model.resize_token_embeddings(len(tokenizer))

    if MOREH_SYSTEM or torch.cuda.is_available():
        model = model.cuda()

    print_number_of_parameters(model)

    def train_and_evaluate(args, model_config, tokenizer, model, unet_number):
        dataloader_kwargs = get_dataloader_kwargs(args, model_config)
        train_data_loader, val_data_loader = get_dataloaders(args, tokenizer, **dataloader_kwargs)

        ## optimizer
        steps_per_epoch = len(train_data_loader) // args.grad_accumulation_steps
        optim = get_optimizer(model, args)
        lr_sched = get_lr_scheduler(optim, args, steps_per_epoch)

        ## resume from source
        start_epoch = 1
        if args.load_checkpoint != '':
            epoch, model, optim, lr_sched, tokenizer = load_checkpoint(args.load_checkpoint, model, optim, lr_sched,
                                                                       tokenizer, args)
            start_epoch = epoch + 1
            if args.checkpoint_iter > 0:
                start_epoch = epoch
            args.epochs = start_epoch + args.epochs - 1
            model.cuda()
            optimizer_to(optim, next(model.parameters()).device)

        ## train
        trainer = MorehTrainer(model, optim, lr_sched, args.model_type)
        trainer.train_epochs(args,
                             start_epoch,
                             tokenizer,
                             train_data_loader,
                             val_data_loader,
                             evaluate,
                             unet_number=unet_number)

    train_and_evaluate(args, config, tokenizer, model, None)

    if args.save_model != '':
        save_model(args.save_model, model, tokenizer)


def main():
    args = parse_args()
    train(args)


if __name__ == '__main__':
    main()
