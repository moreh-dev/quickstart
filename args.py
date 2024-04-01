from argparse import ArgumentParser
from namespace import ModelNames, SchedulerNames, OptimizerNames
from loguru import logger
import os
import json

VALID_DATASET_TYPES = [
    "text",
    "c4",
    "pile",
    "laion",
    "celeba",
    "laion_hf",
    "ls",
    "ls_w2v2",
    "vqav2",
    "llava_instruct_80k",
    "llava_instruct_150k",
    "llava_v1_5_mix665k",
]


def validate_pp_args(args):
    if args.use_pipeline and args.use_advanced_parallelization:
        raise RuntimeError(
            "Do not use both the 'use_pipeline' and 'use_advanced_parallelization' arguments simultaneously.")
    if args.use_pipeline and (not args.encoder_split_layers and not args.decoder_split_layers):
        raise ValueError("using pipeline but split_layers is not set")
    if args.use_pipeline and args.grad_accumulation_steps != 1:
        raise ValueError("do not use grad_accumulation_step while using pipeline. instead use --num-micro-batches")
    if args.model_type in [ModelNames.DIFFUSION]:
        assert not args.use_pipeline, f"{args.model_type} does not support --use-pipeline."


def parse_args():
    parser = ArgumentParser(description="general training script for llm suite")
    parser.add_argument('--model-type', help='type of the model to be used', choices=[x.value for x in ModelNames])
    parser.add_argument('--model-config-path', type=str, default='', help='path to the model configuration file')

    # checkpoint
    parser.add_argument(
        '--load-checkpoint',
        type=str,
        default='',
        help=
        'path to a directory containing model checkpoint, optimizer states, and LR scheduler state. When specified, training resumes from the given checkpoint'
    )
    parser.add_argument(
        '--load-pretrained',
        type=str,
        default='',
        help='load pretrained a local path (model-checkpoint) or a name (for huggingface from_preatrained)')
    parser.add_argument('--checkpoint-iter', type=int, default=0, help='training iterations to restart the training.')
    parser.add_argument(
        '--checkpoint-path',
        type=str,
        default='./checkpoint',
        help=
        'path to save checkpoint. note that (checkpoint summary, model checkpoint, optimizer states, lr scheduler state) will be saved for args.checkpoint_path directory'
    )
    parser.add_argument('--checkpoint-step-interval',
                        type=int,
                        default=10000,
                        help='interval to save checkpoint based on training steps')
    parser.add_argument('--checkpoint-epoch-interval', type=int, default=1, help='checkpoint save interval')
    parser.add_argument('--init-model', type=str, default='', help='path to initial model parameters')
    parser.add_argument('--save-model',
                        type=str,
                        default='',
                        help='path to save the trained model (and its tokenizer if it exists) at the end of training')

    # train
    parser.add_argument('--epochs', '-e', type=int, default=2, help='number of training epochs')
    parser.add_argument('--train-batch-size',
                        '-b',
                        type=int,
                        default=20,
                        help='number of samples for each training iteration')
    parser.add_argument('--val-batch-size', type=int, help='number of samples for each validation iteration')
    parser.add_argument('--dropout', type=float, default=0, help='dropout rate')
    parser.add_argument(
        '--grad-accumulation-steps',
        type=int,
        default=1,
        help=
        'Gradient accumulation steps. the effective batch size is computed as (train-batch-size / grad-accumulation-steps).'
    )
    parser.add_argument('--train-log-interval', type=int, default=1, help='logging interval for training iterations')
    parser.add_argument('--eval-log-interval', type=int, default=1, help='logging interval for evaluation iterations')
    parser.add_argument('--eval-start', type=int, default=0, help='epoch at which evaluation starts')
    parser.add_argument('--eval-interval', type=int, default=1, help='epoch interval for evaluation ')
    parser.add_argument('--seed',
                        '-s',
                        type=int,
                        default=0,
                        help='manually set random seed for torch, numpy and random packages')
    parser.add_argument('--skip-eval', action='store_true', help='skip evaluation at the end of the epoch')
    parser.add_argument('--eval-metrics',
                        type=str,
                        nargs='*',
                        default=['MeanLoss'],
                        help="list of the evaluation metrics")

    # datasets
    parser.add_argument('--train-dataset', type=str, default='sample_news_data.txt', help='path to training data files')
    parser.add_argument('--val-dataset',
                        type=str,
                        default='sample_news_data.txt',
                        help='path to validation data files')
    parser.add_argument('--train-file-idx',
                        type=int,
                        nargs='*',
                        default=[0],
                        help='list of indices specifying the training files (for language corpus datasets)')
    parser.add_argument('--valid-file-idx',
                        type=int,
                        nargs='*',
                        default=[-1],
                        help='list of indices specifying the training files (for language corpus datasets)')
    parser.add_argument('--train-dataset-type',
                        default='c4',
                        choices=VALID_DATASET_TYPES,
                        help='type of the training dataset')
    parser.add_argument('--val-dataset-type',
                        default='text',
                        choices=VALID_DATASET_TYPES,
                        help='type of the validation dataset')
    parser.add_argument('--train-split-ratio',
                        type=float,
                        default=-1,
                        help='ratio of the dataset to allocate for training.')
    parser.add_argument('--loss-reduction', type=str, default='mean', help='loss reduction method: [mean, sum, none]')
    parser.add_argument('--num-workers', type=int, default=16, help='number of dataloader workers')

    # optimizers
    parser.add_argument('--optimizer',
                        choices=[x.value for x in OptimizerNames],
                        default=OptimizerNames.ADAMW,
                        help='select the optimizer to use')
    parser.add_argument('--optimizer-epsilon', type=float, default=None, help='epsilon value for the optimizers')
    parser.add_argument('--optimizer-betas',
                        type=float,
                        nargs='*',
                        default=None,
                        help='betas for the optimizer (e.g. AdamW(*, betas=--optimizer_betas))')
    parser.add_argument('--clip-grad-norm', action='store_true', help='enable gradient clipping through clip_grad_norm')
    parser.add_argument('--max-grad-norm', type=float, default=1.0, help='maximum value for clip_grad_norm')
    parser.add_argument('--warmup-steps',
                        type=int,
                        default=0,
                        help='number of warm-up steps for learning-rate schdulers')
    parser.add_argument('--weight-decay', type=float, default=None)
    parser.add_argument('--lr', type=float, help='learning rate')
    parser.add_argument('--lr-scheduler',
                        type=str,
                        default='Static',
                        choices=[x.value for x in SchedulerNames],
                        help='select the learning rate scheduler to use')

    # language models
    parser.add_argument('--overwrite-train-dataset-cache',
                        type=bool,
                        default=False,
                        help='(language model only) overwrite the tokenization cache for the training datasets')
    parser.add_argument('--overwrite-val-dataset-cache',
                        type=bool,
                        default=False,
                        help='(language model only) overwrite the tokenization cache for the validation datasets')
    parser.add_argument('--vocab-file-path', type=str, default='./data/vocab.json', help='path to vocabulary JSON file')
    parser.add_argument('--num-extra-ids',
                        type=int,
                        default=0,
                        help='number of extra ids added to the vocabulary for use as sentinels')
    parser.add_argument('--tokenizer-name-or-path',
                        type=str,
                        default='google/flan-t5-xxl',
                        help='tokenizer name or path')
    parser.add_argument('--tokenizer-max-length',
                        type=int,
                        default=None,
                        help='the maximum length for the input tokens to the language models')
    parser.add_argument('--update-tokenizer', action='store_true', help='allow to update the tokenizer after loading')
    parser.add_argument('--fixed-vocab',
                        action='store_true',
                        default=False,
                        help='enable huggingface-style tokenization: <pad> token id will be identical with <eos>')
    parser.add_argument('--block-size', type=int, default=512, help='maximum length of token for each sample')

    # moreh
    parser.add_argument('--use-moreh-attention', action='store_true', help='(MoAI) enable Moreh attention forward')
    parser.add_argument('--use-pipeline',
                        action='store_true',
                        help='(MoAI) enable pipeline parallelism for large models')
    parser.add_argument('--use-tensor-parallel',
                        action='store_true',
                        help='(MoAI) enable tensor parallelism for large models')
    parser.add_argument('--tensor-parallel-group-size',
                        type=int,
                        default=2,
                        help='(MoAI) set tensor parallel group size')
    parser.add_argument(
        '--distribute-parameter',
        action='store_true',
        help='(MoAI) distribute model parameters through multiple GPUs, allowing larger avaliable GPU memory')
    parser.add_argument(
        '--decoder-split-layers',
        type=int,
        nargs='*',
        default=[],
        help='(MoAI; language model only) indices of decoder layers to be splitted for the pipeline parallel.')
    parser.add_argument('--gpu-init', action='store_true', help='(MoAI) enable direct model-initialization on GPU')
    parser.add_argument(
        '--use-advanced-parallelization',
        action='store_true',
        help=
        '(MoAI) enable Moreh Advanced-Parallelization (AP), which efficiently parallelizes the large-models without cumbersome parallelization packages and configurations'
    )
    parser.add_argument('--bfloat16', action='store_true', help='enable mixed-precision training with bfloat16')
    parser.add_argument('--num-micro-batches',
                        type=int,
                        default=1,
                        help='split batch to N steps for the model-parallelism (micro batches)')
    parser.add_argument(
        '--use-fast-tokenizer',
        action='store_true',
        help=
        'enable a fast Rust-based tokenizer, which allows substantially faster tokenization than vanillas, if it is supported for a given model.'
    )

    # LLaVA
    parser.add_argument('--freeze-text-model', action='store_true', help='freeze the text model of LLaVA')
    parser.add_argument(
        '--group-by-modality-length',
        action='store_true',
        help='samples indices in a way that groups together features of the dataset of roughly the same length')

    parser.set_defaults(bfloat16=True)

    args = parser.parse_args()

    if args.use_pipeline:
        validate_pp_args(args)

    if args.val_batch_size is None:
        args.val_batch_size = args.train_batch_size

    if args.grad_accumulation_steps < 1:
        logger.error('The gradient accumulation step should be greater than one.')
        exit(1)

    if args.train_split_ratio > 0:
        if 'llava' in args.train_dataset_type:
            logger.info(
                f'Arguments val_dataset and val_dataset_type are ignored for the {args.train_dataset_type} dataset.')
            args.val_dataset = ''
            args.val_dataset_type = ''
        else:
            logger.error(f'Dataset splitting is not supported for the {args.train_dataset_type} dataset.')
            exit(1)

    if args.train_batch_size % args.grad_accumulation_steps != 0:
        logger.error('The batch size must be divided by grad accumulation steps.')
        exit(1)

    if (args.load_checkpoint != '' or args.init_model != '') and not args.bfloat16:
        logger.warning(f"load_checkpoint: {args.load_checkpoint}, init_model: {args.init_model}")
        logger.warning(f"setting bfloat16 from {args.bfloat16} to False")
        args.bfloat16 = False

    for k, v in vars(args).items():
        logger.info(f'PARAMETER | {k} : {v}')
    logger.info('')
    logger.info('')

    return args
