from dataset.text_dataset import C4Dataset, TextDataset

import torch
from transformers import AutoTokenizer
from namespace import ModelNames


def get_dataloader_kwargs(args, model_config, **kwargs):
    dataloader_kwargs = {}

    return dataloader_kwargs


def get_dataset_class_and_kwargs(dataset_type, valid_file_index=None):
    if dataset_type == 'c4':
        dataset_cls = C4Dataset
        dataset_kwargs = {
            'dataset_type': dataset_type,
            'file_index': valid_file_index,
        }
    elif dataset_type == 'pile':
        dataset_cls = C4Dataset
        dataset_kwargs = {'extention': 'jsonl', 'file_index': valid_file_index, 'dataset_type': dataset_type}
    elif dataset_type == 'text':
        dataset_cls = TextDataset
        dataset_kwargs = {'dataset_type': dataset_type}
    else:
        raise ValueError(f"train dataset type {dataset_type} is not valid")

    return dataset_cls, dataset_kwargs


def get_dataloaders(args, tokenizer, **kwargs):

    train_batch_size = args.train_batch_size // args.grad_accumulation_steps
    val_batch_size = args.val_batch_size // args.grad_accumulation_steps

    train_dataset_cls, train_dataset_kwargs = get_dataset_class_and_kwargs(args.train_dataset_type,
                                                                            args.train_file_idx)
    val_dataset_cls, val_dataset_kwargs = get_dataset_class_and_kwargs(args.val_dataset_type, args.valid_file_idx)
    val_dataset = val_dataset_cls(tokenizer,
                                    args.overwrite_val_dataset_cache,
                                    args.val_dataset,
                                    args.block_size,
                                    cleaning=True,
                                    **val_dataset_kwargs)

    val_data_loader = torch.utils.data.DataLoader(val_dataset,
                                                    batch_size=val_batch_size,
                                                    num_workers=args.num_workers)

    assert len(val_data_loader) != 0, f"Number of the validation dataset is zero. Check '--val-dataset.'"

    train_dataset = train_dataset_cls(tokenizer, args.overwrite_train_dataset_cache, args.train_dataset,
                                        args.block_size, **train_dataset_kwargs)

    train_data_loader = torch.utils.data.DataLoader(train_dataset,
                                                    batch_size=train_batch_size,
                                                    num_workers=args.num_workers,
                                                    drop_last=True)

    assert len(train_data_loader) != 0, f"Number of the training dataset is zero. Check '--train-dataset.'"

    return train_data_loader, val_data_loader
