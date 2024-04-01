import torch
from torch.nn.utils.rnn import pad_sequence
from typing import List, Tuple
from datasets import load_dataset, load_from_disk
import os


def cycle(dl):
    while True:
        for data in dl:
            yield data


def collate_fn(tokenizer):

    def collate(examples: List[torch.Tensor]):
        if tokenizer._pad_token is None:
            return pad_sequence(examples, batch_first=True)
        return pad_sequence(examples, batch_first=True, padding_value=tokenizer.pad_token_id)

    return collate


def load_hf_dataset(path, split=None):
    if os.path.exists(path):
        dataset = load_from_disk(path)
        if split is not None and split in dataset:
            dataset = dataset[split]
    else:
        dataset = load_dataset(path, split=split)

    return dataset
