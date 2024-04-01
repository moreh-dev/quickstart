import copy
import glob
import os
import random
import tarfile
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import torch
import webdataset as wds
from albumentations.pytorch.transforms import ToTensorV2
from datasets.utils.file_utils import get_datasets_user_agent
from huggingface_hub import hf_hub_download
from loguru import logger
from torch.utils.data import (
    ChainDataset,
    ConcatDataset,
    DataLoader,
    Dataset,
    IterableDataset,
    random_split,
)
from torchvision import datasets
from torchvision import transforms as T

from dataset.tokenizer import Tokenizer
from utils.img_utils import rectangle_img_to_square
from transformers import AutoProcessor

from .llava_dataset import (
    DataArguments,
    DataCollatorForSupervisedDataset,
    LazySupervisedDataset,
    LengthGroupedSampler,
)

MixedPrecisionTrainer = None
if hasattr(torch, 'moreh'):
    MixedPrecisionTrainer = torch.moreh.MixedPrecisionTrainer
else:
    logger.warning(f"Moreh AI framework is not found")

USER_AGENT = get_datasets_user_agent()

######################### Text image datasets #####################################


class ImageDataset(Dataset):

    def __init__(self, folder, image_size, exts=['jpg', 'jpeg', 'png', 'tiff'], convert_image_to_type=None):
        super().__init__()
        exts = exts + [ext.upper() for ext in exts]
        self.folder = folder
        self.image_size = image_size
        self.paths = [p for ext in exts for p in Path(f'{folder}').rglob(f'*.{ext}')]
        self.indices = np.arange(len(self.paths))
        height, width = image_size
        self.transform = A.Compose([
            A.Lambda(lambda img, **kwargs: rectangle_img_to_square(img)),
            A.Resize(height=height, width=width),
            A.HorizontalFlip(),
            ToTensorV2(),
        ])

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        path = self.paths[self.indices[index]]
        img = cv2.imread(str(path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return self.transform(image=img)['image']


class WebDataset(IterableDataset):

    def __init__(self, urls, shuffle=False):
        self.urls = urls
        if shuffle:
            random.shuffle(self.urls)
        self.length = None

    def __iter__(self):
        for url in self.urls:
            dataset = wds.WebDataset(url, cache_dir=None, cache_size=10**10,
                                     handler=wds.handlers.warn_and_continue).decode('rgb')
            dataset_iter = iter(dataset)
            while True:  # WARNING:
                try:
                    sample = next(dataset_iter, None)
                except OSError as e:
                    logger.warning(f"Skip the sample since '{e}' is raised.")
                    sample = None
                if sample is None:
                    break
                image, text = sample['jpg'], sample['txt']
                yield (None, image), text

    def __len__(self):
        if self.length is None:
            with tarfile.open(self.urls[0], 'r') as tar:
                intar_file_names = tar.getnames()
            self.length = len([path for path in intar_file_names if path.endswith('.jpg')])
        return self.length * len(self.urls)


######################### dataloader getters #####################################


def get_llava_data_paths(dataset_name: str, data_path: str):
    # Check the existence of data JSON
    llava_data_path = os.path.join(data_path, f'{dataset_name}.json')
    if not os.path.exists(llava_data_path):
        repo_id = ('liuhaotian/LLaVA-Instruct-150K' if 'llava' in dataset_name else 'liuhaotian/LLaVA-Pretrain')
        llava_data_path = hf_hub_download(repo_id=repo_id, filename=f'{dataset_name}.json', repo_type="dataset")

    # Check the existence of image folders
    if 'llava' in dataset_name:
        image_folders = [
            "coco/train2017",
            "gqa/images",
            "ocr_vqa/images",
            "textvqa/train_images",
            "vg/VG_100K",
            "vg/VG_100K_2",
        ]
        if dataset_name != "llava_v1_5_mix665k":
            image_folders = image_folders[:1]
        for dir in image_folders:
            img_dir = os.path.join(data_path, dir)
            assert os.path.exists(img_dir), (
                f"Image folder {img_dir} not found",
                "Please prepare the images based on the following: "
                "https://github.com/haotian-liu/LLaVA#visual-instruction-tuning",
            )

        image_folder = (data_path if dataset_name == "llava_v1_5_mix665k" else f"{data_path}/coco/train2017")
    else:
        image_folder = os.path.join(data_path, 'images')
        assert os.path.exists(image_folder), (f"Image folder {image_folder} not found",
                                              "Please download the images from "
                                              "https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain")

    return llava_data_path, image_folder


def get_text_image_pair_dataloaders(tokenizer: Tokenizer,
                                    processor: AutoProcessor = None,
                                    path='data',
                                    dataset_name='laion',
                                    pretrained_encoder=None,
                                    train_batch_size=32,
                                    val_batch_size=32,
                                    shuffle_train=False,
                                    num_workers=8,
                                    pin_memory=False,
                                    drop_last=True,
                                    prefetch_factor=2,
                                    train_split_ratio=0.8,
                                    seed=42,
                                    **kwargs):
    r"""Get data loaders for text-image pair datasets.

    Args:
        tokenizer (Tokenizer): The tokenizer object used for processing text data.
        processor (AutoProcessor): The processor object used for processing text and image data.
        path (str, optional): The root path to the dataset. Defaults to 'data'.
        dataset_name (str, optional): The name of the dataset. Defaults to 'laion'.
        pretrained_encoder (object, optional): The pretrained encoder model for text encoding.
                                               Defaults to None. If it is None, tokenized texts
                                               will be output instead of the encoded text.
        image_size (tuple, optional): The desired size for image resizing (width, height).
                                      Defaults to (256, 256).
        train_batch_size (int, optional): The batch size for the training data loader.
                                          Defaults to 32.
        val_batch_size (int, optional): The batch size for the validation data loader.
                                        Defaults to 32.
        num_workers (int, optional): The number of workers for data loading.
                                     Defaults to 8.
                                     (**WARNING**) If num_workers > 0 and pretrained_encoder != None,
                                     it will raise error.
        pin_memory (bool, optional): Whether to pin memory for faster data transfer to the GPU.
                                     Defaults to True.

    Return:
        DataLoader, DataLoader: The training and validation data loaders.
    """
    assert not (num_workers > 0 and pretrained_encoder is not None), \
        "computing text encoding features during collate_fn cannot uses multiple workers."
    sampler = None

    # dataset can be multiple paths
    data_paths = path if type(path) in [list, tuple] else [path]
    total_train_dataset = []
    total_val_dataset = []

    # choose datasets
    for data_path in data_paths:
        if dataset_name in ['laion_hf']:
            tars = glob.glob(f"{data_path}/*.tar")
            size_train = int(len(tars) * 0.95)
            train_dataset = WebDataset(tars[:size_train])
            val_dataset = WebDataset(tars[size_train:])
            shuffle_train = False

        elif 'llava' in dataset_name or dataset_name == 'blip_laion_cc_sbu_558k':
            llava_data_path, image_folder = get_llava_data_paths(dataset_name, data_path)
            data_args = DataArguments(llava_data_path, is_multimodal=True, image_folder=image_folder)
            data_args.image_processor = processor.image_processor
            data_args.mm_use_im_start_end = False
            dataset = LazySupervisedDataset(tokenizer=tokenizer, data_path=llava_data_path, data_args=data_args)

            # Split the dataset according to split ratio
            train_split_ratio = (1.0 if train_split_ratio < 0 or train_split_ratio > 1 else train_split_ratio)
            num_samples = len(dataset)
            num_train = int(train_split_ratio * num_samples)
            num_val = num_samples - num_train
            generator = torch.Generator().manual_seed(seed)
            train_dataset, val_dataset = random_split(dataset, [num_train, num_val], generator=generator)

            # Initialize sampler based on modality lengths
            train_list_data_dict = [train_dataset.dataset.list_data_dict[idx] for idx in train_dataset.indices]
            lengths = train_dataset.dataset.modality_lengths(train_list_data_dict)
            sampler = LengthGroupedSampler(batch_size=train_batch_size,
                                           world_size=1,
                                           lengths=lengths,
                                           group_by_modality=kwargs['group_by_modality'])

        elif dataset_name.lower() == 'celeba':
            transformer = T.Compose([
                T.CenterCrop(178),
                T.Resize((64, 64)),
                T.RandomHorizontalFlip(0.5),
                T.ToTensor(),
            ])
            transformer_eval = T.Compose([
                T.CenterCrop(178),
                T.Resize((64, 64)),
                T.ToTensor(),
            ])

            logger.info("Ignore integrity check (via CelebA._check_integrity) "
                        "to allow using only subset of the CelebA as in 'data_example'")
            datasets.CelebA._check_integrity = lambda _: True

            train_dataset = datasets.CelebA(root=path,
                                            split='train',
                                            download=False,
                                            transform=transformer,
                                            target_transform=lambda x: x[20])  # get baldness attribute
            val_dataset = datasets.CelebA(root=path,
                                          split='valid',
                                          download=False,
                                          transform=transformer_eval,
                                          target_transform=lambda x: x[20])  # get baldness attribute

            shuffle_train = True
        else:
            raise NotImplementedError(f"{dataset_name} is not implemented.")

        total_train_dataset.append(train_dataset)
        total_val_dataset.append(val_dataset)

    if isinstance(total_train_dataset[0], IterableDataset):
        concated_train_dataset = ChainDataset(total_train_dataset)
    elif isinstance(total_train_dataset[0], Dataset):
        concated_train_dataset = ConcatDataset(total_train_dataset)
    else:
        logger.info("Current train dataset is neither IterableDataset nore Dataset.")
        concated_train_dataset = ConcatDataset(total_train_dataset)

    if isinstance(total_val_dataset[0], IterableDataset):
        concated_val_dataset = ChainDataset(total_val_dataset)
    elif isinstance(total_val_dataset[0], Dataset):
        concated_val_dataset = ConcatDataset(total_val_dataset)
    else:
        logger.info("Current validation dataset is neither IterableDataset nore Dataset.")
        concated_val_dataset = ConcatDataset(total_val_dataset)

    # choose collator
    if 'llava' in dataset_name or dataset_name == 'blip_laion_cc_sbu_558k':
        collate_fn = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    else:
        logger.warning(f"There is no implemnted collate_fn for 'dataset_name={dataset_name}', set to None.")
        collate_fn = None

    # get dataloaders
    train_data_loader = DataLoader(concated_train_dataset,
                                   batch_size=train_batch_size,
                                   shuffle=shuffle_train,
                                   sampler=sampler,
                                   pin_memory=pin_memory,
                                   num_workers=num_workers,
                                   collate_fn=collate_fn,
                                   prefetch_factor=prefetch_factor,
                                   drop_last=drop_last)

    val_data_loader = DataLoader(concated_val_dataset,
                                 batch_size=val_batch_size,
                                 shuffle=False,
                                 pin_memory=pin_memory,
                                 num_workers=num_workers,
                                 collate_fn=collate_fn,
                                 drop_last=False)

    return train_data_loader, val_data_loader


def get_image_dataloaders(path, image_size=(128, 128), train_batch_size=32, val_batch_size=32, num_workers=8):

    train_dataset = ImageDataset(path, image_size)
    val_dataset = copy.deepcopy(train_dataset)

    train_dataset.indices = train_dataset.indices[len(val_dataset) // 10:]
    val_dataset.indices = val_dataset.indices[:len(val_dataset) // 10]

    train_data_loader = DataLoader(train_dataset, batch_size=train_batch_size, num_workers=num_workers)
    val_data_loader = DataLoader(val_dataset, batch_size=val_batch_size, num_workers=num_workers)

    return train_data_loader, val_data_loader
