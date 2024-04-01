from loguru import logger
from torch.utils.data import Dataset, IterableDataset
from typing import List
from multiprocessing import Pool
from functools import partial
from jsonstream import loads

import os
import pickle
import torch
import numpy as np
import random
import gzip


def group_and_tokenize(lines, tokenizer, input_length=512):
    # by not returning last concated_line, we
    # drop remainder in file-level
    # so dropped line length, in maximum 49 * num_files
    result = [tokenizer.convert_tokens_to_ids(tokenizer.tokenize(line)) for line in lines]
    result = sum([], result)
    result_ = [result[i:i + input_length] for i in range(0, len(result) // input_length * input_length, input_length)]

    assert len(result_) % input_length < 50

    return result


def clean_and_tokenize(line, tokenizer):
    docs = line.split("UNUSED4997")
    doc = docs[np.argmax([len(d) for d in docs])]
    doc = TextDataset.cleaning(doc)
    return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(doc))


class TextDataset(Dataset):

    def __init__(self,
                 tokenizer,
                 overwrite_cache,
                 path: str,
                 block_size: int = 128,
                 cache_file_name: str = None,
                 **kwargs):
        self.READ_CHUNK_SIZE = 1000000

        directory = './cache'
        filename = os.path.basename(path)
        os.makedirs(directory, exist_ok=True)
        self.block_size = block_size

        cache_file_name = f"{tokenizer.name_or_path}_cached_lm_{str(block_size)}_{filename}" if not cache_file_name else cache_file_name
        cached_features_file = os.path.join(directory, cache_file_name)

        if os.path.exists(cached_features_file) and not overwrite_cache:
            logger.info(f'Loading features from cached file {cached_features_file}')
            self.examples = np.load(cached_features_file, allow_pickle=True)["arr"]
        else:
            logger.info(f'Creating features from dataset file at {directory}')

            self.examples = []
            arrs = self._get_tokenized_text(path, tokenizer, **kwargs)

            logger.info(f'Saving features into cached file {cached_features_file}')
            self.examples = arrs
            os.makedirs(os.path.dirname(cached_features_file), exist_ok=True)
            with open(cached_features_file, "wb") as handle:
                np.savez_compressed(handle, arr=self.examples)

    def _get_tokenized_text(self, path, tokenizer, cleaning=False, **kwargs) -> List[int]:
        file_path = path
        assert os.path.isfile(file_path), f"cannot find filepath {file_path}"
        arrs = None
        with open(file_path, encoding="utf-8") as f:
            lines = []
            pool = Pool(processes=64)
            line_idx = 0
            for line in f:
                line = line.strip()
                lines.append(line)
                if len(lines) == self.READ_CHUNK_SIZE:
                    line_idx += 1
                    logger.info(f"Read lines {line_idx * self.READ_CHUNK_SIZE}")
                    text = pool.map(partial(clean_and_tokenize, tokenizer=tokenizer), lines)
                    text = [t for t in text if t]
                    self._truncate_input(tokenizer, text, self.block_size)

                    if arrs is None:
                        arrs = np.asarray(self.examples)
                    else:
                        arrs = np.concatenate((arrs, np.asarray(self.examples)))
                    self.examples = []
                    lines = []
            if lines:
                text = pool.map(partial(clean_and_tokenize, tokenizer=tokenizer), lines)
                text = [t for t in text if t]
                self._truncate_input(tokenizer, text, self.block_size)
                if arrs is None:
                    arrs = np.asarray(self.examples)
                else:
                    arrs = np.concatenate((arrs, np.asarray(self.examples)))
                self.examples = []

        return arrs

    def _truncate_input(self, tokenizer, tokenized_texts, block_size):
        eos_id = tokenizer.eos_token_id
        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        block_size -= 1  # eos position
        for tokenized_text in tokenized_texts:
            for i in range(0, len(tokenized_text), block_size):  # Truncate in block of block_size
                if i > 0:
                    continue
                if i + block_size >= len(tokenized_text):
                    remain = len(tokenized_text) - i
                    padded = tokenized_text[i:] + [eos_id] + [pad_token_id] * (block_size - remain)
                    self.examples.append(padded)
                else:
                    example = tokenized_text[i:i + block_size] + [eos_id]
                    self.examples.append(example)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return torch.tensor(self.examples[item], dtype=torch.long)

    @classmethod
    def cleaning(cls, line: str) -> str:
        line = line.replace("\n", "")
        line = line.replace("<|endoftext|>", "")
        line = line.replace(" ,", ",").replace(" .", ".").replace(" %", "%")
        line = line.replace(" - ", "-").replace(" : ", ":").replace(" / ", "/")
        line = line.replace("( ", "(").replace(" )", ")")
        line = line.replace("`` ", "\"").replace(" ''", "\"")
        line = line.replace(" 's", "'s").replace("s ' ", "s' ")
        punct_loc = TextDataset.get_last_punctuation(line)
        if punct_loc <= 20:
            return ""
        return line[:punct_loc + 1]

    @classmethod
    def get_last_punctuation(cls, text):
        for i, c in enumerate(text[::-1]):
            if c == "!" or c == "?" or c == ".":
                return len(text) - 1 - i
        return -1


class InferenceTextDataset(Dataset):

    def __init__(self, tokenizer, overwrite_cache, file_path: str, cache_file_name: str = None):
        assert os.path.isfile(file_path)

        directory = './cache'
        filename = os.path.basename(file_path)
        os.makedirs(directory, exist_ok=True)

        cache_file_name = f"t5_cached_lm_inference_{filename}" if not cache_file_name else cache_file_name
        cached_features_file = os.path.join(directory, cache_file_name)

        if os.path.exists(cached_features_file) and not overwrite_cache:
            logger.info(f'Loading features from cached file {cached_features_file}')
            with open(cached_features_file, "rb") as handle:
                self.examples = pickle.load(handle)
        else:
            logger.info(f'Creating features from dataset file at {directory}')

            self.examples = []
            text = []
            with open(file_path, encoding="utf-8") as f:
                lines = f.readlines()
                for line in lines:
                    line = line.replace('\n', '')
                    tokens = tokenizer.tokenize(line)
                    self.examples.append(tokenizer.convert_tokens_to_ids(tokens))

            logger.info(f'Saving features into cached file {cached_features_file}')
            with open(cached_features_file, "wb") as handle:
                pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return torch.tensor(self.examples[item], dtype=torch.long)


class LargeTextDataset(IterableDataset):

    def __init__(self,
                 tokenizer,
                 overwrite_cache,
                 path: str,
                 block_size: int = 128,
                 cache_file_name: str = None,
                 directory: str = './cache',
                 num_files: int = 100,
                 **kwargs):
        self.num_files = num_files
        filename = os.path.basename(path)
        os.makedirs(directory, exist_ok=True)

        cache_dir_name = f"t5_cached_lm_{str(block_size)}_{filename}_dir" if not cache_file_name else cache_file_name
        cache_dir_name = os.path.join(directory, cache_dir_name)
        os.makedirs(cache_dir_name, exist_ok=True)
        self.cache_dir_name = cache_dir_name
        self.total_lines = 0
        self.block_size = block_size

        if not self.exist_cache(cache_dir_name) and not overwrite_cache:
            logger.info(f'Creating features from dataset file at {directory}')

            self.examples = []
            arrs = self._get_tokenized_text(path, tokenizer, block_size, **kwargs)

            logger.info(f'Saving features into cached file {cache_dir_name}')
            self.examples = np.concatenate(arrs)
            self.total_lines = len(self.examples)

            for i in range(num_files):
                with open(os.path.join(cache_dir_name, str(i)), "wb") as handle:
                    start = i * len(self.examples) // num_files
                    end = (i + 1) * len(self.examples) // num_files if i < num_files - 1 else len(self.examples)
                    np.savez_compressed(handle, arr=self.examples[start:end])

    def exist_cache(self, cache_dir_name):
        for i in range(self.num_files):
            if not os.path.exists(os.path.join(cache_dir_name, str(i))):
                return False
        return True

    def __len__(self):
        if self.total_lines > 0:
            return self.total_lines
        for i in range(self.num_files):
            with open(os.path.join(self.cache_dir_name, str(i)), "rb") as handle:
                self.total_lines += len(np.load(handle)["arr"])
        return self.total_lines

    def __iter__(self):
        worker = torch.utils.data.get_worker_info()
        worker_id = worker.id if worker is not None else -1
        num_workers = worker.num_workers if worker is not None else 1

        for i in range(self.num_files):
            if i % num_workers != worker_id and num_workers > 1:
                continue
            with open(os.path.join(self.cache_dir_name, str(i)), "rb") as handle:
                lines = np.load(handle)["arr"]
                for line in lines:
                    yield torch.tensor(line, dtype=torch.long)

    def _get_tokenized_text(self, path, tokenizer, cleaning=False, **kwargs) -> List[int]:
        file_path = path
        assert os.path.isfile(file_path)
        arrs = []
        with open(file_path, encoding="utf-8") as f:
            lines = []
            pool = Pool(processes=64)
            for line in f:
                line = line.strip()
                lines.append(line)
                if len(lines) == 1000000:
                    text = pool.map(partial(clean_and_tokenize, tokenizer=tokenizer), lines)
                    text = [t for t in text if t]
                    self._truncate_input(tokenizer, text, self.block_size)
                    arrs.append(np.asarray(self.examples))
                    self.examples = []
                    lines = []
            if lines:
                text = pool.map(partial(clean_and_tokenize, tokenizer=tokenizer), lines)
                text = [t for t in text if t]
                self._truncate_input(tokenizer, text, self.block_size)
                arrs.append(np.asarray(self.examples))
                self.examples = []

        return arrs

    def _truncate_input(self, tokenizer, tokenized_texts, block_size):
        eos_id = tokenizer.convert_tokens_to_ids('</s>')
        block_size -= 1  # eos position
        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        for tokenized_text in tokenized_texts:
            for i in range(0, len(tokenized_text), block_size):  # Truncate in block of block_size
                if i > 0:
                    continue
                if i + block_size >= len(tokenized_text):
                    remain = len(tokenized_text) - i
                    padded = tokenized_text[i:] + [eos_id] + [pad_token_id] * (block_size - remain)
                    self.examples.append(padded)
                else:
                    example = tokenized_text[i:i + block_size] + [eos_id]
                    self.examples.append(example)

    @classmethod
    def cleaning(cls, line: str) -> str:
        line = line.replace("\n", "")
        line = line.replace("<|endoftext|>", "")
        line = line.replace(" ,", ",").replace(" .", ".").replace(" %", "%")
        line = line.replace(" - ", "-").replace(" : ", ":").replace(" / ", "/")
        line = line.replace("( ", "(").replace(" )", ")")
        line = line.replace("`` ", "\"").replace(" ''", "\"")
        line = line.replace(" 's", "'s").replace("s ' ", "s' ")
        punct_loc = TextDataset.get_last_punctuation(line)
        if punct_loc <= 20:
            return ""
        return line[:punct_loc + 1]

    @classmethod
    def get_last_punctuation(cls, text):
        for i, c in enumerate(text[::-1]):
            if c == "!" or c == "?" or c == ".":
                return len(text) - 1 - i
        return -1


class C4Dataset(LargeTextDataset):

    def __init__(self,
                 tokenizer,
                 overwrite_cache,
                 path: str,
                 block_size: int = 512,
                 cache_dir_name: str = None,
                 directory: str = './cache',
                 **kwargs):
        filename = os.path.basename(path)
        os.makedirs(directory, exist_ok=True)
        self.block_size = block_size

        self.dataset_type = kwargs.get('dataset_type', 'c4')
        cache_dir_name = f"{self.dataset_type}_cached_{str(tokenizer.name_or_path.replace('/', '_'))}_{str(block_size)}_{filename}_dir" if not cache_dir_name else cache_dir_name
        cache_dir_name = os.path.join(directory, cache_dir_name)
        os.makedirs(cache_dir_name, exist_ok=True)
        self.cache_dir_name = cache_dir_name
        self.total_lines = 0

        self.extention = kwargs.get('extention', 'gz')
        files = [file_name for file_name in os.listdir(path) if file_name.endswith(self.extention)]

        if kwargs.get('file_index', []):
            file_index = kwargs['file_index']
            num_files = len(file_index)
            logger.info(f'using {num_files} / {len(files)} for testing')
            files = [files[idx] for idx in file_index]

        num_files = len(files)
        self.num_files = num_files

        if not self.exist_cache(cache_dir_name) and not overwrite_cache:
            logger.info(f'Creating features from dataset file at {directory}')

            for i in range(num_files):
                self.examples = []
                arrs = self._get_tokenized_text(os.path.join(path, files[i]), tokenizer, **kwargs)

                logger.info(f'Saving features into cached file {cache_dir_name}')
                self.examples = np.concatenate(arrs)
                self.total_lines += len(self.examples)
                with open(os.path.join(cache_dir_name, str(i)), "wb") as handle:
                    start = i * len(self.examples) // num_files
                    end = (i + 1) * len(self.examples) // num_files if i < num_files - 1 else len(self.examples)
                    # np.savez_compressed(handle, arr=self.examples[start:end])
                    np.savez_compressed(handle, arr=self.examples)

    def exist_cache(self, cache_dir_name):
        for i in range(self.num_files):
            if not os.path.exists(os.path.join(cache_dir_name, str(i))):
                return False
        return True

    def __len__(self):
        if self.total_lines > 0:
            return self.total_lines
        for i in range(self.num_files):
            with open(os.path.join(self.cache_dir_name, str(i)), "rb") as handle:
                self.total_lines += len(np.load(handle)["arr"])
        return self.total_lines

    def __iter__(self):
        worker = torch.utils.data.get_worker_info()
        worker_id = worker.id if worker is not None else -1
        num_workers = worker.num_workers if worker is not None else 1

        file_indicies = list(range(self.num_files))
        random.shuffle(file_indicies)
        for i in file_indicies:
            if i % num_workers != worker_id and num_workers > 1:
                continue
            with open(os.path.join(self.cache_dir_name, str(i)), "rb") as handle:
                lines = np.load(handle)["arr"]
                np.random.shuffle(lines)
                for line in lines:
                    yield torch.tensor(line, dtype=torch.long)

    def _get_tokenized_text(self, path, tokenizer, cleaning=False, **kwargs) -> List[int]:
        file_path = path
        assert os.path.isfile(file_path)
        arrs = []

        if self.extention == 'gz':
            rf = gzip.open(path)
            json_str_list = rf.readlines()
        elif self.extention == 'jsonl':
            with open(path, 'r') as json_file:
                json_str_list = list(json_file)

        lines = []
        pool = Pool(processes=32)
        for json_str in json_str_list:
            json_str_list = loads(json_str)
            obj = next(json_str_list)

            lines.append(obj['text'])
            if len(lines) == 1000000:
                text = pool.map(partial(clean_and_tokenize, tokenizer=tokenizer), lines)
                text = [t for t in text if t]
                self._truncate_input(tokenizer, text, self.block_size)
                arrs.append(np.asarray(self.examples))
                self.examples = []
                lines = []

        if lines:
            text = pool.map(partial(clean_and_tokenize, tokenizer=tokenizer), lines)
            text = [t for t in text if t]
            self._truncate_input(tokenizer, text, self.block_size)
            arrs.append(np.asarray(self.examples))
            self.examples = []

        return arrs

    def _truncate_input(self, tokenizer, tokenized_texts, block_size):
        eos_id = tokenizer.convert_tokens_to_ids('</s>')
        block_size -= 1  # eos position
        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        for tokenized_text in tokenized_texts:
            for i in range(0, len(tokenized_text), block_size):  # Truncate in block of block_size
                if i > 0:
                    continue
                if i + block_size >= len(tokenized_text):
                    remain = len(tokenized_text) - i
                    padded = tokenized_text[i:] + [eos_id] + [pad_token_id] * (block_size - remain)
                    self.examples.append(padded)
                else:
                    example = tokenized_text[i:i + block_size] + [eos_id]
                    self.examples.append(example)

    @classmethod
    def cleaning(cls, line: str) -> str:
        line = line.replace("\n", "")
        line = line.replace("<|endoftext|>", "")
        line = line.replace(" ,", ",").replace(" .", ".").replace(" %", "%")
        line = line.replace(" - ", "-").replace(" : ", ":").replace(" / ", "/")
        line = line.replace("( ", "(").replace(" )", ")")
        line = line.replace("`` ", "\"").replace(" ''", "\"")
        line = line.replace(" 's", "'s").replace("s ' ", "s' ")
        punct_loc = TextDataset.get_last_punctuation(line)
        if punct_loc <= 20:
            return ""
        return line[:punct_loc + 1]

    @classmethod
    def get_last_punctuation(cls, text):
        for i, c in enumerate(text[::-1]):
            if c == "!" or c == "?" or c == ".":
                return len(text) - 1 - i
        return -1


def main():
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('gpt2')

    ds = C4Dataset(tokenizer, False, 'data/llama_pretrain_en2', 1024)
    from torch.utils.data import DataLoader

    train_data_loader = DataLoader(ds, batch_size=128, num_workers=1, drop_last=True)

    for batch in train_data_loader:
        print(batch)


if __name__ == '__main__':
    main()
