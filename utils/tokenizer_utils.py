from loguru import logger
from transformers import AutoTokenizer


def maybe_add_special_tokens(tokenizer):
    r"""
    add required special tokens if none
    """
    added_tokens = []
    if tokenizer.pad_token_id is None:
        token_dict = {'pad_token': '<pad>'}
        tokenizer.add_special_tokens(token_dict)
        added_tokens.append(token_dict)

    if tokenizer.eos_token_id is None:
        token_dict = {'eos_token': '<endoftext>'}
        tokenizer.add_special_tokens(token_dict)
        added_tokens.append(token_dict)

    for tokens in added_tokens:
        token_names = tokens.keys()
        for token_name in token_names:
            tokenizer_key = f'{token_name}_id'
            logger.info(f"added {token_name}: {getattr(tokenizer, tokenizer_key)}")

    return tokenizer


def get_tokenizer(
    tokenizer_name_or_path,
    fixed_vocab=False,
    update_tokenizer=False,
    model_max_length=None,
    use_fast=False,
    num_extra_ids=0,
):

    msg = "Using huggingface tokenizer"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, use_fast=use_fast, trust_remote_code=True)
    if update_tokenizer:
        if fixed_vocab:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        else:
            msg = "Using modified huggingface tokenizer"
            tokenizer = maybe_add_special_tokens(tokenizer)

    if num_extra_ids:
        tokenizer = add_extra_ids_to_tokenizer(tokenizer, num_extra_ids)

    if model_max_length is not None:
        tokenizer.model_max_length = model_max_length

    logger.info(f'{msg} {tokenizer_name_or_path}')

    return tokenizer


def add_extra_ids_to_tokenizer(tokenizer, num_extra_ids):
    sentinels = [f'<extra_{i}>' for i in range(num_extra_ids)]
    tokenizer.add_tokens(sentinels)
    return tokenizer
