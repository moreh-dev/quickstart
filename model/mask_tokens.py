from typing import Tuple

import torch
import numpy as np
import copy

BEGIN_NSG_TOKEN = "UNUSED4998"  #must be removed
END_NSG_TOKEN = "UNUSED4999"


def mask_tokens(inputs: torch.Tensor, tokenizer, corruption_rate: float, extra_id_start: int,
                num_extra_ids: int) -> Tuple[torch.Tensor, torch.Tensor]:
    labels = inputs.clone()

    probability_matrix = torch.full(labels.shape, corruption_rate)

    unk_token_id = tokenizer.convert_tokens_to_ids('<unk>')
    eos_token_id = tokenizer.convert_tokens_to_ids('</s>')

    special_token_mask = labels.eq(unk_token_id)
    special_token_mask += labels.eq(tokenizer.pad_token_id)
    special_token_mask += labels.eq(eos_token_id)

    probability_matrix.masked_fill_(special_token_mask, value=0.0)

    mask_inputs = torch.bernoulli(probability_matrix).bool()
    mask_inputs_sum = torch.sum(mask_inputs, 1)
    mask_labels_sum = mask_inputs_sum - (mask_inputs[:, 0] == 1).long() \
                                      + (mask_inputs[:, -1] != 1).long()
    mask_sum = torch.max(mask_inputs_sum, mask_labels_sum)
    exceeded = (mask_sum > num_extra_ids)

    for exceeded_idx in torch.nonzero(exceeded):
        # exceeded_idx is not a scalar tensor
        # but it's a 1D tensor with only one element
        indices = torch.nonzero(mask_inputs[exceeded_idx[0]])
        perm = torch.randperm(indices.size(0))
        num_to_remove = mask_sum[exceeded_idx] - num_extra_ids
        mask_inputs[exceeded_idx, indices[perm[:num_to_remove]]] = 0

    # preserve eos and pad token
    mask_labels = torch.logical_not(mask_inputs)

    def _mask(t, mask):
        # mask tokens
        t = torch.where(mask, extra_id_start, t)

        # remove consecutive mask
        # e.g. [A lion is <mask> <mask> in the <mask> river.] -> [A lion is <mask> in the <mask> river.]
        mask_rshifted = torch.roll(mask, 1, 1)
        mask_rshifted[:, 0].zero_()
        mask_removed = torch.logical_and(mask, mask_rshifted)
        offset = torch.cumsum(mask_removed, 1)
        idx = torch.arange(t.size(1))
        idx = idx - offset
        masked = torch.full(t.shape, tokenizer.pad_token_id)
        masked.scatter_(1, idx, t)

        # give each masks different token number
        # e.g.  [A lion is <mask_1> in the <mask_1> river.] -> [A lion is <mask_1> in the <mask_2> river.]
        is_masked = (masked == extra_id_start)
        is_masked_rshifted = torch.roll(is_masked, 1, 1)
        is_masked_rshifted[:, 0].zero_()
        is_masked_cumsum = torch.cumsum(is_masked_rshifted, 1)
        masked = torch.where(is_masked, masked + is_masked_cumsum, masked)

        return masked

    inputs = _mask(inputs, mask_inputs)
    labels = _mask(labels, mask_labels)

    labels_rshifted = torch.roll(labels, 1, 1)
    labels_rshifted[:, 0] = 1
    labels_eos = torch.logical_xor(labels_rshifted > 0, labels > 0)
    labels = labels + labels_eos * eos_token_id

    attention_mask = (inputs != tokenizer.pad_token_id)

    return inputs, labels, attention_mask


def scatter_numpy(self, dim, index, src):
    r"""
    Writes all values from the Tensor src into self at the indices specified in the index Tensor.
    Args:
        dim: The axis along which to index
        param index: The indices of elements to scatter
        param src: The source element(s) to scatter

    Return: 
        self
    """
    if index.dtype != np.dtype('int_'):
        raise TypeError("The values of index must be integers")
    if self.ndim != index.ndim:
        raise ValueError("Index should have the same number of dimensions as output")
    if dim >= self.ndim or dim < -self.ndim:
        raise IndexError("dim is out of range")
    if dim < 0:
        # Not sure why scatter should accept dim < 0, but that is the behavior in PyTorch's scatter
        dim = self.ndim + dim
    idx_xsection_shape = index.shape[:dim] + index.shape[dim + 1:]
    self_xsection_shape = self.shape[:dim] + self.shape[dim + 1:]
    if idx_xsection_shape != self_xsection_shape:
        raise ValueError("Except for dimension " + str(dim) +
                         ", all dimensions of index and output should be the same size")
    if (index >= self.shape[dim]).any() or (index < 0).any():
        raise IndexError("The values of index must be between 0 and (self.shape[dim] -1)")

    def make_slice(arr, dim, i):
        slc = [slice(None)] * arr.ndim
        slc[dim] = i
        return slc

    # We use index and dim parameters to create idx
    # idx is in a form that can be used as a NumPy advanced index for scattering of src param. in self
    idx = [[
        *np.indices(idx_xsection_shape).reshape(index.ndim - 1, -1), index[tuple(make_slice(index, dim,
                                                                                            i))].reshape(1, -1)[0]
    ] for i in range(index.shape[dim])]
    idx = list(np.concatenate(idx, axis=1))
    idx.insert(dim, idx.pop())

    if not np.isscalar(src):
        if index.shape[dim] > src.shape[dim]:
            raise IndexError("Dimension " + str(dim) + "of index can not be bigger than that of src ")
        src_xsection_shape = src.shape[:dim] + src.shape[dim + 1:]
        if idx_xsection_shape != src_xsection_shape:
            raise ValueError("Except for dimension " + str(dim) +
                             ", all dimensions of index and src should be the same size")
        # src_idx is a NumPy advanced index for indexing of elements in the src
        src_idx = list(idx)
        src_idx.pop(dim)
        src_idx.insert(dim, np.repeat(np.arange(index.shape[dim]), np.prod(idx_xsection_shape)))
        self[tuple(idx)] = src[tuple(src_idx)]

    else:
        self[idx] = src

    return self


def mask_tokens_numpy(inputs: np.ndarray, tokenizer, corruption_rate: float, extra_id_start: int,
                      num_extra_ids: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    labels = copy.deepcopy(inputs)

    probability_matrix = np.full(labels.shape, corruption_rate)

    unk_token_id = tokenizer.convert_tokens_to_ids('<unk>')
    eos_token_id = tokenizer.convert_tokens_to_ids('</s>')

    special_token_mask = np.equal(labels, unk_token_id)
    special_token_mask += np.equal(labels, tokenizer.pad_token_id)
    special_token_mask += np.equal(labels, eos_token_id)

    probability_matrix = np.ma.array(probability_matrix, mask=special_token_mask)
    probability_matrix = probability_matrix.filled(fill_value=0.0)

    mask_inputs = np.random.binomial(1., probability_matrix).astype(bool)
    mask_inputs_sum = np.sum(mask_inputs, 1)
    mask_labels_sum = mask_inputs_sum - (mask_inputs[:, 0] == 1).astype(np.int32) \
                                      + (mask_inputs[:, -1] != 1).astype(np.int32)
    mask_sum = np.maximum(mask_inputs_sum, mask_labels_sum)
    exceeded = (mask_sum > num_extra_ids)

    for exceeded_idx in np.nonzero(exceeded)[0]:
        # exceeded_idx is not a scalar tensor
        # but it's a 1D tensor with only one element
        indices = np.nonzero(mask_inputs[exceeded_idx])[0]
        perm = np.random.permutation(indices.shape[0])
        num_to_remove = mask_sum[exceeded_idx] - num_extra_ids
        mask_inputs[exceeded_idx, indices[perm[:num_to_remove]]] = 0

    # preserve eos and pad token
    mask_labels = np.logical_not(mask_inputs)

    def _mask(t, mask):
        # mask tokens
        t = np.where(mask, extra_id_start, t)

        # remove consecutive mask
        # e.g. [A lion is <mask> <mask> in the <mask> river.] -> [A lion is <mask> in the <mask> river.]
        mask_rshifted = np.roll(mask, 1, 1)
        mask_rshifted[:, 0] = 0
        mask_removed = np.logical_and(mask, mask_rshifted)
        offset = np.cumsum(mask_removed, 1)
        idx = np.arange(t.shape[1])
        idx = idx - offset
        masked = np.full(t.shape, tokenizer.pad_token_id)
        np.put_along_axis(masked, idx, t, 1)

        # give each masks different token number
        # e.g.  [A lion is <mask_1> in the <mask_1> river.] -> [A lion is <mask_1> in the <mask_2> river.]
        is_masked = (masked == extra_id_start)
        is_masked_rshifted = np.roll(is_masked, 1, 1)
        is_masked_rshifted[:, 0] = 0
        is_masked_cumsum = np.cumsum(is_masked_rshifted, 1)
        masked = np.where(is_masked, masked + is_masked_cumsum, masked)

        return masked

    inputs = _mask(inputs, mask_inputs)
    labels = _mask(labels, mask_labels)

    labels = put_eos_token(labels, eos_token_id=eos_token_id)

    return inputs, labels


def put_eos_token(inputs, eos_token_id: int = 1):
    inputs_rshifted = np.roll(inputs, 1, 1)
    inputs_rshifted[:, 0] = 1
    inputs_eos = np.logical_xor(inputs_rshifted > 0, inputs > 0)
    inputs = inputs + inputs_eos * eos_token_id
    return inputs


def filter_unused_tokens(inputs, tokenizer):
    kt_tokens = ["UNUSED4997", "UNUSED4998", "UNUSED4999"]
    kt_tokens = [tokenizer.convert_tokens_to_ids(x) for x in kt_tokens]
    pad_token_id = tokenizer.pad_token_id
    for i in range(inputs.shape[0]):
        delete_ids = np.where((inputs[i] == kt_tokens[0]) | (inputs[i] == kt_tokens[1]) | (inputs[i] == kt_tokens[2]))
        deleted = np.delete(inputs[i], delete_ids)
        inputs[i] = np.pad(deleted, (0, len(inputs[i]) - len(deleted)), "constant", constant_values=pad_token_id)
    return inputs


def preprocess_inputs(args, batch, extra_id_start, tokenizer):
    data = batch.numpy()

    data = filter_unused_tokens(data, tokenizer)
    inputs, labels = \
        mask_tokens_numpy(data, tokenizer, args.corruption_rate, extra_id_start, args.num_extra_ids)

    attention_mask = (inputs != tokenizer.pad_token_id)

    inputs, labels, attention_mask = torch.tensor(inputs), torch.tensor(labels), torch.tensor(attention_mask)

    labels[labels == tokenizer.pad_token_id] = -100
    return attention_mask, inputs, labels


if __name__ == '__main__':
    import numpy as np
    from transformers import AutoTokenizer
    from loguru import logger
    tokenizer = AutoTokenizer.from_pretrained("KETI-AIR/ke-t5-large")
    logger.info('start mask token')
    for i in range(100):
        print(i)
        #fake_input = np.random.randint(0, 10, size=(128, 512))
        test_input = tokenizer.batch_encode_plus([
            "딥러닝은 머신 러닝의 한 방법으로, 학습 과정 동안 인공 신경망으로서 예시 데이터에서 얻은 일반적인 규칙을 독립적으로 구축(훈련)합니다. 특히 머신 비전 분야에서 신경망은 일반적으로 데이터와 예제 데이터에 대한 사전 정의된 결과와 같은 지도 학습을 통해 학습됩니다.",
            "기계 학습(機械學習) 또는 머신 러닝(영어: machine learning)은 경험을 통해 자동으로 개선하는 컴퓨터 알고리즘의 연구이다. 인공지능의 한 분야로 간주된다. 컴퓨터가 학습할 수 있도록 하는 알고리즘과 기술을 개발하는 분야이다.",
            "감나무"
        ],
                                                 padding=True)
        test_input = test_input['input_ids']
        test_input = torch.tensor(test_input).numpy()
        inputs, labels = mask_tokens_numpy(test_input, tokenizer, 0.15, 64100, 60)
        #inputs, labels, attention_mask = mask_tokens(fake_input, tokenizer, 0.15, 64100, 60)
    logger.info('mask token finished')
