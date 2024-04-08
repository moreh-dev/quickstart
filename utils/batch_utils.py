from dataclasses import astuple, dataclass
import torch

from model.mask_tokens import preprocess_inputs
from utils import create_attn_mask, mask_pads
import namespace


@dataclass
class BatchTensors:
    inputs: torch.Tensor
    labels: torch.Tensor
    attention_mask: torch.Tensor
    kwargs: dict = None

    def __init__(self, inputs: torch.Tensor, labels: torch.Tensor, attention_mask: torch.Tensor, kwargs: dict = {}):
        self.inputs = self.to_cuda(inputs)
        self.labels = self.to_cuda(labels)
        self.attention_mask = self.to_cuda(attention_mask)
        self.kwargs = kwargs
        for key in self.kwargs.keys():
            self.kwargs[key] = self.to_cuda(self.kwargs[key])

    def __iter__(self):
        return iter(astuple(self))

    def to_cuda(self, x: torch.Tensor):
        if x is not None:
            x = x.cuda()
        return x


class BatchProcessor:

    def __init__(self, *args, **kwargs):
        pass

    @classmethod
    def get_processed_batch(cls, model_name, parsed_args, batch, tokenizer, *args, **kwargs):
        ModelNames = namespace.ModelNames
        pixel_values = None
        batch_kwargs = {}
        if model_name in [ModelNames.LLAVA]:
            dataset_name = parsed_args.train_dataset_type
            if "llava" in dataset_name or dataset_name == 'blip_laion_cc_sbu_558k':
                inputs, labels, attention_mask, batch_kwargs = cls.llava_batch_processing(parsed_args, batch, tokenizer)
            else:
                raise NotImplementedError(f"{dataset_name} is not implemented for {model_name}.")
        elif model_name in [ModelNames.DIFFUSION]:  # Image generation models
            inputs, labels, attention_mask = cls.image_batch_processing(parsed_args, batch)
        else:
            inputs, labels, attention_mask = cls.default_text_batch_processing(parsed_args, batch, tokenizer)

        if pixel_values is not None:
            batch_kwargs['pixel_values'] = pixel_values

        batch_tensors = BatchTensors(inputs, labels, attention_mask, batch_kwargs)
        return batch_tensors

    
    @classmethod
    def contiguous_text_batch_processing(cls, parsed_args, batch, tokenizer, extra_id_start):
        attention_mask_cpu, inputs_cpu, labels_cpu = preprocess_inputs(parsed_args, batch, extra_id_start, tokenizer)
        attention_mask = attention_mask_cpu.contiguous()
        inputs = inputs_cpu.cuda()
        labels = labels_cpu.contiguous().cuda()
        return inputs, labels, attention_mask

    @classmethod
    def default_text_batch_processing(cls, parsed_args, batch, tokenizer):
        inputs, labels = (batch, mask_pads(batch, tokenizer))
        attention_mask = create_attn_mask(inputs, tokenizer).cuda()
        return inputs, labels, attention_mask
