from dataclasses import astuple, dataclass
import torch

from hub_model.utils.img_utils import (
    cast_uint8_images_to_float,
    normalize_neg_one_to_one,
    resize_image_to,
)
from hub_model.model.mask_tokens import preprocess_inputs
from hub_model.utils import create_attn_mask, mask_pads
from hub_model import namespace


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
    def text_to_image_batch_processing(cls, parsed_args, batch, tokenizer, *args, **kwargs):
        img, text_tokens = batch
        img = img.cuda()
        text_tokens = (text_tokens.cuda() if torch.is_tensor(text_tokens) else text_tokens)
        encoder = kwargs.pop("encoder", None)

        # text_tokens is missing
        if text_tokens is None:
            text_emb, attn_mask = None, None
        # text_tokens - simple text
        elif type(text_tokens[0]) == str:
            raise NotImplementedError("Raw text input is not supported. batch should include tokenized"
                                      "text or encoded text.")
        # text_tokens - simple class, long tensor
        elif len(text_tokens.shape) == 1:  # simple class
            text_emb = text_tokens
            attn_mask = None
        # text_tokens - tokenized text
        elif len(text_tokens.shape) == 2:
            assert encoder is not None, (
                "Given text input is tokenized text. "
                "To encode text with pre-trained models (e.g. T5, CLIP), 'encoder' should be specified")
            attn_mask = text_tokens != tokenizer.pad_token_id
            with torch.no_grad():
                text_emb = encoder.encode_text(token_ids=text_tokens, attn_mask=attn_mask)
        # text_tokens - encoded feature
        elif len(text_tokens.shape) == 3:
            text_emb = text_tokens
            text_emb = text_emb[:, :parsed_args.block_size, :]  # TODO: max seq len setting with the configuration
            with torch.no_grad():
                attn_mask = (text_emb.sum(-1) != tokenizer.pad_token_id).detach()
        else:
            raise IndexError("text in the batch is neither encoded text (bsz, seq_len, dim)"
                             " nor tokenized text (bsz, seq_len)")

        assert hasattr(parsed_args,
                       "image_size"), "args.image_size is not set up. check [dataset_factory.get_dataloader_kwargs]"

        if img.dtype == torch.uint8:
            img = cast_uint8_images_to_float(img)
        img = normalize_neg_one_to_one(img)
        img = resize_image_to(img, parsed_args.image_size)
        if text_emb is not None:
            text_emb = text_emb.float()

        return img, text_emb, attn_mask

    @classmethod
    def llava_batch_processing(cls, parsed_args, batch, tokenizer):
        input_ids, labels, attention_mask, pixel_values = (
            batch["input_ids"],
            batch["labels"],
            batch["attention_mask"],
            batch["images"],
        )
        batch_kwargs = {"pixel_values": pixel_values}
        return input_ids, labels, attention_mask, batch_kwargs

    @classmethod
    def contiguous_text_batch_processing(cls, parsed_args, batch, tokenizer, extra_id_start):
        attention_mask_cpu, inputs_cpu, labels_cpu = preprocess_inputs(parsed_args, batch, extra_id_start, tokenizer)
        attention_mask = attention_mask_cpu.contiguous()
        inputs = inputs_cpu.cuda()
        labels = labels_cpu.contiguous().cuda()
        return inputs, labels, attention_mask

    @classmethod
    def image_batch_processing(cls, parsed_args, batch):
        img = batch
        img.cuda()

        assert hasattr(parsed_args,
                       "image_size"), "args.image_size is not set up. check [dataset_factory.get_dataloader_kwargs]"

        if img.dtype == torch.uint8:
            img = cast_uint8_images_to_float(img)
        img = normalize_neg_one_to_one(img)
        img = resize_image_to(img, parsed_args.image_size)
        return img, None, None

    @classmethod
    def default_text_batch_processing(cls, parsed_args, batch, tokenizer):
        inputs, labels = (batch, mask_pads(batch, tokenizer))
        attention_mask = create_attn_mask(inputs, tokenizer).cuda()
        return inputs, labels, attention_mask
