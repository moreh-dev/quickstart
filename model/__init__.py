r"""Model initialization classes are defined in this source."""

import json
from abc import ABC, abstractmethod

import torch
from loguru import logger

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM
from model.modeling_gpt import GPTLMHeadModel
from model.configuration_gpt import GPTConfig


from namespace import ModelNames

try:
    moreh_ops = torch.ops.moreh

except:
    moreh_ops = None


def args_getattr(args, attr, default=None):
    if getattr(args, attr, None) is None:
        logger.info(f"args does not have {attr}, set to {default}")

    attr = getattr(args, attr, default)
    return attr


class ModelInitializer(ABC):
    r"""Abstract base class for initializing models.

    Methods:
        initialize(args, config_path, bfloat16=False, **kwargs) -> Model:
            Initializes and returns a model based on the provided arguments and configuration.

        get_model(config, **kwargs) -> Model:
            Abstract method to get the model based on the given configuration and additional keyword arguments.

        get_config(config_path) -> Configuration:
            Abstract method to retrieve a model configuration from a specified path.

        add_additional_config(args, config, **kwargs) -> Configuration:
            Abstract method to add additional configurations to the model's base configuration.

    Note:
        - This class defines the interface for initializing models with various configurations.
        - Inherit from this class to implement a specific model initialization process.

    Examples:
        class MyModelInitializer(ModelInitializer):
            def get_model(self, config, **kwargs):
                # Implementation for getting the model based on the configuration.
                # ...
                return model

            def get_config(self, config_path):
                # Implementation for retrieving the configuration from the specified path.
                # ...
                return config

            def add_additional_config(self, args, config, **kwargs):
                # Implementation for adding additional configurations to the base configuration.
                # ...
                return config

        # Instantiate the custom model initializer
        my_initializer = MyModelInitializer()
        # Initialize the model with specific arguments and configuration
        model = my_initializer.initialize(args, config_path, bfloat16=True)
    """

    def initialize(self, args, **kwargs):
        r"""Construct model from json file"""
        config = self.get_config(args.model_config_path)

        config.use_moreh_attention = args_getattr(args, "use_moreh_attention", False)
        config.loss_reduction = args_getattr(args, "loss_reduction", 'mean')
        config.tensor_parallel = args_getattr(args, "use_tensor_parallel", False)
        config.use_advanced_parallelization = args_getattr(args, "use_advanced_parallelization", False)

        config = self.add_additional_config(args, config, **kwargs)

        if args.dropout > 0:
            config.dropout_rate = args.dropout

        model = self.get_model(config)

        return model, config

    @abstractmethod
    def get_model(self, config, **kwargs):
        pass

    @abstractmethod
    def get_config(self, config_path):
        pass

    @abstractmethod
    def add_additional_config(self, args, config, **kwargs):
        r"""
        additional configuration can be added in this method.
        For instance, LM configurations such as eos_token_id can be added in here.
        """

        return config


class LLMInitializer(ModelInitializer):

    def add_additional_config(self, args, config, **kwargs):
        r"""
        additional configurations for language models
        """

        self.tokenizer_vocab_size = kwargs.pop('vocab_size', config.vocab_size)
        config.pad_token_id = kwargs.pop('pad_token_id', config.pad_token_id)
        config.eos_token_id = kwargs.pop('eos_token_id', config.eos_token_id)
        config.decoder_start_token_id = kwargs.pop('decoder_start_token_id', config.decoder_start_token_id)

        assert config.vocab_size is not None and\
            config.pad_token_id is not None and\
            config.eos_token_id is not None and\
            config.decoder_start_token_id is not None,\
            f"vocab_size = {config.vocab_size}, \
            pad_token_id= {config.pad_token_id}, \
            eos_token_id= {config.eos_token_id}, \
            decoder_start_token_id= {config.decoder_start_token_id} must be provided for LLMs."

        config.encoder_split_layers = args.encoder_split_layers if args.use_pipeline else []
        config.decoder_split_layers = args.decoder_split_layers if args.use_pipeline else []
        config.absolute_position_embedding = getattr(config, "absolute_position_embedding", False)

        # add extra arguments specified in kwargs
        for key, value in kwargs.items():
            logger.info(f'setting extra config {key}: {value}')
            setattr(config, key, value)

        return config

    def initialize(self, args, **kwargs):
        model, config = super().initialize(args, **kwargs)

        model_vocab_size = model.config.vocab_size
        if self.tokenizer_vocab_size > model_vocab_size:
            logger.warning(
                f'Vocab size mismatch between model ({model_vocab_size}) and tokenizer ({self.tokenizer_vocab_size}).')
            logger.warning(f'Setting model embedding from {model_vocab_size} to {self.tokenizer_vocab_size}')
            model.resize_token_embeddings(self.tokenizer_vocab_size)

        return model, config


class LLama2Initializer(LLMInitializer):

    def get_model(self, config):
        if config.load_pretrained:
            return Llama2.from_pretrained(config.load_pretrained, config=config)
        else:
            return Llama2(config)

    def get_config(self, config_path):
        with open(config_path) as f:
            model_config_dict = json.load(f)
        return Llama2Config(**model_config_dict)

    def add_additional_config(self, args, config, **kwargs):
        config = super().add_additional_config(args, config, **kwargs)
        config.use_pipeline = args.use_pipeline
        config.bfloat16 = args.bfloat16
        config.load_pretrained = args.load_pretrained
        return config


class QWENInitializer(LLMInitializer):

    def get_model(self, config):
        if config.load_pretrained:
            return Qwen.from_pretrained(config.load_pretrained, config=config)
        else:
            return Qwen(config)

    def get_config(self, config_path):
        with open(config_path) as f:
            model_config_dict = json.load(f)
        return QWenConfig(**model_config_dict)

    def add_additional_config(self, args, config, **kwargs):
        config = super().add_additional_config(args, config, **kwargs)
        config.use_pipeline = args.use_pipeline
        config.bfloat16 = args.bfloat16
        config.load_pretrained = args.load_pretrained
        return config


class GPTInitializer(LLMInitializer):

    def get_model(self, config, **kwargs):
        model = GPTLMHeadModel(config)
        return model

    def get_config(self, config_path):
        return GPTConfig.from_pretrained(config_path)

    def add_additional_config(self, args, config, **kwargs):
        config = super().add_additional_config(args, config, **kwargs)
        config.use_pipeline = args.use_pipeline
        config.bfloat16 = args.bfloat16
        return config


class MistralInitializer(LLMInitializer):

    def get_model(self, config):
        if config.load_pretrained:
            return MistralForCausalLM.from_pretrained(config.load_pretrained, config=config)
        else:
            return MistralForCausalLM(config)

    def get_config(self, config_path):
        with open(config_path) as f:
            model_config_dict = json.load(f)
        return MistralConfig(**model_config_dict)

    def add_additional_config(self, args, config, **kwargs):
        config = super().add_additional_config(args, config, **kwargs)
        config.use_pipeline = args.use_pipeline
        config.bfloat16 = args.bfloat16
        config.load_pretrained = args.load_pretrained
        return config


class DiffusionInitializer(ModelInitializer):

    def get_model(self, config, **kwargs):
        return DiffusionModel(config)

    def get_config(self, config_path):
        return TextToImageModelsConfig.from_json_file(config_path)

    def add_additional_config(self, args, config, **kwargs):
        return config


class LlavaInitializer(LLMInitializer):

    def get_model(self, config, *args, **kwargs):
        model = Llava(config)

        if not (config.load_checkpoint or config.init_model):
            # Get HuggingFace repo id
            vision_repo_id = config.vision_config._name_or_path
            text_repo_id = config.text_config._name_or_path

            # Load pretrained vision model
            if vision_repo_id:
                logger.info(f"Loading pretrained weight for LLaVA vision model from {vision_repo_id}")
                model.vision_tower = AutoModel.from_pretrained(vision_repo_id, config=config.vision_config)
                logger.success("Loading complete")

            # Load pretrained text model
            if text_repo_id:
                logger.info(f"Loading pretrained weight for LLaVA text model from {text_repo_id}")
                model.language_model = AutoModelForCausalLM.from_pretrained(text_repo_id, config=config.text_config)
                logger.success("Loading complete")

        # Freeze layers
        model.vision_tower.requires_grad_(False)
        if config.freeze_text_model:
            model.language_model.requires_grad_(False)

        # Cast model to bfloat16
        if config.bfloat16:
            model.to(torch.bfloat16)

        return model

    def get_config(self, config_path):
        return AutoConfig.from_pretrained(config_path)

    def add_additional_config(self, args, config, **kwargs):
        config = super().add_additional_config(args, config, **kwargs)
        config.bfloat16 = args.bfloat16
        config.load_checkpoint = args.load_checkpoint
        config.init_model = args.init_model
        config.freeze_text_model = args.freeze_text_model
        config.tokenizer_model_max_length = args.tokenizer_max_length
        config.use_pipeline = args.use_pipeline
        config.text_config.use_pipeline = args.use_pipeline
        config.text_config.decoder_split_layers = args.decoder_split_layers if args.use_pipeline else []
        return config


MODEL_INITIALIZER = {
    ModelNames.LLAMA2: LLama2Initializer,
    ModelNames.LLAVA: LlavaInitializer,
    ModelNames.QWEN: QWENInitializer,
    ModelNames.DIFFUSION: DiffusionInitializer,
    ModelNames.GPT: GPTInitializer,
    ModelNames.MISTRAL: MistralInitializer,
}


def get_initializer(model_type):
    try:
        model_initializer = MODEL_INITIALIZER[model_type]
    except:
        raise ValueError(f"not a valid model_type {model_type}")

    return model_initializer()


def init_model(args, tokenizer):
    r"""Original initialization code for train_t5.py"""
    initializer = get_initializer(args.model_type)

    return initializer.initialize(
        args,
        vocab_size=len(tokenizer),
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        decoder_start_token_id=tokenizer.pad_token_id,
    )
