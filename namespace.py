from transformers.utils import ExplicitEnum


class ModelNames(ExplicitEnum):
    r"""Stores the acceptable string identifiers for models."""
    LLAMA2 = "llama2"
    LLAVA = "llava"
    QWEN = "qwen"
    DIFFUSION = "diffusion"
    GPT = "gpt"
    MISTRAL = "mistral"


class OptimizerNames(ExplicitEnum):
    r"""Stores the acceptable string identifiers for optimizers."""
    ADAM = "Adam"
    ADAMW = "AdamW"
    ADAFACTOR = "Adafactor"
    MOREH_ADAFACTOR = "MorehAdafactor"
    SOPHIA = "Sophia"


class SchedulerNames(ExplicitEnum):
    r"""Stores the acceptable string identifiers for models."""
    Static = "Static"
    InverseSquareRoot = "InverseSquareRoot"
    LinearWarmupPolyDecay = "LinearWarmupPolyDecay"
    LinearWarmup = "LinearWarmup"
    CosineWarmup = "CosineWarmup"


class MetricNames(ExplicitEnum):
    r"""Stores name for metric"""
    Perplexity = "Perplexity"
    Accuracy = "Accuracy"
    VQAAccuracy = "VQAAccuracy"
    MeanLoss = "MeanLoss"
    DivLoss = "DivLoss"
