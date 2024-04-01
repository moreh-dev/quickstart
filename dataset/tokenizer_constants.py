from dataclasses import dataclass


@dataclass
class BaseTokenizerConstants:
    IGNORE_INDEX: int = -100


@dataclass
class LlavaConstants(BaseTokenizerConstants):
    IMAGE_TOKEN_INDEX: int = 32000
    DEFAULT_IMAGE_TOKEN: str = "<image>"
    DEFAULT_IMAGE_PATCH_TOKEN: str = "<im_patch>"
    DEFAULT_IM_START_TOKEN: str = "<im_start>"
    DEFAULT_IM_END_TOKEN: str = "<im_end>"
