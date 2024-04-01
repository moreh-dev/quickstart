import base64
import io
from typing import List

import albumentations as A
import numpy as np
import torch
import torch.nn.functional as F
from albumentations.pytorch.transforms import ToTensorV2
from PIL import Image, ImageFile

IMAGE_FILE_EXTENSIONS = ('jpg', 'png', 'gif')


def img_to_tensor(path, img_size=(64, 64), transform=None):
    height, width = img_size
    img = Image.open(path)
    transform = (A.Compose([
        A.Resize(height=height, width=width),
        A.CenterCrop(height=height, width=width),
        ToTensorV2(),
    ]) if transform is None else transform)
    tensor = transform(img)["image"] if transform is None else transform(img)
    return tensor


def cast_uint8_images_to_float(images):
    if not images.dtype == torch.uint8:
        return images
    return images / 255


def normalize_neg_one_to_one(img):
    # image normalization functions
    # ddpms expect images to be in the range of -1 to 1
    return img * 2 - 1


def unnormalize_zero_to_one(normed_img):
    # image normalization functions
    # ddpms expect images to be in the range of -1 to 1
    return (normed_img + 1) * 0.5


def resize_image_to(image, target_image_size, clamp_range=None, mode='nearest'):
    orig_image_size = image.shape[-1]

    if orig_image_size == target_image_size:
        return image

    out = F.interpolate(image, target_image_size, mode=mode)

    if clamp_range is not None:
        out = out.clamp(*clamp_range)

    return out


def calc_all_frame_dims(downsample_factors: List[int], frames):
    if not (frames is not None):
        return (tuple(), ) * len(downsample_factors)

    all_frame_dims = []

    for divisor in downsample_factors:
        assert (frames % divisor) == 0
        all_frame_dims.append((frames // divisor, ))

    return all_frame_dims


def expand2square(pil_img: Image, background_color: tuple = (122, 116, 104)):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


def rectangle_img_to_square(img):
    if isinstance(img, np.ndarray):
        height, width, _ = img.shape

        if height != width:
            size = min(height, width)

            x_start = (width - size) // 2
            x_end = x_start + size
            y_start = (height - size) // 2
            y_end = y_start + size

            square_img = img[y_start:y_end, x_start:x_end]
            return square_img
        else:
            return img
    elif isinstance(img, ImageFile):
        width, height = img.size

        if height != width:
            size = min(height, width)

            left = (width - size) / 2
            top = (height - size) / 2
            right = (width + size) / 2
            bottom = (height + size) / 2

            # Crop the center of the image
            square_img = img.crop((left, top, right, bottom))
            return square_img
        else:
            return img
    else:
        raise NotImplementedError("image type must be numpy.ndarray and PIL.ImageFile")


def convert_image_to(img_type, image):
    if image.mode != img_type:
        return image.convert(img_type)
    return image


def pil_to_bytes(img: Image, format_: str):
    img_byte_array = io.BytesIO()
    img.save(img_byte_array, format=format_)
    img_byte_array = img_byte_array.getvalue()
    return base64.b64encode(img_byte_array).decode()


def bytes_to_pil(bytes_str):
    decoded_bytes = base64.b64decode(bytes_str)
    image = Image.open(io.BytesIO(decoded_bytes))
    return image
