"""
轻量级 lazy_image shim
为 deepdoc/vision/operators.py 提供 ensure_pil_image 接口
"""
from PIL import Image
import io


def ensure_pil_image(img):
    """
    Ensure the input is a PIL Image object,
    but if it's already a numpy array, return it as-is so we don't destroy float32 tensors.
    """
    import numpy as np
    if isinstance(img, np.ndarray):
        return img
    elif isinstance(img, Image.Image):
        return img
    elif isinstance(img, bytes):
        return Image.open(io.BytesIO(img)).convert("RGB")
    else:
        raise ValueError(f"Cannot convert type {type(img)}")


def open_image_for_processing(img):
    return ensure_pil_image(img)


def is_image_like(img):
    return isinstance(img, (Image.Image, bytes))
