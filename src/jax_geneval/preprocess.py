from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from PIL import Image, ImageOps


IMAGENET_MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
IMAGENET_STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)


@dataclass(frozen=True)
class ImageMeta:
    ori_shape: Tuple[int, int]
    img_shape: Tuple[int, int]
    pad_shape: Tuple[int, int]
    scale_factor: float


def resize_keep_ratio(image: Image.Image, target_hw: tuple[int, int] = (800, 800)) -> tuple[np.ndarray, float]:
    """Resize to a fixed square GenEval detector input.

    The official mmdet config resizes square 512x512 images to 800x800 before
    padding. This helper intentionally targets that fixed shape first; broader
    aspect-ratio support will keep the same static padded output.
    """

    target_h, target_w = target_hw
    width, height = image.size
    scale = min(target_w / width, target_h / height)
    new_w = int(round(width * scale))
    new_h = int(round(height * scale))
    image_arr = np.asarray(image, dtype=np.uint8)
    try:
        import cv2

        resized = cv2.resize(image_arr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    except ImportError:
        resized = np.asarray(image.resize((new_w, new_h), Image.BILINEAR), dtype=np.uint8)
    return resized, scale


def preprocess_image(path: str, target_hw: tuple[int, int] = (800, 800)) -> tuple[np.ndarray, ImageMeta]:
    image = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    ori_h, ori_w = image.height, image.width
    resized, scale = resize_keep_ratio(image, target_hw)
    target_h, target_w = target_hw
    arr = np.zeros((target_h, target_w, 3), dtype=np.float32)
    resized_arr = resized.astype(np.float32)
    img_h, img_w = resized_arr.shape[:2]
    arr[:img_h, :img_w] = resized_arr
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    meta = ImageMeta(
        ori_shape=(ori_h, ori_w),
        img_shape=(img_h, img_w),
        pad_shape=(target_h, target_w),
        scale_factor=scale,
    )
    return arr, meta
