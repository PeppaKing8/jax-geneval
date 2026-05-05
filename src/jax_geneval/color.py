from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Sequence

import jax.numpy as jnp
import numpy as np
from PIL import Image

from .evaluation import COLORS


class JaxClipColorClassifier:
    """Lazy JAX CLIP zero-shot color classifier compatible with GenEval."""

    def __init__(
        self,
        *,
        model_name: str = "ViT-L/14",
        batch_size: int = 16,
        repo_root: str | None = None,
        bgcolor: str = "#999",
        crop: bool = True,
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.bgcolor = bgcolor
        self.crop = crop
        clip_repo = repo_root or os.environ.get("JAX_CLIP_REPO")
        if clip_repo:
            clip_path = Path(clip_repo).expanduser().resolve()
            if str(clip_path) not in sys.path:
                sys.path.insert(0, str(clip_path))
        try:
            from jclip.clip import CLIPTokenizer, create_clip_encode_fn
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "Could not import `jclip`. Install the JAX CLIP port on PYTHONPATH, "
                "or pass --clip-repo /path/to/legacy/jax-clip, or set JAX_CLIP_REPO."
            ) from exc

        self.image_encode_fn, self.image_params, _, self.preprocess = create_clip_encode_fn(
            model_name,
            modality="image",
        )
        self.text_encode_fn, self.text_params, _ = create_clip_encode_fn(
            model_name,
            modality="text",
        )
        self.tokenizer = CLIPTokenizer(context_length=77)
        self._classifiers: dict[str, np.ndarray] = {}

    def _classifier(self, classname: str) -> np.ndarray:
        if classname in self._classifiers:
            return self._classifiers[classname]
        templates = [
            f"a photo of a {{c}} {classname}",
            f"a photo of a {{c}}-colored {classname}",
            "a photo of a {c} object",
        ]
        texts = [template.format(c=color) for color in COLORS for template in templates]
        tokens = self.tokenizer.tokenize_batch(texts).astype(np.int32)
        features = np.asarray(self.text_encode_fn(self.text_params, jnp.asarray(tokens)))
        features = features.reshape(len(COLORS), len(templates), -1)
        features = features / np.linalg.norm(features, axis=-1, keepdims=True)
        features = features.mean(axis=1)
        features = features / np.linalg.norm(features, axis=-1, keepdims=True)
        self._classifiers[classname] = features.astype(np.float32)
        return self._classifiers[classname]

    def _crop_to_clip_input(
        self,
        image: Image.Image,
        obj: tuple[np.ndarray, np.ndarray | None],
    ) -> np.ndarray:
        box, mask = obj
        image = image.convert("RGB")
        if mask is not None:
            if self.bgcolor == "original":
                blank = image.copy()
            else:
                blank = Image.new("RGB", image.size, color=self.bgcolor)
            image = Image.composite(image, blank, Image.fromarray(mask))
        if self.crop:
            image = image.crop(tuple(box[:4]))
        tensor = self.preprocess(image)
        arr = tensor.detach().cpu().numpy().transpose(1, 2, 0)
        return arr.astype(np.float32)

    def __call__(
        self,
        image: Image.Image,
        objects: Sequence[tuple[np.ndarray, np.ndarray | None]],
        classname: str,
    ) -> list[str]:
        classifier = self._classifier(classname)
        batches = []
        for start in range(0, len(objects), self.batch_size):
            chunk = objects[start : start + self.batch_size]
            arr = np.stack([self._crop_to_clip_input(image, obj) for obj in chunk], axis=0)
            if arr.shape[0] < self.batch_size:
                pad = np.repeat(arr[-1:], self.batch_size - arr.shape[0], axis=0)
                arr = np.concatenate([arr, pad], axis=0)
            features = np.asarray(self.image_encode_fn(self.image_params, jnp.asarray(arr)))
            batches.append(features[: len(chunk)])
        image_features = np.concatenate(batches, axis=0)
        image_features = image_features / np.linalg.norm(image_features, axis=-1, keepdims=True)
        logits = image_features @ classifier.T
        return [COLORS[int(index)] for index in logits.argmax(axis=1)]
