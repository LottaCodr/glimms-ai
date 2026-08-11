"""CLIP image features with an opt-in, deterministic local fallback."""

from __future__ import annotations

import hashlib
import logging
import os
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

STYLE_TAGS = [
    "casual",
    "formal",
    "smart-casual",
    "athletic",
    "bohemian",
    "minimalist",
    "maximalist",
    "vintage",
    "streetwear",
    "luxury",
    "traditional",
    "modest",
]


class CLIPExtractor:
    """Extract normalized 512-dimensional image features.

    Loading a transformer at module import time made every API health check
    attempt a model download.  The real model is now lazy and opt-in through
    ``CLIP_ENABLED=true``.  The fallback is stable for the same bytes, which is
    important when those vectors are indexed or compared later.
    """

    dimension = 512

    def __init__(self) -> None:
        self.model: Any | None = None
        self.processor: Any | None = None
        self._torch: Any | None = None
        self._model_attempted = False
        self.enabled = os.getenv("CLIP_ENABLED", "false").strip().lower() in {"1", "true", "yes", "on"}
        self.model_name = os.getenv("CLIP_MODEL", "openai/clip-vit-base-patch32")

    def _ensure_model(self) -> bool:
        if self.model is not None and self.processor is not None:
            return True
        if self._model_attempted or not self.enabled:
            return False
        self._model_attempted = True
        try:
            import torch
            from transformers import CLIPModel, CLIPProcessor

            kwargs = {}
            if os.getenv("CLIP_LOCAL_ONLY", "false").lower() in {"1", "true", "yes", "on"}:
                kwargs["local_files_only"] = True
            self.model = CLIPModel.from_pretrained(self.model_name, **kwargs)
            self.processor = CLIPProcessor.from_pretrained(self.model_name, **kwargs)
            self.model.eval()
            self._torch = torch
            logger.info("CLIP model loaded: %s", self.model_name)
            return True
        except Exception as exc:  # noqa: BLE001 - model loading must not break health checks
            self.model = None
            self.processor = None
            logger.warning("CLIP unavailable (%s); using deterministic embeddings", exc)
            return False

    def embed(self, image_bytes: bytes) -> list[float]:
        if not image_bytes:
            raise ValueError("image is empty")
        if self._ensure_model():
            import io

            from PIL import Image

            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            inputs = self.processor(images=image, return_tensors="pt")
            with self._torch.no_grad():
                features = self.model.get_image_features(**inputs)
                features = features / features.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            return [float(value) for value in features[0].tolist()]
        return self._fallback_embedding(image_bytes)

    @classmethod
    def _fallback_embedding(cls, image_bytes: bytes) -> list[float]:
        # SHA-512 gives 64 stable bytes.  Repeating and centering the digest is
        # not a semantic substitute for CLIP, but it is deterministic and has
        # the same shape, so local pipelines remain testable without a model.
        digest = hashlib.sha512(image_bytes).digest()
        raw = np.frombuffer((digest * ((cls.dimension // len(digest)) + 1))[: cls.dimension], dtype=np.uint8)
        values = raw.astype(np.float32) / 127.5 - 1.0
        values /= max(float(np.linalg.norm(values)), 1e-12)
        return [float(value) for value in values]

    def get_style_tags(self, embedding: list[float]) -> list[str]:
        if not embedding:
            return []
        if self._ensure_model():
            import torch

            texts = [f"a {tag} outfit" for tag in STYLE_TAGS]
            inputs = self.processor(text=texts, return_tensors="pt", padding=True)
            with torch.no_grad():
                text_features = self.model.get_text_features(**inputs)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            vector = torch.tensor(embedding, dtype=text_features.dtype).unsqueeze(0)
            scores = (vector @ text_features.T).squeeze(0)
            return [STYLE_TAGS[index] for index in scores.topk(min(3, len(STYLE_TAGS))).indices.tolist()]

        vector = np.asarray(embedding, dtype=np.float32)
        scores = []
        for tag in STYLE_TAGS:
            prototype = np.frombuffer(
                (hashlib.sha512(tag.encode("utf-8")).digest() * 9)[: self.dimension],
                dtype=np.uint8,
            ).astype(np.float32)
            prototype = prototype / 127.5 - 1.0
            prototype /= max(float(np.linalg.norm(prototype)), 1e-12)
            scores.append(float(np.dot(vector, prototype)))
        return [STYLE_TAGS[index] for index in np.argsort(scores)[::-1][:3]]
