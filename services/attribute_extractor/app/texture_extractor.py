from __future__ import annotations

import io

import numpy as np
from PIL import Image


class TextureExtractor:
    def extract(self, image_bytes: bytes) -> dict:
        if not image_bytes:
            raise ValueError("image is empty")
        with Image.open(io.BytesIO(image_bytes)) as source:
            image = source.convert("L").resize((128, 128))
        pixels = np.asarray(image, dtype=np.float32)
        variance = float(np.var(pixels))
        edges = self._sobel(pixels)
        density = float(np.mean(edges > 50))
        return {
            "roughness": "smooth" if variance < 500 else "medium" if variance < 2000 else "rough",
            "pattern": "solid" if density < 0.1 else "subtle" if density < 0.25 else "patterned",
            "edge_density": round(density, 3),
            "contrast": round(float(np.std(pixels)), 2),
        }

    @staticmethod
    def _sobel(image: np.ndarray) -> np.ndarray:
        from scipy.ndimage import sobel

        return np.hypot(sobel(image, axis=1), sobel(image, axis=0))
