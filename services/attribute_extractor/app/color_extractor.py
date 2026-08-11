from __future__ import annotations

import io

import numpy as np
from PIL import Image


class ColorExtractor:
    """Extract a deterministic dominant palette using small-image k-means."""

    def extract(self, image_bytes: bytes, n: int = 5) -> dict:
        if not image_bytes:
            raise ValueError("image is empty")
        if n < 1:
            raise ValueError("n must be at least 1")
        with Image.open(io.BytesIO(image_bytes)) as source:
            image = source.convert("RGB").resize((150, 150))
        pixels = np.asarray(image, dtype=np.float32).reshape(-1, 3)
        centers = self._kmeans(pixels, n)
        palette = [
            {
                "hex": f"#{int(np.clip(color[0], 0, 255)):02x}{int(np.clip(color[1], 0, 255)):02x}{int(np.clip(color[2], 0, 255)):02x}",
                "rgb": {
                    "r": int(np.clip(color[0], 0, 255)),
                    "g": int(np.clip(color[1], 0, 255)),
                    "b": int(np.clip(color[2], 0, 255)),
                },
            }
            for color in centers
        ]
        return {
            "dominant": palette[0],
            "palette": palette,
            "mood": self._mood(palette[0]["hex"]),
        }

    @staticmethod
    def _kmeans(data: np.ndarray, k: int, iters: int = 20) -> np.ndarray:
        if data.ndim != 2 or data.shape[0] == 0:
            raise ValueError("image contains no pixels")
        k = min(max(int(k), 1), data.shape[0])

        # Evenly spaced deterministic seeds avoid a palette changing between
        # identical requests because of np.random's global state.
        seed_indices = np.linspace(0, data.shape[0] - 1, k, dtype=int)
        centers = data[seed_indices].copy()
        for _ in range(max(1, iters)):
            distances = np.linalg.norm(data[:, None, :] - centers[None, :, :], axis=2)
            labels = np.argmin(distances, axis=1)
            new_centers = np.array(
                [data[labels == index].mean(axis=0) if np.any(labels == index) else centers[index] for index in range(k)]
            )
            if np.allclose(centers, new_centers, atol=0.25):
                centers = new_centers
                break
            centers = new_centers

        counts = np.bincount(labels, minlength=k)
        # The old implementation treated whichever random cluster happened to
        # be first as dominant.  Sort by actual population, then color values
        # to make ties stable.
        order = sorted(range(k), key=lambda index: (-int(counts[index]), tuple(centers[index].tolist())))
        return centers[order]

    @staticmethod
    def _mood(hex_color: str) -> str:
        value = hex_color.lstrip("#")
        if len(value) != 6:
            return "balanced"
        red, green, blue = (int(value[index : index + 2], 16) for index in (0, 2, 4))
        brightness = (red * 299 + green * 587 + blue * 114) / 1000
        saturation = max(red, green, blue) - min(red, green, blue)
        if brightness > 200 and saturation < 30:
            return "neutral"
        if brightness < 60:
            return "dark"
        if blue > red and blue > green:
            return "cool"
        if red > green and red > blue:
            return "warm"
        return "balanced"
