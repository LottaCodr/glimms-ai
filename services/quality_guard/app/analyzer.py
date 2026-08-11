"""Cheap image-quality checks used before expensive model inference."""

from __future__ import annotations

import io
import os
from typing import Any

import numpy as np
from PIL import Image


class QualityAnalyzer:
    def __init__(self) -> None:
        self.blur_threshold = max(float(os.getenv("BLUR_THRESHOLD", "100")), 0.0)
        self.low_light_threshold = min(max(float(os.getenv("LOW_LIGHT_THRESHOLD", "35")), 0.0), 255.0)
        self.overexposure_threshold = min(max(float(os.getenv("OVEREXPOSURE_THRESHOLD", "245")), 0.0), 255.0)
        self.min_dimension = max(int(os.getenv("MIN_IMAGE_DIMENSION", "128")), 1)

    @staticmethod
    def _laplacian_variance(gray: np.ndarray) -> float:
        from scipy.ndimage import laplace

        return float(np.var(laplace(gray)))

    def analyze(self, image_bytes: bytes) -> dict[str, Any]:
        if not image_bytes:
            raise ValueError("image is empty")
        with Image.open(io.BytesIO(image_bytes)) as source:
            image = source.convert("RGB")
        width, height = image.size
        # A bounded analysis image keeps request cost predictable for large
        # phone photos while retaining enough detail for blur/exposure checks.
        image.thumbnail((1024, 1024))
        rgb = np.asarray(image, dtype=np.float32)
        gray = np.dot(rgb[..., :3], [0.299, 0.587, 0.114])
        blur_score = self._laplacian_variance(gray)
        brightness = float(np.mean(gray))
        contrast = float(np.std(gray))
        dark_fraction = float(np.mean(gray <= self.low_light_threshold))
        bright_fraction = float(np.mean(gray >= self.overexposure_threshold))

        issues: list[str] = []
        if min(width, height) < self.min_dimension:
            issues.append("resolution_low")
        if blur_score < self.blur_threshold:
            issues.append("blur")
        if brightness < self.low_light_threshold or dark_fraction >= 0.55:
            issues.append("low_light")
        if bright_fraction >= 0.55:
            issues.append("overexposed")
        if contrast < 12:
            issues.append("low_contrast")

        quality = 100.0
        if self.blur_threshold:
            quality -= min(40.0, max(0.0, 1.0 - blur_score / self.blur_threshold) * 40.0)
        if brightness < self.low_light_threshold:
            quality -= 25.0
        elif dark_fraction >= 0.55:
            quality -= 15.0
        if bright_fraction >= 0.55:
            quality -= 20.0
        if contrast < 12:
            quality -= min(15.0, (12.0 - contrast) * 1.25)
        if min(width, height) < self.min_dimension:
            quality -= 20.0

        guidance = self._guidance(issues)
        return {
            "width": width,
            "height": height,
            "blur_score": round(blur_score, 3),
            "brightness": round(brightness, 3),
            "contrast": round(contrast, 3),
            "quality_score": round(max(0.0, min(100.0, quality)), 2),
            "acceptable": not issues,
            "issues": issues,
            "guidance": guidance,
        }

    @staticmethod
    def _guidance(issues: list[str]) -> list[str]:
        guidance: list[str] = []
        if "resolution_low" in issues:
            guidance.append("Use a larger image and keep the subject fully inside the frame.")
        if "blur" in issues:
            guidance.append("Hold the camera steady, tap to focus, and take the photo in good light.")
        if "low_light" in issues:
            guidance.append("Move to brighter, even lighting and avoid strong backlighting.")
        if "overexposed" in issues:
            guidance.append("Reduce direct glare or exposure and avoid pointing at a bright window.")
        if "low_contrast" in issues:
            guidance.append("Use lighting with some contrast between the subject and its background.")
        return guidance or ["Image quality is suitable for processing."]
