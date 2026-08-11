"""Pillow compositor for item mockups."""

from __future__ import annotations

import io
import math
from typing import Any

from PIL import Image, ImageColor, ImageOps


class Compositor:
    def compose(
        self,
        layers: list[tuple[dict[str, Any], bytes]],
        width: int = 1200,
        height: int = 900,
        background: str = "#f7f4ef",
        remove_background: bool = False,
        output_format: str = "JPEG",
    ) -> tuple[bytes, dict[str, Any]]:
        if not layers:
            raise ValueError("at least one layer is required")
        if not (1 <= width <= 4000 and 1 <= height <= 4000):
            raise ValueError("canvas dimensions must be between 1 and 4000 pixels")

        try:
            background_rgb = ImageColor.getrgb(background)
        except (ValueError, TypeError) as exc:
            raise ValueError("background must be a valid color") from exc
        canvas = Image.new("RGBA", (width, height), (*background_rgb[:3], 255))
        placed = []
        cell_columns = max(1, math.ceil(math.sqrt(len(layers))))
        cell_width = max(1, width // cell_columns)
        cell_rows = math.ceil(len(layers) / cell_columns)
        cell_height = max(1, height // cell_rows)

        for index, (layer, data) in enumerate(layers):
            with Image.open(io.BytesIO(data)) as source:
                image = source.convert("RGBA")
            if remove_background:
                image = self._remove_background(image)

            bbox = layer.get("bbox") or {}
            position = layer.get("position") or {}
            x = self._number(position.get("x", bbox.get("x")), None)
            y = self._number(position.get("y", bbox.get("y")), None)
            target_width = self._number(position.get("width", bbox.get("width")), None)
            target_height = self._number(position.get("height", bbox.get("height")), None)

            if target_width is None or target_height is None:
                column = index % cell_columns
                row = index // cell_columns
                padding = max(8, min(cell_width, cell_height) // 12)
                max_width = max(1, cell_width - padding * 2)
                max_height = max(1, cell_height - padding * 2)
                resized = ImageOps.contain(image, (max_width, max_height), method=Image.Resampling.LANCZOS)
                x = column * cell_width + (cell_width - resized.width) // 2
                y = row * cell_height + (cell_height - resized.height) // 2
            else:
                target_width = max(1, min(width, int(target_width)))
                target_height = max(1, min(height, int(target_height)))
                resized = ImageOps.contain(
                    image,
                    (target_width, target_height),
                    method=Image.Resampling.LANCZOS,
                )
                x = int(x or 0) + (target_width - resized.width) // 2
                y = int(y or 0) + (target_height - resized.height) // 2

            x = max(0, min(width - resized.width, int(x or 0)))
            y = max(0, min(height - resized.height, int(y or 0)))
            canvas.alpha_composite(resized, (x, y))
            placed.append(
                {
                    "image_key": layer.get("image_key", ""),
                    "x": x,
                    "y": y,
                    "width": resized.width,
                    "height": resized.height,
                }
            )

        output = io.BytesIO()
        output_format = output_format.upper()
        if output_format == "PNG":
            canvas.save(output, format="PNG", optimize=True)
        else:
            canvas.convert("RGB").save(output, format="JPEG", quality=90, optimize=True)
        return output.getvalue(), {"width": width, "height": height, "layers": placed}

    @staticmethod
    def _number(value: Any, default: float | None) -> float | None:
        if value is None:
            return default
        try:
            number = float(value)
            return number if math.isfinite(number) else default
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _remove_background(image: Image.Image) -> Image.Image:
        try:
            from rembg import remove

            output = remove(image)
            return output.convert("RGBA") if isinstance(output, Image.Image) else Image.open(io.BytesIO(output)).convert("RGBA")
        except Exception:  # noqa: BLE001 - optional enhancement must be fail-open
            # Background removal is optional.  A regular image is preferable
            # to making the entire composition unavailable if its model is not
            # installed or downloaded.
            return image
