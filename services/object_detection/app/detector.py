"""Image object detection with a safe, deterministic fallback.

A custom YOLOv8 ONNX model can be enabled with ``MODEL_PATH``.  The fallback is
only intended to keep the gateway usable in development when the model is not
mounted; it is deterministic and is clearly reported in logs rather than
returning a different answer for every request.
"""

from __future__ import annotations

import hashlib
import io
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps

logger = logging.getLogger(__name__)

LABELS = {
    "wardrobe": [
        "shirt",
        "pants",
        "dress",
        "jacket",
        "shoes",
        "bag",
        "hat",
        "skirt",
        "coat",
        "sweater",
        "jeans",
        "shorts",
        "blazer",
        "suit",
    ],
    "room": [
        "chair",
        "sofa",
        "table",
        "desk",
        "lamp",
        "shelf",
        "bed",
        "cabinet",
        "rug",
        "curtain",
        "pillow",
        "bookcase",
        "wardrobe",
        "mirror",
    ],
    "garden": [
        "plant",
        "flower",
        "tree",
        "shrub",
        "grass",
        "succulent",
        "herb",
        "pot",
        "fence",
        "path",
    ],
}

CATEGORIES = {
    "shirt": "top",
    "jacket": "top",
    "sweater": "top",
    "coat": "top",
    "dress": "full",
    "pants": "bottom",
    "jeans": "bottom",
    "shorts": "bottom",
    "skirt": "bottom",
    "shoes": "footwear",
    "bag": "accessory",
    "hat": "accessory",
    "blazer": "top",
    "suit": "full",
}

# The standard COCO names are useful when a stock YOLO model is mounted.  A
# domain-specific model should set MODEL_LABELS to its labels instead.
COCO_LABELS = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]

COCO_TO_DOMAIN = {
    "couch": "sofa",
    "dining table": "table",
    "potted plant": "plant",
    "backpack": "bag",
    "handbag": "bag",
    "tie": "accessory",
    "suitcase": "bag",
}


class Detector:
    def __init__(self) -> None:
        try:
            configured_threshold = float(os.getenv("CONFIDENCE_THRESHOLD", "0.5"))
        except ValueError:
            configured_threshold = 0.5
        self.threshold = min(max(configured_threshold, 0.0), 1.0)
        self.model = self._load_model()
        configured_labels = os.getenv("MODEL_LABELS", "").strip()
        self.model_labels = [label.strip() for label in configured_labels.split(",") if label.strip()]
        if not self.model_labels:
            self.model_labels = COCO_LABELS
        try:
            configured_input_size = int(os.getenv("MODEL_INPUT_SIZE", "640"))
        except ValueError:
            configured_input_size = 640
        self.input_size = max(configured_input_size, 32)
        try:
            configured_nms = float(os.getenv("NMS_IOU_THRESHOLD", "0.45"))
        except ValueError:
            configured_nms = 0.45
        self.nms_threshold = min(max(configured_nms, 0.0), 1.0)

    def _load_model(self) -> Any | None:
        path_value = os.getenv("MODEL_PATH", "models/yolov8n.onnx").strip()
        if not path_value:
            logger.info("MODEL_PATH is empty; using deterministic development detector")
            return None

        path = Path(path_value)
        if not path.is_absolute() and not path.exists():
            # Docker runs from /app while local development usually runs from
            # the repository root.  Try the service directory as well.
            path = Path(__file__).resolve().parents[1] / path_value
        if not path.exists():
            logger.warning("No detector model found at %s; using development fallback", path)
            return None

        try:
            import onnxruntime as ort

            logger.info("Loading ONNX detector from %s", path)
            session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
            if not session.get_inputs():
                raise ValueError("ONNX detector has no inputs")
            return session
        except Exception as exc:  # noqa: BLE001 - model errors must not prevent health checks
            logger.warning("ONNX detector could not be loaded: %s", exc)
            return None

    def detect(self, image_bytes: bytes, vertical: str, image_key: str) -> list[dict]:
        if vertical not in LABELS:
            raise ValueError(f"unsupported vertical: {vertical}")
        image = self._to_numpy(image_bytes)
        if self.model is not None:
            return self._infer(image, LABELS[vertical], image_key)
        return self._mock(LABELS[vertical], image_key, image.shape[:2], image_bytes)

    def _to_numpy(self, data: bytes) -> np.ndarray:
        if not data:
            raise ValueError("image is empty")
        with Image.open(io.BytesIO(data)) as source:
            image = self._auto_rotate(source).convert("RGB")
            return np.asarray(image, dtype=np.uint8)

    @staticmethod
    def _auto_rotate(image: Image.Image) -> Image.Image:
        """Apply EXIF orientation, including mirrored orientations."""

        try:
            return ImageOps.exif_transpose(image)
        except Exception:  # noqa: BLE001 - malformed EXIF should not reject valid pixels
            # Some malformed images expose invalid EXIF.  The pixels are still
            # useful, so keep the original rather than failing the request.
            return image

    def _infer(self, image: np.ndarray, valid_labels: list[str], image_key: str) -> list[dict]:
        """Run common YOLOv8 ONNX output layouts and clamp boxes to the image."""

        height, width = image.shape[:2]
        pil_image = Image.fromarray(image, mode="RGB").resize((self.input_size, self.input_size))
        tensor = np.asarray(pil_image, dtype=np.float32) / 255.0
        tensor = tensor.transpose(2, 0, 1)[np.newaxis, ...]
        input_name = self.model.get_inputs()[0].name
        outputs = self.model.run(None, {input_name: tensor})
        if not outputs:
            return []

        predictions = np.asarray(outputs[0])
        if predictions.ndim == 3:
            predictions = predictions[0]
        if predictions.ndim == 1:
            predictions = predictions.reshape(1, -1)
        if predictions.ndim != 2:
            logger.warning("Unsupported detector output shape: %s", predictions.shape)
            return []
        # YOLOv8 commonly returns [84, 8400], while some exporters return the
        # transposed [8400, 84] layout.
        if (
            6 <= predictions.shape[0] <= 256
            and (predictions.shape[0] < predictions.shape[1] or predictions.shape[1] < 6)
        ):
            predictions = predictions.T

        results: list[dict] = []
        for row in predictions:
            parsed = self._parse_prediction(row, width, height, valid_labels)
            if parsed is None:
                continue
            confidence, label, box = parsed
            if confidence < self.threshold:
                continue
            results.append(
                {
                    "label": label,
                    "confidence": round(float(confidence), 3),
                    "bbox": box,
                    "category": CATEGORIES.get(label, "item"),
                    "image_key": image_key,
                }
            )
        return self._nms(results)

    def _nms(self, detections: list[dict]) -> list[dict]:
        """Suppress overlapping boxes of the same class."""

        kept: list[dict] = []
        for label in sorted({str(item.get("label")) for item in detections}):
            candidates = sorted(
                (item for item in detections if item.get("label") == label),
                key=lambda item: (-float(item["confidence"]), str(item["image_key"]), item["bbox"]["x"]),
            )
            while candidates:
                best = candidates.pop(0)
                kept.append(best)
                candidates = [
                    candidate
                    for candidate in candidates
                    if self._iou(best["bbox"], candidate["bbox"]) <= self.nms_threshold
                ]
        return sorted(kept, key=lambda item: (-float(item["confidence"]), item["label"]))

    @staticmethod
    def _iou(first: dict, second: dict) -> float:
        left = max(first["x"], second["x"])
        top = max(first["y"], second["y"])
        right = min(first["x"] + first["width"], second["x"] + second["width"])
        bottom = min(first["y"] + first["height"], second["y"] + second["height"])
        intersection = max(0, right - left) * max(0, bottom - top)
        first_area = max(0, first["width"]) * max(0, first["height"])
        second_area = max(0, second["width"]) * max(0, second["height"])
        union = first_area + second_area - intersection
        return intersection / union if union else 0.0

    def _parse_prediction(
        self,
        row: np.ndarray,
        width: int,
        height: int,
        valid_labels: list[str],
    ) -> tuple[float, str, dict] | None:
        if row.size < 6:
            return None

        # Six-column exports are normally x1,y1,x2,y2,confidence,class_id.
        direct_xyxy = row.size == 6
        if direct_xyxy:
            confidence = float(row[4])
            class_index = int(row[5])
            coords = row[:4]
        else:
            scores = row[4:]
            if row.size == 85:  # YOLOv5-style objectness + 80 class scores
                scores = row[5:] * row[4]
            class_index = int(np.argmax(scores))
            confidence = float(scores[class_index])
            coords = row[:4]

        if not np.isfinite(confidence) or confidence <= 0:
            return None
        label = self._label_for_class(class_index, valid_labels)
        if label is None:
            return None

        if direct_xyxy:
            x1, y1, x2, y2 = [float(value) for value in coords]
        else:
            cx, cy, box_width, box_height = [float(value) for value in coords]
            x1, y1 = cx - box_width / 2, cy - box_height / 2
            x2, y2 = cx + box_width / 2, cy + box_height / 2

        # Exporters differ on whether coordinates are normalized or in input
        # pixels.  Detect normalized values and scale both forms correctly.
        max_coordinate = max(abs(x1), abs(y1), abs(x2), abs(y2))
        if max_coordinate <= 1.5:
            x1, x2 = x1 * width, x2 * width
            y1, y2 = y1 * height, y2 * height
        else:
            x1, x2 = x1 * width / self.input_size, x2 * width / self.input_size
            y1, y2 = y1 * height / self.input_size, y2 * height / self.input_size

        left = max(0, min(width - 1, round(x1)))
        top = max(0, min(height - 1, round(y1)))
        right = max(left + 1, min(width, round(x2)))
        bottom = max(top + 1, min(height, round(y2)))
        if right <= left or bottom <= top:
            return None
        return min(max(confidence, 0.0), 1.0), label, {
            "x": left,
            "y": top,
            "width": right - left,
            "height": bottom - top,
        }

    def _label_for_class(self, class_index: int, valid_labels: list[str]) -> str | None:
        if class_index < 0:
            return None
        raw_label = self.model_labels[class_index] if class_index < len(self.model_labels) else None
        if raw_label is None:
            return None
        normalized = COCO_TO_DOMAIN.get(raw_label, raw_label)
        if normalized in valid_labels:
            return normalized
        # A custom model may use category names not present in the starter
        # vocabulary.  Keep known domain objects, but do not turn a person or a
        # car into a random clothing item.
        if raw_label in valid_labels:
            return raw_label
        return None

    @staticmethod
    def _mock(
        valid_labels: list[str],
        image_key: str,
        image_shape: tuple[int, int] = (480, 640),
        image_bytes: bytes = b"",
    ) -> list[dict]:
        """Return stable prototype detections when no model is mounted."""

        seed = hashlib.sha256(image_key.encode("utf-8") + image_bytes[:4096]).digest()
        count = min(3, len(valid_labels))
        order = sorted(range(len(valid_labels)), key=lambda index: seed[index % len(seed)])
        height, width = image_shape
        results = []
        for position, index in enumerate(order[:count]):
            box_width = max(1, min(width, int(width * (0.28 + position * 0.04))))
            box_height = max(1, min(height, int(height * (0.42 + position * 0.03))))
            x = min(max(0, int(width * (0.12 + position * 0.24))), max(0, width - box_width))
            y = min(max(0, int(height * (0.12 + position * 0.12))), max(0, height - box_height))
            label = valid_labels[index]
            results.append(
                {
                    "label": label,
                    "confidence": round(0.72 + (seed[(index + 7) % len(seed)] / 2550), 3),
                    "bbox": {"x": x, "y": y, "width": box_width, "height": box_height},
                    "category": CATEGORIES.get(label, "item"),
                    "image_key": image_key,
                }
            )
        return results
