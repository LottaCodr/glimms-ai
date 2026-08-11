import os, io, numpy as np, logging
from PIL import Image, ExifTags

logger = logging.getLogger(__name__)

LABELS = {
    "wardrobe": ["shirt","pants","dress","jacket","shoes","bag","hat","skirt","coat","sweater","jeans","shorts","blazer","suit"],
    "room":     ["chair","sofa","table","desk","lamp","shelf","bed","cabinet","rug","curtain","pillow","bookcase","wardrobe","mirror"],
    "garden":   ["plant","flower","tree","shrub","grass","succulent","herb","pot","fence","path"],
}

CATEGORIES = {
    "shirt":"top","jacket":"top","sweater":"top","coat":"top","dress":"full",
    "pants":"bottom","jeans":"bottom","shorts":"bottom","skirt":"bottom",
    "shoes":"footwear","bag":"accessory","hat":"accessory","blazer":"top","suit":"top",
}

class Detector:
    def __init__(self):
        self.threshold = float(os.getenv("CONFIDENCE_THRESHOLD", "0.5"))
        self.model = self._load_model()

    def _load_model(self):
        path = os.getenv("MODEL_PATH", "models/yolov8n.onnx")
        if os.path.exists(path):
            try:
                import onnxruntime as ort
                logger.info(f"Loading ONNX model from {path}")
                return ort.InferenceSession(path, providers=["CPUExecutionProvider"])
            except Exception as e:
                logger.warning(f"ONNX load failed: {e}")
        logger.warning("No model found — using mock detector")
        return None

    def detect(self, image_bytes: bytes, vertical: str, image_key: str) -> list[dict]:
        img_np = self._to_numpy(image_bytes)
        valid_labels = LABELS.get(vertical, LABELS["wardrobe"])

        if self.model:
            return self._infer(img_np, valid_labels, image_key)
        return self._mock(valid_labels, image_key)

    def _to_numpy(self, data: bytes):
        import cv2
        img = Image.open(io.BytesIO(data)).convert("RGB")
        img = self._auto_rotate(img)
        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    def _auto_rotate(self, img: Image.Image) -> Image.Image:
        try:
            exif = img._getexif()
            if not exif: return img
            k = next(k for k, v in ExifTags.TAGS.items() if v == "Orientation")
            rot = {3:180, 6:270, 8:90}.get(exif.get(k, 1))
            if rot: return img.rotate(rot, expand=True)
        except Exception: pass
        return img

    def _infer(self, img, valid_labels, image_key):
        import cv2
        h, w = img.shape[:2]
        inp = cv2.resize(img, (640, 640)).astype(np.float32) / 255.0
        inp = inp.transpose(2, 0, 1)[np.newaxis]
        out = self.model.run(None, {"images": inp})[0]
        results = []
        for pred in out[0].T:
            conf = float(np.max(pred[4:]))
            if conf < self.threshold: continue
            cls = int(np.argmax(pred[4:]))
            label = valid_labels[cls % len(valid_labels)]
            cx, cy, bw, bh = pred[:4]
            results.append({
                "label": label, "confidence": round(conf, 3),
                "bbox": {"x": max(0, int((cx-bw/2)*w)), "y": max(0, int((cy-bh/2)*h)),
                         "width": int(bw*w), "height": int(bh*h)},
                "category": CATEGORIES.get(label, "item"),
                "image_key": image_key,
            })
        return results

    def _mock(self, valid_labels, image_key):
        import random
        return [{"label": l, "confidence": round(random.uniform(0.72, 0.97), 3),
                 "bbox": {"x":60,"y":80,"width":220,"height":320},
                 "category": CATEGORIES.get(l, "item"), "image_key": image_key}
                for l in random.sample(valid_labels, min(3, len(valid_labels)))]
