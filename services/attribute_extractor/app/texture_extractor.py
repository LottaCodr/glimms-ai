import numpy as np, io
from PIL import Image

class TextureExtractor:
    def extract(self, image_bytes: bytes) -> dict:
        img = np.array(Image.open(io.BytesIO(image_bytes)).convert("L").resize((128,128))).astype(float)
        var = float(np.var(img))
        edges = self._sobel(img)
        density = float(np.mean(edges > 50))
        return {
            "roughness": "smooth" if var < 500 else "medium" if var < 2000 else "rough",
            "pattern":   "solid" if density < 0.1 else "subtle" if density < 0.25 else "patterned",
            "edge_density": round(density, 3),
        }

    def _sobel(self, img):
        from scipy.ndimage import sobel
        return np.hypot(sobel(img, axis=1), sobel(img, axis=0))
