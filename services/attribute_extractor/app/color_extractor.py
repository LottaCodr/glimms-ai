import numpy as np, io
from PIL import Image

class ColorExtractor:
    def extract(self, image_bytes: bytes, n: int = 5) -> dict:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize((150,150))
        pixels = np.array(img).reshape(-1, 3).astype(float)
        centers = self._kmeans(pixels, n)
        palette = [{"hex": "#{:02x}{:02x}{:02x}".format(int(c[0]),int(c[1]),int(c[2])),
                    "rgb": {"r":int(c[0]),"g":int(c[1]),"b":int(c[2])}} for c in centers]
        return {"dominant": palette[0], "palette": palette, "mood": self._mood(palette[0]["hex"])}

    def _kmeans(self, data, k, iters=10):
        centers = data[np.random.choice(len(data), k, replace=False)].copy()
        for _ in range(iters):
            labels = np.argmin(np.linalg.norm(data[:,None] - centers[None,:], axis=2), axis=1)
            new_centers = np.array([data[labels==i].mean(0) if (labels==i).any() else centers[i] for i in range(k)])
            if np.allclose(centers, new_centers): break
            centers = new_centers
        return centers

    def _mood(self, h: str) -> str:
        r,g,b = int(h[1:3],16), int(h[3:5],16), int(h[5:7],16)
        bright = (r*299+g*587+b*114)/1000
        sat = max(r,g,b) - min(r,g,b)
        if bright > 200 and sat < 30: return "neutral"
        if bright < 60: return "dark"
        if b > r and b > g: return "cool"
        if r > g and r > b: return "warm"
        return "balanced"
