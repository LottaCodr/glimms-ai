import numpy as np, logging
logger = logging.getLogger(__name__)

STYLE_TAGS = ["casual","formal","smart-casual","athletic","bohemian",
              "minimalist","maximalist","vintage","streetwear","luxury","traditional","modest"]

class CLIPExtractor:
    def __init__(self):
        self.model = self.processor = None
        try:
            from transformers import CLIPModel, CLIPProcessor
            import torch
            self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            self.model.eval()
            logger.info("CLIP model loaded")
        except Exception as e:
            logger.warning(f"CLIP unavailable: {e} — using mock embeddings")

    def embed(self, image_bytes: bytes) -> list[float]:
        if not self.model:
            return list(np.random.randn(512).astype(float))
        from PIL import Image
        import torch, io
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        inputs = self.processor(images=img, return_tensors="pt")
        with torch.no_grad():
            f = self.model.get_image_features(**inputs)
            f = f / f.norm(dim=-1, keepdim=True)
        return f[0].tolist()

    def get_style_tags(self, embedding: list[float]) -> list[str]:
        if not self.model:
            import random; return random.sample(STYLE_TAGS, 3)
        import torch
        texts = [f"a {t} outfit" for t in STYLE_TAGS]
        inputs = self.processor(text=texts, return_tensors="pt", padding=True)
        with torch.no_grad():
            tf = self.model.get_text_features(**inputs)
            tf = tf / tf.norm(dim=-1, keepdim=True)
        v = torch.tensor(embedding).unsqueeze(0)
        scores = (v @ tf.T).squeeze()
        return [STYLE_TAGS[i] for i in scores.topk(3).indices.tolist()]
