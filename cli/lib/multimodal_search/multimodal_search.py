from PIL import Image
from sentence_transformers import SentenceTransformer

class MultimodalSearch:
    def __init__(self, model_name="clip-ViT-B-32"):
        self.model = SentenceTransformer(model_name)

    def embed_image(self, image_path: str):
        image = Image.open(image_path)
        image_embeddings = self.model.encode([image])

        return image_embeddings
