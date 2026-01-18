from PIL import Image
from sentence_transformers import SentenceTransformer
from lib.utils.utils import get_cosine_similarity

class MultimodalSearch:
    def __init__(self, documents, model_name="clip-ViT-B-32"):
        self.model = SentenceTransformer(model_name)
        self.documents = documents
        self.texts = []
        for doc in documents:
            self.texts.append(f"{doc.title}: {doc.description}")
        self.text_embeddings = self.model.encode(self.texts, show_progress_bar=True)

    def embed_image(self, image_path: str):
        image = Image.open(image_path)
        image_embeddings = self.model.encode([image], show_progress_bar=True)

        return image_embeddings

    def search_with_image(self, image_path: str):
        image_embeddings = self.embed_image(image_path)
        image_vec = image_embeddings[0]
        scores = []

        for i, text_embed in enumerate(self.text_embeddings):
            score = get_cosine_similarity(image_vec, text_embed)
            scores.append((score, i))

        scores.sort(key=lambda x: x[0], reverse=True)
        top_scores = scores[:5]
        results = []

        for score, idx in top_scores:
            doc = self.documents[idx]
            results.append({
                "id": doc.id,
                "title": doc.title,
                "description": doc.description[:100],
                "similarity": score,
            })
        return results
