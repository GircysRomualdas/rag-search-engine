from sentence_transformers import SentenceTransformer


class SemanticSearch:
    def __init__(self) -> None:
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def generate_embedding(self, text: str):
        if len(text.strip()) == 0:
            raise ValueError("cannot generate embedding for empty text")

        embeddings = self.model.encode([text])
        return embeddings[0]
