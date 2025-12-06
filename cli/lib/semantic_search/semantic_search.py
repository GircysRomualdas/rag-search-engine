import os

import numpy as np
from lib.utils.constants import CACHE_MOVIE_EMBEDDINGS_PATH
from sentence_transformers import SentenceTransformer


class SemanticSearch:
    def __init__(self) -> None:
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.embeddings: np.ndarray = None
        self.documents: list[dict] = []
        self.document_map: dict[int, dict] = {}

    def generate_embedding(self, text: str) -> np.ndarray:
        if len(text.strip()) == 0:
            raise ValueError("cannot generate embedding for empty text")

        embeddings = self.model.encode([text])
        return embeddings[0]

    def build_embeddings(self, documents: list[dict]) -> np.ndarray:
        self.documents = documents

        docs = []
        for document in documents:
            self.document_map[document["id"]] = document

            docs.append(f"{document['title']}: {document['description']}")

        self.embeddings = self.model.encode(docs, show_progress_bar=True)

        np.save(CACHE_MOVIE_EMBEDDINGS_PATH, self.embeddings)

        return self.embeddings

    def load_or_create_embeddings(self, documents: list[dict]) -> np.ndarray:
        self.documents = documents

        for document in documents:
            self.document_map[document["id"]] = document

        if os.path.exists(CACHE_MOVIE_EMBEDDINGS_PATH):
            self.embeddings = np.load(CACHE_MOVIE_EMBEDDINGS_PATH)

            if len(self.embeddings) == len(documents):
                return self.embeddings

        return self.build_embeddings(documents)
