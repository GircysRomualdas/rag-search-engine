import os
import re

import numpy as np
from lib.utils.constants import (
    CACHE_MOVIE_EMBEDDINGS_PATH,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SENTENCE_SIZE,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_MODEL,
)
from sentence_transformers import SentenceTransformer


class SemanticSearch:
    def __init__(self, model: str = DEFAULT_MODEL) -> None:
        self.model = SentenceTransformer(model)
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

    def search(self, query: str, limit: int):
        if self.embeddings is None:
            raise ValueError(
                "No embeddings loaded. Call `load_or_create_embeddings` first."
            )

        query_embedding = self.generate_embedding(query)

        similarities: list[tuple[float, dict]] = []

        for i, doc_embedding in enumerate(self.embeddings):
            similarity_score = cosine_similarity(query_embedding, doc_embedding)
            document = self.documents[i]
            similarities.append((similarity_score, document))

        similarities.sort(reverse=True, key=lambda item: item[0])

        results = []

        for score, doc in similarities[:limit]:
            results.append(
                {
                    "score": score,
                    "title": doc["title"],
                    "description": doc["description"],
                }
            )

        return results


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)


def get_text_chunks(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[str]:
    words = text.split()
    chunks = []
    n_words = chunk_size - overlap

    for i in range(0, len(words), n_words):
        chunk_words = words[i : i + chunk_size]
        if chunks and len(chunk_words) <= overlap:
            break

        chunks.append(" ".join(chunk_words))

    return chunks


def get_sentences_chunks(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SENTENCE_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks = []
    step = chunk_size - overlap

    for i in range(0, len(sentences), step):
        chunk_sentences = sentences[i : i + chunk_size]
        if chunks and len(chunk_sentences) <= overlap:
            break

        chunks.append(" ".join(chunk_sentences))

    return chunks
