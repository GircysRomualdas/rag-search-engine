import os

import numpy as np
from lib.utils.constants import (
    CACHE_CHUNK_EMBEDDINGS_PATH,
    CHCHE_CHUNK_METADATA_PATH,
    DEFAULT_MODEL,
    SCORE_PRECISION,
)
from lib.utils.data_models import (
    ChunkIndex,
    ChunkMetadata,
    ChunkScore,
    ChunkSearchResult,
    Movie,
)
from lib.utils.load import load_chunk_metadata
from lib.utils.write import write_json

from .semantic_search import SemanticSearch, get_cosine_similarity, get_sentences_chunks


class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        super().__init__(model_name)
        self.chunk_embeddings: np.ndarray = None
        self.chunk_metadata: list[ChunkMetadata] = []

    def build_chunk_embeddings(self, documents: list[Movie]) -> np.ndarray:
        self.documents = documents

        all_chunks = []
        chunks_metadata = []
        for doc_idx, document in enumerate(documents):
            self.document_map[document.id] = document

            if len(document.description) == 0:
                continue

            chunks = get_sentences_chunks(document.description, 4, 1)
            all_chunks.extend(chunks)

            for chu_idx, chunk in enumerate(chunks):
                chunks_metadata.append(
                    ChunkMetadata(
                        movie_index=doc_idx,
                        chunk_index=chu_idx,
                        total_chunks=len(chunks),
                    )
                )

        self.chunk_embeddings = self.model.encode(all_chunks, show_progress_bar=True)
        self.chunk_metadata = chunks_metadata

        np.save(CACHE_CHUNK_EMBEDDINGS_PATH, self.chunk_embeddings)
        write_json(
            path=CHCHE_CHUNK_METADATA_PATH,
            data={
                "chunks": [
                    {
                        "movie_idx": m.movie_index,
                        "chunk_idx": m.chunk_index,
                        "total_chunks": m.total_chunks,
                    }
                    for m in chunks_metadata
                ],
                "total_chunks": len(all_chunks),
            },
        )

        return self.chunk_embeddings

    def load_or_create_chunk_embeddings(self, documents: list[Movie]) -> np.ndarray:
        self.documents = documents

        for document in documents:
            self.document_map[document.id] = document

        if os.path.exists(CACHE_CHUNK_EMBEDDINGS_PATH) and os.path.exists(
            CHCHE_CHUNK_METADATA_PATH
        ):
            self.chunk_embeddings = np.load(CACHE_CHUNK_EMBEDDINGS_PATH)
            self.chunk_metadata = load_chunk_metadata()
            return self.chunk_embeddings

        return self.build_chunk_embeddings(documents)

    def search_chunks(self, query: str, limit: int = 10) -> list[ChunkSearchResult]:
        if self.chunk_embeddings is None or self.chunk_metadata is None:
            raise ValueError(
                "No chunk embeddings loaded. Call `load_or_create_chunk_embeddings` first."
            )

        embedding = self.generate_embedding(query)
        chunk_scores: list[ChunkScore] = []

        for i, chunk_embedding in enumerate(self.chunk_embeddings):
            chunk_score = get_cosine_similarity(embedding, chunk_embedding)
            meta = self.chunk_metadata[i]
            chunk_scores.append(
                ChunkScore(
                    movie_index=meta.movie_index,
                    chunk_index=ChunkIndex(meta.chunk_index),
                    score=chunk_score,
                )
            )

        movie_scores: dict[int, ChunkScore] = {}

        for chunk_score in chunk_scores:
            movie_idx = chunk_score.movie_index
            if (
                movie_idx not in movie_scores
                or chunk_score.score > movie_scores[movie_idx].score
            ):
                movie_scores[movie_idx] = chunk_score

        sorted_movies = sorted(
            movie_scores.values(),
            key=lambda cs: cs.score,
            reverse=True,
        )
        filtered_movies = sorted_movies[:limit]

        results: list[ChunkSearchResult] = []

        for movie_score in filtered_movies:
            movie_idx = movie_score.movie_index
            doc: Movie = self.documents[movie_idx]
            results.append(
                ChunkSearchResult(
                    movie_id=doc.id,
                    title=doc.title,
                    document=doc.description[:100],
                    score=round(movie_score.score, SCORE_PRECISION),
                    movie_index=movie_idx,
                    best_chunk_index=movie_score.chunk_index,
                )
            )

        return results
