import os

from lib.utils.constants import CACHE_INDEX_PATH, DEFAULT_ALPHA, DEFAULT_SEARCH_LIMIT
from lib.utils.data_models import (
    Bm25Result,
    ChunkSearchResult,
    HybridResult,
    Movie,
    MovieId,
)

from ..keyword_search.inverted_index import InvertedIndex
from ..semantic_search.chunked_semantic_search import ChunkedSemanticSearch


class HybridSearch:
    def __init__(self, documents) -> None:
        self.documents: list[Movie] = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx = InvertedIndex()
        if not os.path.exists(CACHE_INDEX_PATH):
            self.idx.build()
            self.idx.save()

    def _bm25_search(self, query: str, limit: int) -> list[Bm25Result]:
        self.idx.load()
        return self.idx.bm25_search(query, limit)

    def weighted_search(
        self,
        query: str,
        alpha: float = DEFAULT_ALPHA,
        limit: int = DEFAULT_SEARCH_LIMIT,
    ) -> list[HybridResult]:
        bm25_results: list[Bm25Result] = self._bm25_search(query, limit * 500)
        bm25_scores = [r.score for r in bm25_results]
        bm25_normalized = get_normalize_scores(bm25_scores)

        chunk_results: list[ChunkSearchResult] = self.semantic_search.search_chunks(
            query, limit * 500
        )
        chunk_scores = [r.score for r in chunk_results]
        chunk_normalized = get_normalize_scores(chunk_scores)

        combined: dict[MovieId, HybridResult] = {}
        movie_by_id: dict[MovieId, Movie] = {m.id: m for m in self.documents}

        for i, bm25_result in enumerate(bm25_results):
            doc_id_int = bm25_result.movie_id
            doc_id = MovieId(doc_id_int)
            movie = movie_by_id.get(doc_id)
            if movie is None:
                continue

            normalized_bm25 = bm25_normalized[i]

            combined[doc_id] = HybridResult(
                movie_id=doc_id,
                title=movie.title,
                document=movie.description[:100],
                bm25_score=normalized_bm25,
                semantic_score=0.0,
                score=0.0,
            )

        for i, result in enumerate(chunk_results):
            doc_id = result.movie_id
            normalized_sem = chunk_normalized[i]

            if doc_id not in combined:
                movie = movie_by_id.get(doc_id)
                if movie is None:
                    continue
                combined[doc_id] = HybridResult(
                    movie_id=doc_id,
                    title=movie.title,
                    document=movie.description[:100],
                    bm25_score=0.0,
                    semantic_score=0.0,
                    score=0.0,
                )

            combined[doc_id].semantic_score = normalized_sem

        for val in combined.values():
            val.score = alpha * val.bm25_score + (1 - alpha) * val.semantic_score

        weighted_results = list(combined.values())
        weighted_results.sort(key=lambda r: r.score, reverse=True)
        return weighted_results[:limit]

    def rrf_search(self, query, k, limit=10):
        raise NotImplementedError("RRF hybrid search is not implemented yet.")


def get_normalize_scores(scores: list[float]) -> list[float]:
    if not scores:
        return []

    min_score = min(scores)
    max_score = max(scores)

    if max_score == min_score:
        return [1.0] * len(scores)

    normalized_scores = []
    for s in scores:
        normalized_scores.append((s - min_score) / (max_score - min_score))

    return normalized_scores
