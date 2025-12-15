import os

from lib.utils.constants import CACHE_INDEX_PATH, DEFAULT_ALPHA, DEFAULT_SEARCH_LIMIT

from ..keyword_search.inverted_index import InvertedIndex
from ..semantic_search.chunked_semantic_search import ChunkedSemanticSearch


class HybridSearch:
    def __init__(self, documents):
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx = InvertedIndex()
        if not os.path.exists(CACHE_INDEX_PATH):
            self.idx.build()
            self.idx.save()

    def _bm25_search(self, query: str, limit: int) -> list[tuple[int, float]]:
        self.idx.load()
        return self.idx.bm25_search(query, limit)

    def weighted_search(
        self,
        query: str,
        alpha: float = DEFAULT_ALPHA,
        limit: int = DEFAULT_SEARCH_LIMIT,
    ) -> list[dict]:
        bm25_results = self._bm25_search(query, limit * 500)
        bm25_scores = [score for (_, score) in bm25_results]
        bm25_normalized = normalize_scores(bm25_scores)

        chunk_results = self.semantic_search.search_chunks(query, limit * 500)
        chunk_scores = [r["score"] for r in chunk_results]
        chunk_normalized = normalize_scores(chunk_scores)

        combined = {}

        for i, (doc_id, score) in enumerate(bm25_results):
            normalized = bm25_normalized[i]
            title = ""
            document = {}
            for doc in self.documents:
                if doc["id"] == doc_id:
                    title = doc["title"]
                    document = doc["description"][:100]

            combined[doc_id] = {
                "id": doc_id,
                "title": title,
                "document": document,
                "bm25_score": normalized,
                "semantic_score": 0.0,
            }

        for i, result in enumerate(chunk_results):
            doc_id = result["id"]
            normalized = chunk_normalized[i]

            if combined.get(doc_id, None) is None:
                combined[doc_id] = {
                    "id": doc_id,
                    "title": result["title"],
                    "document": result["document"],
                    "bm25_score": 0.0,
                }

            combined[doc_id]["semantic_score"] = normalized

        for val in combined.values():
            hybrid = alpha * val["bm25_score"] + (1 - alpha) * val["semantic_score"]
            val["score"] = hybrid

        weighted_results = list(combined.values())
        weighted_results.sort(key=lambda d: d["score"], reverse=True)
        return weighted_results[:limit]

    def rrf_search(self, query, k, limit=10):
        raise NotImplementedError("RRF hybrid search is not implemented yet.")


def normalize_scores(scores: list[float]) -> list[float]:
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
