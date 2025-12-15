from lib.utils.load import load_movies

from ..hybrid_search import HybridSearch


def weighted_search_command(query: str, alpha: float, limit: int) -> None:
    documents = load_movies()
    hybrid_search = HybridSearch(documents)
    results = hybrid_search.weighted_search(query, alpha, limit)

    for i, result in enumerate(results, start=1):
        title = result.title
        hybrid_score = result.score
        bm25_score = result.bm25_score
        semantic_score = result.semantic_score
        snippet = result.document

        print(f"{i}. {title}")
        print(f"   Hybrid Score: {hybrid_score:.3f}")
        print(f"   BM25: {bm25_score:.3f}, Semantic: {semantic_score:.3f}")
        print(f"   {snippet}")
