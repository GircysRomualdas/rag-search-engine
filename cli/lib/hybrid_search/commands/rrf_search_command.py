from lib.utils.constants import (
    DEFAULT_K,
    DEFAULT_SEARCH_LIMIT,
)
from lib.utils.load import load_movies
from ..hybrid_search import HybridSearch
from ..query_enhancement import enhance_query


def rrf_search_command(
    query: str, k: int = DEFAULT_K, limit: int = DEFAULT_SEARCH_LIMIT, enhance: str = None
) -> None:
    if enhance:
        enhanced_query = enhance_query(query, enhance)
        print(f"Enhanced query ({enhance}): '{query}' -> '{enhanced_query}'\n")
        query = enhanced_query

    documents = load_movies()
    hybrid_search = HybridSearch(documents)
    results = hybrid_search.rrf_search(query, k, limit)

    for i, result in enumerate(results, 1):
        title = result.title
        score = result.rrf_score
        doc = result.document
        bm25_rank = result.bm25_rank
        semantic_rank = result.semantic_rank

        print(f"{i}. {title}")
        print(f"   RRF Score: {score:.3f}")
        print(f"   BM25 Rank: {bm25_rank}, Semantic Rank: {semantic_rank}")
        print(f"   {doc[:100]}...")
        print()
