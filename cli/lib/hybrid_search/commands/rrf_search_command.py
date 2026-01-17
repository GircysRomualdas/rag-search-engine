from lib.utils.constants import (
    DEFAULT_K,
    DEFAULT_SEARCH_LIMIT,
)
from ..query_enhancement import enhance_query, rerank_results, evaluated_results


def rrf_search_command(
    query: str,
    k: int = DEFAULT_K,
    limit: int = DEFAULT_SEARCH_LIMIT,
    enhance: str = None,
    rerank_method: str = None,
    evaluate: bool = False
) -> None:
    print(f"original query: {query}")
    if enhance:
        enhanced_query = enhance_query(query, enhance)
        print(f"Enhanced query ({enhance}): '{query}' -> '{enhanced_query}'\n")
        query = enhanced_query

    results = rerank_results(query, k, limit, rerank_method)

    if rerank_method:
        print(f"Reranking top {limit} results using {rerank_method} method...\n")

    print(f"Reciprocal Rank Fusion Results for '{query}' (k={k}):\n")

    for i, result in enumerate(results, 1):
        print("\nEvaluation:\n")
        print(f"{i}. {result.title}")
        if rerank_method == "individual":
            print(f"   Rerank Score: {result.rerank_score:.3f}/10")
        elif rerank_method == "batch":
            print(f"   Rerank Rank: {i}")
        elif rerank_method == "cross_encoder":
            print(f"   Cross Encoder Score: {result.rerank_score}")
        print(f"   RRF Score: {result.rrf_score:.3f}")
        print(f"   BM25 Rank: {result.bm25_rank}, Semantic Rank: {result.semantic_rank}")
        print(f"   {result.document[:100]}...")

    if evaluate:
        eval_results = evaluated_results(results, query)

        for i, (score, result) in enumerate(eval_results, 1):
            print(f"{i}. {result.title}: {score}/3")
