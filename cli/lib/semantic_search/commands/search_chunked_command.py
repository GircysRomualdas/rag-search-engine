from lib.utils.constants import DEFAULT_SEARCH_LIMIT
from lib.utils.load import load_movies

from ..chunked_semantic_search import ChunkedSemanticSearch


def search_chunked_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> None:
    chunked_semantic_search = ChunkedSemanticSearch()
    documents = load_movies()
    chunked_semantic_search.load_or_create_chunk_embeddings(documents)
    results = chunked_semantic_search.search_chunks(query, limit)

    for i, result in enumerate(results, start=1):
        title = result["title"]
        score = result["score"]
        description = result["document"]

        print(f"\n{i}. {title} (score: {score:.4f})")
        print(f"   {description}...")
