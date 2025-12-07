from lib.utils.constants import DEFAULT_SEARCH_LIMIT
from lib.utils.load import load_movies

from ..semantic_search import SemanticSearch


def search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> None:
    semantic_search = SemanticSearch()
    documents = load_movies()
    semantic_search.load_or_create_embeddings(documents)

    results = semantic_search.search(query, limit)

    for i, result in enumerate(results, 1):
        print(f"{i}. {result['title']} (score: {result['score']:.4f})")
        print(f"   {result['description'][:100]}...")
        print()
