from lib.utils.constants import DEFAULT_SEARCH_LIMIT

from ..inverted_index import InvertedIndex


def bm25_search_command(
    query: str, limit: int = DEFAULT_SEARCH_LIMIT
) -> tuple[list[tuple[int, float]], dict[int, dict]]:
    inverted_index = InvertedIndex()
    inverted_index.load()

    results = inverted_index.bm25_search(query, limit)
    return results, inverted_index.docmap
