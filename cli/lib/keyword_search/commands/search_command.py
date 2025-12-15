from lib.utils.constants import DEFAULT_SEARCH_LIMIT
from lib.utils.data_models import Movie
from lib.utils.utils import tokenize_text

from ..inverted_index import InvertedIndex


def search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> None:
    print(f"Searching for: {query}")
    results = search(query, limit)

    for i, result in enumerate(results):
        print(f"{i + 1}. {result.title}")


def search(query: str, limit: int) -> list[Movie]:
    inverted_index = InvertedIndex()
    inverted_index.load()
    results = []
    preprocessed_query = tokenize_text(query)

    for token in preprocessed_query:
        doc_ids = inverted_index.get_documents(token)

        for doc_id in doc_ids:
            movie = inverted_index.docmap[doc_id]
            results.append(movie)

            if len(results) >= limit:
                return results

    return results
