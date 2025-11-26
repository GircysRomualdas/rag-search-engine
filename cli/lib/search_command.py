from .inverted_index import InvertedIndex
from .utils.constants import DEFAULT_SEARCH_LIMIT
from .utils.utils import text_process


def search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    inverted_index = InvertedIndex()
    inverted_index.load()
    results = []
    preprocessed_query = text_process(query)

    for token in preprocessed_query:
        doc_ids = inverted_index.get_documents(token)

        for doc_id in doc_ids:
            movie = inverted_index.docmap[doc_id]
            results.append(movie)

            if len(results) >= limit:
                return results

    return results
