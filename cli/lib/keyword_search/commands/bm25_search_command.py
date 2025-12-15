from lib.utils.constants import DEFAULT_SEARCH_LIMIT
from lib.utils.data_models import Bm25Result, Movie, MovieId

from ..inverted_index import InvertedIndex


def bm25_search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> None:
    inverted_index = InvertedIndex()
    inverted_index.load()

    results: list[Bm25Result] = inverted_index.bm25_search(query, limit)

    print(f"Searching for: {query}")
    docmap = inverted_index.docmap

    for rank, result in enumerate(results, start=1):
        doc_id_int = result.movie_id
        doc_id = MovieId(doc_id_int)
        movie: Movie = docmap[doc_id]
        title = movie.title
        score = result.score
        print(f"{rank}. ({doc_id_int}) {title} - Score: {score:.2f}")
