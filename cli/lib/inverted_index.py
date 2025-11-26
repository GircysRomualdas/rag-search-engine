from .utils.load import (
    load_cache_docmap,
    load_cache_index,
    load_movies,
)
from .utils.utils import text_process
from .utils.write import (
    write_cache_docmap,
    write_cache_index,
)


class InvertedIndex:
    def __init__(self) -> None:
        self.index: dict[str, set[int]] = {}
        self.docmap: dict[int, dict] = {}

    def __add_document(self, doc_id: int, text: str) -> None:
        preprocessed_text = text_process(text)

        for token in preprocessed_text:
            if token not in self.index:
                self.index[token] = set()

            self.index[token].add(doc_id)

    def get_documents(self, term: str) -> list[int]:
        doc_ids = self.index.get(term, set())
        return sorted(doc_ids)

    def build(self) -> None:
        movies = load_movies()

        for movie in movies:
            doc_id = movie["id"]
            self.docmap[doc_id] = movie
            text = f"{movie['title']} {movie['description']}"
            self.__add_document(doc_id, text)

    def save(self) -> None:
        write_cache_index(self.index)
        write_cache_docmap(self.docmap)

    def load(self) -> None:
        self.index = load_cache_index()
        self.docmap = load_cache_docmap()
