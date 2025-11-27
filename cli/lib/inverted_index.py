from collections import Counter

from .utils.load import (
    load_cache_docmap,
    load_cache_index,
    load_cache_term_frequencies,
    load_movies,
)
from .utils.utils import text_process
from .utils.write import (
    write_cache_docmap,
    write_cache_index,
    write_cache_term_frequencies,
)


class InvertedIndex:
    def __init__(self) -> None:
        self.index: dict[str, set[int]] = {}
        self.docmap: dict[int, dict] = {}
        self.term_frequencies: dict[int, Counter[str]] = {}

    def __add_document(self, doc_id: int, text: str) -> None:
        preprocessed_text = text_process(text)

        if doc_id not in self.term_frequencies:
            self.term_frequencies[doc_id] = Counter()
        self.term_frequencies[doc_id].update(preprocessed_text)

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
        write_cache_term_frequencies(self.term_frequencies)

    def load(self) -> None:
        self.index = load_cache_index()
        self.docmap = load_cache_docmap()
        self.term_frequencies = load_cache_term_frequencies()

    def get_tf(self, doc_id: int, term: str) -> int:
        if doc_id not in self.term_frequencies:
            raise KeyError(f"Document ID {doc_id} not found in term frequencies")
        term_frequencies = self.term_frequencies[doc_id]
        return term_frequencies[term]
