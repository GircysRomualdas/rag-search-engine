import math
from collections import Counter

from .utils.load import (
    load_cache_docmap,
    load_cache_index,
    load_cache_term_frequencies,
    load_movies,
)
from .utils.utils import tokenize_text
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
        preprocessed_text = tokenize_text(text)

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

    def get_term_frequencies(self, doc_id: int, term: str) -> int:
        tokens = tokenize_text(term)
        if len(tokens) != 1:
            raise ValueError("term must be a single token")
        token = tokens[0]

        return self.term_frequencies[doc_id][token]

    def get_inverse_document_frequency(self, term: str) -> float:
        tokens = tokenize_text(term)
        if len(tokens) != 1:
            raise ValueError("term must be a single token")
        token = tokens[0]

        doc_count = len(self.docmap)
        term_doc_count = len(self.get_documents(token))
        inverse_document_frequency = math.log((doc_count + 1) / (term_doc_count + 1))
        return inverse_document_frequency

    def get_bm25_idf(self, term: str) -> float:
        tokens = tokenize_text(term)
        if len(tokens) != 1:
            raise ValueError("term must be a single token")
        token = tokens[0]
        doc_count = len(self.docmap)
        term_doc_count = len(self.get_documents(token))
        inverse_document_frequency = math.log(
            (doc_count - term_doc_count + 0.5) / (term_doc_count + 0.5) + 1
        )
        return inverse_document_frequency
