import math
from collections import Counter

from lib.utils.constants import BM25_B, BM25_K1, DEFAULT_SEARCH_LIMIT
from lib.utils.data_models import Bm25Result, Movie, MovieId, Token, TokenCount
from lib.utils.load import (
    load_cache_doc_lengths,
    load_cache_docmap,
    load_cache_index,
    load_cache_term_frequencies,
    load_movies,
)
from lib.utils.utils import tokenize_text
from lib.utils.write import (
    write_cache_doc_lengths,
    write_cache_docmap,
    write_cache_index,
    write_cache_term_frequencies,
)


class InvertedIndex:
    def __init__(self) -> None:
        self.index: dict[Token, set[MovieId]] = {}
        self.docmap: dict[MovieId, Movie] = {}
        self.term_frequencies: dict[MovieId, Counter[Token]] = {}
        self.doc_lengths: dict[MovieId, TokenCount] = {}

    def __add_document(self, raw_doc_id: int, text: str) -> None:
        preprocessed_text = tokenize_text(text)
        doc_id = MovieId(raw_doc_id)

        if doc_id not in self.term_frequencies:
            self.term_frequencies[doc_id] = Counter()
        self.term_frequencies[doc_id].update(preprocessed_text)

        self.doc_lengths[doc_id] = TokenCount(len(preprocessed_text))

        for token in preprocessed_text:
            if token not in self.index:
                self.index[token] = set()
            self.index[token].add(doc_id)

    def __get_avg_doc_length(self) -> float:
        if len(self.doc_lengths) == 0:
            return 0.0

        full_length = sum(self.doc_lengths.values())
        return full_length / len(self.doc_lengths)

    def get_documents(self, term: str) -> list[MovieId]:
        doc_ids = self.index.get(Token(term), set())
        return sorted(doc_ids)

    def build(self) -> None:
        movies = load_movies()
        for movie in movies:
            doc_id = movie.id
            self.docmap[doc_id] = movie
            text = f"{movie.title} {movie.description}"
            self.__add_document(doc_id, text)

    def save(self) -> None:
        write_cache_index(self.index)
        write_cache_docmap(self.docmap)
        write_cache_term_frequencies(self.term_frequencies)
        write_cache_doc_lengths(self.doc_lengths)

    def load(self) -> None:
        self.index = load_cache_index()
        self.docmap = load_cache_docmap()
        self.term_frequencies = load_cache_term_frequencies()
        self.doc_lengths = load_cache_doc_lengths()

    def get_term_frequencies(self, raw_doc_id: int, term: str) -> int:
        doc_id = MovieId(raw_doc_id)
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

    def get_bm25_tf(
        self, doc_id: MovieId, term: str, k1: float = BM25_K1, b: float = BM25_B
    ) -> float:
        term_frequency = self.get_term_frequencies(doc_id, term)
        avg_doc_length = self.__get_avg_doc_length()
        length_ratio = 0.0
        if avg_doc_length != 0:
            length_ratio = self.doc_lengths[doc_id] / avg_doc_length

        length_norm = 1 - b + b * length_ratio
        bm25_tf = (term_frequency * (k1 + 1)) / (term_frequency + k1 * length_norm)
        return bm25_tf

    def bm25(self, doc_id: MovieId, term: str) -> float:
        return self.get_bm25_tf(doc_id, term) * self.get_bm25_idf(term)

    def bm25_search(
        self, query: str, limit: int = DEFAULT_SEARCH_LIMIT
    ) -> list[Bm25Result]:
        tokens = tokenize_text(query)
        bm25_scores = {}

        for doc_id in self.docmap:
            total_score = 0.0
            for token in tokens:
                total_score += self.bm25(doc_id, token)
            bm25_scores[doc_id] = total_score

        sorted_scores = sorted(
            bm25_scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )
        top_scores = sorted_scores[:limit]

        return [
            Bm25Result(movie_id=doc_id, score=score) for doc_id, score in top_scores
        ]
