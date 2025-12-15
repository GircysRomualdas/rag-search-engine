import json
import os
import pickle
from collections import Counter

from .constants import (
    CACHE_DOC_LENGTHS_PATH,
    CACHE_DOCMAP_PATH,
    CACHE_INDEX_PATH,
    CACHE_TERM_FREQUENCIES_PATH,
    PROJECT_ROOT,
)
from .data_models import Movie, MovieId, Token, TokenCount


def write_cache_index(index: dict[Token, set[MovieId]]) -> None:
    write_pickle(CACHE_INDEX_PATH, index)


def write_cache_docmap(docmap: dict[MovieId, Movie]) -> None:
    write_pickle(CACHE_DOCMAP_PATH, docmap)


def write_cache_term_frequencies(
    term_frequencies: dict[MovieId, Counter[Token]],
) -> None:
    write_pickle(CACHE_TERM_FREQUENCIES_PATH, term_frequencies)


def write_cache_doc_lengths(doc_lengths: dict[MovieId, TokenCount]) -> None:
    write_pickle(CACHE_DOC_LENGTHS_PATH, doc_lengths)


def write_pickle(path: str, data: dict) -> None:
    with open(path, "wb") as f:
        pickle.dump(data, f)

    relative_path = os.path.relpath(path, PROJECT_ROOT)
    print(f"written to {relative_path}")


def write_json(path: str, data) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
