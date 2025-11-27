import os
import pickle
from collections import Counter

from .constants import (
    CACHE_DOCMAP_PATH,
    CACHE_INDEX_PATH,
    CACHE_TERM_FREQUENCIES,
    PROJECT_ROOT,
)


def write_cache_index(index: dict[str, set[int]]) -> None:
    write_pickle(CACHE_INDEX_PATH, index, "Inverted index")


def write_cache_docmap(docmap: dict[int, dict]) -> None:
    write_pickle(CACHE_DOCMAP_PATH, docmap, "Document map")


def write_cache_term_frequencies(term_frequencies: dict[int, Counter[str]]) -> None:
    write_pickle(CACHE_TERM_FREQUENCIES, term_frequencies, "Term frequencies")


def write_pickle(path: str, data: dict, msg: str) -> None:
    with open(path, "wb") as f:
        pickle.dump(data, f)

    relative_path = os.path.relpath(path, PROJECT_ROOT)
    print(f"{msg} written to {relative_path}")
