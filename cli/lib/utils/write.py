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


def write_cache_index(index: dict[str, set[int]]) -> None:
    write_pickle(CACHE_INDEX_PATH, index, "Inverted index")


def write_cache_docmap(docmap: dict[int, dict]) -> None:
    write_pickle(CACHE_DOCMAP_PATH, docmap, "Document map")


def write_cache_term_frequencies(term_frequencies: dict[int, Counter[str]]) -> None:
    write_pickle(CACHE_TERM_FREQUENCIES_PATH, term_frequencies, "Term frequencies")


def write_cache_doc_lengths(doc_lengths: dict[int, int]) -> None:
    write_pickle(CACHE_DOC_LENGTHS_PATH, doc_lengths, "Document lengths")


def write_pickle(path: str, data: dict, msg: str) -> None:
    with open(path, "wb") as f:
        pickle.dump(data, f)

    relative_path = os.path.relpath(path, PROJECT_ROOT)
    print(f"{msg} written to {relative_path}")


def write_json(path: str, data) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
