import json
import pickle
from collections import Counter

from .constants import (
    CACHE_DOCMAP_PATH,
    CACHE_INDEX_PATH,
    CACHE_TERM_FREQUENCIES,
    MOVIES_DATA_PATH,
    STOP_WORDS_DATA_PATH,
)


def load_cache_index() -> dict[str, set[int]]:
    index = load_pickle(CACHE_INDEX_PATH, "Inverted index")
    return index


def load_cache_docmap() -> dict[int, dict]:
    docmap = load_pickle(CACHE_DOCMAP_PATH, "Document map")
    return docmap


def load_cache_term_frequencies() -> dict[int, Counter[str]]:
    term_frequencies = load_pickle(CACHE_TERM_FREQUENCIES, "Term frequencies")
    return term_frequencies


def load_movies() -> list[dict]:
    with open(MOVIES_DATA_PATH, "r") as f:
        data = json.load(f)
    return data["movies"]


def load_stop_words() -> list[str]:
    with open(STOP_WORDS_DATA_PATH, "r") as f:
        data = f.read().splitlines()
    return data


def load_pickle(path: str, msg: str) -> dict:
    try:
        with open(path, "rb") as f:
            data = pickle.load(f)
        return data
    except FileNotFoundError:
        raise FileNotFoundError(f"{msg} file not found: {path}")
