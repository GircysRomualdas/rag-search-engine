import json
import pickle

from .constants import (
    CACHE_DOCMAP_PATH,
    CACHE_INDEX_PATH,
    MOVIES_DATA_PATH,
    STOP_WORDS_DATA_PATH,
)


def load_cache_index() -> dict[str, set[int]]:
    try:
        with open(CACHE_INDEX_PATH, "rb") as f:
            index = pickle.load(f)
        return index
    except FileNotFoundError:
        raise FileNotFoundError(f"Cache index file not found: {CACHE_INDEX_PATH}")


def load_cache_docmap() -> dict[int, dict]:
    try:
        with open(CACHE_DOCMAP_PATH, "rb") as f:
            docmap = pickle.load(f)
        return docmap
    except FileNotFoundError:
        raise FileNotFoundError(f"Cache docmap file not found: {CACHE_DOCMAP_PATH}")


def load_movies() -> list[dict]:
    with open(MOVIES_DATA_PATH, "r") as f:
        data = json.load(f)
    return data["movies"]


def load_stop_words() -> list[str]:
    with open(STOP_WORDS_DATA_PATH, "r") as f:
        data = f.read().splitlines()
    return data
