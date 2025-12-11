import json
import pickle
from collections import Counter

from .constants import (
    CACHE_DOC_LENGTHS_PATH,
    CACHE_DOCMAP_PATH,
    CACHE_INDEX_PATH,
    CACHE_TERM_FREQUENCIES_PATH,
    CHCHE_CHUNK_METADATA_PATH,
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
    term_frequencies = load_pickle(CACHE_TERM_FREQUENCIES_PATH, "Term frequencies")
    return term_frequencies


def load_cache_doc_lengths() -> dict[int, int]:
    doc_lengths = load_pickle(CACHE_DOC_LENGTHS_PATH, "Document lengths")
    return doc_lengths


def load_movies() -> list[dict]:
    data = load_json(MOVIES_DATA_PATH)
    return data["movies"]


def load_chunk_metadata() -> list[dict]:
    data = load_json(CHCHE_CHUNK_METADATA_PATH)
    return data["chunks"]


def load_json(path: str) -> dict:
    with open(path, "r") as f:
        data = json.load(f)
    return data


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
