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
from .data_models import ChunkMetadata, Movie, MovieId, Token, TokenCount


def load_cache_index() -> dict[Token, set[MovieId]]:
    index = load_pickle(CACHE_INDEX_PATH)
    return index


def load_cache_docmap() -> dict[MovieId, Movie]:
    raw_docmap = load_pickle(CACHE_DOCMAP_PATH)
    return {MovieId(doc_id): movie for doc_id, movie in raw_docmap.items()}


def load_cache_term_frequencies() -> dict[MovieId, Counter[Token]]:
    term_frequencies = load_pickle(CACHE_TERM_FREQUENCIES_PATH)
    return term_frequencies


def load_cache_doc_lengths() -> dict[MovieId, TokenCount]:
    doc_lengths = load_pickle(CACHE_DOC_LENGTHS_PATH)
    return doc_lengths


def load_movies() -> list[Movie]:
    data = load_json(MOVIES_DATA_PATH)
    return [
        Movie(
            id=MovieId(m["id"]),
            title=m["title"],
            description=m["description"],
        )
        for m in data["movies"]
    ]


def load_chunk_metadata() -> list[ChunkMetadata]:
    data = load_json(CHCHE_CHUNK_METADATA_PATH)
    return [
        ChunkMetadata(
            movie_index=c["movie_idx"],
            chunk_index=c["chunk_idx"],
            total_chunks=c["total_chunks"],
        )
        for c in data["chunks"]
    ]


def load_json(path: str) -> dict:
    with open(path, "r") as f:
        data = json.load(f)
    return data


def load_stop_words() -> list[str]:
    with open(STOP_WORDS_DATA_PATH, "r") as f:
        data = f.read().splitlines()
    return data


def load_pickle(path: str) -> dict:
    try:
        with open(path, "rb") as f:
            data = pickle.load(f)
        return data
    except FileNotFoundError:
        raise FileNotFoundError(f"file not found: {path}")
