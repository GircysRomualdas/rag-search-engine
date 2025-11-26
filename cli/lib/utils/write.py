import os
import pickle

from .constants import CACHE_DOCMAP_PATH, CACHE_INDEX_PATH, PROJECT_ROOT


def write_cache_index(index: dict[str, set[int]]) -> None:
    with open(CACHE_INDEX_PATH, "wb") as f:
        pickle.dump(index, f)

    relative_path = os.path.relpath(CACHE_INDEX_PATH, PROJECT_ROOT)
    print(f"Inverted index written to {relative_path}")


def write_cache_docmap(docmap: dict[int, dict]) -> None:
    with open(CACHE_DOCMAP_PATH, "wb") as f:
        pickle.dump(docmap, f)

    relative_path = os.path.relpath(CACHE_DOCMAP_PATH, PROJECT_ROOT)
    print(f"Document map written to {relative_path}")
