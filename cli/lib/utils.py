import json
import os
import pickle
import string

from nltk.stem import PorterStemmer

DEFAULT_SEARCH_LIMIT = 5

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
MOVIES_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "movies.json")
STOP_WORDS_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "stopwords.txt")

CACHE_DIR = os.path.join(PROJECT_ROOT, "cache")
os.makedirs(CACHE_DIR, exist_ok=True)
CACHE_INDEX_PATH = os.path.join(CACHE_DIR, "index.pkl")
CACHE_DOCMAP_PATH = os.path.join(CACHE_DIR, "docmap.pkl")


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


def load_movies() -> list[dict]:
    with open(MOVIES_DATA_PATH, "r") as f:
        data = json.load(f)
    return data["movies"]


def load_stop_words() -> list[str]:
    with open(STOP_WORDS_DATA_PATH, "r") as f:
        data = f.read().splitlines()
    return data


def text_process(text: str) -> list[str]:
    lowered_text = text.lower()
    clean_text = remove_punctuation(lowered_text)
    words = clean_text.split()
    filtered_words = remove_stop_words(words)
    return get_stems(filtered_words)


def remove_punctuation(text: str) -> str:
    translator = str.maketrans("", "", string.punctuation)
    clean_text = text.translate(translator)
    return clean_text


def remove_stop_words(words: list[str]) -> list[str]:
    stop_words = load_stop_words()
    return [word for word in words if word.lower() not in stop_words]


def get_stems(words: list[str]) -> list[str]:
    stemmer = PorterStemmer()
    return [stemmer.stem(word) for word in words]
