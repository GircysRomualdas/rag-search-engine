import os

BM25_K1: float = 1.5
BM25_B: float = 0.75
DEFAULT_SEARCH_LIMIT: int = 5

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
MOVIES_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "movies.json")
STOP_WORDS_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "stopwords.txt")

CACHE_DIR = os.path.join(PROJECT_ROOT, "cache")
os.makedirs(CACHE_DIR, exist_ok=True)
CACHE_INDEX_PATH = os.path.join(CACHE_DIR, "index.pkl")
CACHE_DOCMAP_PATH = os.path.join(CACHE_DIR, "docmap.pkl")
CACHE_TERM_FREQUENCIES = os.path.join(CACHE_DIR, "term_frequencies.pkl")
CACHE_DOC_LENGTHS = os.path.join(CACHE_DIR, "doc_lengths.pkl")
