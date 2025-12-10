import os

BM25_K1 = 1.5
BM25_B = 0.75
DEFAULT_SEARCH_LIMIT = 5
DEFAULT_CHUNK_SIZE = 200
DEFAULT_CHUNK_OVERLAP = 0
DEFAULT_CHUNK_SENTENCE_SIZE = 4

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

MOVIES_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "movies.json")
STOP_WORDS_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "stopwords.txt")

CACHE_DIR = os.path.join(PROJECT_ROOT, "cache")
os.makedirs(CACHE_DIR, exist_ok=True)
CACHE_INDEX_PATH = os.path.join(CACHE_DIR, "index.pkl")
CACHE_DOCMAP_PATH = os.path.join(CACHE_DIR, "docmap.pkl")
CACHE_TERM_FREQUENCIES_PATH = os.path.join(CACHE_DIR, "term_frequencies.pkl")
CACHE_DOC_LENGTHS_PATH = os.path.join(CACHE_DIR, "doc_lengths.pkl")
CACHE_MOVIE_EMBEDDINGS_PATH = os.path.join(CACHE_DIR, "movie_embeddings.npy")
