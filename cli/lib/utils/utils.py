import string
from nltk.stem import PorterStemmer
from .data_models import Token
from .load import load_stop_words
from .data_models import HybridRRFResult
import numpy as np

def tokenize_text(text: str) -> list[Token]:
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

def get_stems(words: list[str]) -> list[Token]:
    stemmer = PorterStemmer()
    return [stemmer.stem(Token(word)) for word in words]

def get_cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)
