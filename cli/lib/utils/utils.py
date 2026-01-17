import string

from nltk.stem import PorterStemmer

from .data_models import Token
from .load import load_stop_words
from .data_models import HybridRRFResult


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

def get_rrf_doc_list(rrf_results: list[HybridRRFResult]) -> str:
    doc_lines = []
    for r in rrf_results:
        line = f"ID: {r.movie_id}, Title: {r.title}, Description: {r.document}"
        doc_lines.append(line)

    doc_list_str = "\n".join(doc_lines)
    return doc_list_str
