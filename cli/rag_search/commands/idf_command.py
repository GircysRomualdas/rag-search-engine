import math

from ..inverted_index import InvertedIndex
from ..utils.utils import text_process


def idf_command(text: str) -> float:
    inverted_index = InvertedIndex()
    inverted_index.load()

    doc_count = len(inverted_index.docmap)
    preprocessed_text = text_process(text)[0]
    term_doc_count = len(inverted_index.get_documents(preprocessed_text))
    inverse_document_frequency = math.log((doc_count + 1) / (term_doc_count + 1))
    return inverse_document_frequency
