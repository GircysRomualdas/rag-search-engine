from ..inverted_index import InvertedIndex


def idf_command(term: str) -> float:
    inverted_index = InvertedIndex()
    inverted_index.load()
    return inverted_index.get_inverse_document_frequency(term)
