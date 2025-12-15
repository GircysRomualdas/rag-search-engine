from ..inverted_index import InvertedIndex


def idf_command(term: str) -> None:
    inverted_index = InvertedIndex()
    inverted_index.load()
    idf = inverted_index.get_inverse_document_frequency(term)

    print(f"Inverse document frequency of '{term}': {idf:.2f}")
