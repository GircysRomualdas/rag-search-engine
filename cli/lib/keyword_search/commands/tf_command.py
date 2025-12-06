from ..inverted_index import InvertedIndex


def tf_command(doc_id: int, term: str) -> int:
    inverted_index = InvertedIndex()
    inverted_index.load()

    return inverted_index.get_term_frequencies(doc_id, term)
