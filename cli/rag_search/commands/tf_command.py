from ..inverted_index import InvertedIndex


def tf_command(doc_id: int, term: str) -> int:
    inverted_index = InvertedIndex()
    inverted_index.load()

    term_frequency = inverted_index.get_tf(doc_id, term)
    return term_frequency
