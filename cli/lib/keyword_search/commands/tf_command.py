from ..inverted_index import InvertedIndex


def tf_command(doc_id: int, term: str) -> None:
    inverted_index = InvertedIndex()
    inverted_index.load()
    print(
        f"Term frequency of '{term}' in document {doc_id}: {inverted_index.get_term_frequencies(doc_id, term)}"
    )
