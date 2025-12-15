from ..inverted_index import InvertedIndex


def bm25_idf_command(term: str) -> None:
    inverted_index = InvertedIndex()
    inverted_index.load()

    bm25_idf_score = inverted_index.get_bm25_idf(term)

    print(f"BM25 IDF score of '{term}': {bm25_idf_score:.2f}")
