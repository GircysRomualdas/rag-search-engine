from ..inverted_index import InvertedIndex


def bm25_idf_command(term: str) -> float:
    inverted_index = InvertedIndex()
    inverted_index.load()

    bm25_idf_score = inverted_index.get_bm25_idf(term)
    return bm25_idf_score
