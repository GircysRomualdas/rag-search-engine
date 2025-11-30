from ..inverted_index import InvertedIndex
from ..utils.constants import BM25_K1


def bm25_tf_command(doc_id: int, term: str, k1: float = BM25_K1) -> float:
    inverted_index = InvertedIndex()
    inverted_index.load()

    bm25_tf_score = inverted_index.get_bm25_tf(doc_id, term, k1)
    return bm25_tf_score
