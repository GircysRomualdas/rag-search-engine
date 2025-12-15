from lib.utils.constants import BM25_B, BM25_K1
from lib.utils.data_models import MovieId

from ..inverted_index import InvertedIndex


def bm25_tf_command(
    doc_id: int, term: str, k1: float = BM25_K1, b: float = BM25_B
) -> None:
    inverted_index = InvertedIndex()
    inverted_index.load()

    bm25_tf_score = inverted_index.get_bm25_tf(MovieId(doc_id), term, k1, b)

    print(f"BM25 TF score of '{term}' in document '{doc_id}': {bm25_tf_score:.2f}")
