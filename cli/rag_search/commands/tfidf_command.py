from ..inverted_index import InvertedIndex


def tfidf_command(doc_id: int, term: str):
    inverted_index = InvertedIndex()
    inverted_index.load()
    tf = inverted_index.get_term_frequencies(doc_id, term)
    idf = inverted_index.get_inverse_document_frequency(term)
    tf_idf_score = tf * idf
    return tf_idf_score
