from lib.utils.load import load_movies

from ..chunked_semantic_search import ChunkedSemanticSearch


def embed_chunks_command() -> None:
    documents = load_movies()
    chunk_semantic_search = ChunkedSemanticSearch()
    embeddings = chunk_semantic_search.load_or_create_chunk_embeddings(documents)
    print(f"Generated {len(embeddings)} chunked embeddings")
