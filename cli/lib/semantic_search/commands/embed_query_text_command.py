from ..semantic_search import SemanticSearch


def embed_query_text(query: str) -> None:
    semantic_search = SemanticSearch()

    embedding = semantic_search.generate_embedding(query)

    print(f"Query: {query}")
    print(f"First 5 dimensions: {embedding[:5]}")
    print(f"Shape: {embedding.shape}")
