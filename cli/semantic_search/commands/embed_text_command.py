from ..semantic_search import SemanticSearch


def embed_text_command(text: str) -> None:
    semantic_search = SemanticSearch()

    embedding = semantic_search.generate_embedding(text)

    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")
