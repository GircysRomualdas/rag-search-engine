from ..multimodal_search import MultimodalSearch


def verify_image_embedding_command(image_path: str) -> None:
    documents = load_movies()
    multimodal_search = MultimodalSearch(documents)

    image_embeddings = multimodal_search.embed_image(image_path)

    print(f"Embedding shape: {image_embeddings[0].shape[0]} dimensions")
