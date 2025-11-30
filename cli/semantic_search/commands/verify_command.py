from ..semantic_search import SemanticSearch


def verify_model_command() -> None:
    semantic_search = SemanticSearch()

    print(f"Model loaded: {semantic_search.model}")
    print(f"Max sequence length: {semantic_search.model.max_seq_length}")
