from ..semantic_search import get_sentences_chunks


def semantic_chunk_command(text: str, max_chunk_size: int, overlap: int) -> None:
    chunks = get_sentences_chunks(text, max_chunk_size, overlap)

    total_chars = len(text)
    print(f"Semantically chunking {total_chars} characters")
    for index, chunk in enumerate(chunks, start=1):
        print(f"{index}. {chunk}")
