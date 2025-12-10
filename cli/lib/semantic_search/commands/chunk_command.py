from ..semantic_search import get_text_chunks


def chunk_command(text: str, chunk_size: int, overlap: int) -> None:
    chunks = get_text_chunks(text, chunk_size, overlap)

    total_chars = len(text)
    print(f"Chunking {total_chars} characters")
    for index, chunk in enumerate(chunks, start=1):
        print(f"{index}. {chunk}")
