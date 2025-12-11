import argparse

from lib.semantic_search.commands.chunk_command import chunk_command
from lib.semantic_search.commands.embed_chunks_command import embed_chunks_command
from lib.semantic_search.commands.embed_query_text_command import embed_query_text
from lib.semantic_search.commands.embed_text_command import embed_text_command
from lib.semantic_search.commands.search_command import search_command
from lib.semantic_search.commands.semantic_chunk_command import semantic_chunk_command
from lib.semantic_search.commands.verify_command import verify_model_command
from lib.semantic_search.commands.verify_embeddings_command import (
    verify_embeddings_command,
)
from lib.utils.constants import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SENTENCE_SIZE,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_SEARCH_LIMIT,
)


def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # verify
    subparsers.add_parser("verify", help="Verify that the embedding model is loaded")

    # embed_text
    single_embed_parser = subparsers.add_parser(
        "embed_text", help="Generate an embedding for a single text"
    )
    single_embed_parser.add_argument("text", type=str, help="Text to embed")

    # verify_embeddings
    subparsers.add_parser(
        "verify_embeddings", help="Verify embeddings for the movie dataset"
    )

    # embedquery
    embed_query_parser = subparsers.add_parser(
        "embedquery", help="Generate an embedding for a search query"
    )
    embed_query_parser.add_argument("query", type=str, help="Query to embed")

    # search
    search_parser = subparsers.add_parser(
        "search", help="Search for movies using semantic search"
    )
    search_parser.add_argument("query", type=str, help="Search query")
    search_parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_SEARCH_LIMIT,
        help="Number of results to return",
    )

    # chunk
    chunk_parser = subparsers.add_parser(
        "chunk", help="Split text into fixed-size chunks with optional overlap"
    )
    chunk_parser.add_argument("text", type=str, help="Text to chunk")
    chunk_parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help="Size of each chunk in words",
    )
    chunk_parser.add_argument(
        "--overlap",
        type=int,
        default=DEFAULT_CHUNK_OVERLAP,
        help="Number of words to overlap between chunks",
    )

    # semantic_chunk
    semantic_chunk_parser = subparsers.add_parser(
        "semantic_chunk", help="Split text on sentence boundaries to preserve meaning"
    )
    semantic_chunk_parser.add_argument("text", type=str, help="Text to chunk")
    semantic_chunk_parser.add_argument(
        "--max-chunk-size",
        type=int,
        default=4,
        help="Maximum size of each chunk in sentences",
    )
    semantic_chunk_parser.add_argument(
        "--overlap",
        type=int,
        default=0,
        help="Number of sentences to overlap between chunks",
    )

    # embed_chunks
    subparsers.add_parser("embed_chunks", help="embed_chunks")

    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model_command()
        case "embed_text":
            embed_text_command(args.text)
        case "verify_embeddings":
            verify_embeddings_command()
        case "embedquery":
            embed_query_text(args.query)
        case "search":
            search_command(args.query, args.limit)
        case "chunk":
            chunk_command(args.text, args.chunk_size, args.overlap)
        case "semantic_chunk":
            semantic_chunk_command(args.text, args.max_chunk_size, args.overlap)
        case "embed_chunks":
            embed_chunks_command()
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
