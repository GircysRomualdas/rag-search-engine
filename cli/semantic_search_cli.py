import argparse

from lib.semantic_search.commands.embed_text_command import embed_text_command
from lib.semantic_search.commands.verify_command import verify_model_command
from lib.semantic_search.commands.verify_embeddings_command import (
    verify_embeddings_command,
)


def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("verify", help="Verify that the embedding model is loaded")

    embed_text_parser = subparsers.add_parser(
        "embed_text", help="Generate an embedding for a single text"
    )
    embed_text_parser.add_argument("text", type=str, help="Text to embed")

    subparsers.add_parser("verify_embeddings", help="Verify embeddings")

    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model_command()
        case "embed_text":
            embed_text_command(args.text)
        case "verify_embeddings":
            verify_embeddings_command()
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
