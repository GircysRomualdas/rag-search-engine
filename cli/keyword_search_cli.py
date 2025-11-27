import argparse

from lib.build_command import build_command
from lib.search_command import search_command
from lib.tf_command import tf_command


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    subparsers.add_parser("build", help="Build the inverted index")

    term_frequency_parser = subparsers.add_parser(
        "tf", help="term frequency for that term in the document"
    )
    term_frequency_parser.add_argument("doc_id", type=int, help="Document ID")
    term_frequency_parser.add_argument("term", type=str, help="Search term")

    args = parser.parse_args()

    match args.command:
        case "search":
            print(f"Searching for: {args.query}")
            results = search_command(args.query)

            for i, result in enumerate(results):
                print(f"{i + 1}. {result['title']}")
        case "build":
            print("Building inverted index...")
            build_command()
            print("Inverted index built successfully.")
        case "tf":
            term_frequency = tf_command(args.doc_id, args.term)
            print(
                f"Term frequency of '{args.term}' in document {args.doc_id}: {term_frequency}"
            )
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
