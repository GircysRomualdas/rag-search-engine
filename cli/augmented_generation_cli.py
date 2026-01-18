import argparse
from lib.augmented_generation.commands.rag_command import rag_command
from lib.augmented_generation.commands.summarize_command import summarize_command
from lib.augmented_generation.commands.citations_command import citations_command

def main():
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    rag_parser = subparsers.add_parser(
        "rag", help="Perform RAG (search + generate answer)"
    )
    rag_parser.add_argument("query", type=str, help="Search query for RAG")

    summarize_parser = subparsers.add_parser(
        "summarize", help="Summarize results for a query"
    )
    summarize_parser.add_argument("query", type=str, help="Query to summarize")
    summarize_parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Maximum number of items to summarize (default: 5)",
    )

    citations_parser = subparsers.add_parser(
        "citations", help="Return citations for a query"
    )
    citations_parser.add_argument("query", type=str, help="Query for citations")
    citations_parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Maximum number of citations to return (default: 5)",
    )


    args = parser.parse_args()

    match args.command:
        case "rag":
            rag_command(args.query)
        case "summarize":
            summarize_command(args.query, args.limit)
        case "citations":
            citations_command(args.query, args.limit)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
