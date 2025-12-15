import argparse

from lib.hybrid_search.commands.normalize_command import normalize_command
from lib.hybrid_search.commands.rrf_search_command import rrf_search_command
from lib.hybrid_search.commands.weighted_search_command import weighted_search_command
from lib.utils.constants import (
    DEFAULT_ALPHA,
    DEFAULT_SEARCH_LIMIT,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # normalize
    normalize_parser = subparsers.add_parser(
        "normalize", help="Normalize a list of scores"
    )
    normalize_parser.add_argument(
        "scores", nargs="+", type=float, help="List of scores to normalize"
    )

    # weighted search
    weighted_search_parser = subparsers.add_parser(
        "weighted-search",
        help="Run a weighted hybrid search combining vector and keyword relevance",
    )
    weighted_search_parser.add_argument(
        "query",
        type=str,
        help="Search query text",
    )
    weighted_search_parser.add_argument(
        "--alpha",
        type=float,
        default=DEFAULT_ALPHA,
        help="Weighting factor between semantic and keyword search (default: 0.5)",
    )
    weighted_search_parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_SEARCH_LIMIT,
        help="Number of results to return (default: 5)",
    )

    # rrf-search
    rrf_parser = subparsers.add_parser(
        "rrf-search", help="Perform Reciprocal Rank Fusion search"
    )
    rrf_parser.add_argument("query", type=str, help="Search query")
    rrf_parser.add_argument(
        "-k",
        type=int,
        default=60,
        help="RRF k parameter controlling weight distribution (default=60)",
    )
    rrf_parser.add_argument(
        "--limit", type=int, default=5, help="Number of results to return (default=5)"
    )

    args = parser.parse_args()

    match args.command:
        case "normalize":
            normalize_command(args.scores)
        case "weighted-search":
            weighted_search_command(args.query, args.alpha, args.limit)
        case "rrf-search":
            rrf_search_command(args.query, args.k, args.limit)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
