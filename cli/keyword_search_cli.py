import argparse

from lib.keyword_search.commands.bm25_idf_command import bm25_idf_command
from lib.keyword_search.commands.bm25_search_command import bm25_search_command
from lib.keyword_search.commands.bm25_tf_command import bm25_tf_command
from lib.keyword_search.commands.build_command import build_command
from lib.keyword_search.commands.idf_command import idf_command
from lib.keyword_search.commands.search_command import search_command
from lib.keyword_search.commands.tf_command import tf_command
from lib.keyword_search.commands.tfidf_command import tfidf_command
from lib.utils.constants import BM25_B, BM25_K1, DEFAULT_SEARCH_LIMIT


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # search
    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    # build
    subparsers.add_parser("build", help="Build the inverted index")

    # tf
    tf_parser = subparsers.add_parser(
        "tf", help="Term frequency for that term in the document"
    )
    tf_parser.add_argument("doc_id", type=int, help="Document ID")
    tf_parser.add_argument("term", type=str, help="Search term")

    # idf
    idf_parser = subparsers.add_parser("idf", help="Inverse document frequency")
    idf_parser.add_argument("term", type=str, help="Search term")

    # tfidf
    tfidf_parser = subparsers.add_parser(
        "tfidf", help="Term Frequency-Inverse Document Frequency"
    )
    tfidf_parser.add_argument("doc_id", type=int, help="Document ID")
    tfidf_parser.add_argument("term", type=str, help="Search term")

    # bm25idf
    bm25_idf_parser = subparsers.add_parser(
        "bm25idf", help="Get BM25 IDF score for a given term"
    )
    bm25_idf_parser.add_argument(
        "term", type=str, help="Term to get BM25 IDF score for"
    )

    # bm25tf
    bm25_tf_parser = subparsers.add_parser(
        "bm25tf", help="Get BM25 TF score for a given document ID and term"
    )
    bm25_tf_parser.add_argument("doc_id", type=int, help="Document ID")
    bm25_tf_parser.add_argument("term", type=str, help="Term to get BM25 TF score for")
    bm25_tf_parser.add_argument(
        "k1", type=float, nargs="?", default=BM25_K1, help="Tunable BM25 K1 parameter"
    )
    bm25_tf_parser.add_argument(
        "b", type=float, nargs="?", default=BM25_B, help="Tunable BM25 b parameter"
    )

    # bm25search
    bm25search_parser = subparsers.add_parser(
        "bm25search", help="Search movies using full BM25 scoring"
    )
    bm25search_parser.add_argument("query", type=str, help="Search query")
    bm25search_parser.add_argument(
        "limit",
        type=int,
        nargs="?",
        default=DEFAULT_SEARCH_LIMIT,
        help="Search result limit",
    )

    args = parser.parse_args()

    match args.command:
        case "search":
            search_command(args.query)
        case "build":
            build_command()
        case "tf":
            tf_command(args.doc_id, args.term)
        case "idf":
            idf_command(args.term)
        case "tfidf":
            tfidf_command(args.doc_id, args.term)
        case "bm25idf":
            bm25_idf_command(args.term)
        case "bm25tf":
            bm25_tf_command(args.doc_id, args.term, args.k1, args.b)
        case "bm25search":
            bm25_search_command(args.query, args.limit)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
