import argparse
from lib.evaluation_search.commands.evaluation_command import evaluation_command

def main():
    parser = argparse.ArgumentParser(description="Search Evaluation CLI")
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of results to evaluate (k for precision@k, recall@k)",
    )

    args = parser.parse_args()

    evaluation_command(args.limit)


if __name__ == "__main__":
    main()
