import argparse
from lib.multimodal_search.commands.verify_image_embedding_command import verify_image_embedding_command

def main():
    parser = argparse.ArgumentParser(description="Multimodal Search CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    verify_parser = subparsers.add_parser(
        "verify_image_embedding",
        help="Generate an embedding for an image and print its shape",
    )
    verify_parser.add_argument(
        "image_path",
        type=str,
        help="Path to the image file",
    )

    args = parser.parse_args()

    match args.command:
        case "verify_image_embedding":
            verify_image_embedding_command(args.image_path)
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()
