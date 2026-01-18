import argparse
from lib.describe_image.commands.describe_image_command import describe_image_command

def main():
    parser = argparse.ArgumentParser(description="Image-based Query Rewriter CLI")

    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to an image file",
    )

    parser.add_argument(
        "--query",
        type=str,
        required=True,
        help="Text query to rewrite based on the image",
    )

    args = parser.parse_args()

    describe_image_command(args.image, args.query)


if __name__ == "__main__":
    main()
