from ..inverted_index import InvertedIndex


def build_command() -> None:
    print("Building inverted index...")
    inverted_index = InvertedIndex()
    inverted_index.build()
    inverted_index.save()
    print("Inverted index built successfully.")
