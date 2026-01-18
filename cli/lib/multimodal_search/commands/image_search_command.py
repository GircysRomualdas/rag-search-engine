from ..multimodal_search import MultimodalSearch
from lib.utils.load import load_movies

def image_search_command(image_path: str):
    documents = load_movies()
    multimodal_search = MultimodalSearch(documents)

    results = multimodal_search.search_with_image(image_path)

    for i, result in enumerate(results, start=1):
        print(f"{i}. {result['title']} (similarity: {result['similarity']:.3f})")
        print(f"   {result['description']}...\n")
