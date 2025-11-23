from .utils import DEFAULT_SEARCH_LIMIT, load_movies, text_process


def search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    movies = load_movies()
    results = []
    for movie in movies:
        preprocessed_title = text_process(movie["title"])
        preprocessed_query = text_process(query)
        if any(
            item in title_word
            for title_word in preprocessed_title
            for item in preprocessed_query
        ):
            results.append(movie)

            if len(results) >= limit:
                break

    return results
