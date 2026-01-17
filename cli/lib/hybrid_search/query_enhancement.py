import os
from dotenv import load_dotenv
from google import genai
from lib.utils.constants import DEFAULT_GEMINI_MODEL
from .hybrid_search import HybridSearch
from lib.utils.load import load_movies
from lib.utils.data_models import HybridRRFResult
from lib.utils.utils import get_rrf_doc_list
import time
import json

load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)
model = DEFAULT_GEMINI_MODEL

def spell_correct(query: str) -> str:
    prompt = f"""Fix any spelling errors in this movie search query.

    Only correct obvious typos. Don't change correctly spelled words.

    Query: "{query}"

    If no errors, return the original query.
    Corrected:"""

    result = prompt_model(prompt)
    return result if result else query

def rewrite_query(query: str) -> str:
    prompt = f"""Rewrite this movie search query to be more specific and searchable.

    Original: "{query}"

    Consider:
    - Common movie knowledge (famous actors, popular films)
    - Genre conventions (horror = scary, animation = cartoon)
    - Keep it concise (under 10 words)
    - It should be a google style search query that's very specific
    - Don't use boolean logic

    Examples:

    - "that bear movie where leo gets attacked" -> "The Revenant Leonardo DiCaprio bear attack"
    - "movie about bear in london with marmalade" -> "Paddington London marmalade"
    - "scary movie with bear from few years ago" -> "bear horror movie 2015-2020"

    Rewritten query:"""

    result = prompt_model(prompt)
    return result if result else query

def expand_query(query: str) -> str:
    prompt = f"""Expand this movie search query with related terms.

    Add synonyms and related concepts that might appear in movie descriptions.
    Keep expansions relevant and focused.
    This will be appended to the original query.

    Examples:

    - "scary bear movie" -> "scary horror grizzly bear movie terrifying film"
    - "action movie with bear" -> "action thriller bear chase fight adventure"
    - "comedy with bear" -> "comedy funny bear humor lighthearted"

    Query: "{query}"
    """

    result = prompt_model(prompt)
    return result if result else query

def individual(results: list[HybridRRFResult], query: str) -> list[HybridRRFResult]:
    for result in results:
        prompt = f"""Rate how well this movie matches the search query.

        Query: "{query}"
        Movie: {result.title} - {result.document}

        Consider:
        - Direct relevance to query
        - User intent (what they're looking for)
        - Content appropriateness

        Rate 0-10 (10 = perfect match).
        Give me ONLY the number in your response, no other text or explanation.

        Score:"""

        prompt_result = prompt_model(prompt)
        result.rerank_score = float(prompt_result) if prompt_result else 0.0
        time.sleep(4)

    results.sort(key=lambda r: r.rerank_score, reverse=True)
    return results

def batch(results: list[HybridRRFResult], query: str) -> list[HybridRRFResult]:
    prompt = f"""Rank these movies by relevance to the search query.

    Query: "{query}"

    Movies:
    {get_rrf_doc_list(results)}

    Return ONLY the IDs in order of relevance (best match first). Return a valid JSON list, nothing else. For example:

    [75, 12, 34, 2, 1]
    """

    try:
        prompt_result = json.loads(prompt_model(prompt))
        id_to_result = {int(r.movie_id): r for r in results}
        ordered_results = []
        for movie_id in prompt_result:
            movie_id_int = int(movie_id)
            if movie_id_int in id_to_result:
                ordered_results.append(id_to_result[movie_id_int])
    except:
        ordered_results = results

    return ordered_results


def prompt_model(prompt: str) -> str | None:
    response = client.models.generate_content(model=model, contents=prompt)
    corrected = (response.text or "").strip().strip('"')
    return corrected

def rerank_results(query: str, k: int, limit: int, method: str = None) -> list[HybridRRFResult]:
    documents = load_movies()
    hybrid_search = HybridSearch(documents)
    match method:
        case "individual":
            results = individual(hybrid_search.rrf_search(query, k, limit * 5), query)
            return results[:limit]
        case "batch":
            results = batch(hybrid_search.rrf_search(query, k, limit * 5), query)
            return results[:limit]
        case _:
            return hybrid_search.rrf_search(query, k, limit)

def enhance_query(query: str, method: str = None) -> str:
    match method:
        case "spell":
            return spell_correct(query)
        case "rewrite":
            return rewrite_query(query)
        case "expand":
            return expand_query(query)
        case _:
            return query
