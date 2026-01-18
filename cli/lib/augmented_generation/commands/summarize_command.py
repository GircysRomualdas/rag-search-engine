from lib.hybrid_search.hybrid_search import HybridSearch
from lib.utils.load import load_movies
from lib.utils.constants import DEFAULT_K
from lib.utils.prompt import get_rrf_doc_list, prompt_model
from lib.utils.data_models import HybridRRFResult

def summarize_command(query: str, limit: int) -> None:
    documents = load_movies()
    hybrid_search = HybridSearch(documents)
    results = hybrid_search.rrf_search(query, k=DEFAULT_K, limit=5)

    prompt_result = summarize_prompt(results, query)

    print("Search Results:")
    for result in results:
        print(f"     -{result.title}")

    print("\nLLM Summary:")
    print(prompt_result)


def summarize_prompt(results: list[HybridRRFResult], query: str) -> str:
    prompt = f"""
    Provide information useful to this query by synthesizing information from multiple search results in detail.
    The goal is to provide comprehensive information so that users know what their options are.
    Your response should be information-dense and concise, with several key pieces of information about the genre, plot, etc. of each movie.
    This should be tailored to Hoopla users. Hoopla is a movie streaming service.
    Query: {query}
    Search Results:
    {get_rrf_doc_list(results)}
    Provide a comprehensive 3–4 sentence answer that combines information from multiple sources:
    """

    return prompt_model(prompt)
