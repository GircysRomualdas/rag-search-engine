from lib.hybrid_search.hybrid_search import HybridSearch
from lib.utils.load import load_movies
from lib.utils.constants import DEFAULT_K
from lib.utils.prompt import get_rrf_doc_list, prompt_model
from lib.utils.data_models import HybridRRFResult

def citations_command(query: str, limit: int) -> None:
    documents = load_movies()
    hybrid_search = HybridSearch(documents)
    results = hybrid_search.rrf_search(query, k=DEFAULT_K, limit=5)

    prompt_result = augmented_prompt(results, query)

    print("Search Results:")
    for result in results:
        print(f"     -{result.title}")

    print("\nLLM Answer:")
    print(prompt_result)


def augmented_prompt(results: list[HybridRRFResult], query: str) -> str:
    prompt = f"""Answer the question or provide information based on the provided documents.

    This should be tailored to Hoopla users. Hoopla is a movie streaming service.

    If not enough information is available to give a good answer, say so but give as good of an answer as you can while citing the sources you have.

    Query: {query}

    Documents:
    {get_rrf_doc_list(results)}

    Instructions:
    - Provide a comprehensive answer that addresses the query
    - Cite sources using [1], [2], etc. format when referencing information
    - If sources disagree, mention the different viewpoints
    - If the answer isn't in the documents, say "I don't have enough information"
    - Be direct and informative

    Answer:"""

    return prompt_model(prompt)
