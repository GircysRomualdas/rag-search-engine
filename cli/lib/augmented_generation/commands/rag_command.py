from lib.hybrid_search.hybrid_search import HybridSearch
from lib.utils.load import load_movies
from lib.utils.constants import DEFAULT_K
from lib.utils.prompt import get_rrf_doc_list, prompt_model
from lib.utils.data_models import HybridRRFResult

def rag_command(query: str) -> None:
    documents = load_movies()
    hybrid_search = HybridSearch(documents)
    results = hybrid_search.rrf_search(query, k=DEFAULT_K, limit=5)

    prompt_result = augmented_prompt(results, query)

    print("Search Results:")
    for result in results:
        print(f"     -{result.title}")

    print("\nRAG Response:")
    print(prompt_result)


def augmented_prompt(results: list[HybridRRFResult], query: str) -> str:
    prompt = f"""Answer the question or provide information based on the provided documents. This should be tailored to Hoopla users. Hoopla is a movie streaming service.

    Query: {query}

    Documents:
    {get_rrf_doc_list(results)}

    Provide a comprehensive answer that addresses the query:"""

    return prompt_model(prompt)
