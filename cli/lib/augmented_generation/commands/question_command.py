from lib.hybrid_search.hybrid_search import HybridSearch
from lib.utils.load import load_movies
from lib.utils.constants import DEFAULT_K
from lib.utils.prompt import get_rrf_doc_list, prompt_model
from lib.utils.data_models import HybridRRFResult

def question_command(query: str, limit: int) -> None:
    documents = load_movies()
    hybrid_search = HybridSearch(documents)
    results = hybrid_search.rrf_search(query, k=DEFAULT_K, limit=limit)

    prompt_result = question_prompt(results, query)

    print("Search Results:")
    for result in results:
        print(f"     -{result.title}")

    print("\nAnswer: ")
    print(prompt_result)


def question_prompt(results: list[HybridRRFResult], query: str) -> str:
    prompt = f"""Answer the user's question based on the provided movies that are available on Hoopla.

    This should be tailored to Hoopla users. Hoopla is a movie streaming service.

    Question: {query}

    Documents:
    {get_rrf_doc_list(results)}

    Instructions:
    - Answer questions directly and concisely
    - Be casual and conversational
    - Don't be cringe or hype-y
    - Talk like a normal person would in a chat conversation
    - Use information from the documents when possible, but you may also use your general knowledge about movies if it helps answer the question better
    - If the question asks who the main characters are in a movie, list the main characters by name, even if some names are not explicitly mentioned in the documents

    Answer:"""

    return prompt_model(prompt)
