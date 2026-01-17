from lib.utils.load import load_golden_dataset, load_movies
from lib.utils.data_models import GoldenDataset, HybridRRFResult
from lib.hybrid_search.hybrid_search import HybridSearch


def evaluation_command(limit: int) -> None:
    golden_datasets = load_golden_dataset()
    documents = load_movies()
    hybrid_search = HybridSearch(documents)
    results: list[tuple[GoldenDataset, list[HybridRRFResult]]] = []

    for data in golden_datasets:
        results.append((data, hybrid_search.rrf_search(data.query, k=60, limit=limit)))

    print(f"k={limit}\n")

    for golden, search_results in results:
        retrieved_titles = [r.title for r in search_results]

        relevant_set = set(golden.relevant_docs)

        relevant_retrieved = 0
        for title in retrieved_titles:
            if title in relevant_set:
                relevant_retrieved += 1

        total_retrieved = len(retrieved_titles)
        precision = relevant_retrieved / total_retrieved if total_retrieved > 0 else 0.0

        print(f"- Query: {golden.query}")
        print(f"  - Precision@{limit}: {precision:.4f}")
        print(f"  - Retrieved: {', '.join(retrieved_titles)}")
        print(f"  - Relevant: {', '.join(golden.relevant_docs)}")
        print()
