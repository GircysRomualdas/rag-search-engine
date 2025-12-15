from ..hybrid_search import normalize_scores


def normalize_command(scores: list[float]) -> None:
    normalized_score = normalize_scores(scores)
    for score in normalized_score:
        print(f"* {score:.4f}")
