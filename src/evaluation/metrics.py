def recall_at_k(
    retrieved_ids: list[str],
    relevant_ids: set[str],
    k: int
) -> float:
    if not relevant_ids:
        return 0.0

    retrieved_at_k = set(retrieved_ids[:k])
    return len(retrieved_at_k.intersection(relevant_ids)) / len(relevant_ids)


def precision_at_k(
    retrieved_ids: list[str],
    relevant_ids: set[str],
    k: int
) -> float:
    if k <= 0:
        return 0.0

    retrieved_at_k = retrieved_ids[:k]

    if not retrieved_at_k:
        return 0.0

    hits = sum(
        1
        for item in retrieved_at_k
        if item in relevant_ids
    )
    return hits / k


def mean_reciprocal_rank(
    ranked_results: list[list[str]],
    relevant_ids_by_query: list[set[str]]
) -> float:
    if not ranked_results:
        return 0.0

    reciprocal_ranks = []

    for retrieved_ids, relevant_ids in zip(ranked_results, relevant_ids_by_query):
        reciprocal_ranks.append(
            _reciprocal_rank(
                retrieved_ids=retrieved_ids,
                relevant_ids=relevant_ids
            )
        )

    return sum(reciprocal_ranks) / len(ranked_results)


def _reciprocal_rank(
    retrieved_ids: list[str],
    relevant_ids: set[str]
) -> float:
    for index, retrieved_id in enumerate(retrieved_ids, start=1):
        if retrieved_id in relevant_ids:
            return 1 / index

    return 0.0
