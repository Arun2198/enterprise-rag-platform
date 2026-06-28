from evaluation.metrics import mean_reciprocal_rank
from evaluation.metrics import precision_at_k
from evaluation.metrics import recall_at_k


def test_retrieval_metrics():

    retrieved = ["chunk-1", "chunk-2", "chunk-3"]
    relevant = {"chunk-2", "chunk-4"}

    assert recall_at_k(retrieved, relevant, 2) == 0.5
    assert precision_at_k(retrieved, relevant, 2) == 0.5


def test_mean_reciprocal_rank():

    result = mean_reciprocal_rank(
        ranked_results=[
            ["a", "b"],
            ["c", "d"],
        ],
        relevant_ids_by_query=[
            {"b"},
            {"c"},
        ]
    )

    assert result == 0.75
