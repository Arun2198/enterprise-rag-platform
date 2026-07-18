from evaluation.metrics import average_rank
from evaluation.metrics import average_retrieved_documents
from evaluation.metrics import hit_rate_at_k
from evaluation.metrics import mean_reciprocal_rank
from evaluation.metrics import ndcg_at_k
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


def test_hit_rate_at_k_true_when_any_relevant_present():

    assert hit_rate_at_k(["a", "b", "c"], {"b"}, 2) == 1.0
    assert hit_rate_at_k(["a", "b", "c"], {"c"}, 2) == 0.0


def test_hit_rate_at_k_with_no_relevant_ids_is_zero():

    assert hit_rate_at_k(["a", "b"], set(), 2) == 0.0


def test_ndcg_at_k_perfect_ranking_is_one():

    assert ndcg_at_k(["a", "b"], {"a", "b"}, 2) == 1.0


def test_ndcg_at_k_no_relevant_hits_is_zero():

    assert ndcg_at_k(["a", "b"], {"z"}, 2) == 0.0


def test_ndcg_at_k_rewards_earlier_hits():

    high = ndcg_at_k(["a", "b", "c"], {"a"}, 3)
    low = ndcg_at_k(["b", "c", "a"], {"a"}, 3)

    assert high > low


def test_average_rank_uses_first_hit_position_and_skips_misses():

    result = average_rank(
        ranked_results=[["a", "b"], ["c", "d"], ["x", "y"]],
        relevant_ids_by_query=[{"b"}, {"c"}, {"z"}]
    )

    # query 1: hit at rank 2, query 2: hit at rank 1, query 3: no hit (excluded)
    assert result == 1.5


def test_average_rank_with_no_hits_is_zero():

    result = average_rank(
        ranked_results=[["a", "b"]],
        relevant_ids_by_query=[{"z"}]
    )

    assert result == 0.0


def test_average_retrieved_documents():

    result = average_retrieved_documents([["a", "b"], ["c"], []])

    assert result == 1.0


def test_average_retrieved_documents_empty_input_is_zero():

    assert average_retrieved_documents([]) == 0.0
