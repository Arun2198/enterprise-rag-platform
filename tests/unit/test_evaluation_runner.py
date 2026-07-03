from app.services.rag_service import RAGService
from evaluation.runner import EvaluationExample
from evaluation.runner import EvaluationRunner


def test_evaluation_runner_scores_service_retrieval(tmp_path):

    file_path = tmp_path / "leave_policy.md"
    file_path.write_text(
        "Contractors receive 10 days of leave.",
        encoding="utf-8"
    )
    service = RAGService()
    service.ingest([str(file_path)], metadata={"department": "hr"})
    runner = EvaluationRunner(service)

    result = runner.run(
        examples=[
            EvaluationExample(
                query="How many leave days do contractors receive?",
                relevant_chunk_ids={"leave_policy:0"},
                metadata_filter={"department": "hr"}
            )
        ],
        k=1
    )

    assert result.query_count == 1
    assert result.recall_at_k == 1.0
    assert result.precision_at_k == 1.0
    assert result.mean_reciprocal_rank == 1.0
