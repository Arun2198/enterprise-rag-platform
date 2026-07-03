from pydantic import BaseModel
from pydantic import Field

from app.services.rag_service import RAGService
from evaluation.metrics import mean_reciprocal_rank
from evaluation.metrics import precision_at_k
from evaluation.metrics import recall_at_k


class EvaluationExample(BaseModel):
    query: str
    relevant_chunk_ids: set[str] = Field(default_factory=set)
    metadata_filter: dict[str, str] | None = None


class EvaluationResult(BaseModel):
    query_count: int
    recall_at_k: float
    precision_at_k: float
    mean_reciprocal_rank: float


class EvaluationRunner:

    def __init__(
        self,
        service: RAGService
    ) -> None:
        self.service = service

    def run(
        self,
        examples: list[EvaluationExample],
        k: int = 5
    ) -> EvaluationResult:
        ranked_results: list[list[str]] = []
        relevant_sets: list[set[str]] = []
        recalls = []
        precisions = []

        for example in examples:
            response = self.service.ask(
                query=example.query,
                top_k=k,
                metadata_filter=example.metadata_filter
            )
            retrieved_ids = [
                source.chunk_id
                for source in response.sources
            ]
            ranked_results.append(retrieved_ids)
            relevant_sets.append(example.relevant_chunk_ids)
            recalls.append(
                recall_at_k(
                    retrieved_ids=retrieved_ids,
                    relevant_ids=example.relevant_chunk_ids,
                    k=k
                )
            )
            precisions.append(
                precision_at_k(
                    retrieved_ids=retrieved_ids,
                    relevant_ids=example.relevant_chunk_ids,
                    k=k
                )
            )

        query_count = len(examples)

        if query_count == 0:
            return EvaluationResult(
                query_count=0,
                recall_at_k=0.0,
                precision_at_k=0.0,
                mean_reciprocal_rank=0.0
            )

        return EvaluationResult(
            query_count=query_count,
            recall_at_k=sum(recalls) / query_count,
            precision_at_k=sum(precisions) / query_count,
            mean_reciprocal_rank=mean_reciprocal_rank(
                ranked_results=ranked_results,
                relevant_ids_by_query=relevant_sets
            )
        )
