from app.services.rag_service import RERANKER_FLAG_NAME
from app.services.rag_service import RAGService
from mlops.feature_flags import FeatureFlagManager
from rag.chunking.chunk import Chunk
from rag.chunking.recursive_chunker import RecursiveChunker
from rag.guardrails.base import Action
from rag.guardrails.manager import GuardrailResult
from rag.retrieval.hybrid_retrieval import RetrievedChunk


class _StubRetriever:

    def __init__(self):
        self.calls = []

    def retrieve(self, query, top_k, metadata_filter=None):
        self.calls.append(top_k)
        return []


class _StubReranker:

    def __init__(self):
        self.called = False

    def rerank(self, query, candidates, top_k):
        self.called = True
        return []


def test_rag_service_ingests_and_answers_from_markdown(tmp_path):

    file_path = tmp_path / "leave_policy.md"
    file_path.write_text(
        "# Leave Policy\n"
        "Employees receive 20 days of paid leave annually. "
        "Contractors receive 10 days of leave.",
        encoding="utf-8"
    )
    service = RAGService(
        chunker=RecursiveChunker(
            chunk_size=120,
            chunk_overlap=20,
            minimum_chunk_size=10
        )
    )

    ingest_response = service.ingest([str(file_path)])
    ask_response = service.ask("How many leave days do contractors receive?")

    assert ingest_response.indexed_documents == 1
    assert ingest_response.indexed_chunks >= 1
    assert ingest_response.errors == []
    assert "Contractors receive 10 days of leave." in ask_response.answer
    assert ask_response.sources


def test_ask_bypasses_reranking_when_no_reranker_configured():

    class StubRetriever:

        def __init__(self):
            self.calls = []

        def retrieve(self, query, top_k, metadata_filter=None):
            self.calls.append(top_k)
            return []

    service = RAGService()
    service.retriever = StubRetriever()

    service.ask("query", top_k=5)

    assert service.retriever.calls == [5]


def test_ask_requests_top_k_times_candidate_multiplier_from_retriever():

    class StubRetriever:

        def __init__(self):
            self.calls = []

        def retrieve(self, query, top_k, metadata_filter=None):
            self.calls.append(top_k)
            return []

    class StubReranker:

        def rerank(self, query, candidates, top_k):
            return []

    service = RAGService(reranker=StubReranker(), candidate_multiplier=4)
    service.retriever = StubRetriever()

    service.ask("query", top_k=5)

    assert service.retriever.calls == [20]


def test_ask_forwards_reranked_chunks_unchanged_to_answerer(tmp_path):

    file_path = tmp_path / "policy.md"
    file_path.write_text("# Policy\nSome policy content here.", encoding="utf-8")

    captured = {}

    class RecordingAnswerer:

        def answer(self, query, retrieved_chunks):
            captured["chunks"] = retrieved_chunks
            captured["query"] = query
            return "recorded answer"

    reranked_chunk = Chunk(
        chunk_id="fixed:0",
        document_id="fixed",
        chunk_index=0,
        text="fixed reranked text",
        source="fixed.md",
        document_type="markdown"
    )
    expected = [
        RetrievedChunk(
            chunk=reranked_chunk,
            vector_score=0.1,
            keyword_score=0.1,
            score=0.99
        )
    ]

    class StubReranker:

        def __init__(self):
            self.received_top_k = None

        def rerank(self, query, candidates, top_k):
            self.received_top_k = top_k
            return expected

    reranker = StubReranker()
    service = RAGService(
        answerer=RecordingAnswerer(),
        reranker=reranker,
        candidate_multiplier=4
    )
    service.ingest([str(file_path)])

    response = service.ask("policy question", top_k=3)

    assert captured["chunks"] == expected
    assert captured["query"] == "policy question"
    assert reranker.received_top_k == 3
    assert response.answer == "recorded answer"


class _FixedAnswerer:

    def __init__(self, answer: str):
        self._answer = answer

    def answer(self, query, retrieved_chunks):
        return self._answer


def test_ask_redacts_pii_in_final_answer(tmp_path):

    file_path = tmp_path / "policy.md"
    file_path.write_text("# Policy\nSome policy content here.", encoding="utf-8")

    service = RAGService(answerer=_FixedAnswerer("Contact john@company.com for help."))
    service.ingest([str(file_path)])

    response = service.ask("policy question", top_k=3)

    assert "[REDACTED_EMAIL]" in response.answer
    assert "john@company.com" not in response.answer
    assert response.guardrail_flags["pii_detected"] is True


def test_ask_flags_hallucination_in_guardrail_flags(tmp_path):

    file_path = tmp_path / "policy.md"
    file_path.write_text("# Policy\nContractors receive 10 days of leave.", encoding="utf-8")

    service = RAGService(
        answerer=_FixedAnswerer("Completely unrelated statement about astronomy.")
    )
    service.ingest([str(file_path)])

    response = service.ask("policy question", top_k=3)

    assert response.guardrail_flags["hallucination"] is True
    assert "groundedness" in response.guardrail_flags


def test_ask_clean_answer_passes_guardrails_unchanged(tmp_path):

    file_path = tmp_path / "policy.md"
    file_path.write_text(
        "# Policy\nContractors receive 10 days of leave.",
        encoding="utf-8"
    )

    service = RAGService(answerer=_FixedAnswerer("Contractors receive 10 days of leave."))
    service.ingest([str(file_path)])

    response = service.ask("How many leave days do contractors receive?", top_k=3)

    assert response.answer == "Contractors receive 10 days of leave."
    assert response.guardrail_flags["pii_detected"] is False
    assert response.guardrail_flags["hallucination"] is False


def test_ask_bypasses_guardrails_when_disabled(tmp_path):

    file_path = tmp_path / "policy.md"
    file_path.write_text("# Policy\nSome policy content here.", encoding="utf-8")

    service = RAGService(
        answerer=_FixedAnswerer("Contact john@company.com for help."),
        guardrails_enabled=False
    )
    service.ingest([str(file_path)])

    response = service.ask("policy question", top_k=3)

    assert response.answer == "Contact john@company.com for help."
    assert response.guardrail_flags == {}


class _BlockingGuardrailManager:

    def __init__(self, block_input: bool = False, block_output: bool = False):
        self.block_input = block_input
        self.block_output = block_output
        self.run_input_calls = 0
        self.run_output_calls = 0

    def run_input(self, query):
        self.run_input_calls += 1
        action = Action.BLOCK if self.block_input else Action.ALLOW
        text = "blocked at input" if self.block_input else query
        return GuardrailResult(findings=[], action=action, text=text, flags={"blocked": self.block_input})

    def run_output(self, query, answer, retrieved_chunks):
        self.run_output_calls += 1
        action = Action.BLOCK if self.block_output else Action.ALLOW
        text = "blocked at output" if self.block_output else answer
        return GuardrailResult(findings=[], action=action, text=text, flags={"blocked": self.block_output})


def test_ask_blocks_before_retrieval_when_input_guardrail_blocks():

    class StubRetriever:

        def __init__(self):
            self.calls = 0

        def retrieve(self, query, top_k, metadata_filter=None):
            self.calls += 1
            return []

    guardrail_manager = _BlockingGuardrailManager(block_input=True)
    service = RAGService(guardrail_manager=guardrail_manager)
    service.retriever = StubRetriever()

    response = service.ask("malicious query", top_k=3)

    assert response.answer == "blocked at input"
    assert response.sources == []
    assert response.confidence == 0.0
    assert guardrail_manager.run_output_calls == 0
    assert service.retriever.calls == 0


def test_ask_blocks_and_hides_sources_when_output_guardrail_blocks(tmp_path):

    file_path = tmp_path / "policy.md"
    file_path.write_text("# Policy\nSensitive policy content.", encoding="utf-8")

    guardrail_manager = _BlockingGuardrailManager(block_output=True)
    service = RAGService(
        answerer=_FixedAnswerer("some answer"),
        guardrail_manager=guardrail_manager
    )
    service.ingest([str(file_path)])

    response = service.ask("policy question", top_k=3)

    assert response.answer == "blocked at output"
    assert response.sources == []
    assert response.confidence == 0.0


def test_reranker_runs_unconditionally_without_feature_flags():

    reranker = _StubReranker()
    service = RAGService(reranker=reranker)
    service.retriever = _StubRetriever()

    service.ask("query", top_k=5)

    assert reranker.called is True


def test_feature_flag_disabled_skips_reranker_for_every_request():

    flags = FeatureFlagManager()
    flags.define(RERANKER_FLAG_NAME, enabled=False)
    reranker = _StubReranker()
    service = RAGService(reranker=reranker, feature_flags=flags)
    service.retriever = _StubRetriever()

    service.ask("query", top_k=5, client_id="user-1")

    assert reranker.called is False


def test_feature_flag_enabled_at_full_rollout_runs_reranker():

    flags = FeatureFlagManager()
    flags.define(RERANKER_FLAG_NAME, enabled=True, rollout_percentage=100.0)
    reranker = _StubReranker()
    service = RAGService(reranker=reranker, feature_flags=flags)
    service.retriever = _StubRetriever()

    service.ask("query", top_k=5, client_id="user-1")

    assert reranker.called is True


def test_missing_flag_definition_fails_open_to_reranker_enabled():

    flags = FeatureFlagManager()  # RERANKER_FLAG_NAME never defined
    reranker = _StubReranker()
    service = RAGService(reranker=reranker, feature_flags=flags)
    service.retriever = _StubRetriever()

    service.ask("query", top_k=5, client_id="user-1")

    assert reranker.called is True


def test_same_client_id_gets_stable_canary_bucketing():

    flags = FeatureFlagManager()
    flags.define(RERANKER_FLAG_NAME, enabled=True, rollout_percentage=50.0)
    service = RAGService(reranker=_StubReranker(), feature_flags=flags)

    first = service._reranker_enabled_for("stable-client")
    second = service._reranker_enabled_for("stable-client")

    assert first == second
