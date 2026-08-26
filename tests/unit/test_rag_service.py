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


class _SpyEmbedder:

    dimensions = 384

    def __init__(self):
        self.embed_batch_calls = []
        self.embed_calls = []

    def embed(self, text):
        self.embed_calls.append(text)
        return [0.1] * self.dimensions

    def embed_batch(self, texts):
        self.embed_batch_calls.append(list(texts))
        return [[0.1] * self.dimensions for _ in texts]


def test_ingest_embeds_all_chunks_of_a_document_in_one_batch_call(tmp_path):

    file_path = tmp_path / "leave_policy.md"
    file_path.write_text(
        "# Leave Policy\n"
        "Employees receive 20 days of paid leave annually. "
        "Contractors receive 10 days of leave.",
        encoding="utf-8"
    )
    embedder = _SpyEmbedder()
    service = RAGService(
        embedder=embedder,
        chunker=RecursiveChunker(chunk_size=120, chunk_overlap=20, minimum_chunk_size=10)
    )

    service.ingest([str(file_path)])

    assert len(embedder.embed_batch_calls) == 1
    assert embedder.embed_calls == []
    assert len(embedder.embed_batch_calls[0]) >= 1


def test_ingest_stamps_embedding_lineage_onto_indexed_chunks(tmp_path):

    file_path = tmp_path / "leave_policy.md"
    file_path.write_text("Employees receive 20 days of paid leave annually.", encoding="utf-8")
    service = RAGService(chunker=RecursiveChunker(chunk_size=900, chunk_overlap=50, minimum_chunk_size=10))

    service.ingest([str(file_path)])

    stored_chunk = service.vector_store.get("leave_policy:0")
    assert stored_chunk.embedding_provider == "hashing"
    assert stored_chunk.embedding_model == "hashing-384"
    assert stored_chunk.embedding_version == "384"
    assert stored_chunk.indexed_at is not None
    assert stored_chunk.content_hash is not None
    assert stored_chunk.chunking_version == "recursive:900:50:10"


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


def test_ingest_allows_any_path_when_no_allowed_dir_configured(tmp_path):

    file_path = tmp_path / "notes.md"
    file_path.write_text("# Notes\nSome content here for the pipeline.", encoding="utf-8")
    service = RAGService()

    response = service.ingest([str(file_path)])

    assert response.indexed_documents == 1
    assert response.errors == []


def test_ingest_allows_files_inside_the_configured_directory(tmp_path):

    allowed_dir = tmp_path / "allowed"
    allowed_dir.mkdir()
    file_path = allowed_dir / "notes.md"
    file_path.write_text("# Notes\nSome content here for the pipeline.", encoding="utf-8")
    service = RAGService(ingest_allowed_dir=str(allowed_dir))

    response = service.ingest([str(file_path)])

    assert response.indexed_documents == 1
    assert response.errors == []


def test_ingest_rejects_a_path_outside_the_configured_directory(tmp_path):

    allowed_dir = tmp_path / "allowed"
    allowed_dir.mkdir()
    outside_file = tmp_path / "secret.md"
    outside_file.write_text("# Secret\nShould never be readable via ingest.", encoding="utf-8")
    service = RAGService(ingest_allowed_dir=str(allowed_dir))

    response = service.ingest([str(outside_file)])

    assert response.indexed_documents == 0
    assert response.indexed_chunks == 0
    assert "PATH_NOT_ALLOWED" in response.errors[0]


def test_ingest_rejects_a_traversal_path_that_escapes_the_allowed_directory(tmp_path):

    allowed_dir = tmp_path / "allowed"
    allowed_dir.mkdir()
    outside_file = tmp_path / "secret.md"
    outside_file.write_text("# Secret\nShould never be readable via ingest.", encoding="utf-8")
    service = RAGService(ingest_allowed_dir=str(allowed_dir))

    traversal_path = str(allowed_dir / ".." / "secret.md")
    response = service.ingest([traversal_path])

    assert response.indexed_documents == 0
    assert "PATH_NOT_ALLOWED" in response.errors[0]


def test_ingest_rejects_an_absolute_path_outside_the_allowed_directory(tmp_path):

    allowed_dir = tmp_path / "allowed"
    allowed_dir.mkdir()
    service = RAGService(ingest_allowed_dir=str(allowed_dir))

    response = service.ingest(["/etc/passwd"])

    assert response.indexed_documents == 0
    assert "PATH_NOT_ALLOWED" in response.errors[0]


def test_ingest_allows_a_file_directly_at_the_allowed_directory_root(tmp_path):

    allowed_dir = tmp_path / "allowed"
    allowed_dir.mkdir()
    file_path = allowed_dir / "root.md"
    file_path.write_text("# Root\nContent directly in the allowed directory.", encoding="utf-8")
    service = RAGService(ingest_allowed_dir=str(allowed_dir))

    response = service.ingest([str(file_path)])

    assert response.indexed_documents == 1


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
        candidate_multiplier=4,
        hallucination_guard_enabled=False
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

    service = RAGService(
        answerer=_FixedAnswerer("Contact john@company.com for help."),
        hallucination_guard_enabled=False
    )
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


def _chunk_with_access(chunk_id, text, access_groups=None):
    return Chunk(
        chunk_id=chunk_id, document_id="doc", chunk_index=0, text=text,
        source="doc.md", document_type="markdown",
        access_groups=access_groups or []
    )


def test_unrestricted_chunks_are_visible_to_any_caller(tmp_path):

    file_path = tmp_path / "public.md"
    file_path.write_text("Public information anyone can read about the office.", encoding="utf-8")
    service = RAGService(chunker=RecursiveChunker(chunk_size=900, chunk_overlap=50, minimum_chunk_size=10))
    service.ingest([str(file_path)])

    response = service.ask("office", access_groups=None)

    assert response.sources


def test_restricted_chunk_is_excluded_when_caller_has_no_matching_group():

    service = RAGService()
    restricted_chunk = _chunk_with_access("doc:0", "Confidential salary information for engineering.", access_groups=["finance"])
    service.vector_store.add(restricted_chunk, service.embedder.embed(restricted_chunk.text))

    response = service.ask("salary information", access_groups=["engineering"])

    assert response.sources == []
    assert "salary" not in response.answer.lower() or "could not find" in response.answer.lower()


def test_restricted_chunk_is_included_when_caller_has_a_matching_group():

    service = RAGService()
    restricted_chunk = _chunk_with_access("doc:0", "Confidential salary information for engineering.", access_groups=["finance"])
    service.vector_store.add(restricted_chunk, service.embedder.embed(restricted_chunk.text))

    response = service.ask("salary information", access_groups=["finance", "engineering"])

    assert response.sources
    assert response.sources[0].chunk_id == "doc:0"


def test_restricted_chunk_is_excluded_when_no_access_groups_provided():

    service = RAGService()
    restricted_chunk = _chunk_with_access("doc:0", "Confidential salary information.", access_groups=["finance"])
    service.vector_store.add(restricted_chunk, service.embedder.embed(restricted_chunk.text))

    response = service.ask("salary information", access_groups=None)

    assert response.sources == []


def test_confidence_uses_groundedness_not_raw_retrieval_score(tmp_path):
    """
    A near-perfect retrieval match (high vector/rerank score) paired with
    a low-groundedness answer must NOT be reported as high confidence -
    that's exactly the "retrieval score as answer confidence" mistake
    this separation exists to avoid.
    """
    file_path = tmp_path / "policy.md"
    file_path.write_text("Contractors receive 10 days of leave per year.", encoding="utf-8")
    service = RAGService(
        hallucination_guard_enabled=True,
        groundedness_threshold=0.9  # deliberately strict so groundedness comes back low
    )
    service.ingest([str(file_path)])

    response = service.ask("How many leave days do contractors receive?")

    assert response.groundedness is not None
    assert response.confidence == round(max(0.0, min(response.groundedness, 1.0)), 4)


def test_confidence_falls_back_to_retrieval_score_when_hallucination_guard_disabled(tmp_path):

    file_path = tmp_path / "policy.md"
    file_path.write_text("Contractors receive 10 days of leave per year.", encoding="utf-8")
    service = RAGService(hallucination_guard_enabled=False)
    service.ingest([str(file_path)])

    response = service.ask("How many leave days do contractors receive?")

    assert response.groundedness is None
    assert response.sources
    top_retrieval_score = max(s.score for s in response.sources)
    assert response.confidence == round(max(0.0, min(top_retrieval_score, 1.0)), 4)


def test_sources_expose_document_version_section_and_retrieval_method(tmp_path):

    file_path = tmp_path / "policy.md"
    file_path.write_text(
        "# Leave Policy\nContractors receive 10 days of leave per year.",
        encoding="utf-8"
    )
    service = RAGService()
    service.ingest([str(file_path)])

    response = service.ask("How many leave days do contractors receive?")

    source = response.sources[0]
    assert source.document_version == 1
    assert source.section == "Leave Policy"
    assert source.retrieval_method in ("dense", "bm25", "both")
    assert source.rank >= 1


def test_low_groundedness_answer_is_replaced_with_an_abstention_message(tmp_path):

    from app.services.rag_service import ABSTENTION_MESSAGE

    file_path = tmp_path / "policy.md"
    file_path.write_text("# Policy\nContractors receive 10 days of leave.", encoding="utf-8")
    service = RAGService(answerer=_FixedAnswerer("Completely unrelated statement about astronomy."))
    service.ingest([str(file_path)])

    response = service.ask("policy question", top_k=3)

    assert response.answer == ABSTENTION_MESSAGE
    assert response.guardrail_flags["hallucination"] is True


def test_abstention_keeps_sources_for_auditability(tmp_path):

    file_path = tmp_path / "policy.md"
    file_path.write_text("# Policy\nContractors receive 10 days of leave.", encoding="utf-8")
    service = RAGService(answerer=_FixedAnswerer("Completely unrelated statement about astronomy."))
    service.ingest([str(file_path)])

    response = service.ask("policy question", top_k=3)

    assert response.sources


def test_abstention_can_be_disabled(tmp_path):

    file_path = tmp_path / "policy.md"
    file_path.write_text("# Policy\nContractors receive 10 days of leave.", encoding="utf-8")
    service = RAGService(
        answerer=_FixedAnswerer("Completely unrelated statement about astronomy."),
        abstention_enabled=False
    )
    service.ingest([str(file_path)])

    response = service.ask("policy question", top_k=3)

    assert response.answer == "Completely unrelated statement about astronomy."


def test_grounded_answer_is_not_replaced_by_abstention(tmp_path):

    file_path = tmp_path / "policy.md"
    file_path.write_text("Contractors receive 10 days of leave per year.", encoding="utf-8")
    service = RAGService()
    service.ingest([str(file_path)])

    response = service.ask("How many leave days do contractors receive?")

    assert "10 days" in response.answer


def test_delete_document_removes_all_its_chunks(tmp_path):

    file_path = tmp_path / "policy.md"
    file_path.write_text("# Policy\nSome policy content here that is long enough.", encoding="utf-8")
    service = RAGService(chunker=RecursiveChunker(chunk_size=30, chunk_overlap=5, minimum_chunk_size=5))
    ingest_response = service.ingest([str(file_path)])

    deleted_count = service.delete_document("policy")

    assert deleted_count == ingest_response.indexed_chunks
    assert len(service.vector_store) == 0


def test_delete_document_that_was_never_indexed_returns_zero():

    service = RAGService()

    assert service.delete_document("does-not-exist") == 0


def test_deleted_document_chunks_no_longer_appear_in_ask_sources(tmp_path):

    file_path = tmp_path / "policy.md"
    file_path.write_text("Contractors receive 10 days of leave per year.", encoding="utf-8")
    service = RAGService()
    service.ingest([str(file_path)])

    service.delete_document("policy")
    response = service.ask("How many leave days do contractors receive?")

    assert response.sources == []


def test_reindex_document_replaces_old_chunks_with_new_ones(tmp_path):

    file_path = tmp_path / "policy.md"
    file_path.write_text("Contractors receive 10 days of leave.", encoding="utf-8")
    service = RAGService()
    service.ingest([str(file_path)])
    assert len(service.vector_store) == 1

    file_path.write_text("Contractors receive 15 days of leave now.", encoding="utf-8")
    result = service.reindex_document(str(file_path))

    assert result.indexed_documents == 1
    assert len(service.vector_store) == 1
    response = service.ask("How many leave days do contractors receive?")
    assert "15 days" in response.answer


def test_reindex_document_reports_error_for_a_missing_file(tmp_path):

    service = RAGService()

    result = service.reindex_document(str(tmp_path / "does-not-exist.md"))

    assert result.indexed_documents == 0
    assert result.errors


def test_ask_with_trace_returns_matching_ask_response(tmp_path):

    file_path = tmp_path / "policy.md"
    file_path.write_text("Contractors receive 10 days of leave per year.", encoding="utf-8")
    service = RAGService()
    service.ingest([str(file_path)])

    plain_response = service.ask("How many leave days do contractors receive?")
    traced_response, trace = service.ask_with_trace("How many leave days do contractors receive?")

    assert traced_response.answer == plain_response.answer
    assert traced_response.confidence == plain_response.confidence
    assert [s.chunk_id for s in traced_response.sources] == [s.chunk_id for s in plain_response.sources]


def test_ask_with_trace_records_dense_and_bm25_candidates(tmp_path):

    file_path = tmp_path / "policy.md"
    file_path.write_text("Contractors receive 10 days of leave per year.", encoding="utf-8")
    service = RAGService()
    service.ingest([str(file_path)])

    _, trace = service.ask_with_trace("How many leave days do contractors receive?")

    assert trace.query == "How many leave days do contractors receive?"
    assert len(trace.dense_candidates) >= 1
    assert len(trace.bm25_candidates) >= 1
    assert len(trace.fused_candidates) >= 1
    assert trace.final_chunk_ids
    assert "embedding" in trace.stage_timings_ms
    assert "dense_search" in trace.stage_timings_ms
    assert "bm25_search" in trace.stage_timings_ms
    assert "rrf_fusion" in trace.stage_timings_ms
    assert "generation" in trace.stage_timings_ms
    assert "total" in trace.stage_timings_ms


def test_ask_with_trace_records_reranker_candidates_when_reranker_configured(tmp_path):

    file_path = tmp_path / "policy.md"
    file_path.write_text("Contractors receive 10 days of leave per year.", encoding="utf-8")
    service = RAGService(reranker=_StubReranker())
    service.ingest([str(file_path)])

    _, trace = service.ask_with_trace("How many leave days do contractors receive?")

    assert trace.reranker_used is True
    assert "rerank" in trace.stage_timings_ms


def test_ask_with_trace_records_groundedness_and_guardrail_findings(tmp_path):

    file_path = tmp_path / "policy.md"
    file_path.write_text("Contractors receive 10 days of leave per year.", encoding="utf-8")
    service = RAGService()
    service.ingest([str(file_path)])

    _, trace = service.ask_with_trace("How many leave days do contractors receive?")

    assert trace.groundedness is not None
    assert isinstance(trace.guardrail_findings, list)


def test_ask_with_trace_short_circuits_on_blocked_input():

    from rag.guardrails.base import GuardrailFinding
    from rag.guardrails.base import GuardrailStage
    from rag.guardrails.base import Severity
    from rag.guardrails.manager import GuardrailManager

    class _AlwaysBlockInput:
        stage = GuardrailStage.INPUT
        name = "always_block"

        def check(self, context):
            return GuardrailFinding(
                guardrail_name=self.name,
                triggered=True,
                severity=Severity.CRITICAL,
                action=Action.BLOCK,
                message="blocked for test"
            )

    service = RAGService(guardrail_manager=GuardrailManager(guardrails=[_AlwaysBlockInput()]))

    response, trace = service.ask_with_trace("anything")

    assert response.sources == []
    assert trace.query == "anything"
    assert "total" in trace.stage_timings_ms


class _MarkerEmbedder:
    """
    Deterministic 2D embedder for retrieval-relevance tests: text
    containing "MARKER" embeds to [1, 0], everything else to [0, 1] - lets
    a test construct an exact "query matches indexed content" (cosine 1.0)
    or "query is unrelated to indexed content" (cosine 0.0) scenario
    without depending on a real embedding model's actual output.
    """
    dimensions = 2
    provider_name = "test_dense"
    model_name = "marker-embedder"

    def embed(self, text):
        return [1.0, 0.0] if "MARKER" in text else [0.0, 1.0]

    def embed_batch(self, texts):
        return [self.embed(t) for t in texts]


def _service_with_retrieval_relevance_guard(tmp_path, threshold=0.5):
    file_path = tmp_path / "policy.md"
    file_path.write_text(
        "MARKER Contractors receive 10 days of leave per year.", encoding="utf-8"
    )
    service = RAGService(
        embedder=_MarkerEmbedder(),
        retrieval_relevance_guard_enabled=True,
        retrieval_relevance_threshold=threshold
    )
    service.ingest([str(file_path)])
    return service


def test_low_retrieval_relevance_triggers_abstention_even_with_high_groundedness(tmp_path):

    from app.services.rag_service import ABSTENTION_MESSAGE

    service = _service_with_retrieval_relevance_guard(tmp_path)

    # query has no "MARKER" - unrelated to the only indexed content by
    # construction (cosine similarity 0.0) - but ExtractiveAnswerer will
    # still copy verbatim from the retrieved chunk, so groundedness stays
    # high (the known gap this guard exists to catch)
    response = service.ask("completely unrelated topic")

    assert response.answer == ABSTENTION_MESSAGE
    assert response.guardrail_flags["low_retrieval_relevance"] is True


def test_high_retrieval_relevance_does_not_trigger_abstention(tmp_path):

    service = _service_with_retrieval_relevance_guard(tmp_path)

    response = service.ask("MARKER how many leave days")

    assert response.guardrail_flags["low_retrieval_relevance"] is False
    assert "10 days" in response.answer


def test_retrieval_relevance_guard_disabled_by_default(tmp_path):

    file_path = tmp_path / "policy.md"
    file_path.write_text("MARKER Contractors receive 10 days of leave.", encoding="utf-8")
    service = RAGService(embedder=_MarkerEmbedder())
    service.ingest([str(file_path)])

    response = service.ask("completely unrelated topic")

    assert "low_retrieval_relevance" not in response.guardrail_flags


def test_should_abstain_combines_hallucination_and_low_relevance_signals():

    service = RAGService()

    assert service._should_abstain({"hallucination": True, "low_retrieval_relevance": False}) is True
    assert service._should_abstain({"hallucination": False, "low_retrieval_relevance": True}) is True
    assert service._should_abstain({"hallucination": False, "low_retrieval_relevance": False}) is False
    assert service._should_abstain({}) is False


def test_ask_extracts_valid_citations_from_llm_style_answer(tmp_path):

    file_path = tmp_path / "policy.md"
    file_path.write_text("Contractors receive 10 days of leave per year.", encoding="utf-8")
    service = RAGService(answerer=_FixedAnswerer(
        "Contractors receive 10 days of leave [Source 1]."
    ))
    service.ingest([str(file_path)])

    response = service.ask("How many leave days do contractors receive?")

    assert len(response.citations) == 1
    assert response.citations[0].valid is True
    assert response.citations[0].source_number == 1
    assert response.guardrail_flags["has_invalid_citations"] is False


def test_ask_flags_invalid_citation_to_a_nonexistent_source(tmp_path):

    file_path = tmp_path / "policy.md"
    file_path.write_text("Contractors receive 10 days of leave per year.", encoding="utf-8")
    service = RAGService(answerer=_FixedAnswerer(
        "Contractors receive 10 days of leave per year [Source 1] [Source 9]."
    ))
    service.ingest([str(file_path)])

    response = service.ask("How many leave days do contractors receive?")

    assert len(response.citations) == 2
    assert response.citations[1].source_number == 9
    assert response.citations[1].valid is False
    assert response.guardrail_flags["has_invalid_citations"] is True


def test_ask_returns_no_citations_for_extractive_answers(tmp_path):

    file_path = tmp_path / "policy.md"
    file_path.write_text("Contractors receive 10 days of leave per year.", encoding="utf-8")
    service = RAGService()
    service.ingest([str(file_path)])

    response = service.ask("How many leave days do contractors receive?")

    assert response.citations == []
    assert "has_invalid_citations" not in response.guardrail_flags
