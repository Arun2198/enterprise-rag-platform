from app.services.rag_service import ABSTENTION_MESSAGE
from app.services.rag_service import RAGService
from evaluation.robustness import load_robustness_dataset
from evaluation.robustness import run_robustness_eval
from rag.guardrails.manager import GuardrailManager
from rag.guardrails.pii_guard import PIIGuard
from rag.guardrails.hallucination_detector import HallucinationDetector
from rag.guardrails.prompt_injection_guard import PromptInjectionGuard


def _service_with_prompt_injection_guard() -> RAGService:
    """
    Bare RAGService() only wires PIIGuard/HallucinationDetector by default
    (see GuardrailManager.default) - PromptInjectionGuard is opt-in,
    normally wired by service_factory for the live app. The robustness
    dataset's adversarial cases need it present to mean anything, so build
    the same guardrail set service_factory would.
    """
    service = RAGService()
    service.guardrail_manager = GuardrailManager(guardrails=[
        PromptInjectionGuard(),
        PIIGuard(),
        HallucinationDetector(threshold=0.60, embedder=service.embedder),
    ])
    return service


def test_robustness_dataset_loads_and_validates():

    dataset = load_robustness_dataset("evaluation/robustness_dataset.json")

    assert dataset.name == "ai-rmf-1stdraft-robustness"
    assert len(dataset.cases) == 8
    assert {case.category for case in dataset.cases} == {"unanswerable", "adversarial"}


def test_adversarial_cases_are_reliably_blocked_against_the_real_ai_rmf_pdf():
    """
    Real end-to-end check, same spirit as
    test_evaluation_integration.py's Layer 1 test: ingest the actual
    sample_documents/AI-RMF-1stdraft.pdf through a real RAGService and
    exercise every adversarial case in the dataset against it. This is the
    part of robustness that's provider-independent - PromptInjectionGuard
    runs on the raw query before retrieval/generation even happen, so it
    doesn't matter which Answerer is configured.
    """
    service = _service_with_prompt_injection_guard()
    service.ingest(["sample_documents/AI-RMF-1stdraft.pdf"])
    dataset = load_robustness_dataset("evaluation/robustness_dataset.json")

    report = run_robustness_eval(service, dataset, ABSTENTION_MESSAGE)

    adversarial_failures = [
        r for r in report.results if r.category == "adversarial" and not r.passed
    ]
    assert not adversarial_failures, f"adversarial robustness failures: {adversarial_failures}"


def test_known_gap_with_hash_quality_embeddings_extractive_answerer_does_not_abstain():
    """
    Groundedness measures whether the ANSWER matches the retrieved
    CHUNKS, not whether the chunks are actually relevant to the QUERY.
    ExtractiveAnswerer's "answer" is always copied verbatim from a
    retrieved chunk, so it's tautologically high-groundedness even when
    every retrieved chunk is topically irrelevant.

    A real fix now exists: RetrievalRelevanceGuard
    (rag/guardrails/retrieval_relevance_guard.py) adds the missing
    signal - cosine similarity between the query and its best-matching
    retrieved chunk - and RAGService abstains on either groundedness or
    this new signal (see RAGService._should_abstain). Verified with a
    genuine dense embedder (BAAI/bge-small-en-v1.5) via
    scripts/retrieval_relevance_guard_verification.py: zero false
    positives across all 24 real golden_dataset.json queries, 3 of 4
    known-gap unanswerable queries now correctly caught.

    This specific test still shows the gap because HashingEmbedder (the
    default here, and what every test in this suite effectively runs on -
    tests/unit/conftest.py's SentenceTransformer mock also delegates to
    HashingEmbedder for speed) does not separate relevant from irrelevant
    queries reliably enough for this signal to be safe: real answerable
    golden-dataset queries score as low as 0.29 cosine similarity,
    overlapping with unanswerable queries up to 0.43 - verified by the
    same script. RetrievalRelevanceGuard is therefore deliberately left
    disabled by default everywhere (GuardrailManager.default(),
    RETRIEVAL_RELEVANCE_GUARD_ENABLED) and is a real no-op at its
    HashingEmbedder-appropriate default threshold rather than a
    falsely-confident guess. This test intentionally keeps the guard
    off, so it documents the honest floor: without a genuine dense
    embedder, this specific gap is not fixable by this signal.
    """
    service = _service_with_prompt_injection_guard()
    service.ingest(["sample_documents/AI-RMF-1stdraft.pdf"])
    dataset = load_robustness_dataset("evaluation/robustness_dataset.json")

    report = run_robustness_eval(service, dataset, ABSTENTION_MESSAGE)

    unanswerable_results = [r for r in report.results if r.category == "unanswerable"]
    assert unanswerable_results
    assert all(r.observed_action == "answered" for r in unanswerable_results), (
        "if this now fails, HashingEmbedder's separation has somehow "
        "improved - re-run scripts/retrieval_relevance_guard_verification.py "
        "and update this test and its docstring"
    )


def test_robustness_eval_reports_a_failure_when_a_case_is_not_met():

    class _AlwaysAnswersConfidently:
        def ask(self, query, top_k=5):
            from app.schemas import AskResponse
            from app.schemas import Source
            return AskResponse(
                answer="a confident but wrong answer",
                sources=[Source(
                    document_id="d1", chunk_id="d1:0", source="test",
                    score=1.0, text="irrelevant"
                )],
                confidence=0.9,
                guardrail_flags={}
            )

    dataset = load_robustness_dataset("evaluation/robustness_dataset.json")
    report = run_robustness_eval(_AlwaysAnswersConfidently(), dataset, ABSTENTION_MESSAGE)

    assert report.pass_rate < 1.0
    assert any(not r.passed for r in report.results)
