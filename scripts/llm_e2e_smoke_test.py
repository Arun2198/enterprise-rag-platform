"""
End-to-end smoke test for OpenAICompatibleAnswerer against a real LLM endpoint.

Not part of the pytest suite (unit tests must not make network calls) - run
this directly, with GENERATION_PROVIDER/LLM_* config supplied via environment
variables. See .github/workflows/llm-e2e-smoke-test.yml for the CI wiring.
"""
import sys

from app.config import load_settings
from app.service_factory import build_rag_service
from rag.generation.openai_compatible_answerer import FALLBACK_ERROR
from rag.generation.openai_compatible_answerer import FALLBACK_NO_CONTEXT

SAMPLE_DOCUMENT = "sample_documents/AI-RMF-1stdraft.pdf"
QUESTION = "What does NIST mean by trustworthy AI?"


def main() -> int:
    settings = load_settings()
    service = build_rag_service(settings)

    print(f"answerer: {type(service.answerer).__name__}")
    print(f"model: {settings.llm_model_name}")

    ingest = service.ingest([SAMPLE_DOCUMENT])
    print(f"ingested: {ingest.indexed_documents} docs, {ingest.indexed_chunks} chunks")

    if ingest.errors:
        print(f"ingest errors: {ingest.errors}")
        return 1

    response = service.ask(QUESTION, top_k=4)
    print(f"question: {QUESTION}")
    print(f"answer: {response.answer}")

    if response.answer in (FALLBACK_NO_CONTEXT, FALLBACK_ERROR):
        print("FAILED: got a fallback answer instead of a real LLM response")
        return 1

    print("OK: received a grounded answer from the live endpoint")
    return 0


if __name__ == "__main__":
    sys.exit(main())
