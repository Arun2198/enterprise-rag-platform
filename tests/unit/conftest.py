from unittest.mock import patch


class _FakeCrossEncoder:
    """
    Session-wide stand-in for sentence_transformers.CrossEncoder. Reranking
    is enabled by default (RERANKER_ENABLED defaults to true), so anything
    that builds a RAGService through the normal wiring - including
    app.main's module-level build_rag_service() call - would otherwise
    download a real model. Individual tests that care about specific
    CrossEncoder behavior patch over this with their own mock.
    """

    def __init__(self, *args, **kwargs) -> None:
        pass

    def predict(self, pairs, *args, **kwargs):
        return [0.0 for _ in pairs]


patch("rag.retrieval.reranker.CrossEncoder", _FakeCrossEncoder).start()
