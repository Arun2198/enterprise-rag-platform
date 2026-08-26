# Deliberately no re-exports here. rag.vector_store.in_memory_store
# imports rag.retrieval.bm25 (for real BM25 scoring); if this package's
# __init__ eagerly imported hybrid_retrieval (which imports
# rag.vector_store.base), importing rag.vector_store before rag.retrieval
# triggers a circular import - rag.vector_store.base ends up mid-init
# when hybrid_retrieval asks for VectorStore from it. Verified against a
# real script (scripts/opensearch_smoke_test.py) hitting exactly this
# import order; the test suite's own import order happened not to.
# Import submodules directly instead: from rag.retrieval.hybrid_retrieval
# import HybridRetriever.
