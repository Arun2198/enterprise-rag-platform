class EmbeddingProviderError(RuntimeError):
    """
    Raised when an API-based embedder exhausts retries or hits a
    non-retryable failure. Deliberately not swallowed into a fallback
    vector the way generation falls back to a fixed string - a bad
    embedding silently corrupts the index, so ingestion should fail loudly
    instead.
    """
    pass
