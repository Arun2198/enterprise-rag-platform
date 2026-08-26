class RerankerProviderError(RuntimeError):
    """
    Raised when an API-based reranker exhausts retries or hits a
    non-retryable failure - not swallowed into "just return the
    unreranked candidates", since that would silently change ranking
    quality without anyone noticing. Callers that want a fallback should
    wrap the reranker explicitly (the same pattern FallbackAnswerer uses
    for generation), not have one baked in silently here.
    """
    pass
