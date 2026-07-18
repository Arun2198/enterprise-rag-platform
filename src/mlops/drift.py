from typing import Protocol

from mlops.schemas import DriftReport

DRIFT_TYPES = (
    "data",
    "embedding",
    "retrieval",
    "prompt",
    "model",
    "user_query",
)


class DriftDetector(Protocol):
    """
    Not implemented - design-only interface so future detectors can
    register with PlatformManager (register_provider) without it or
    anything downstream changing:

    - data drift: distribution shift in ingested document content
    - embedding drift: centroid/cosine-distance shift in the embedding
      space over time
    - retrieval drift: shift in the retrieval score distribution or
      hit-rate against a fixed evaluation set (evaluation.runner is the
      natural source of the numbers a detector like this would watch)
    - prompt drift: unintended changes to a prompt template between
      artifact versions (artifacts.ArtifactRegistry.history() is the
      natural diff source)
    - model drift: shift in an LLM/embedding provider's output
      distribution after a provider-side model update
    - user query drift: shift in the distribution/clustering of
      incoming user queries over time

    A concrete detector takes whatever data it needs and returns a
    DriftReport; nothing about how it's invoked is prescribed here.
    """
    drift_type: str

    def detect(self, *args: object, **kwargs: object) -> DriftReport:
        ...
