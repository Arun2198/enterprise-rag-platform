from typing import Protocol


class Embedder(Protocol):

    dimensions: int
    provider_name: str

    def embed(
        self,
        text: str
    ) -> list[float]:
        ...

    def embed_batch(
        self,
        texts: list[str]
    ) -> list[list[float]]:
        ...
