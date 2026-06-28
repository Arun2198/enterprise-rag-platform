import hashlib
import math
import re


class HashingEmbedder:
    """
    Local deterministic embedding baseline.

    This keeps the MVP runnable without external model credentials. The vector
    store interface can later be backed by BGE/OpenAI/Bedrock embeddings.
    """

    def __init__(
        self,
        dimensions: int = 384
    ) -> None:
        self.dimensions = dimensions

    def embed(
        self,
        text: str
    ) -> list[float]:
        vector = [0.0] * self.dimensions

        for token in self._tokens(text):
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign

        norm = math.sqrt(sum(value * value for value in vector))

        if norm == 0:
            return vector

        return [
            value / norm
            for value in vector
        ]

    def _tokens(
        self,
        text: str
    ) -> list[str]:
        return re.findall(r"[a-z0-9]+", text.lower())
