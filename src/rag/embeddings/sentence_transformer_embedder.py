from sentence_transformers import SentenceTransformer


class SentenceTransformerEmbedder:
    """
    Embeds text using a local sentence-transformers model.

    Loads the model once and reuses it for every call, since spinning up a
    transformer per request would make this unusably slow. Default model is
    BAAI/bge-small-en-v1.5 - small enough to run on CPU, good enough for a
    single-tenant / demo-scale corpus.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5"
    ) -> None:
        self.model_name = model_name
        self._model = SentenceTransformer(model_name)
        self.dimensions = self._model.get_sentence_embedding_dimension()

    def embed(
        self,
        text: str
    ) -> list[float]:
        embedding = self._model.encode(
            text,
            normalize_embeddings=True
        )
        return embedding.tolist()
