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
        self.provider_name = "sentence_transformer"
        self.model_name = model_name
        self._model = SentenceTransformer(model_name)
        # get_sentence_embedding_dimension() was renamed to
        # get_embedding_dimension() in sentence-transformers 5.x - use
        # whichever the installed version actually has rather than
        # pinning to one and eating a FutureWarning (or breaking outright
        # on a version where the old name is finally removed).
        if hasattr(self._model, "get_embedding_dimension"):
            self.dimensions = self._model.get_embedding_dimension()
        else:
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

    def embed_batch(
        self,
        texts: list[str]
    ) -> list[list[float]]:
        if not texts:
            return []

        embeddings = self._model.encode(
            texts,
            normalize_embeddings=True
        )
        return embeddings.tolist()
