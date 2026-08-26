from unittest.mock import MagicMock
from unittest.mock import patch

from rag.embeddings.sentence_transformer_embedder import SentenceTransformerEmbedder


@patch("rag.embeddings.sentence_transformer_embedder.SentenceTransformer")
def test_constructs_the_underlying_model_with_the_given_name(mock_model_class):

    SentenceTransformerEmbedder(model_name="BAAI/bge-base-en-v1.5")

    mock_model_class.assert_called_once_with("BAAI/bge-base-en-v1.5")


@patch("rag.embeddings.sentence_transformer_embedder.SentenceTransformer")
def test_defaults_to_bge_small_when_no_model_name_given(mock_model_class):

    SentenceTransformerEmbedder()

    mock_model_class.assert_called_once_with("BAAI/bge-small-en-v1.5")


@patch("rag.embeddings.sentence_transformer_embedder.SentenceTransformer")
def test_dimensions_reflects_the_loaded_models_embedding_size(mock_model_class):

    mock_model_class.return_value.get_embedding_dimension.return_value = 768

    embedder = SentenceTransformerEmbedder(model_name="BAAI/bge-base-en-v1.5")

    assert embedder.dimensions == 768


@patch("rag.embeddings.sentence_transformer_embedder.SentenceTransformer")
def test_dimensions_falls_back_to_the_old_method_name_on_older_versions(mock_model_class):
    """
    get_sentence_embedding_dimension() was renamed to
    get_embedding_dimension() in sentence-transformers 5.x - an older
    installed version won't have the new name at all.
    """
    del mock_model_class.return_value.get_embedding_dimension
    mock_model_class.return_value.get_sentence_embedding_dimension.return_value = 384

    embedder = SentenceTransformerEmbedder(model_name="BAAI/bge-base-en-v1.5")

    assert embedder.dimensions == 384


@patch("rag.embeddings.sentence_transformer_embedder.SentenceTransformer")
def test_embed_encodes_with_normalization_and_returns_a_plain_list(mock_model_class):

    mock_model = mock_model_class.return_value
    fake_vector = MagicMock()
    fake_vector.tolist.return_value = [0.1, 0.2, 0.3]
    mock_model.encode.return_value = fake_vector

    embedder = SentenceTransformerEmbedder()
    result = embedder.embed("some text")

    mock_model.encode.assert_called_once_with("some text", normalize_embeddings=True)
    assert result == [0.1, 0.2, 0.3]


@patch("rag.embeddings.sentence_transformer_embedder.SentenceTransformer")
def test_model_is_loaded_once_and_reused_across_calls(mock_model_class):

    fake_vector = MagicMock()
    fake_vector.tolist.return_value = [0.0]
    mock_model_class.return_value.encode.return_value = fake_vector

    embedder = SentenceTransformerEmbedder()
    embedder.embed("first")
    embedder.embed("second")

    mock_model_class.assert_called_once()
    assert mock_model_class.return_value.encode.call_count == 2


@patch("rag.embeddings.sentence_transformer_embedder.SentenceTransformer")
def test_embed_batch_encodes_all_texts_in_one_call(mock_model_class):

    mock_model = mock_model_class.return_value
    fake_matrix = MagicMock()
    fake_matrix.tolist.return_value = [[0.1, 0.2], [0.3, 0.4]]
    mock_model.encode.return_value = fake_matrix

    embedder = SentenceTransformerEmbedder()
    result = embedder.embed_batch(["first", "second"])

    mock_model.encode.assert_called_once_with(["first", "second"], normalize_embeddings=True)
    assert result == [[0.1, 0.2], [0.3, 0.4]]


@patch("rag.embeddings.sentence_transformer_embedder.SentenceTransformer")
def test_embed_batch_of_empty_list_makes_no_model_call(mock_model_class):

    embedder = SentenceTransformerEmbedder()

    result = embedder.embed_batch([])

    assert result == []
    mock_model_class.return_value.encode.assert_not_called()
