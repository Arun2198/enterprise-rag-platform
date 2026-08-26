from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from botocore.credentials import ReadOnlyCredentials

from rag.vector_store.opensearch_client_factory import OpenSearchHTTPError
from rag.vector_store.opensearch_client_factory import SignedOpenSearchClient
from rag.vector_store.opensearch_client_factory import build_opensearch_client


@patch("rag.vector_store.opensearch_client_factory.boto3.Session")
def test_build_opensearch_client_returns_a_signed_client(mock_session_class):

    mock_session_class.return_value.get_credentials.return_value = MagicMock()

    client = build_opensearch_client(
        host="search-domain.us-east-1.es.amazonaws.com",
        region="us-east-1"
    )

    assert isinstance(client, SignedOpenSearchClient)


@patch("rag.vector_store.opensearch_client_factory.boto3.Session")
def test_build_opensearch_client_raises_a_clear_error_with_no_credentials(mock_session_class):

    mock_session_class.return_value.get_credentials.return_value = None

    with pytest.raises(RuntimeError, match="No AWS credentials"):
        build_opensearch_client(
            host="search-domain.us-east-1.es.amazonaws.com",
            region="us-east-1"
        )


@patch("rag.vector_store.opensearch_client_factory.boto3.Session")
def test_signed_client_signs_and_sends_a_search_request(mock_session_class):

    mock_credentials = MagicMock()
    mock_credentials.get_frozen_credentials.return_value = ReadOnlyCredentials(
        "fake-access-key", "fake-secret-key", None
    )
    mock_session_class.return_value.get_credentials.return_value = mock_credentials

    client = build_opensearch_client(host="search-domain.us-east-1.es.amazonaws.com", region="us-east-1")

    fake_response = MagicMock()
    fake_response.ok = True
    fake_response.json.return_value = {"hits": {"hits": []}}

    with patch.object(client._session, "request", return_value=fake_response) as mock_request:
        result = client.search(index="chunks", body={"query": {"match_all": {}}})

    assert result == {"hits": {"hits": []}}
    called_url = mock_request.call_args.args[1]
    assert called_url.endswith("/chunks/_search")
    assert mock_request.call_args.kwargs["headers"]["Content-Type"] == "application/json"


@patch("rag.vector_store.opensearch_client_factory.boto3.Session")
def test_signed_client_sends_bulk_as_ndjson(mock_session_class):

    mock_credentials = MagicMock()
    mock_credentials.get_frozen_credentials.return_value = ReadOnlyCredentials(
        "fake-access-key", "fake-secret-key", None
    )
    mock_session_class.return_value.get_credentials.return_value = mock_credentials

    client = build_opensearch_client(host="search-domain.us-east-1.es.amazonaws.com", region="us-east-1")

    fake_response = MagicMock()
    fake_response.ok = True
    fake_response.json.return_value = {"errors": False, "items": []}

    with patch.object(client._session, "request", return_value=fake_response) as mock_request:
        client.bulk(body=[{"index": {"_index": "chunks", "_id": "a"}}, {"text": "hello"}])

    sent_body = mock_request.call_args.kwargs["data"]
    assert sent_body.count("\n") == 2
    assert mock_request.call_args.kwargs["headers"]["Content-Type"] == "application/x-ndjson"


@patch("rag.vector_store.opensearch_client_factory.boto3.Session")
def test_signed_client_raises_opensearch_http_error_on_failure(mock_session_class):

    mock_credentials = MagicMock()
    mock_credentials.get_frozen_credentials.return_value = ReadOnlyCredentials(
        "fake-access-key", "fake-secret-key", None
    )
    mock_session_class.return_value.get_credentials.return_value = mock_credentials

    client = build_opensearch_client(host="search-domain.us-east-1.es.amazonaws.com", region="us-east-1")

    fake_response = MagicMock()
    fake_response.ok = False
    fake_response.status_code = 400
    fake_response.text = "mapper_parsing_exception"

    with patch.object(client._session, "request", return_value=fake_response):
        with pytest.raises(OpenSearchHTTPError):
            client.search(index="chunks", body={})


@patch("rag.vector_store.opensearch_client_factory.boto3.Session")
def test_signed_client_delete_honors_ignore_status_codes(mock_session_class):

    mock_credentials = MagicMock()
    mock_credentials.get_frozen_credentials.return_value = ReadOnlyCredentials(
        "fake-access-key", "fake-secret-key", None
    )
    mock_session_class.return_value.get_credentials.return_value = mock_credentials

    client = build_opensearch_client(host="search-domain.us-east-1.es.amazonaws.com", region="us-east-1")

    fake_response = MagicMock()
    fake_response.ok = False
    fake_response.status_code = 404
    fake_response.text = "not_found"

    with patch.object(client._session, "request", return_value=fake_response):
        result = client.delete(index="chunks", id="missing", ignore=[404])

    assert result is None


@patch("rag.vector_store.opensearch_client_factory.boto3.Session")
def test_signed_client_indices_exists_checks_status_code(mock_session_class):

    mock_credentials = MagicMock()
    mock_credentials.get_frozen_credentials.return_value = ReadOnlyCredentials(
        "fake-access-key", "fake-secret-key", None
    )
    mock_session_class.return_value.get_credentials.return_value = mock_credentials

    client = build_opensearch_client(host="search-domain.us-east-1.es.amazonaws.com", region="us-east-1")

    fake_response = MagicMock()
    fake_response.status_code = 200

    with patch.object(client._session, "request", return_value=fake_response):
        assert client.indices.exists(index="chunks") is True


@patch("rag.vector_store.opensearch_client_factory.boto3.Session")
def test_signed_client_get_alias_returns_empty_dict_on_404(mock_session_class):

    mock_credentials = MagicMock()
    mock_credentials.get_frozen_credentials.return_value = ReadOnlyCredentials(
        "fake-access-key", "fake-secret-key", None
    )
    mock_session_class.return_value.get_credentials.return_value = mock_credentials

    client = build_opensearch_client(host="search-domain.us-east-1.es.amazonaws.com", region="us-east-1")

    fake_response = MagicMock()
    fake_response.status_code = 404

    with patch.object(client._session, "request", return_value=fake_response):
        assert client.indices.get_alias("rag-prod") == {}


@patch("rag.vector_store.opensearch_client_factory.boto3.Session")
def test_signed_client_update_aliases_posts_the_actions(mock_session_class):

    mock_credentials = MagicMock()
    mock_credentials.get_frozen_credentials.return_value = ReadOnlyCredentials(
        "fake-access-key", "fake-secret-key", None
    )
    mock_session_class.return_value.get_credentials.return_value = mock_credentials

    client = build_opensearch_client(host="search-domain.us-east-1.es.amazonaws.com", region="us-east-1")

    fake_response = MagicMock()
    fake_response.ok = True
    fake_response.json.return_value = {"acknowledged": True}

    with patch.object(client._session, "request", return_value=fake_response) as mock_request:
        client.indices.update_aliases([{"add": {"index": "rag-v1", "alias": "rag-prod"}}])

    called_url = mock_request.call_args.args[1]
    assert called_url.endswith("/_aliases")
    assert mock_request.call_args.kwargs["data"] is not None


@patch("rag.vector_store.opensearch_client_factory.boto3.Session")
def test_signed_client_list_names_includes_query_params_in_the_signed_url(mock_session_class):

    mock_credentials = MagicMock()
    mock_credentials.get_frozen_credentials.return_value = ReadOnlyCredentials(
        "fake-access-key", "fake-secret-key", None
    )
    mock_session_class.return_value.get_credentials.return_value = mock_credentials

    client = build_opensearch_client(host="search-domain.us-east-1.es.amazonaws.com", region="us-east-1")

    fake_response = MagicMock()
    fake_response.ok = True
    fake_response.json.return_value = [{"index": "rag-v1"}, {"index": "rag-v2"}]

    with patch.object(client._session, "request", return_value=fake_response) as mock_request:
        result = client.indices.list_names("rag-v*")

    assert result == ["rag-v1", "rag-v2"]
    called_url = mock_request.call_args.args[1]
    assert "format=json" in called_url
    assert "/_cat/indices/rag-v*" in called_url


@patch("rag.vector_store.opensearch_client_factory.boto3.Session")
def test_delete_by_query_passes_conflicts_as_a_query_param(mock_session_class):

    mock_credentials = MagicMock()
    mock_credentials.get_frozen_credentials.return_value = ReadOnlyCredentials(
        "fake-access-key", "fake-secret-key", None
    )
    mock_session_class.return_value.get_credentials.return_value = mock_credentials

    client = build_opensearch_client(host="search-domain.us-east-1.es.amazonaws.com", region="us-east-1")

    fake_response = MagicMock()
    fake_response.ok = True
    fake_response.json.return_value = {"deleted": 1}

    with patch.object(client._session, "request", return_value=fake_response) as mock_request:
        client.delete_by_query(index="chunks", body={"query": {"match_all": {}}}, conflicts="proceed")

    called_url = mock_request.call_args.args[1]
    assert "conflicts=proceed" in called_url
