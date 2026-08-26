import json
from typing import Any
from urllib.parse import urlencode

import boto3
import requests
from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest
from requests.adapters import HTTPAdapter
from urllib3.util import Retry

RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})


class OpenSearchHTTPError(RuntimeError):

    def __init__(self, response: requests.Response) -> None:
        self.status_code = response.status_code
        self.body = response.text
        super().__init__(f"OpenSearch returned {response.status_code}: {response.text[:500]}")


class _IndicesClient:

    def __init__(self, parent: "SignedOpenSearchClient") -> None:
        self._parent = parent

    def exists(self, index: str) -> bool:
        response = self._parent._request("HEAD", f"/{index}")
        return response.status_code == 200

    def create(self, index: str, body: dict[str, Any]) -> dict[str, Any]:
        response = self._parent._request("PUT", f"/{index}", json_body=body)
        return self._parent._json_or_raise(response)

    def delete(self, index: str, ignore: list[int] | None = None) -> dict[str, Any] | None:
        response = self._parent._request("DELETE", f"/{index}")

        if ignore and response.status_code in ignore:
            return None

        return self._parent._json_or_raise(response)

    def get_alias(self, name: str) -> dict[str, Any]:
        """Returns {index_name: {...}} for every index the alias currently
        points to - empty dict if the alias doesn't exist yet."""
        response = self._parent._request("GET", f"/_alias/{name}")

        if response.status_code == 404:
            return {}

        return self._parent._json_or_raise(response)

    def update_aliases(self, actions: list[dict[str, Any]]) -> dict[str, Any]:
        """Atomic alias repoint - OpenSearch applies every action in one
        request, so a caller never observes the alias missing or pointing
        at both the old and new index mid-switch."""
        response = self._parent._request("POST", "/_aliases", json_body={"actions": actions})
        return self._parent._json_or_raise(response)

    def list_names(self, pattern: str) -> list[str]:
        response = self._parent._request(
            "GET", f"/_cat/indices/{pattern}", params={"format": "json"}
        )

        if response.status_code == 404:
            return []

        body = self._parent._json_or_raise(response)
        return sorted(item["index"] for item in body)


class _ClusterClient:

    def __init__(self, parent: "SignedOpenSearchClient") -> None:
        self._parent = parent

    def health(self, index: str | None = None) -> dict[str, Any]:
        path = f"/_cluster/health/{index}" if index else "/_cluster/health"
        response = self._parent._request("GET", path)
        return self._parent._json_or_raise(response)


class SignedOpenSearchClient:
    """
    Minimal, direct OpenSearch REST client, SigV4-signed via botocore.

    opensearch-py ships its own AWS-signed connection path (AWSV4SignerAuth
    paired with RequestsHttpConnection, which is the library's own documented
    pairing - the alternative, Urllib3HttpConnection, raises a TypeError
    immediately since AWSV4SignerAuth's call signature doesn't match it).
    That documented pairing was verified against a real OpenSearch domain to
    hang indefinitely on every request. The exact same SigV4 signature, sent
    as a plain `requests` call instead of through opensearch-py's internal
    Session/connection-pooling layer, returns 200 immediately - so the
    signing logic is correct and the bug is inside opensearch-py's HTTP
    connection handling in this environment. Rather than depend on a library
    integration that's demonstrably broken here, this talks to OpenSearch's
    REST API directly with the same, verified-working signing path. It only
    implements the handful of operations OpenSearchVectorStore actually
    calls - it is not a general-purpose OpenSearch client.
    """

    def __init__(
        self,
        host: str,
        region: str,
        port: int = 443,
        use_ssl: bool = True,
        verify_certs: bool = True,
        connect_timeout: float = 10.0,
        max_retries: int = 3,
        service: str = "es"
    ) -> None:
        scheme = "https" if use_ssl else "http"
        self._base_url = f"{scheme}://{host}:{port}"
        self._region = region
        self._service = service
        self._verify_certs = verify_certs
        self._timeout = connect_timeout

        credentials = boto3.Session().get_credentials()

        if credentials is None:
            raise RuntimeError(
                "No AWS credentials available to sign OpenSearch requests. "
                "Configure them the same way as any other boto3 client in "
                "this process (IAM role, AWS CLI profile, or env vars)."
            )

        self._credentials = credentials
        self._session = requests.Session()
        # A plain HTTPAdapter(max_retries=N) only retries connection-level
        # failures (DNS, refused, reset) - it does NOT retry HTTP error
        # responses like 429/503 at all, which is the failure mode that
        # actually matters for a shared OpenSearch domain under load. This
        # Retry object adds status_forcelist so those get retried too, with
        # exponential backoff + jitter. POST is included in allowed_methods
        # (urllib3 excludes it by default, treating POST as generally
        # unsafe to retry blindly) because every POST this client makes -
        # _search (read-only), _bulk/_update/_delete_by_query (all keyed by
        # deterministic chunk_id, naturally idempotent via upsert) - is
        # actually safe to retry in this application's specific usage.
        retry = Retry(
            total=max_retries,
            status_forcelist=RETRYABLE_STATUS_CODES,
            allowed_methods=frozenset({"GET", "HEAD", "PUT", "DELETE", "POST"}),
            backoff_factor=0.5,
            backoff_jitter=0.2
        )
        adapter = HTTPAdapter(max_retries=retry)
        self._session.mount("https://", adapter)
        self._session.mount("http://", adapter)

        self.indices = _IndicesClient(self)
        self.cluster = _ClusterClient(self)

    def index(
        self,
        index: str,
        id: str,
        body: dict[str, Any]
    ) -> dict[str, Any]:
        response = self._request("PUT", f"/{index}/_doc/{id}", json_body=body)
        return self._json_or_raise(response)

    def bulk(
        self,
        body: list[dict[str, Any]]
    ) -> dict[str, Any]:
        ndjson = "\n".join(json.dumps(item) for item in body) + "\n"
        response = self._request("POST", "/_bulk", raw_body=ndjson)
        return self._json_or_raise(response)

    def search(
        self,
        index: str,
        body: dict[str, Any]
    ) -> dict[str, Any]:
        response = self._request("POST", f"/{index}/_search", json_body=body)
        return self._json_or_raise(response)

    def delete(
        self,
        index: str,
        id: str,
        ignore: list[int] | None = None
    ) -> dict[str, Any] | None:
        response = self._request("DELETE", f"/{index}/_doc/{id}")

        if ignore and response.status_code in ignore:
            return None

        return self._json_or_raise(response)

    def delete_by_query(
        self,
        index: str,
        body: dict[str, Any],
        conflicts: str | None = None
    ) -> dict[str, Any]:
        params = {"conflicts": conflicts} if conflicts else None
        response = self._request("POST", f"/{index}/_delete_by_query", json_body=body, params=params)
        return self._json_or_raise(response)

    def update(
        self,
        index: str,
        id: str,
        body: dict[str, Any]
    ) -> dict[str, Any]:
        response = self._request("POST", f"/{index}/_update/{id}", json_body=body)
        return self._json_or_raise(response)

    def _request(
        self,
        method: str,
        path: str,
        json_body: dict[str, Any] | None = None,
        raw_body: str | None = None,
        params: dict[str, str] | None = None
    ) -> requests.Response:
        url = f"{self._base_url}{path}"

        if params:
            url = f"{url}?{urlencode(params)}"

        data = raw_body if raw_body is not None else (
            json.dumps(json_body) if json_body is not None else None
        )

        # the query string must already be part of the URL before signing -
        # SigV4 signs it as part of the canonical request, so appending it
        # to an already-signed URL afterward would invalidate the signature.
        aws_request = AWSRequest(method=method, url=url, data=data)
        SigV4Auth(
            self._credentials.get_frozen_credentials(),
            self._service,
            self._region
        ).add_auth(aws_request)
        headers = dict(aws_request.headers)

        if data is not None:
            headers["Content-Type"] = "application/x-ndjson" if raw_body else "application/json"

        return self._session.request(
            method,
            url,
            data=data,
            headers=headers,
            timeout=self._timeout,
            verify=self._verify_certs
        )

    def _json_or_raise(
        self,
        response: requests.Response
    ) -> dict[str, Any]:
        if not response.ok:
            raise OpenSearchHTTPError(response)

        return response.json()


def build_opensearch_client(
    host: str,
    region: str,
    port: int = 443,
    use_ssl: bool = True,
    verify_certs: bool = True,
    connect_timeout: float = 10.0,
    max_retries: int = 3
) -> SignedOpenSearchClient:
    """
    Builds an OpenSearch client authenticated via AWS SigV4 request signing,
    using whatever credentials boto3 resolves ambiently (IAM role on ECS,
    local AWS CLI profile, env vars, ...) - never a stored username/password.
    No network call happens here; the client only connects on first request.
    """
    return SignedOpenSearchClient(
        host=host,
        region=region,
        port=port,
        use_ssl=use_ssl,
        verify_certs=verify_certs,
        connect_timeout=connect_timeout,
        max_retries=max_retries
    )
