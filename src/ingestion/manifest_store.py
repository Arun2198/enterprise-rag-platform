import json
from typing import Any
from typing import Protocol

from ingestion.contracts.manifest import DocumentManifest


class ManifestStore(Protocol):
    """
    Durable record of what was previously ingested for each document, so
    IncrementalIndexer has something to diff the current ingest against.
    Deliberately a small key-value shape (get/put/delete by document_id) -
    same "smallest thing that actually works" spirit as IngestionJobStore.
    """

    def get(
        self,
        document_id: str
    ) -> DocumentManifest | None:
        ...

    def put(
        self,
        manifest: DocumentManifest
    ) -> None:
        ...

    def delete(
        self,
        document_id: str
    ) -> None:
        ...


class InMemoryManifestStore:
    """
    Default, zero-config manifest store - an in-process dict. This is
    exactly the same durability tradeoff InMemoryRateLimiter already
    documents for this project: correct and sufficient for a single ECS
    task, gone on a task restart/redeploy, and not shared across replicas.
    Good enough to make incremental ingestion work out of the box in local
    dev/tests without requiring S3; service_factory upgrades to
    S3ManifestStore automatically once S3_BUCKET is configured, the same
    opt-in-upgrade pattern used for conversations/backups.
    """

    def __init__(self) -> None:
        self._manifests: dict[str, DocumentManifest] = {}

    def get(
        self,
        document_id: str
    ) -> DocumentManifest | None:
        return self._manifests.get(document_id)

    def put(
        self,
        manifest: DocumentManifest
    ) -> None:
        self._manifests[manifest.document_id] = manifest

    def delete(
        self,
        document_id: str
    ) -> None:
        self._manifests.pop(document_id, None)


class S3ManifestStore:
    """
    Durable manifest storage for the live app - one JSON object per
    document (same "S3 instead of a new database" pattern as
    IngestionJobStore/ConversationStore/S3BackupTarget). Without this, an
    ECS task restart or redeploy would lose all ingestion history and
    every next ingest would look like a brand-new document again.
    """

    def __init__(
        self,
        client: Any,
        bucket_name: str,
        prefix: str = "ingestion_manifests/"
    ) -> None:
        self.client = client
        self.bucket_name = bucket_name
        self.prefix = prefix

    def get(
        self,
        document_id: str
    ) -> DocumentManifest | None:
        try:
            response = self.client.get_object(Bucket=self.bucket_name, Key=self._key(document_id))
        except self.client.exceptions.NoSuchKey:
            return None

        return DocumentManifest.model_validate(json.loads(response["Body"].read()))

    def put(
        self,
        manifest: DocumentManifest
    ) -> None:
        self.client.put_object(
            Bucket=self.bucket_name,
            Key=self._key(manifest.document_id),
            Body=manifest.model_dump_json().encode("utf-8"),
            ContentType="application/json"
        )

    def delete(
        self,
        document_id: str
    ) -> None:
        self.client.delete_object(Bucket=self.bucket_name, Key=self._key(document_id))

    def _key(
        self,
        document_id: str
    ) -> str:
        return f"{self.prefix}{document_id}.json"
