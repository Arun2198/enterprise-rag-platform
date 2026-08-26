import logging
import os
import tempfile
from datetime import datetime
from datetime import timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_ALLOWED_EXTENSIONS = (".pdf", ".docx", ".md", ".markdown")
DEFAULT_MAX_FILE_SIZE_BYTES = 25 * 1024 * 1024  # 25 MiB


class S3ValidationError(ValueError):
    pass


class S3DocumentStore:
    """
    S3-backed document storage for production ingestion - upload, download
    to a local temp file for the existing (unchanged) file-based parsers to
    read, and prefix-based lifecycle tracking (raw/processed/failed).

    The S3 client is injected (an authenticated boto3 "s3" client), same
    pattern as OpenSearchVectorStore/BedrockAnswerer - this class doesn't
    build its own credentials, it just uses whatever client it's given.
    """

    def __init__(
        self,
        client: Any,
        bucket_name: str,
        raw_prefix: str = "raw/",
        processed_prefix: str = "processed/",
        failed_prefix: str = "failed/",
        max_file_size_bytes: int = DEFAULT_MAX_FILE_SIZE_BYTES,
        allowed_extensions: tuple[str, ...] = DEFAULT_ALLOWED_EXTENSIONS
    ) -> None:
        self.client = client
        self.bucket_name = bucket_name
        self.raw_prefix = raw_prefix
        self.processed_prefix = processed_prefix
        self.failed_prefix = failed_prefix
        self.max_file_size_bytes = max_file_size_bytes
        self.allowed_extensions = allowed_extensions

    def upload(
        self,
        local_path: str,
        document_id: str,
        original_filename: str | None = None
    ) -> str:
        """
        Validates file type/size, uploads to the raw/ prefix, and preserves
        the original filename as object metadata (S3 keys are opaque - the
        real name matters for parser dispatch and for showing the user
        something recognizable later).
        """
        original_filename = original_filename or Path(local_path).name
        size_bytes = os.path.getsize(local_path)
        self.validate(original_filename, size_bytes)

        extension = Path(original_filename).suffix.lower()
        key = f"{self.raw_prefix}{document_id}{extension}"

        self.client.upload_file(
            local_path,
            self.bucket_name,
            key,
            ExtraArgs={
                "Metadata": {
                    "original_filename": original_filename,
                    "document_id": document_id,
                    "uploaded_at": datetime.now(timezone.utc).isoformat()
                }
            }
        )
        logger.info(
            "s3_document_uploaded",
            extra={"bucket": self.bucket_name, "key": key, "size_bytes": size_bytes}
        )
        return key

    def validate(
        self,
        filename: str,
        size_bytes: int
    ) -> None:
        extension = Path(filename).suffix.lower()

        if extension not in self.allowed_extensions:
            raise S3ValidationError(
                f"file type {extension!r} is not allowed - "
                f"must be one of {self.allowed_extensions}"
            )

        if size_bytes > self.max_file_size_bytes:
            raise S3ValidationError(
                f"file is {size_bytes} bytes, exceeds the "
                f"{self.max_file_size_bytes} byte limit"
            )

        if size_bytes == 0:
            raise S3ValidationError("file is empty")

    def download_to_temp(
        self,
        key: str
    ) -> str:
        """
        Downloads an S3 object to a local temp file so the existing
        file-path-based parsers (PDFParser/DOCXParser/MarkdownParser) can
        read it completely unchanged - the parser layer has no idea the
        file came from S3. Caller is responsible for deleting the temp
        file when done (see IngestionPipeline.ingest_from_s3).
        """
        suffix = Path(key).suffix
        fd, local_path = tempfile.mkstemp(suffix=suffix)
        os.close(fd)

        self.client.download_file(self.bucket_name, key, local_path)
        return local_path

    def mark_processed(
        self,
        key: str
    ) -> str:
        return self._move(key, self.processed_prefix)

    def mark_failed(
        self,
        key: str,
        reason: str | None = None
    ) -> str:
        extra_metadata = {"failure_reason": reason} if reason else {}
        return self._move(key, self.failed_prefix, extra_metadata=extra_metadata)

    def get_metadata(
        self,
        key: str
    ) -> dict[str, Any]:
        response = self.client.head_object(Bucket=self.bucket_name, Key=key)
        return response.get("Metadata", {})

    def _move(
        self,
        key: str,
        target_prefix: str,
        extra_metadata: dict[str, str] | None = None
    ) -> str:
        """
        S3 has no atomic "move" - copy to the new prefix, then delete the
        original. If extra_metadata is given, it's merged into the object's
        existing metadata on the copy (S3 requires a full metadata
        replacement on copy when adding new keys, hence REPLACE below).
        """
        filename = key.rsplit("/", maxsplit=1)[-1]
        new_key = f"{target_prefix}{filename}"

        copy_kwargs: dict[str, Any] = {
            "Bucket": self.bucket_name,
            "CopySource": {"Bucket": self.bucket_name, "Key": key},
            "Key": new_key,
        }

        if extra_metadata:
            existing = self.get_metadata(key)
            copy_kwargs["Metadata"] = {**existing, **extra_metadata}
            copy_kwargs["MetadataDirective"] = "REPLACE"

        self.client.copy_object(**copy_kwargs)
        self.client.delete_object(Bucket=self.bucket_name, Key=key)
        return new_key
