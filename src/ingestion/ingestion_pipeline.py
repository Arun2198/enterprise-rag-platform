import os
from pathlib import Path

from ingestion.cleaners.text_cleaner import TextCleaner
from ingestion.contracts.document import Document
from ingestion.contracts.result import Result
from ingestion.parsers.factory import ParserFactory
from ingestion.s3_document_store import S3DocumentStore


class IngestionPipeline:

    def __init__(
        self,
        parser_factory: ParserFactory | None = None,
        cleaner: TextCleaner | None = None
    ) -> None:
        self.parser_factory = parser_factory or ParserFactory()
        self.cleaner = cleaner or TextCleaner()

    def ingest_from_s3(
        self,
        s3_store: S3DocumentStore,
        key: str,
        document_id: str | None = None
    ) -> Result[Document]:
        """
        Downloads the S3 object to a local temp file and reuses ingest_file()
        completely unchanged - the parser layer never needs to know the
        source was S3 rather than a local path. The temp file is always
        removed, success or failure.

        Parsers derive document_id/source from the local file path, which
        for a downloaded temp file is a meaningless random name - this
        overrides both with the real S3 key (or an explicit document_id)
        afterward so identity survives the S3 -> local temp file -> parser
        round trip.
        """
        local_path = s3_store.download_to_temp(key)
        filename = key.rsplit("/", maxsplit=1)[-1]
        resolved_document_id = document_id or Path(filename).stem

        try:
            result = self.ingest_file(local_path)

            if not result.success or result.data is None:
                return result

            document = result.data.model_copy(update={
                "document_id": resolved_document_id,
                "source": f"s3://{s3_store.bucket_name}/{key}",
                "metadata": {**result.data.metadata, "file_name": filename}
            })
            return Result(success=True, data=document)
        finally:
            if os.path.exists(local_path):
                os.remove(local_path)

    def ingest_file(
        self,
        file_path: str
    ) -> Result[Document]:
        parser_result = self.parser_factory.get_parser(file_path)

        if not parser_result.success or parser_result.data is None:
            return Result(
                success=False,
                error=parser_result.error
            )

        parsed_result = parser_result.data.parse(file_path)

        if not parsed_result.success or parsed_result.data is None:
            return parsed_result

        document = parsed_result.data.model_copy(
            update={
                "content": self.cleaner.clean(parsed_result.data.content)
            }
        )

        return Result(
            success=True,
            data=document
        )
