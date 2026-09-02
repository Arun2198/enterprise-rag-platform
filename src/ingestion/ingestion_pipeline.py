import os

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
        document_id: str
    ) -> Result[Document]:
        """
        Downloads the S3 object to a local temp file and reuses ingest_file()
        completely unchanged - the parser layer never needs to know the
        source was S3 rather than a local path. The temp file is always
        removed, success or failure.

        document_id is mandatory, not filename-derived: every caller
        (the async ingestion worker, in practice) already tracks a real
        stable identity for the document being ingested (the job's own
        document_id), and the S3 key alone is not a safe identity source -
        a temp download path is meaningless, and the key itself can
        change across a rename the same way a local filename can.
        """
        local_path = s3_store.download_to_temp(key)
        filename = key.rsplit("/", maxsplit=1)[-1]

        try:
            result = self.ingest_file(local_path, document_id=document_id)

            if not result.success or result.data is None:
                return result

            document = result.data.model_copy(update={
                "source": f"s3://{s3_store.bucket_name}/{key}",
                "metadata": {**result.data.metadata, "file_name": filename}
            })
            return Result(success=True, data=document)
        finally:
            if os.path.exists(local_path):
                os.remove(local_path)

    def ingest_file(
        self,
        file_path: str,
        document_id: str
    ) -> Result[Document]:
        """
        document_id is mandatory - overrides the parser's own
        filename-derived id (Path(file_path).stem). Every caller must
        supply the document's real stable identity, on every ingest of
        it, including after a rename: renaming a file must not be
        indistinguishable from ingesting a brand-new document, and
        incremental re-embedding (IncrementalIndexer) diffs purely on
        document_id, so a caller-supplied filename-derived id would
        silently defeat that the moment a file gets renamed.
        """
        parser_result = self.parser_factory.get_parser(file_path)

        if not parser_result.success or parser_result.data is None:
            return Result(
                success=False,
                error=parser_result.error
            )

        parsed_result = parser_result.data.parse(file_path)

        if not parsed_result.success or parsed_result.data is None:
            return parsed_result

        # Clean per-page too, not just the flattened content - the
        # chunker needs cleaned per-page text so a page's hash and its
        # chunks are computed from the same normalized text the rest of
        # the pipeline sees, not raw pre-cleaning text.
        cleaned_pages = (
            [self.cleaner.clean(page) for page in parsed_result.data.pages]
            if parsed_result.data.pages is not None
            else None
        )
        document = parsed_result.data.model_copy(update={
            "content": self.cleaner.clean(parsed_result.data.content),
            "pages": cleaned_pages,
            "document_id": document_id
        })

        return Result(
            success=True,
            data=document
        )
