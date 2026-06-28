from ingestion.cleaners.text_cleaner import TextCleaner
from ingestion.contracts.document import Document
from ingestion.contracts.result import Result
from ingestion.parsers.factory import ParserFactory


class IngestionPipeline:

    def __init__(
        self,
        parser_factory: ParserFactory | None = None,
        cleaner: TextCleaner | None = None
    ) -> None:
        self.parser_factory = parser_factory or ParserFactory()
        self.cleaner = cleaner or TextCleaner()

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
