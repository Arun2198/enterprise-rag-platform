from pathlib import Path

from ingestion.contracts.document import Document
from ingestion.contracts.result import Error
from ingestion.contracts.result import Result
from ingestion.parsers.base_parser import BaseParser


class MarkdownParser(BaseParser):

    def parse(
        self,
        file_path: str
    ) -> Result[Document]:

        try:
            path = Path(file_path)

            if not path.exists():
                return Result(
                    success=False,
                    error=Error(
                        code="FILE_NOT_FOUND",
                        message=f"{file_path} does not exist"
                    )
                )

            content = path.read_text(encoding="utf-8").strip()

            if not content:
                return Result(
                    success=False,
                    error=Error(
                        code="EMPTY_DOCUMENT",
                        message="No text extracted from Markdown"
                    )
                )

            document = Document(
                document_id=path.stem,
                source=str(path),
                document_type="markdown",
                content=content,
                metadata={
                    "file_name": path.name
                }
            )

            return Result(
                success=True,
                data=document
            )

        except UnicodeDecodeError as ex:
            return Result(
                success=False,
                error=Error(
                    code="MARKDOWN_DECODE_ERROR",
                    message=str(ex)
                )
            )
