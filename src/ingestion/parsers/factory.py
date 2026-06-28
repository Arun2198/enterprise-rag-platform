from pathlib import Path

from ingestion.contracts.result import Error
from ingestion.contracts.result import Result
from ingestion.parsers.base_parser import BaseParser
from ingestion.parsers.docx_parser import DOCXParser
from ingestion.parsers.markdown_parser import MarkdownParser
from ingestion.parsers.pdf_parser import PDFParser


class ParserFactory:

    def __init__(self) -> None:
        self._parsers: dict[str, BaseParser] = {
            ".pdf": PDFParser(),
            ".docx": DOCXParser(),
            ".md": MarkdownParser(),
            ".markdown": MarkdownParser(),
        }

    def get_parser(
        self,
        file_path: str
    ) -> Result:
        extension = Path(file_path).suffix.lower()
        parser = self._parsers.get(extension)

        if parser is None:
            return Result(
                success=False,
                error=Error(
                    code="UNSUPPORTED_FILE_TYPE",
                    message=f"No parser registered for extension: {extension}"
                )
            )

        return Result(
            success=True,
            data=parser
        )
