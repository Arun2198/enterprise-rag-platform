from pathlib import Path

from pypdf import PdfReader

from ingestion.contracts.document import Document
from ingestion.contracts.result import Error
from ingestion.contracts.result import Result
from ingestion.parsers.base_parser import BaseParser


class PDFParser(BaseParser):

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

            reader = PdfReader(file_path)

            # One entry per physical page, empty string for a page with no
            # extractable text (e.g. a scanned image) rather than skipping
            # it - downstream chunking needs pages[i] to always mean
            # "page i+1", not a compacted list that silently shifts once
            # any page in the middle turns out blank.
            pages = [page.extract_text() or "" for page in reader.pages]
            content = "\n".join(text for text in pages if text)

            if not content.strip():
                return Result(
                    success=False,
                    error=Error(
                        code="EMPTY_DOCUMENT",
                        message="No text extracted from PDF"
                    )
                )

            document = Document(
                document_id=path.stem,
                source=str(path),
                document_type="pdf",
                content=content,
                pages=pages,
                metadata={
                    "page_count": len(reader.pages),
                    "file_name": path.name
                }
            )

            return Result(
                success=True,
                data=document
            )

        except Exception as ex:

            return Result(
                success=False,
                error=Error(
                    code="PDF_PARSE_ERROR",
                    message=str(ex)
                )
            )
