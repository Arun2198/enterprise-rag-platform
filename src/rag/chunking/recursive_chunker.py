import re

from ingestion.contracts.document import Document
from ingestion.contracts.result import Error
from ingestion.contracts.result import Result
from rag.chunking.chunk import Chunk


class RecursiveChunker:

    def __init__(
        self,
        chunk_size: int = 900,
        chunk_overlap: int = 120,
        minimum_chunk_size: int = 80
    ) -> None:
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.minimum_chunk_size = minimum_chunk_size

    def chunk(
        self,
        document: Document
    ) -> Result[list[Chunk]]:
        if not document.content.strip():
            return Result(
                success=False,
                error=Error(
                    code="EMPTY_DOCUMENT",
                    message="Cannot chunk an empty document"
                )
            )

        sections = self._split_sections(document.content)
        chunks: list[Chunk] = []

        for section_title, section_text in sections:
            for text in self._split_text(section_text):
                chunks.append(
                    Chunk(
                        chunk_id=f"{document.document_id}:{len(chunks)}",
                        document_id=document.document_id,
                        chunk_index=len(chunks),
                        text=text,
                        source=document.source,
                        document_type=document.document_type,
                        owner=document.owner,
                        created_at=document.created_at,
                        updated_at=document.updated_at,
                        parent_section=section_title,
                        metadata={
                            **document.metadata,
                            "document_id": document.document_id,
                            "document_type": document.document_type,
                            "source": document.source,
                            "section": section_title,
                        }
                    )
                )

        return Result(
            success=True,
            data=chunks
        )

    def _split_sections(
        self,
        content: str
    ) -> list[tuple[str | None, str]]:
        sections: list[tuple[str | None, list[str]]] = []
        current_title: str | None = None
        current_lines: list[str] = []

        for line in content.splitlines():
            stripped = line.strip()
            if not stripped:
                continue

            if self._looks_like_heading(stripped) and current_lines:
                sections.append((current_title, current_lines))
                current_title = stripped.strip("# ").strip()
                current_lines = [stripped]
            else:
                if self._looks_like_heading(stripped):
                    current_title = stripped.strip("# ").strip()
                current_lines.append(stripped)

        if current_lines:
            sections.append((current_title, current_lines))

        return [
            (title, "\n".join(lines))
            for title, lines in sections
        ]

    def _split_text(
        self,
        text: str
    ) -> list[str]:
        if len(text) <= self.chunk_size:
            return [text]

        sentences = self._split_sentences(text)
        chunks: list[str] = []
        current = ""

        for sentence in sentences:
            candidate = f"{current} {sentence}".strip()

            if len(candidate) <= self.chunk_size:
                current = candidate
                continue

            if current:
                chunks.append(current)
                current = self._with_overlap(current, sentence)
            else:
                chunks.extend(self._split_long_sentence(sentence))
                current = ""

        if current:
            chunks.append(current)

        return self._merge_tiny_chunks(chunks)

    def _split_sentences(
        self,
        text: str
    ) -> list[str]:
        return [
            sentence.strip()
            for sentence in re.split(r"(?<=[.!?])\s+", text.replace("\n", " "))
            if sentence.strip()
        ]

    def _split_long_sentence(
        self,
        sentence: str
    ) -> list[str]:
        chunks = []
        start = 0

        while start < len(sentence):
            end = start + self.chunk_size
            chunks.append(sentence[start:end].strip())
            start = max(end - self.chunk_overlap, end)

        return [
            chunk
            for chunk in chunks
            if chunk
        ]

    def _with_overlap(
        self,
        previous: str,
        sentence: str
    ) -> str:
        overlap = previous[-self.chunk_overlap:].strip()
        return f"{overlap} {sentence}".strip()

    def _merge_tiny_chunks(
        self,
        chunks: list[str]
    ) -> list[str]:
        if len(chunks) <= 1:
            return chunks

        merged: list[str] = []

        for chunk in chunks:
            if merged and len(chunk) < self.minimum_chunk_size:
                merged[-1] = f"{merged[-1]} {chunk}".strip()
            else:
                merged.append(chunk)

        return merged

    def _looks_like_heading(
        self,
        line: str
    ) -> bool:
        if line.startswith("#"):
            return True

        if len(line) > 90 or line.endswith("."):
            return False

        return bool(re.match(r"^(\d+(\.\d+)*\s+)?[A-Z][A-Za-z0-9 ,&:/()'-]+$", line))
