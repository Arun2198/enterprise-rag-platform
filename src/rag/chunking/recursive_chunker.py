import hashlib
import re

from ingestion.contracts.document import Document
from ingestion.contracts.result import Error
from ingestion.contracts.result import Result
from rag.chunking.chunk import Chunk


class RecursiveChunker:
    """
    The retrievable/embeddable unit is a single sentence, not a
    ~900-char packed group - each Chunk record is one sentence. The old
    chunk-level grouping (heading-aware sections, greedy sentence
    packing up to chunk_size, character overlap between adjacent
    groups) still runs internally as a "window" (_build_windows()) and
    survives purely as parent_chunk_id metadata on each sentence, so
    generation-time context expansion can pull a matched sentence's
    surrounding window back in without the window itself ever being
    independently embedded or stored as its own vector-store record.
    """

    def __init__(
        self,
        chunk_size: int = 900,
        chunk_overlap: int = 120,
        minimum_chunk_size: int = 80,
        minimum_sentence_size: int = 20
    ) -> None:
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.minimum_chunk_size = minimum_chunk_size
        # A sentence under this length ("See Section 4.2.") carries
        # almost no retrievable signal on its own - merged into the
        # adjacent sentence instead of becoming its own embedded unit.
        self.minimum_sentence_size = minimum_sentence_size
        # Identifies exactly which chunking parameters (and which
        # granularity - "sentence" vs. the old "recursive" chunk-level
        # scheme) produced a chunk - any change invalidates positional/
        # content-addressed comparisons against a previous ingest, so a
        # chunk carrying the version it was cut under makes that
        # incompatibility detectable (IncrementalIndexer forces a full
        # re-embed) instead of silently mixing granularities.
        self.chunking_version = (
            f"sentence:{chunk_size}:{chunk_overlap}:{minimum_chunk_size}:{minimum_sentence_size}"
        )

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

        # Chunk each page independently rather than the flattened
        # document.content - this is what makes chunk identity stable
        # under a page-localized edit (see class docstring) and, as a
        # side effect, means no chunk can ever span two pages: a
        # sentence split across a page break becomes two separate
        # chunks instead of one that silently straddles the boundary.
        # Formats with no native page concept (docx, markdown) get a
        # single virtual page 1 covering the whole document - identical
        # output to the old flattened behavior for those formats, just
        # under the new id scheme.
        pages = document.pages if document.pages is not None else [document.content]

        chunks: list[Chunk] = []
        global_index = 0

        for page_index, page_text in enumerate(pages):
            page_number = page_index + 1

            if not page_text.strip():
                continue

            sections = self._split_sections(page_text)
            sentence_index = 0
            window_index = 0

            for section_title, section_text in sections:
                for window_text in self._build_windows(section_text):
                    parent_chunk_id = f"{document.document_id}:p{page_number}:w{window_index}"
                    sentences = self._merge_tiny_sentences(self._split_sentences(window_text))

                    for sentence_text in sentences:
                        chunks.append(
                            Chunk(
                                chunk_id=f"{document.document_id}:p{page_number}:s{sentence_index}",
                                document_id=document.document_id,
                                chunk_index=global_index,
                                page_number=page_number,
                                text=sentence_text,
                                source=document.source,
                                document_type=document.document_type,
                                owner=document.owner,
                                created_at=document.created_at,
                                updated_at=document.updated_at,
                                parent_section=section_title,
                                parent_chunk_id=parent_chunk_id,
                                content_hash=hashlib.sha256(sentence_text.encode("utf-8")).hexdigest(),
                                chunking_version=self.chunking_version,
                                metadata={
                                    **document.metadata,
                                    "document_id": document.document_id,
                                    "document_type": document.document_type,
                                    "source": document.source,
                                    "section": section_title,
                                }
                            )
                        )
                        sentence_index += 1
                        global_index += 1

                    window_index += 1

        if not chunks:
            return Result(
                success=False,
                error=Error(
                    code="EMPTY_DOCUMENT",
                    message="Cannot chunk an empty document"
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
        """
        Groups lines into sections on heading boundaries. Only closes a
        section once it has real body content beyond the heading line
        itself - a run of consecutive heading-like lines (a table of
        contents, a stack of repeated running headers) keeps
        accumulating into the same pending section instead of each
        becoming its own near-empty section. Without this, every TOC
        entry ("Attributes of the AI RMF 3", page number and all)
        becomes a standalone one-sentence chunk that can outrank real
        content on an exact-phrase query, since it IS that exact
        phrase.
        """
        sections: list[tuple[str | None, list[str]]] = []
        current_title: str | None = None
        current_lines: list[str] = []

        for line in content.splitlines():
            stripped = line.strip()
            if not stripped:
                continue

            is_heading = self._looks_like_heading(stripped)

            if is_heading and self._has_body_content(current_lines):
                sections.append((current_title, current_lines))
                current_title = stripped.strip("# ").strip()
                current_lines = [stripped]
            else:
                if is_heading:
                    current_title = stripped.strip("# ").strip()
                current_lines.append(stripped)

        if current_lines:
            sections.append((current_title, current_lines))

        return [
            (title, "\n".join(lines))
            for title, lines in sections
        ]

    def _has_body_content(
        self,
        lines: list[str]
    ) -> bool:
        return any(not self._looks_like_heading(line) for line in lines)

    def _build_windows(
        self,
        text: str
    ) -> list[str]:
        """
        The old chunk-level unit, unchanged: greedy sentence-packing up
        to chunk_size, with a trailing character-overlap carried into
        the next window for continuity. No longer the retrievable unit
        itself - each window's text is split into individual sentences
        right after this returns (see chunk()), and only those
        sentences are independently embedded/stored. A window survives
        only as parent_chunk_id metadata for generation-time context
        expansion.
        """
        if len(text) <= self.chunk_size:
            return [text]

        sentences = self._split_sentences(text)
        windows: list[str] = []
        current = ""

        for sentence in sentences:
            candidate = f"{current} {sentence}".strip()

            if len(candidate) <= self.chunk_size:
                current = candidate
                continue

            if current:
                windows.append(current)
                current = self._with_overlap(current, sentence)
            else:
                windows.extend(self._split_long_sentence(sentence))
                current = ""

        if current:
            windows.append(current)

        return self._merge_tiny_chunks(windows)

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

    def _merge_tiny_sentences(
        self,
        sentences: list[str]
    ) -> list[str]:
        """
        Same spirit as _merge_tiny_chunks, at sentence granularity - a
        sentence under minimum_sentence_size ("See Section 4.2.") gets
        folded into the previous sentence rather than becoming its own
        standalone embedded unit. The first sentence in a window always
        stays standalone even if tiny (nothing earlier to merge into
        yet), matching _merge_tiny_chunks' existing behavior exactly.
        """
        if len(sentences) <= 1:
            return sentences

        merged: list[str] = []

        for sentence in sentences:
            if merged and len(sentence) < self.minimum_sentence_size:
                merged[-1] = f"{merged[-1]} {sentence}".strip()
            else:
                merged.append(sentence)

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
