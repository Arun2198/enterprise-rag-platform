import re
from dataclasses import dataclass

from rag.retrieval.hybrid_retrieval import RetrievedChunk

CITATION_PATTERN = re.compile(r"\[Source\s+(\d+)\]", re.IGNORECASE)


@dataclass(frozen=True)
class Citation:
    """
    One [Source N] marker found in a generated answer, resolved against
    the same numbered source list build_grounded_prompt() gave the model
    (Source 1 = retrieved_chunks[0], etc.). `valid=False` means the model
    cited a source number that was never actually provided - a real
    hallucination signal distinct from HallucinationDetector's whole-
    answer groundedness score, since it catches a specific fabricated
    claim of provenance rather than a generic overlap shortfall.
    """
    source_number: int
    valid: bool
    document_id: str | None = None
    document_version: int | None = None
    chunk_id: str | None = None
    section: str | None = None


def extract_citations(
    answer: str,
    retrieved_chunks: list[RetrievedChunk]
) -> list[Citation]:
    """
    Finds every [Source N] marker in the answer and resolves it against
    the numbered source list. Returns one Citation per marker found (in
    order of appearance, duplicates included - the same source cited
    twice yields two entries), not deduplicated, since a caller counting
    "how many claims cited source 2" needs the raw occurrences.

    Extractive answers won't contain [Source N] markers at all (the
    answer is copied verbatim from a chunk, not generated with citation
    instructions) - this correctly returns an empty list for those,
    not an error.
    """
    citations = []

    for match in CITATION_PATTERN.finditer(answer):
        source_number = int(match.group(1))
        index = source_number - 1

        if 0 <= index < len(retrieved_chunks):
            chunk = retrieved_chunks[index].chunk
            citations.append(Citation(
                source_number=source_number,
                valid=True,
                document_id=chunk.document_id,
                document_version=chunk.document_version,
                chunk_id=chunk.chunk_id,
                section=chunk.parent_section
            ))
        else:
            citations.append(Citation(source_number=source_number, valid=False))

    return citations
