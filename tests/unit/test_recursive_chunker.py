import hashlib

from ingestion.contracts.document import Document
from rag.chunking.recursive_chunker import RecursiveChunker


def test_recursive_chunker_preserves_metadata_and_order():

    document = Document(
        document_id="doc-1",
        source="policy.md",
        document_type="markdown",
        content="# Leave Policy\nEmployees get 20 days. Contractors get 10 days.",
        owner="HR",
        metadata={"category": "policy"}
    )
    chunker = RecursiveChunker(chunk_size=45, chunk_overlap=10, minimum_chunk_size=5)

    result = chunker.chunk(document)

    assert result.success is True
    assert result.data is not None
    assert len(result.data) >= 1
    assert result.data[0].chunk_index == 0
    assert result.data[0].metadata["category"] == "policy"
    assert result.data[0].metadata["document_id"] == "doc-1"


def test_consecutive_toc_like_lines_do_not_each_become_their_own_chunk():
    """
    A table of contents is a run of short, capitalized, no-trailing-period
    lines - exactly what _looks_like_heading() matches. Each TOC entry
    must not become its own near-empty chunk (a "Section Name 12" chunk
    that's just a heading and a page number, with zero real content to
    answer a query about that section).
    """
    document = Document(
        document_id="doc-1",
        source="report.pdf",
        document_type="pdf",
        content=(
            "Table of Contents\n"
            "Introduction 1\n"
            "Key Challenges 3\n"
            "System Design Decisions 5\n"
            "Data Contracts 12\n"
            "1 Introduction\n"
            "This document describes a scalable retrieval system for enterprise documents. "
            "It covers ingestion, chunking, retrieval, and generation in detail."
        ),
        metadata={}
    )
    chunker = RecursiveChunker(chunk_size=900, chunk_overlap=100, minimum_chunk_size=80)

    result = chunker.chunk(document)

    assert result.success is True
    heading_only_chunks = [
        chunk for chunk in result.data
        if chunk.text.strip() in (
            "Introduction 1", "Key Challenges 3",
            "System Design Decisions 5", "Data Contracts 12"
        )
    ]
    assert heading_only_chunks == []
    assert any("scalable retrieval system" in chunk.text for chunk in result.data)


def test_heading_immediately_followed_by_body_still_splits_normally():

    document = Document(
        document_id="doc-1",
        source="policy.md",
        document_type="markdown",
        content=(
            "# Leave Policy\n"
            "Employees get 20 days of paid leave.\n"
            "# Travel Policy\n"
            "Business class is not allowed below director level."
        ),
        metadata={}
    )
    chunker = RecursiveChunker(chunk_size=900, chunk_overlap=50, minimum_chunk_size=10)

    result = chunker.chunk(document)

    assert result.success is True
    sections = {chunk.parent_section for chunk in result.data}
    assert "Leave Policy" in sections
    assert "Travel Policy" in sections


def test_content_hash_matches_the_chunk_text():

    document = Document(
        document_id="doc-1",
        source="policy.md",
        document_type="markdown",
        content="Employees get 20 days of paid leave every year without exception.",
        metadata={}
    )
    chunker = RecursiveChunker(chunk_size=900, chunk_overlap=50, minimum_chunk_size=10)

    result = chunker.chunk(document)

    chunk = result.data[0]
    assert chunk.content_hash == hashlib.sha256(chunk.text.encode("utf-8")).hexdigest()


def test_chunking_version_reflects_the_configured_parameters():

    document = Document(
        document_id="doc-1",
        source="policy.md",
        document_type="markdown",
        content="Some policy content here.",
        metadata={}
    )
    chunker = RecursiveChunker(chunk_size=500, chunk_overlap=50, minimum_chunk_size=20)

    result = chunker.chunk(document)

    assert result.data[0].chunking_version == "sentence:500:50:20:20"


def test_different_chunking_parameters_produce_different_chunking_versions():

    document = Document(
        document_id="doc-1",
        source="policy.md",
        document_type="markdown",
        content="Some policy content here.",
        metadata={}
    )
    chunker_a = RecursiveChunker(chunk_size=500, chunk_overlap=50, minimum_chunk_size=20)
    chunker_b = RecursiveChunker(chunk_size=900, chunk_overlap=120, minimum_chunk_size=80)

    version_a = chunker_a.chunk(document).data[0].chunking_version
    version_b = chunker_b.chunk(document).data[0].chunking_version

    assert version_a != version_b


def test_retrievable_unit_is_a_single_sentence_not_a_packed_window():

    document = Document(
        document_id="doc-1",
        source="policy.md",
        document_type="markdown",
        content=(
            "Employees receive twenty days of paid leave per year. "
            "Contractors receive ten days of leave per year. "
            "All requests must be submitted two weeks in advance."
        ),
        metadata={}
    )
    chunker = RecursiveChunker(chunk_size=900, chunk_overlap=50, minimum_chunk_size=10)

    result = chunker.chunk(document)

    assert result.success is True
    texts = [chunk.text for chunk in result.data]
    assert texts == [
        "Employees receive twenty days of paid leave per year.",
        "Contractors receive ten days of leave per year.",
        "All requests must be submitted two weeks in advance.",
    ]


def test_sentence_chunk_id_format_is_page_scoped_running_count():

    document = Document(
        document_id="my-doc",
        source="policy.pdf",
        document_type="pdf",
        content="First sentence here. Second sentence here. Third sentence here.",
        pages=["First sentence here. Second sentence here. Third sentence here."],
        metadata={}
    )
    chunker = RecursiveChunker(chunk_size=900, chunk_overlap=50, minimum_chunk_size=10)

    result = chunker.chunk(document)

    ids = [chunk.chunk_id for chunk in result.data]
    assert ids == ["my-doc:p1:s0", "my-doc:p1:s1", "my-doc:p1:s2"]
    # 3 segments exactly - IncrementalIndexer._page_number_from_chunk_id
    # parses via rsplit(":", 2) and silently breaks if a 4th segment is
    # ever added.
    assert all(chunk_id.count(":") == 2 for chunk_id in ids)


def test_content_hash_is_per_sentence_not_per_window():

    document = Document(
        document_id="doc-1",
        source="policy.md",
        document_type="markdown",
        content="Employees receive twenty days of leave. Contractors receive ten days of leave.",
        metadata={}
    )
    chunker = RecursiveChunker(chunk_size=900, chunk_overlap=50, minimum_chunk_size=10)

    result = chunker.chunk(document)

    assert len(result.data) == 2
    for chunk in result.data:
        assert chunk.content_hash == hashlib.sha256(chunk.text.encode("utf-8")).hexdigest()
    assert result.data[0].content_hash != result.data[1].content_hash


def test_parent_chunk_id_groups_sentences_from_the_same_window():

    document = Document(
        document_id="doc-1",
        source="policy.md",
        document_type="markdown",
        content="Employees receive twenty days of leave. Contractors receive ten days of leave.",
        metadata={}
    )
    chunker = RecursiveChunker(chunk_size=900, chunk_overlap=50, minimum_chunk_size=10)

    result = chunker.chunk(document)

    assert len(result.data) == 2
    parent_ids = {chunk.parent_chunk_id for chunk in result.data}
    assert len(parent_ids) == 1  # both sentences packed into the same window
    parent_id = parent_ids.pop()
    assert parent_id == "doc-1:p1:w0"
    assert parent_id not in {chunk.chunk_id for chunk in result.data}  # never its own retrievable record


def test_tiny_sentence_is_merged_into_the_previous_sentence():
    """
    A short, low-signal sentence ("See Section 4.2.") gets folded into
    the sentence before it rather than becoming its own standalone
    embedded unit - same spirit as _merge_tiny_chunks, at sentence
    granularity.
    """
    document = Document(
        document_id="doc-1",
        source="policy.md",
        document_type="markdown",
        content=(
            "Employees must complete mandatory compliance training annually. "
            "See Section 4.2."
        ),
        metadata={}
    )
    chunker = RecursiveChunker(
        chunk_size=900, chunk_overlap=50, minimum_chunk_size=10, minimum_sentence_size=20
    )

    result = chunker.chunk(document)

    assert len(result.data) == 1
    assert "See Section 4.2." in result.data[0].text
    assert "mandatory compliance training" in result.data[0].text


def test_tiny_first_sentence_in_a_window_stays_standalone():
    """
    Matches _merge_tiny_chunks' existing behavior exactly: the first
    sentence in a window has nothing earlier to merge into, so it stays
    standalone even if it's under minimum_sentence_size.
    """
    document = Document(
        document_id="doc-1",
        source="policy.md",
        document_type="markdown",
        content="Note. Employees receive twenty days of paid leave every calendar year without exception.",
        metadata={}
    )
    chunker = RecursiveChunker(
        chunk_size=900, chunk_overlap=50, minimum_chunk_size=10, minimum_sentence_size=20
    )

    result = chunker.chunk(document)

    assert result.data[0].text == "Note."
