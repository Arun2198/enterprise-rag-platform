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

    assert result.data[0].chunking_version == "recursive:500:50:20"


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
