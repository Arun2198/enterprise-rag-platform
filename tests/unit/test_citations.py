from rag.chunking.chunk import Chunk
from rag.generation.citations import extract_citations
from rag.retrieval.hybrid_retrieval import RetrievedChunk


def _mk_retrieved(index: int, chunk_id: str) -> RetrievedChunk:
    chunk = Chunk(
        chunk_id=chunk_id, document_id="d1", document_version=2, chunk_index=index,
        text="some text", source="test", document_type="md", parent_section="Intro"
    )
    return RetrievedChunk(chunk=chunk, vector_score=0.9, keyword_score=0.0, score=0.9, rank=index + 1)


def test_extracts_a_valid_citation():

    retrieved = [_mk_retrieved(0, "d1:0"), _mk_retrieved(1, "d1:1")]

    citations = extract_citations("Contractors get 10 days off [Source 1].", retrieved)

    assert len(citations) == 1
    citation = citations[0]
    assert citation.source_number == 1
    assert citation.valid is True
    assert citation.document_id == "d1"
    assert citation.document_version == 2
    assert citation.chunk_id == "d1:0"
    assert citation.section == "Intro"


def test_flags_a_citation_to_a_source_that_was_never_provided():

    retrieved = [_mk_retrieved(0, "d1:0")]

    citations = extract_citations("According to the docs [Source 5], this is true.", retrieved)

    assert len(citations) == 1
    assert citations[0].source_number == 5
    assert citations[0].valid is False
    assert citations[0].chunk_id is None


def test_extracts_multiple_citations_in_order_including_duplicates():

    retrieved = [_mk_retrieved(0, "d1:0"), _mk_retrieved(1, "d1:1")]

    citations = extract_citations(
        "First point [Source 1]. Second point [Source 2]. Back to first [Source 1].",
        retrieved
    )

    assert [c.source_number for c in citations] == [1, 2, 1]
    assert all(c.valid for c in citations)


def test_returns_empty_list_when_answer_has_no_citations():

    retrieved = [_mk_retrieved(0, "d1:0")]

    citations = extract_citations("Contractors get 10 days off.", retrieved)

    assert citations == []


def test_case_insensitive_matching():

    retrieved = [_mk_retrieved(0, "d1:0")]

    citations = extract_citations("As shown [source 1], this holds.", retrieved)

    assert len(citations) == 1
    assert citations[0].valid is True
