from rag.chunking.chunk import Chunk
from rag.generation.extractive_answerer import ExtractiveAnswerer
from rag.retrieval.hybrid_retrieval import RetrievedChunk


def _retrieved(text: str, chunk_id: str = "doc:0") -> RetrievedChunk:
    chunk = Chunk(
        chunk_id=chunk_id,
        document_id="doc",
        chunk_index=0,
        text=text,
        source="doc.md",
        document_type="markdown"
    )
    return RetrievedChunk(chunk=chunk, vector_score=0.5, keyword_score=0.5, score=0.5)


def test_returns_fallback_message_when_no_chunks_retrieved():

    answerer = ExtractiveAnswerer()

    assert answerer.answer("any question", []) == (
        "I could not find relevant context in the indexed documents."
    )


def test_picks_the_sentence_with_the_most_query_term_overlap():

    answerer = ExtractiveAnswerer()
    retrieved = [_retrieved(
        "Employees receive 20 days of paid leave annually. "
        "Contractors receive 10 days of leave. "
        "All requests must be approved by a manager."
    )]

    answer = answerer.answer("How many leave days do contractors receive?", retrieved)

    assert answer == "Contractors receive 10 days of leave."


def test_searches_across_all_retrieved_chunks_not_just_the_first():

    answerer = ExtractiveAnswerer()
    retrieved = [
        _retrieved("The weather today is sunny with a light breeze.", "doc:0"),
        _retrieved("Contractors receive 10 days of leave per year.", "doc:1"),
    ]

    answer = answerer.answer("How many leave days do contractors get?", retrieved)

    assert answer == "Contractors receive 10 days of leave per year."


def test_ties_are_broken_by_the_first_matching_sentence():

    answerer = ExtractiveAnswerer()
    retrieved = [_retrieved("Leave policy applies to staff. Leave policy applies to contractors.")]

    answer = answerer.answer("leave policy", retrieved)

    assert answer == "Leave policy applies to staff."


def test_falls_back_to_first_chunk_text_when_no_sentences_are_found():

    answerer = ExtractiveAnswerer()
    retrieved = [_retrieved("")]

    answer = answerer.answer("anything", retrieved)

    assert answer == ""


def test_answer_is_grounded_in_retrieved_text_by_construction():
    """
    ExtractiveAnswerer never generates text - whatever it returns is a
    substring of some retrieved chunk's text. This is what makes it
    hallucination-free by construction rather than by a guardrail catching
    it after the fact.
    """
    answerer = ExtractiveAnswerer()
    chunk_text = "Business class is not allowed for employees below director level."
    retrieved = [_retrieved(chunk_text)]

    answer = answerer.answer("Is business class allowed?", retrieved)

    assert answer in chunk_text
