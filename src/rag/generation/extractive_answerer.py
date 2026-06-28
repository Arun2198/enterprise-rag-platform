import re

from rag.retrieval.hybrid_retrieval import RetrievedChunk


class ExtractiveAnswerer:
    """
    Grounded local answerer for MVP development.

    It intentionally answers only from retrieved text. A managed LLM gateway can
    replace this class while keeping the same service-level contract.
    """

    def answer(
        self,
        query: str,
        retrieved_chunks: list[RetrievedChunk]
    ) -> str:
        if not retrieved_chunks:
            return "I could not find relevant context in the indexed documents."

        query_terms = self._tokens(query)
        best_sentence = ""
        best_score = -1

        for item in retrieved_chunks:
            for sentence in self._sentences(item.chunk.text):
                sentence_terms = self._tokens(sentence)
                score = len(query_terms.intersection(sentence_terms))

                if score > best_score:
                    best_score = score
                    best_sentence = sentence

        if not best_sentence:
            return retrieved_chunks[0].chunk.text

        return best_sentence

    def _sentences(
        self,
        text: str
    ) -> list[str]:
        return [
            sentence.strip()
            for sentence in re.split(r"(?<=[.!?])\s+", text.replace("\n", " "))
            if sentence.strip()
        ]

    def _tokens(
        self,
        text: str
    ) -> set[str]:
        return set(re.findall(r"[a-z0-9]+", text.lower()))
