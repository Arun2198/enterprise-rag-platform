import logging
import time

from presidio_analyzer import AnalyzerEngine
from presidio_analyzer import Pattern
from presidio_analyzer import PatternRecognizer
from presidio_analyzer import RecognizerResult
from presidio_analyzer.nlp_engine import NlpEngineProvider

from rag.guardrails.base import Action
from rag.guardrails.base import GuardrailContext
from rag.guardrails.base import GuardrailFinding
from rag.guardrails.base import GuardrailStage
from rag.guardrails.base import Severity

logger = logging.getLogger(__name__)

DEFAULT_ENTITIES: tuple[str, ...] = (
    "PERSON",
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "LOCATION",
    "CREDIT_CARD",
    "US_SSN",
    "IBAN_CODE",
    "US_PASSPORT",
    "AADHAAR",
)


def _build_aadhaar_recognizer(language: str) -> PatternRecognizer:
    # Presidio has no built-in Aadhaar (Indian national ID) recognizer;
    # reuses the same regex as the plain PIIGuard for consistency.
    return PatternRecognizer(
        supported_entity="AADHAAR",
        patterns=[
            Pattern(
                name="aadhaar_pattern",
                regex=r"\b\d{4}[ -]?\d{4}[ -]?\d{4}\b",
                score=0.6
            )
        ],
        supported_language=language
    )


class PresidioPIIGuard:
    """
    Phase 2 example guardrail (output stage): PII detection/redaction
    using Microsoft Presidio's NER + pattern recognizers instead of plain
    regex - catches names, addresses, and other context-dependent PII the
    regex-only PIIGuard structurally can't (it has no notion of "this
    looks like a person's name"). Registered alongside PIIGuard, not
    replacing it. The AnalyzerEngine (and its spaCy model) is loaded once
    in the constructor and reused for every check.
    """
    name = "presidio_pii_guard"
    stage = GuardrailStage.OUTPUT

    def __init__(
        self,
        entities: tuple[str, ...] | None = None,
        score_threshold: float = 0.5,
        language: str = "en",
        spacy_model_name: str = "en_core_web_sm"
    ) -> None:
        self.entities = tuple(entities or DEFAULT_ENTITIES)
        self.score_threshold = score_threshold
        self.language = language
        self.analyzer = self._build_analyzer(language, spacy_model_name)

    def _build_analyzer(
        self,
        language: str,
        spacy_model_name: str
    ) -> AnalyzerEngine:
        provider = NlpEngineProvider(
            nlp_configuration={
                "nlp_engine_name": "spacy",
                "models": [{"lang_code": language, "model_name": spacy_model_name}]
            }
        )
        analyzer = AnalyzerEngine(
            nlp_engine=provider.create_engine(),
            supported_languages=[language]
        )
        analyzer.registry.add_recognizer(_build_aadhaar_recognizer(language))
        return analyzer

    def check(
        self,
        context: GuardrailContext
    ) -> GuardrailFinding:
        text = context.answer or ""

        if not text.strip():
            return GuardrailFinding(
                guardrail_name=self.name,
                triggered=False,
                severity=Severity.INFO,
                action=Action.ALLOW,
                message="no PII detected",
                metadata={"detected_entities": []}
            )

        started_at = time.monotonic()
        results = self.analyzer.analyze(
            text=text,
            language=self.language,
            entities=list(self.entities)
        )
        latency_seconds = round(time.monotonic() - started_at, 4)

        spans = self._select_non_overlapping_spans(results)
        detected = [span.entity_type for span in spans]
        triggered = bool(detected)

        logger.info(
            "presidio_pii_check_completed",
            extra={
                "guardrail": self.name,
                "detected_entity_count": len(detected),
                "latency_seconds": latency_seconds
            }
        )

        return GuardrailFinding(
            guardrail_name=self.name,
            triggered=triggered,
            severity=Severity.HIGH if triggered else Severity.INFO,
            action=Action.REDACT if triggered else Action.ALLOW,
            message=(
                f"redacted {len(detected)} PII entity(ies): {', '.join(detected)}"
                if triggered else "no PII detected"
            ),
            redacted_text=self._redact(text, spans) if triggered else None,
            metadata={"detected_entities": detected}
        )

    def _select_non_overlapping_spans(
        self,
        results: list[RecognizerResult]
    ) -> list[RecognizerResult]:
        eligible = [result for result in results if result.score >= self.score_threshold]
        # highest-confidence, most-specific (longest) spans win when spans
        # overlap (e.g. presidio can flag a URL entity fully inside an
        # EMAIL_ADDRESS entity for the same text)
        eligible.sort(key=lambda result: (result.score, result.end - result.start), reverse=True)

        selected: list[RecognizerResult] = []

        for candidate in eligible:
            if any(self._overlaps(candidate, chosen) for chosen in selected):
                continue

            selected.append(candidate)

        selected.sort(key=lambda result: result.start, reverse=True)
        return selected

    def _overlaps(
        self,
        first: RecognizerResult,
        second: RecognizerResult
    ) -> bool:
        return first.start < second.end and second.start < first.end

    def _redact(
        self,
        text: str,
        spans: list[RecognizerResult]
    ) -> str:
        redacted = text

        # spans are sorted start-descending, so replacing from the end of
        # the string backward never shifts the offsets of spans not yet
        # processed
        for span in spans:
            redacted = (
                redacted[:span.start]
                + f"[REDACTED_{span.entity_type}]"
                + redacted[span.end:]
            )

        return redacted
