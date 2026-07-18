from unittest.mock import MagicMock
from unittest.mock import patch

from rag.guardrails.base import Action
from rag.guardrails.base import GuardrailContext
from rag.guardrails.base import Severity
from rag.guardrails.presidio_pii_guard import PresidioPIIGuard


class FakeRecognizerResult:

    def __init__(self, entity_type, start, end, score):
        self.entity_type = entity_type
        self.start = start
        self.end = end
        self.score = score


def _build_guard(analyze_return_value):
    with patch("rag.guardrails.presidio_pii_guard.NlpEngineProvider"), \
         patch("rag.guardrails.presidio_pii_guard.AnalyzerEngine") as mock_analyzer_class:
        mock_analyzer = mock_analyzer_class.return_value
        mock_analyzer.analyze.return_value = analyze_return_value
        guard = PresidioPIIGuard()
        # analyzer.analyze is re-checked per test via guard.analyzer
        return guard, mock_analyzer


def test_clean_answer_is_not_triggered():

    guard, mock_analyzer = _build_guard([])

    finding = guard.check(GuardrailContext(query="q", answer="Nothing sensitive here."))

    assert finding.triggered is False
    assert finding.action == Action.ALLOW
    assert finding.redacted_text is None
    mock_analyzer.analyze.assert_called_once()


def test_person_name_is_redacted():

    text = "John Smith called yesterday."
    guard, _ = _build_guard([FakeRecognizerResult("PERSON", 0, 10, 0.85)])

    finding = guard.check(GuardrailContext(query="q", answer=text))

    assert finding.triggered is True
    assert finding.action == Action.REDACT
    assert finding.severity == Severity.HIGH
    assert finding.redacted_text == "[REDACTED_PERSON] called yesterday."
    assert "PERSON" in finding.metadata["detected_entities"]


def test_low_score_results_are_filtered_out():

    text = "John Smith called yesterday."
    guard, _ = _build_guard([FakeRecognizerResult("PERSON", 0, 10, 0.2)])
    guard.score_threshold = 0.5

    finding = guard.check(GuardrailContext(query="q", answer=text))

    assert finding.triggered is False


def test_overlapping_spans_prefer_higher_score():

    # EMAIL_ADDRESS (whole) and URL (substring) overlap - EMAIL_ADDRESS
    # has a higher score, so it should win and URL should be dropped
    text = "Contact john@company.com now."
    guard, _ = _build_guard([
        FakeRecognizerResult("EMAIL_ADDRESS", 8, 24, 1.0),
        FakeRecognizerResult("URL", 13, 24, 0.5)
    ])

    finding = guard.check(GuardrailContext(query="q", answer=text))

    assert finding.metadata["detected_entities"] == ["EMAIL_ADDRESS"]
    assert "[REDACTED_EMAIL_ADDRESS]" in finding.redacted_text
    assert "[REDACTED_URL]" not in finding.redacted_text


def test_multiple_non_overlapping_spans_are_all_redacted():

    text = "John Smith lives in Paris."
    guard, _ = _build_guard([
        FakeRecognizerResult("PERSON", 0, 10, 0.85),
        FakeRecognizerResult("LOCATION", 20, 25, 0.85)
    ])

    finding = guard.check(GuardrailContext(query="q", answer=text))

    assert finding.redacted_text == "[REDACTED_PERSON] lives in [REDACTED_LOCATION]."
    assert set(finding.metadata["detected_entities"]) == {"PERSON", "LOCATION"}


def test_empty_answer_is_not_triggered_without_calling_analyzer():

    guard, mock_analyzer = _build_guard([])

    finding = guard.check(GuardrailContext(query="q", answer=""))

    assert finding.triggered is False
    mock_analyzer.analyze.assert_not_called()
