import re


class TextCleaner:
    """
    Deterministic text normalization for ingestion output.
    """

    def clean(
        self,
        content: str
    ) -> str:
        lines = []

        for line in content.splitlines():
            normalized = re.sub(r"\s+", " ", line).strip()
            if normalized:
                lines.append(normalized)

        return "\n".join(lines)
