import re
from re import Pattern

DEFAULT_PATTERNS: tuple[Pattern[str], ...] = (
    re.compile(r"ignore (all |any )?(previous|prior|above) instructions", re.IGNORECASE),
    re.compile(r"disregard (all |any )?(previous|prior|above) (instructions|rules)", re.IGNORECASE),
    re.compile(r"you are now (in )?(developer|dan|jailbreak) mode", re.IGNORECASE),
    re.compile(r"pretend (you are|to be) .*(no rules|unrestricted|without restrictions)", re.IGNORECASE),
    re.compile(r"reveal (your |the )?(system prompt|system instructions)", re.IGNORECASE),
    re.compile(r"act as (if )?(you (have|had) no|there (are|were) no) (restrictions|rules|guardrails)", re.IGNORECASE),
    re.compile(r"\bDAN\b"),
    re.compile(r"bypass (your |the )?(safety|content) (filters?|guardrails?)", re.IGNORECASE),
    # indirect-injection-specific: a retrieved document impersonating a
    # higher-privilege message role, or trying to get the model to call a
    # tool or leak a secret it was never asked about
    re.compile(r"\[?(system|developer)\s*(message|prompt)?\]?\s*:", re.IGNORECASE),
    re.compile(r"new instructions?\s*:", re.IGNORECASE),
    re.compile(r"call (the )?(tool|function|api)\b", re.IGNORECASE),
    re.compile(r"(reveal|print|output|leak)\s+(the\s+)?(api[\s_-]?key|secret|password|credentials?)", re.IGNORECASE),
)
