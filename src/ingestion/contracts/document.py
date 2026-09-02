from datetime import datetime
from typing import Any

from pydantic import BaseModel
from pydantic import Field


class Document(BaseModel):
    """
    Canonical document contract used throughout the platform.
    """

    document_id: str = Field(
        description="Unique identifier for the document"
    )

    source: str = Field(
        description="Original file path or source location"
    )

    document_type: str = Field(
        description="Document category"
    )

    content: str = Field(
        description="Normalized document content"
    )

    pages: list[str] | None = Field(
        default=None,
        description=(
            "Per-page text, 1-indexed by list position (pages[0] is page "
            "1), for formats with a real page concept. A page with no "
            "extractable text is an empty string, not a missing entry - "
            "the list stays aligned with real page numbers. None for "
            "formats with no native pagination (docx, markdown) - "
            "chunking treats the whole document as a single virtual "
            "page 1 in that case."
        )
    )

    owner: str | None = Field(
        default=None,
        description="Document owner"
    )

    created_at: datetime | None = None

    updated_at: datetime | None = None

    metadata: dict[str, Any] = Field(
        default_factory=dict
    )
