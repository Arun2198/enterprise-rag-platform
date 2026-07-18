from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


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

    owner: str | None = Field(
        default=None,
        description="Document owner"
    )

    created_at: datetime | None = None

    updated_at: datetime | None = None

    metadata: dict[str, Any] = Field(
        default_factory=dict
    )
