# App-level schemas - Pydantic models with no dependencies
# This location prevents circular imports as schemas have no app logic dependencies

from app.schemas.tool_schemas import (
    ApplicableLaw,
    LegalReferenceResult,
)

__all__ = [
    "ApplicableLaw",
    "LegalReferenceResult",
]
