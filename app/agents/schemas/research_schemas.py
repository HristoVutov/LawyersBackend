"""Structured output schemas for ResearchAgent."""
from pydantic import BaseModel, Field
from typing import Optional


class ResearchResult(BaseModel):
    """Structured research output for downstream agents.
    
    Used by synthesize_node to return consistent, parseable results.
    """
    answer: str = Field(description="Изчерпателен отговор на изследователския въпрос")
    cited_laws: list[str] = Field(
        default_factory=list, 
        description="Списък с цитирани български закони/членове"
    )
    document_sources: list[str] = Field(
        default_factory=list, 
        description="Локални документи, които са референцирани"
    )
    confidence: float = Field(
        default=0.8, 
        ge=0.0, 
        le=1.0, 
        description="Степен на увереност 0-1"
    )
    caveats: list[str] = Field(
        default_factory=list, 
        description="Важни предупреждения или ограничения"
    )


class AnalysisQueries(BaseModel):
    """Structured output from analyze_node.
    
    Extracts search queries from the user's research request.
    """
    queries: list[str] = Field(description="Заявки за търсене, които да се изпълнят")
    reasoning: str = Field(description="Защо тези заявки са релевантни")
