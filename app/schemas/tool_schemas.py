"""Tool output schemas - Pydantic models for structured tool responses.

Located at app-level (not agents/) to prevent circular imports since tools 
import from here and agents also import from here.
"""
from pydantic import BaseModel, Field
from typing import Optional


class ApplicableLaw(BaseModel):
    """A single applicable law entry."""
    law: str = Field(description="Пълно наименование на закона")
    abbreviation: str = Field(description="Абревиатура (напр. ЗЗД, ТЗ, ГПК)")
    articles: list[str] = Field(default_factory=list, description="Списък с членове (напр. 'чл. 26', 'чл. 87-88')")
    summary: str = Field(description="Кратко обобщение на приложимостта")
    source_url: Optional[str] = Field(default=None, description="Линк към lex.bg или друг официален източник")


class LegalReferenceResult(BaseModel):
    """Structured output from get_legal_references tool.
    
    Guarantees valid structure for legal framework information.
    """
    applicable_laws: list[ApplicableLaw] = Field(
        default_factory=list, 
        description="Списък с приложими закони"
    )
    relevant_case_law: list[str] = Field(
        default_factory=list, 
        description="Релевантна съдебна практика (напр. 'ТР 1/2020 ОСГТК на ВКС')"
    )
    key_provisions: str = Field(
        default="", 
        description="Обобщение на ключовите разпоредби"
    )
    search_terms: list[str] = Field(
        default_factory=list, 
        description="Термини за допълнително търсене"
    )
