"""Structured output schemas for DraftingAgent."""
from pydantic import BaseModel, Field
from typing import Optional


class DraftingIntent(BaseModel):
    """Structured output from analyze_node in DraftingAgent.
    
    Replaces fragile JSON parsing with validated Pydantic model.
    """
    intent: str = Field(
        description="Тип на документа: 'договор', 'молба', 'пълномощно', 'протокол' и т.н."
    )
    document_type: str = Field(
        description="Конкретно наименование на документа на български"
    )
    properties: dict = Field(
        default_factory=dict, 
        description="Извлечени ключ-стойност данни (имена, дати, суми и т.н.)"
    )
    suggested_template: Optional[str] = Field(
        default=None, 
        description="Препоръчан път до шаблон"
    )
    missing_info: list[str] = Field(
        default_factory=list, 
        description="Информация, която е необходима от потребителя"
    )
