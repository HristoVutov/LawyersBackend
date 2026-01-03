"""Structured output schemas for AnalysisAgent."""
from pydantic import BaseModel, Field
from typing import Optional, List, Dict

class DocumentParty(BaseModel):
    role: str = Field(description="Роля (напр. Купувач, Продавач)")
    name: str = Field(description="Име на лице/фирма")

class DocumentRisk(BaseModel):
    risk: str = Field(description="Описание на риска")
    severity: str = Field(description="Ниво на риск: Висок, Среден, Нисък")
    recommendation: str = Field(description="Препоръка за намаляване на риска")

class DocumentSegment(BaseModel):
    segmentType: str = Field(description="Тип на под-документа (напр. Анекс, Протокол)")
    summary: str = Field(description="Кратко резюме на сегмента")
    startText: str = Field(description="Първите 10-15 думи от началото на сегмента")
    endText: str = Field(description="Последните 10-15 думи от края на сегмента")
    segmentStartPage: Optional[int] = Field(description="Номер на начална страница (ако има [Page N] маркери)", default=None)
    segmentEndPage: Optional[int] = Field(description="Номер на крайна страница (ако има [Page N] маркери)", default=None)

class DocumentAnalysis(BaseModel):
    """
    Structured analysis of a legal document.
    """
    documentType: str = Field(description="Вид на документа")
    containsMultipleDocuments: bool = Field(description="Дали файлът съдържа множество логически различни документи")
    documentSegments: List[DocumentSegment] = Field(default_factory=list, description="Списък с откритите под-документи")
    
    summary: str = Field(description="Кратко резюме на целия файл (3-5 изречения)")
    keywords: List[str] = Field(description="Списък с ключови думи")
    parties: List[DocumentParty] = Field(default_factory=list, description="Страни по документа")
    keyClauses: List[str] = Field(default_factory=list, description="Важни клаузи")
    risks: List[DocumentRisk] = Field(default_factory=list, description="Идентифицирани рискове")
    missingElements: List[str] = Field(default_factory=list, description="Липсващи елементи")
    
    indexFields: Dict[str, str] = Field(default_factory=dict, description="Специфични полета за индексиране")
    overallScore: float = Field(description="Оценка за качество/сигурност (1-10)")
