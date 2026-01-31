from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from app.agents.analysis_agent import AnalysisAgent
from app.agents.schemas.analysis_schemas import DocumentAnalysis

router = APIRouter()

class AnalysisRequest(BaseModel):
    """Request to analyze document text."""
    content: str = Field(..., description="The extracted text content of the document.")

@router.post("/analyze", response_model=DocumentAnalysis)
async def analyze_document(request: AnalysisRequest):
    """
    Expert legal analysis of document text.
    
    Returns structured JSON with summary, document type, parties, and risks.
    The UI should use this endpoint during local indexing to generate metadata
    without ever exposing the user's API keys locally.
    """
    if not request.content or len(request.content.strip()) < 10:
        raise HTTPException(status_code=400, detail="Content too short to analyze.")

    try:
        agent = AnalysisAgent()
        # Truncate content to 200k chars for LLM safety
        result = await agent.analyze(request.content[:200000])
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM Analysis failed: {str(e)}")
