"""
Analysis Agent - Expert document analyzer that returns structured JSON.

Capabilities:
- Analyze legal documents
- Extract key information (parties, clauses, risks)
- Return structured JSON output
"""
from app.agents.base_agent import BaseAgent, AgentConfig
from app.agents.prompts.analysis_prompt import ANALYSIS_SYSTEM_PROMPT


from app.agents.schemas.analysis_schemas import DocumentAnalysis

class AnalysisAgent(BaseAgent):
    """
    Analysis Agent - Expert legal document analyzer.
    
    This agent has no tools - it receives document content and returns
    structured JSON analysis using Pydantic schemas.
    
    Used for:
    - Document indexing (extracting metadata)
    - Risk analysis
    - Party extraction
    """
    
    def __init__(self):
        config = AgentConfig(
            name="analysis_agent",
            tools=[],  # No tools needed - pure analysis
            system_prompt=ANALYSIS_SYSTEM_PROMPT,
        )
        super().__init__(config)

    async def analyze(self, content: str) -> DocumentAnalysis:
        """
        Analyze the document content and return a structured Pydantic model.
        Uses with_structured_output for robust JSON generation.
        """
        # Create a specialized LLM for structured output
        structured_llm = self.llm.with_structured_output(DocumentAnalysis)
        
        # Invoke it
        print(f"[AnalysisAgent] Sending request to LLM (structured output)...")
        result = await structured_llm.ainvoke(
            self.system_prompt + "\n\nAnalyze this document:\n" + content
        )
        print(f"[AnalysisAgent] Received response from LLM.")
        return result
