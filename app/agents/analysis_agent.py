"""
Analysis Agent - Expert document analyzer that returns structured JSON.

Capabilities:
- Analyze legal documents
- Extract key information (parties, clauses, risks)
- Return structured JSON output
"""
from app.agents.base_agent import BaseAgent, AgentConfig
from app.agents.prompts.analysis_prompt import ANALYSIS_SYSTEM_PROMPT


class AnalysisAgent(BaseAgent):
    """
    Analysis Agent - Expert legal document analyzer.
    
    This agent has no tools - it receives document content and returns
    structured JSON analysis.
    
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
