"""
Analysis Tools - Tools for the Research Agent's gather_info node.

Provides tools for:
- Comparing documents against criteria (Compliance Check)
"""
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from app.config import get_settings
from langchain_google_genai import ChatGoogleGenerativeAI
from app.agents.prompts.research_prompt import COMPARE_COMPLIANCE_TOOL_PROMPT

# Lazy-load LLM
_llm = None

def _get_llm():
    """Lazy-initialize LLM for analysis."""
    global _llm
    if _llm is None:
        settings = get_settings()
        _llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=settings.google_api_key,
        )
    return _llm

@tool
async def compare_compliance(criteria: str, target_content: str) -> str:
    """
    Compare a document (target_content) against a set of criteria or laws.
    
    Use this tool to:
    - Check if a contract complies with the law
    - Verify if a document follows a checklist
    - Identify risks and missing clauses
    
    Args:
        criteria: The standard to check against (e.g., text of a law, a checklist, or rules).
        target_content: The text of the document/contract to be analyzed.
        
    Returns:
        Structured markdown report containing the comparison analysis.
    """
    try:
        if not criteria or not target_content:
            return "Error: Both 'criteria' and 'target_content' are required for comparison."

        llm = _get_llm()
        
        prompt = f"""{COMPARE_COMPLIANCE_TOOL_PROMPT}

## CRITERIA (КРИТЕРИИ):
{criteria[:20000]} 

## TARGET CONTENT (ЦЕЛЕВО СЪДЪРЖАНИЕ):
{target_content[:30000]}
"""
        # Truncating to avoid context limits if extremely large, but Gemini handles large context well.
        # Adjusted limits conservatively.
        
        print(f"[compare_compliance] ⚖️ Comparing content ({len(target_content)} chars) vs criteria ({len(criteria)} chars)...")
        
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        
        return response.content
        
    except Exception as e:
        print(f"[compare_compliance] ❌ Error: {e}")
        return f"Error performing comparison: {str(e)}"
