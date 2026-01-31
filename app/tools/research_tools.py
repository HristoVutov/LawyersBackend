"""
Research Tools - Tools for the Research Agent's gather_info node.

Provides tools for:
- Getting legal references from LLM knowledge
- Searching local documents via DocumentAgent
"""
from typing import Optional
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from app.config import get_settings
from langchain_google_genai import ChatGoogleGenerativeAI
from app.agents.prompts.research_prompt import LEGAL_REFERENCE_TOOL_PROMPT


# We'll lazy-load these to avoid circular imports
_document_agent = None
_llm = None


def _get_llm():
    """Lazy-initialize LLM for legal reference queries."""
    global _llm
    if _llm is None:
        settings = get_settings()
        _llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=settings.google_api_key,
        )
    return _llm


def _get_document_agent():
    """Lazy-initialize DocumentAgent to avoid circular imports."""
    global _document_agent
    if _document_agent is None:
        from app.agents.document_agent import DocumentAgent
        _document_agent = DocumentAgent()
    return _document_agent


# --- Pydantic Models for Structured Output ---

class ApplicableLaw(BaseModel):
    law_name: str = Field(..., description="Full name of the law (e.g., 'Закон за задълженията и договорите')")
    abbreviation: str = Field(..., description="Common abbreviation (e.g., 'ЗЗД')")
    articles: list[str] = Field(..., description="List of specific articles/provisions (e.g., ['чл. 26', 'чл. 27'])")
    summary: str = Field(..., description="Brief summary of why this law is applicable")
    source_url: Optional[str] = Field(None, description="URL to the official text of the law (e.g., lex.bg link)")

class LegalResearchResponse(BaseModel):
    applicable_laws: list[ApplicableLaw] = Field(..., description="List of laws found relevant to the query")
    relevant_case_law: list[str] = Field(..., description="List of relevant court decisions (e.g., 'ТР 1/2020 ОСГТК на ВКС')")
    key_provisions: str = Field(..., description="Synthesis of the key legal provisions and how they apply")
    search_terms: list[str] = Field(..., description="Suggested search terms for further research")


@tool
async def get_legal_references(query: str) -> str:
    """
    Get legal references (laws, articles, case law) from LLM knowledge.
    
    Use this tool when you need to identify:
    - Applicable Bulgarian or EU laws
    - Specific articles and provisions
    - Relevant court decisions (ВКС, ВАС)
    - Legal framework for a question
    
    Args:
        query: The legal question or topic to research
        
    Returns:
        JSON with applicable laws, articles, case law, and source URLs
    """
    try:
        llm = _get_llm()
        structured_llm = llm.with_structured_output(LegalResearchResponse)
        
        prompt = f"""{LEGAL_REFERENCE_TOOL_PROMPT}

## Правен въпрос:
{query}
"""
        
        response = await structured_llm.ainvoke([HumanMessage(content=prompt)])
        
        print(f"[get_legal_references] ⚖️ Got legal references for: {query[:50]}...")
        
        # Return as JSON string for tool output compatibility
        return response.model_dump_json(indent=2)
        
    except Exception as e:
        print(f"[get_legal_references] ❌ Error: {e}")
        return f"Error getting legal references: {str(e)}"



@tool
async def search_documents(query: str, thread_id: str = "default") -> str:
    """
    Search and read local documents related to a query.
    
    Use this tool when you need to:
    - Find documents in the user's project
    - Read content from local files (.docx, .pdf, .txt, etc.)
    - Search indexed documents for specific information
    
    Args:
        query: What to search for in local documents
        thread_id: Conversation ID for caching
        
    Returns:
        Content from found documents with file paths
    """
    try:
        from app.agents.orchestrator import get_agent
        doc_agent = get_agent("document_agent")
        
        if not doc_agent:
            print("[search_documents] ❌ Document Agent not registered.")
            return "Error: Document Agent not available in registry."
        
        print(f"[search_documents] 🔍 Searching for: {query[:50]}...")
        
        # Invoke DocumentAgent subgraph
        sub_inputs = {
            "messages": [HumanMessage(content=f"Намери и извлечи пълния текст на документи относно: {query}")]
        }
        sub_config = {
            "configurable": {"thread_id": f"search_{thread_id}"},
            "metadata": {"agent": "document_agent"},
            "tags": ["document_agent", "subgraph"],
            "run_name": f"document_agent:search"
        }
        
        result_state = await doc_agent.graph.ainvoke(sub_inputs, sub_config)
        
        # Extract the response from DocumentAgent
        messages = result_state.get("messages", [])
        if messages:
            content = messages[-1].content
            print(f"[search_documents] ✅ Found content ({len(content)} chars)")
            return content
        else:
            return "No documents found matching the query."
            
    except Exception as e:
        print(f"[search_documents] ❌ Error: {e}")
        return f"Error searching documents: {str(e)}"


@tool
async def get_project_overview(include_summaries: bool = True) -> str:
    """
    Get a summary of all files, their pre-computed summaries, and project metadata from the client UI.
    
    Use this tool at the START of a project or when you need a high-level overview
    of all documents in the folder without searching them one-by-one.
    
    Returns:
        JSON/Text containing file list, summaries, and meta info.
    """
    return "Error: get_project_overview must be executed on the client-side via Remote Bridge."


# Export all research tools
RESEARCH_TOOLS = [
    get_legal_references,
    search_documents,
    get_project_overview,
]
