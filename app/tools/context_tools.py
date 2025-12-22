"""
Context Tools - Tools for managing shared session context.

Allows agents to explicitly "commit" files to the conversation context.
This acts as a "Session Vector Store" addition, flagging files as relevant
so subsequent agents don't need to re-search for them.
"""
import json
from langchain_core.tools import tool
from pydantic import BaseModel, Field

# We use a global variable or cache to store this *per session* if needed,
# but for now the Orchestrator will manage the state persistence.
# This tool mainly serves as a SIGNAL to the Orchestrator.

@tool
def add_to_context(
    files: list[str] = Field(description="List of virtual file paths to add to context (e.g. ['/project/contract.pdf'])"),
    summary: str = Field(description="Brief summary of why these files are relevant or what they contain."),
    action: str = "add_to_context" # Hidden field to help parsing if needed
) -> str:
    """
    Add specific files to the Active Conversation Context.
    Use this when you have found RELEVANT documents that should be available to other agents.
    
    This prevents other agents from effectively having to search for them again.
    """
    # structure the output as JSON so the Orchestrator can parse it easily
    result = {
        "action": "add_to_context",
        "files": files,
        "summary": summary
    }
    return json.dumps(result, ensure_ascii=False)

CONTEXT_TOOLS = [add_to_context]
