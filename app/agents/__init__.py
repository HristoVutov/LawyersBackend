# Agents package - LangGraph agents
#
# Uses lazy imports via __getattr__ to avoid circular dependencies.
# The circular chain was: tools.research_tools -> agents.__init__ -> research_agent -> tools.research_tools

import importlib

def __getattr__(name):
    """Lazy import agents to avoid circular dependencies."""
    
    if name in ("BaseAgent", "AgentConfig"):
        module = importlib.import_module("app.agents.base_agent")
        return getattr(module, name)
    
    if name == "DocumentAgent":
        module = importlib.import_module("app.agents.document_agent")
        return module.DocumentAgent
    
    if name == "ResearchAgent":
        module = importlib.import_module("app.agents.research_agent")
        return module.ResearchAgent
    
    if name == "DraftingAgent":
        module = importlib.import_module("app.agents.drafting_agent")
        return module.DraftingAgent
    
    if name == "TemplateAgent":
        module = importlib.import_module("app.agents.template_agent")
        return module.TemplateAgent
    
    if name == "AnalysisAgent":
        module = importlib.import_module("app.agents.analysis_agent")
        return module.AnalysisAgent
    
    if name in ("OrchestratorAgent", "initialize_agent_registry", "register_agent", "get_agent", "list_agent_names"):
        module = importlib.import_module("app.agents.orchestrator")
        return getattr(module, name)
    
    raise AttributeError(f"module 'app.agents' has no attribute '{name}'")


__all__ = [
    "BaseAgent",
    "AgentConfig",
    "DocumentAgent",
    "ResearchAgent",
    "DraftingAgent",
    "TemplateAgent",
    "AnalysisAgent",
    "OrchestratorAgent",
    "initialize_agent_registry",
    "register_agent",
    "get_agent",
    "list_agent_names",
]
