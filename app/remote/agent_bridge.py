from typing import List
from app.remote.file_tools import get_remote_file_tools
from app.tools.file_tools import FILE_TOOLS as LOCAL_FILE_TOOLS

def patch_agent_tools(agent, thread_id: str, is_remote: bool = False):
    """
    Switches an agent's tools between local and remote versions.
    
    Args:
        agent: The agent instance to patch (e.g. DocumentAgent)
        thread_id: The conversation thread ID for tool routing
        is_remote: Whether to use remote tools (Server deployment) or local tools (Local development)
    """
    if not is_remote:
        # Keep local tools (default behavior)
        return

    # 1. Identify which tools to replace (e.g., read_file, glob, grep)
    local_tool_names = ["read_file", "write_file", "list_directory", "glob", "grep", "execute_ocr", "search_indexed", "read_document", "get_project_overview"]
    
    # 2. Get remote alternatives
    remote_tools = get_remote_file_tools(thread_id)
    remote_map = {t.name: t for t in remote_tools}

    # 3. Rebuild the tools list
    new_tools = []
    for tool in agent.tools:
        if hasattr(tool, "name") and tool.name in remote_map:
            new_tools.append(remote_map[tool.name])
        else:
            new_tools.append(tool)
            
    # 4. Update the agent
    agent.tools = new_tools
    
    # Re-bind tools to LLM if necessary (agent classes usually handle this in their internal logic)
    if hasattr(agent, "llm_with_tools"):
        agent.llm_with_tools = agent.llm.bind_tools(new_tools)
    
    # 5. Reset the graph so it re-compiles with the new tools on next access
    agent._graph = None
    
    print(f"🌉 [Bridge] Patched {agent.name} with remote tools and reset graph for thread {thread_id}")

def patch_all_agents(thread_id: str, is_remote: bool = False):
    """
    Convenience function to patch all registered agents (including Orchestrator sub-agents).
    """
    if not is_remote:
        return

    from app.agents.orchestrator import _agent_registry, OrchestratorAgent
    
    # 1. Patch the orchestrator if needed (it usually doesn't have tools, but just in case)
    # Actually, we need to find the orchestrator instance.
    # Typically it's a singleton in app/api/websocket.py
    
    # 2. Patch all workers in the registry
    for name, agent in _agent_registry.items():
        patch_agent_tools(agent, thread_id, is_remote=True)
        
    print(f"🌉 [Bridge] All agents patched for thread {thread_id}")
