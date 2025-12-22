# Tools package - LangChain tools for agents
#
# Uses importlib for lazy imports to avoid circular dependencies.
# The circular chain was: tools.__init__ -> __getattr__ -> from app.tools.X -> re-loads tools

import importlib

def __getattr__(name):
    """Lazy import using importlib to avoid circular dependencies."""
    
    # File tools
    if name in ("read_file", "write_file", "list_directory", "glob", "grep", "create_docx_file", "FILE_TOOLS"):
        module = importlib.import_module("app.tools.file_tools")
        return getattr(module, name)
    
    # Legal tools
    if name in ("read_document", "fill_template", "LEGAL_TOOLS"):
        module = importlib.import_module("app.tools.legal_tools")
        return getattr(module, name)
    
    # Search tools
    if name in ("search_indexed", "SEARCH_TOOLS"):
        module = importlib.import_module("app.tools.search_tools")
        return getattr(module, name)
    
    # Template tools
    if name in ("list_templates", "read_template", "create_template", "TEMPLATE_TOOLS"):
        module = importlib.import_module("app.tools.template_tools")
        return getattr(module, name)
    
    # Research tools
    if name in ("get_legal_references", "search_documents", "RESEARCH_TOOLS"):
        module = importlib.import_module("app.tools.research_tools")
        return getattr(module, name)
    
    # ALL_TOOLS
    if name == "ALL_TOOLS":
        file_tools = importlib.import_module("app.tools.file_tools")
        legal_tools = importlib.import_module("app.tools.legal_tools")
        search_tools = importlib.import_module("app.tools.search_tools")
        template_tools = importlib.import_module("app.tools.template_tools")
        research_tools = importlib.import_module("app.tools.research_tools")
        return (
            file_tools.FILE_TOOLS + 
            legal_tools.LEGAL_TOOLS + 
            search_tools.SEARCH_TOOLS + 
            template_tools.TEMPLATE_TOOLS + 
            research_tools.RESEARCH_TOOLS
        )
    
    raise AttributeError(f"module 'app.tools' has no attribute '{name}'")


__all__ = [
    "read_file", "write_file", "list_directory", "glob", "grep", "create_docx_file", "FILE_TOOLS",
    "read_document", "fill_template", "LEGAL_TOOLS",
    "search_indexed", "SEARCH_TOOLS",
    "list_templates", "read_template", "create_template", "TEMPLATE_TOOLS",
    "get_legal_references", "search_documents", "RESEARCH_TOOLS",
    "ALL_TOOLS",
]
