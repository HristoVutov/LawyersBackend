# Tools package - LangChain tools for agents
from app.tools.file_tools import (
    read_file,
    write_file,
    list_directory,
    glob,
    grep,
    create_docx_file,
    FILE_TOOLS,
)

from app.tools.legal_tools import (
    read_document,
    fill_template,
    LEGAL_TOOLS,
)

from app.tools.search_tools import (
    search_indexed,
    SEARCH_TOOLS,
)

from app.tools.template_tools import (
    list_templates,
    read_template,
    create_template,
    TEMPLATE_TOOLS,
)

from app.tools.research_tools import (
    get_legal_references,
    search_documents,
    RESEARCH_TOOLS,
)

# Combined list of all tools
ALL_TOOLS = FILE_TOOLS + LEGAL_TOOLS + SEARCH_TOOLS + TEMPLATE_TOOLS + RESEARCH_TOOLS

__all__ = [
    # File tools
    "read_file",
    "write_file",
    "list_directory",
    "glob",
    "grep",
    "create_docx_file",
    "FILE_TOOLS",
    # Legal tools
    "read_document",
    "fill_template",
    "LEGAL_TOOLS",
    # Search tools
    "search_indexed",
    "SEARCH_TOOLS",
    # Template tools
    "list_templates",
    "read_template",
    "create_template",
    "TEMPLATE_TOOLS",
    # Research tools
    "get_legal_references",
    "search_documents",
    "RESEARCH_TOOLS",
    # All tools
    "ALL_TOOLS",
]

