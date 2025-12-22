"""
Template Agent - Manages document templates.

Capabilities:
- List available templates
- Read template contents
- Template management
"""
from app.agents.base_agent import BaseAgent, AgentConfig
from app.agents.prompts.template_prompt import TEMPLATE_SYSTEM_PROMPT
# Direct imports to avoid circular import
from app.tools.file_tools import read_file, list_directory, glob


class TemplateAgent(BaseAgent):
    """
    Template Agent - Manages document templates.
    
    Tools:
    - list_directory: List folder contents
    - glob: Search for template files
    - read_file: Read template contents
    """
    
    def __init__(self):
        config = AgentConfig(
            name="template_agent",
            tools=[
                list_directory,
                glob,
                read_file,
            ],
            system_prompt=TEMPLATE_SYSTEM_PROMPT,
        )
        super().__init__(config)
