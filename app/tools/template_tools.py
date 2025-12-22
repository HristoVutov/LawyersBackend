"""
Template Tools - Template management for agents.

Provides tools for:
- Listing available templates
- Reading template contents
- Creating new templates as DOCX + HTML
"""
import os
from pathlib import Path
from langchain_core.tools import tool

from app.middleware.virtual_fs import get_vfs, resolve_path


def get_templates_path() -> Path:
    """Get the templates folder path."""
    return resolve_path("/templates/")


async def ensure_templates_folder() -> Path | None:
    """Ensure templates folder exists."""
    try:
        templates_path = get_templates_path()
        templates_path.mkdir(parents=True, exist_ok=True)
        return templates_path
    except Exception as e:
        print(f"Failed to access templates folder: {e}")
        return None


# --- Tools ---

@tool
async def list_templates() -> str:
    """
    List all available document templates from the LawyersAI/Templates folder.
    Templates can be .docx, .txt, or .md files.
    """
    try:
        templates_path = await ensure_templates_folder()
        if not templates_path:
            return "Error: Could not access templates folder."
        
        items = list(templates_path.iterdir())
        templates = [
            f.name for f in items 
            if f.is_file() and f.suffix.lower() in ['.docx', '.txt', '.md', '.html']
        ]
        
        if not templates:
            return f"No templates found in /templates/. Create one using create_template."
        
        result = "📁 Templates in /templates/:\n"
        result += "\n".join(f"  📄 {t}" for t in sorted(templates))
        return result
    except Exception as e:
        return f"Error listing templates: {str(e)}"


@tool
async def read_template(template_name: str) -> str:
    """
    Read the contents of a template file.
    
    Args:
        template_name: Name of the template file (e.g., 'contract.txt', 'invoice.md')
    """
    try:
        templates_path = await ensure_templates_folder()
        if not templates_path:
            return "Error: Could not access templates folder."
        
        template_path = templates_path / template_name
        
        if not template_path.exists():
            return f'Template "{template_name}" not found. Use list_templates to see available templates.'
        
        # Handle binary DOCX files
        if template_name.lower().endswith('.docx'):
            return f"📄 Template: {template_name}\n(Binary DOCX file at /templates/{template_name}. Use create_docx_file to generate documents based on it.)"
        
        # Read text files
        content = template_path.read_text(encoding="utf-8")
        return f"📄 Template: {template_name}\n\n{content}"
    except Exception as e:
        return f"Error reading template: {str(e)}"


@tool
async def create_template(template_name: str, content: str) -> str:
    """
    Create a new document template as both DOCX and HTML.
    Templates are stored in the LawyersAI/Templates folder for reuse across projects.
    
    Args:
        template_name: Name for the template (e.g., 'contract', 'invoice'). Will be saved as .docx
        content: The template content in HTML format
    """
    try:
        from docx import Document
        import re
        
        templates_path = await ensure_templates_folder()
        if not templates_path:
            return "Error: Could not access templates folder."
        
        # Clean up file name
        base_name = re.sub(r"\.(docx|html|txt|md)$", "", template_name, flags=re.IGNORECASE)
        docx_path = templates_path / f"{base_name}.docx"
        html_path = templates_path / f"{base_name}.docx.html"
        
        # Wrap content in HTML structure
        full_html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <style>
        body {{ font-family: Calibri, sans-serif; font-size: 11pt; }}
        h1 {{ font-size: 16pt; font-weight: bold; }}
        h2 {{ font-size: 14pt; font-weight: bold; }}
        h3 {{ font-size: 12pt; font-weight: bold; }}
    </style>
</head>
<body>{content}</body>
</html>"""
        
        # Save HTML file for reference/editing
        html_path.write_text(full_html, encoding="utf-8")
        
        # Create DOCX using python-docx
        doc = Document()
        
        # Simple HTML to DOCX conversion (basic implementation)
        text_content = re.sub(r"<[^>]+>", "\n", content)
        text_content = re.sub(r"\n+", "\n", text_content).strip()
        
        for line in text_content.split("\n"):
            if line.strip():
                doc.add_paragraph(line.strip())
        
        doc.save(str(docx_path))
        
        return f"""✅ Template created successfully!
📄 DOCX: /templates/{base_name}.docx
📄 HTML: /templates/{base_name}.docx.html"""
    except ImportError:
        return "Error: python-docx not installed. Run: pip install python-docx"
    except Exception as e:
        return f"Error creating template: {str(e)}"


# Export all tools
TEMPLATE_TOOLS = [
    list_templates,
    read_template,
    create_template,
]
