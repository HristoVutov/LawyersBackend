from pydantic import BaseModel, Field
from app.remote.tool import create_remote_tool

# --- Schemas ---

class ReadFileArgs(BaseModel):
    file_path: str = Field(description="Virtual path to the file (e.g., /project/document.pdf)")

class WriteFileArgs(BaseModel):
    file_path: str = Field(description="Virtual path to write to (e.g., /output/notes.txt)")
    content: str = Field(description="The content to write to the file")

class OCRArgs(BaseModel):
    file_path: str = Field(description="Virtual path to the file to process (PDF or Image)")

class SearchArgs(BaseModel):
    query: str = Field(description="Search query - keywords or description of what you're looking for")
    search_type: str = Field(default="hybrid", description="Search type: 'keyword', 'vector', or 'hybrid'")
    limit: int | None = Field(default=None, description="Max results to return")

class ReadDocArgs(BaseModel):
    file_path: str = Field(description="Path to the file (virtual or absolute)")

class ProjectOverviewArgs(BaseModel):
    include_summaries: bool = Field(default=True, description="Whether to include pre-computed file summaries")

# --- Tool Instances ---

def get_remote_file_tools(thread_id: str):
    """Returns a list of tools that execute on the client-side."""
    return [
        create_remote_tool(
            name="read_file",
            description="Read the contents of a local file via the UI.",
            args_schema=ReadFileArgs,
            thread_id=thread_id
        ),
        create_remote_tool(
            name="write_file",
            description="Write content to a local file via the UI (Only /output/ or /templates/).",
            args_schema=WriteFileArgs,
            thread_id=thread_id
        ),
        create_remote_tool(
            name="execute_ocr",
            description="Process a local PDF or Image using PaddleOCR on the client computer.",
            args_schema=OCRArgs,
            thread_id=thread_id
        ),
        create_remote_tool(
            name="search_indexed",
            description="Search through indexed documents on the client PC using semantic and keyword search.",
            args_schema=SearchArgs,
            thread_id=thread_id
        ),
        create_remote_tool(
            name="read_document",
            description="Read text from local document files (.docx, .pdf, .txt, .md) on the client PC.",
            args_schema=ReadDocArgs,
            thread_id=thread_id
        ),
        create_remote_tool(
            name="get_project_overview",
            description="Get a summary of all files, their pre-computed summaries, and project metadata from the client UI.",
            args_schema=ProjectOverviewArgs,
            thread_id=thread_id
        )
    ]
