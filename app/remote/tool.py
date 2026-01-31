from typing import Any, Dict, Type, Optional
from pydantic import BaseModel
from langchain.tools import BaseTool
from app.remote.manager import remote_tool_manager

class RemoteBaseTool(BaseTool):
    """
    Standard LangChain tool that proxies execution to the Client UI.
    Inherit from this and define 'args_schema' to create a remote tool.
    """
    thread_id: str = "default"

    async def _arun(self, **kwargs: Any) -> Any:
        """Execute the tool by calling the client via WebSocket."""
        return await remote_tool_manager.call_remote_tool(
            thread_id=self.thread_id,
            tool_name=self.name,
            args=kwargs
        )

    def _run(self, **kwargs: Any) -> Any:
        """Sync run is not supported for remote tools."""
        raise NotImplementedError("Remote tools must be called asynchronously.")

def create_remote_tool(name: str, description: str, args_schema: Type[BaseModel], thread_id: str = "default") -> RemoteBaseTool:
    """Helper to create a remote tool instance."""
    return RemoteBaseTool(
        name=name,
        description=description,
        args_schema=args_schema,
        thread_id=thread_id
    )
