"""
Base Agent - Foundation class for all LangGraph agents.

Encapsulates common logic for agent types including:
- LangGraph workflow building
- Gemini model integration  
- Tool execution with middleware
- Streaming responses
"""
from typing import AsyncIterator, Any, Sequence
from dataclasses import dataclass, field
import os

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langgraph.graph import StateGraph, MessagesState
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

from app.config import get_settings
from app.tracing import get_run_callbacks, get_current_run_id
from app.services.conversation_logger import print_and_log


@dataclass
class AgentConfig:
    """Configuration for an agent."""
    name: str
    tools: list = field(default_factory=list)
    system_prompt: str = "You are a helpful AI assistant."
    model_name: str = "gemini-2.5-flash"
    max_messages: int = 50
    max_iterations: int = 15


class BaseAgent:
    """
    Base Agent class that all specialized agents inherit from.
    
    Provides:
    - LangGraph workflow building
    - Gemini model integration
    - Tool execution
    - Streaming responses
    - Iteration guards
    """
    
    def __init__(self, config: AgentConfig):
        self.config = config
        self.name = config.name
        self.tools = config.tools
        self.system_prompt = config.system_prompt
        self.model_name = config.model_name
        self.max_messages = config.max_messages
        self.max_iterations = config.max_iterations
        
        self.checkpointer = MemorySaver()
        self._graph = None
        self._llm = None
    
    @property
    def llm(self) -> ChatGoogleGenerativeAI:
        """Lazy-initialize the LLM."""
        if self._llm is None:
            settings = get_settings()
            if not settings.google_api_key:
                raise ValueError("GOOGLE_API_KEY not configured")
            
            self._llm = ChatGoogleGenerativeAI(
                model=self.model_name,
                google_api_key=settings.google_api_key,
            )
        return self._llm
    
    @property
    def graph(self):
        """Lazy-initialize the LangGraph workflow."""
        if self._graph is None:
            self._graph = self._build_graph()
        return self._graph
    
    def _build_graph(self):
        """Build the LangGraph workflow."""
        tool_node = ToolNode(self.tools) if self.tools else None
        
        # Bind tools to LLM
        llm_with_tools = self.llm
        if self.tools:
            llm_with_tools = self.llm.bind_tools(self.tools)
        
        async def agent_node(state: MessagesState) -> dict:
            """Main agent node that calls the LLM."""
            # Track iterations for loop guard
            iterations = state.get("iterations", 0) + 1
            
            if iterations > self.max_iterations:
                print_and_log(f"[{self.name}] ⚠️ Max iterations ({self.max_iterations}) reached, forcing exit")
                return {
                    "messages": [AIMessage(content=f"[Agent forced exit after {self.max_iterations} iterations. Please simplify the task.]")],
                    "iterations": iterations,
                }
            
            # Trim messages to manage context window
            messages = self._trim_messages(state["messages"])
            
            # Add system prompt as first message if not present
            if messages and not any(hasattr(m, 'type') and m.type == 'system' for m in messages):
                # Prepend system context to first human message
                pass  # Gemini handles system_instruction separately
            
            try:
                # Invoke LLM
                response = await llm_with_tools.ainvoke(
                    messages,
                    config={"configurable": {"system_instruction": self.system_prompt}}
                )
                
                print_and_log(f"[{self.name}] Response: {str(response.content)[:100]}...")
                
                if hasattr(response, 'tool_calls') and response.tool_calls:
                    print_and_log(f"[{self.name}] Tool calls: {[tc['name'] for tc in response.tool_calls]}")
                
                return {"messages": [response], "iterations": iterations}
                
            except Exception as e:
                print_and_log(f"[{self.name}] ❌ Error: {e}")
                return {
                    "messages": [AIMessage(content=f"Error in {self.name}: {str(e)}")],
                    "iterations": iterations,
                }
        
        def should_continue(state: MessagesState) -> str:
            """Determine if we should continue to tools or end."""
            last_message = state["messages"][-1]
            
            # Check for tool calls
            if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                return "tools"
            
            return "__end__"
        
        # Build the graph
        graph = StateGraph(MessagesState)
        graph.add_node("agent", agent_node)
        
        if tool_node:
            graph.add_node("tools", tool_node)
            graph.add_conditional_edges("agent", should_continue)
            graph.add_edge("tools", "agent")
        else:
            graph.add_edge("agent", "__end__")
        
        graph.set_entry_point("agent")
        
        return graph.compile(checkpointer=self.checkpointer)
    
    def _trim_messages(self, messages: Sequence[BaseMessage]) -> list[BaseMessage]:
        """Trim messages to manage context window."""
        if len(messages) <= self.max_messages:
            return list(messages)
        
        # Keep last N messages, ensuring we start with a human message if possible
        trimmed = list(messages[-self.max_messages:])
        
        # Find first human message
        for i, msg in enumerate(trimmed):
            if isinstance(msg, HumanMessage):
                return trimmed[i:]
        
        return trimmed
    
    async def stream(
        self,
        message: str,
        thread_id: str = "default",
        context_files: list[str] | None = None
    ) -> AsyncIterator[dict[str, Any]]:
        """
        Process a message and stream the response.
        
        Yields chunks with types:
        - content: Text content from the agent
        - tool-call: Tool invocation
        - tool-result: Result from tool execution
        - done: Stream complete
        """
        # Inject context files into message if present
        final_message = message
        if context_files:
            final_message = f"[Active Context Files]\n{chr(10).join(context_files)}\n\n[User Message]\n{message}"
        
        # Build config with LangSmith tracing callbacks
        callbacks, trace_metadata = get_run_callbacks(
            thread_id=thread_id,
            agent_name=self.name,
            extra_tags=["worker-agent"],
        )
        
        config = {
            "configurable": {"thread_id": thread_id},
            "callbacks": callbacks,
            "metadata": trace_metadata,
            "tags": [self.name, "lawyers-dashboard"],
        }
        
        try:
            async for event in self.graph.astream_events(
                {"messages": [HumanMessage(content=final_message)]},
                config=config,
                version="v2"
            ):
                chunk = self._format_event(event)
                if chunk:
                    yield chunk
            
            yield {
                "type": "done", 
                "agent": self.name,
                "run_id": get_current_run_id()
            }
            
        except Exception as e:
            yield {"type": "error", "error": str(e), "agent": self.name}
    
    def _format_event(self, event: dict) -> dict | None:
        """Convert LangGraph events to frontend-friendly format."""
        kind = event.get("event")
        
        if kind == "on_chat_model_stream":
            # Streaming token
            content = event.get("data", {}).get("chunk", {})
            if hasattr(content, "content") and content.content:
                text_content = content.content
                if isinstance(text_content, list):
                    # Handle multimodal/list content
                    text_content = "".join([
                        c if isinstance(c, str) else 
                        (c.get("text", "") if isinstance(c, dict) and c.get("type") == "text" else str(c))
                        for c in text_content
                    ])
                
                return {
                    "type": "content",
                    "text": text_content,
                    "agent": self.name,
                }
        
        elif kind == "on_tool_start":
            # Tool invocation starting
            tool_name = event.get("name", "unknown")
            tool_input = event.get("data", {}).get("input", {})
            return {
                "type": "tool-call",
                "toolCalls": [{"name": tool_name, "args": tool_input}],
                "agent": self.name,
            }
        
        elif kind == "on_tool_end":
            # Tool completed
            tool_name = event.get("name", "unknown")
            output = event.get("data", {}).get("output", "")
            # Truncate large outputs
            if isinstance(output, str) and len(output) > 500:
                output = output[:500] + "..."
            return {
                "type": "tool-result",
                "toolName": tool_name,
                "result": output,
                "agent": self.name,
            }
        
        return None
    
    async def invoke(self, message: str, thread_id: str = "default") -> str:
        """
        Process a message and return the final response (non-streaming).
        """
        final_response = ""
        async for chunk in self.stream(message, thread_id):
            if chunk.get("type") == "content":
                final_response += chunk.get("text", "")
        return final_response

    async def invoke_with_messages(self, messages: list[BaseMessage], thread_id: str = "default") -> str:
        """
        Process a list of messages (history) and return the final response text.
        """
        final_response = ""
        config = {"configurable": {"thread_id": thread_id}}
        
        try:
            async for event in self.graph.astream_events(
                {"messages": messages},
                config=config,
                version="v2"
            ):
                chunk = self._format_event(event)
                if chunk and chunk.get("type") == "content":
                    final_response += chunk.get("text", "")
            
            return final_response
            
        except Exception as e:
            print_and_log(f"[{self.name}] ❌ Error in invoke_with_messages: {e}")
            return f"Error executing {self.name}: {str(e)}"
