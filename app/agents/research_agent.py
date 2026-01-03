"""
Research Agent - Expert legal researcher with deep analysis capabilities.

Refactored to use "Thinking in LangGraph" Structured State and Command Pattern.
Uses tool-calling gather_info node to let the LLM decide which sources to query.

Enhanced with streaming "thinking" events so users can see agent reasoning.
"""
from typing import Annotated, Literal, TypedDict
import operator
import json

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command
from langgraph.prebuilt import ToolNode

from app.agents.base_agent import BaseAgent, AgentConfig
from app.tools.research_tools import get_legal_references, search_documents
from app.tools.analysis_tools import compare_compliance
from app.middleware import TodoListMiddleware
from app.services.conversation_logger import print_and_log
from app.agents.document_agent import DocumentAgent
from langchain_core.tools import tool

# --- Structured State Definition ---
class ResearchState(TypedDict):
    """
    Domain-specific state for the Research Agent.
    Separates raw data from conversation history.
    """
    # Input/Context
    messages: Annotated[list[BaseMessage], operator.add]  # Compatibility with Orchestrator
    task_description: str
    
    # Working memory
    search_queries: list[str]
    gather_messages: Annotated[list[BaseMessage], operator.add]  # Tool-calling conversation
    legal_references: str       # Legal framework from LLM knowledge
    found_documents: str        # Local document content
    
    # Loop guard
    gather_iterations: int
    
    # Outputs
    draft_answer: str

from app.agents.prompts.research_prompt import (
    RESEARCH_SYSTEM_PROMPT, 
    ANALYZE_PROMPT,
    GATHER_INFO_PROMPT,
    SYNTHESIZE_PROMPT
)

# Research tools for gather_info node
# We will add consult_document_agent dynamically in __init__
RESEARCH_GATHER_TOOLS = [get_legal_references, search_documents, compare_compliance]


class ResearchAgent(BaseAgent):
    """
    Research Agent - Explicit Workflow Implementation.
    
    Flow:
    1. Analyze (LLM) -> Identify search queries
    2. Gather Info (Tool-calling loop) -> LLM decides which tools to use
    3. Synthesize (LLM) -> Generate answer
    
    Enhanced with streaming thinking events - the analyze and synthesize
    nodes stream their reasoning to the UI.
    """
    
    def __init__(self):
        self.todo_middleware = TodoListMiddleware()
        
        config = AgentConfig(
            name="research_agent",
            tools=RESEARCH_GATHER_TOOLS,
            system_prompt=RESEARCH_SYSTEM_PROMPT 
        )
        super().__init__(config)
        
        # Initialize Document Agent as a sub-agent
        self.document_agent = DocumentAgent()
        
        # Create a tool to consult the document agent
        @tool
        async def consult_document_agent(query: str) -> str:
            """
            Consult the Document Agent to find, read, or analyze local files.
            Use this for ANY request involving local project files, searching, or reading documents.
            """
            print_and_log(f"[{self.name}] 📞 Calling Document Agent with: {query}")
            # Identify the current thread from some context if possible, or pass "default"
            # In a real scenario, we'd want to propagate the thread_id.
            # BaseAgent.invoke doesn't easily accept thread_id from here unless we bind it.
            # For now, we'll rely on the fact that DocumentAgent uses 'default' or we can update invoke.
            
            # Since we are inside an async tool, we can await
            result = await self.document_agent.invoke(query)
            return result

        # Update tools list
        self.tools = [*RESEARCH_GATHER_TOOLS, consult_document_agent]
        self.llm_with_tools = self.llm.bind_tools(self.tools)
        
        # Override the graph with our custom one
        self._graph = self._build_graph()

    def _build_graph(self):
        """Builds the explicit Structured State graph with tool-calling gather_info node."""
        
        # LLM with tools bound for gather_info
        # Use self.tools which now includes consult_document_agent
        llm_with_tools = self.llm.bind_tools(self.tools)
        
        # Tool node for executing research tools
        tool_node = ToolNode(self.tools)
        
        # --- Node Definitions ---
        
        async def analyze_node(state: ResearchState, config) -> Command[Literal["gather_info"]]:
            """
            Step 1: Analyze request and generate search queries.
            
            The LLM call here will stream "thinking" tokens to the UI,
            showing the agent's reasoning about what to search for.
            """
            messages = state.get("messages", [])
            task_desc = state.get("task_description")
            if not task_desc and messages:
                task_desc = messages[-1].content
            
            print_and_log(f"[{self.name}] 🔍 Analyzing request: {task_desc[:50]}...")
            
            # Dynamic Model Selection
            model_name = config.get("configurable", {}).get("model_name")
            if model_name:
                print_and_log(f"[{self.name}] 🔄 Switching to requested model: {model_name}")
                llm = self._create_llm(model_name)
            else:
                llm = self.llm

            # The LLM response will be streamed via astream_events in orchestrator
            response = await llm.ainvoke(
                [HumanMessage(content=ANALYZE_PROMPT.format(task_description=task_desc))]
            )
            
            queries = []
            try:
                content = response.content.replace("```json", "").replace("```", "").strip()
                data = json.loads(content)
                queries = data.get("queries", [])
            except Exception as e:
                print_and_log(f"[{self.name}] ⚠️ JSON parse failed, falling back to raw")
                queries = [task_desc]
            
            print_and_log(f"[{self.name}] 📝 Generated {len(queries)} queries")
            
            # Initialize gather_info with the prompt
            gather_prompt = GATHER_INFO_PROMPT.format(
                task_description=task_desc,
                search_queries=", ".join(queries) if queries else task_desc
            )
            
            return Command(
                update={
                    "task_description": task_desc,
                    "search_queries": queries,
                    "gather_messages": [HumanMessage(content=gather_prompt)],
                    "gather_iterations": 0
                },
                goto="gather_info"
            )

        async def gather_info_node(state: ResearchState, config) -> Command[Literal["tools", "synthesize"]]:
            """
            Step 2: LLM decides which tools to use for gathering information.
            
            This is a tool-calling node - the LLM can call:
            - get_legal_references: For laws, articles, case law from LLM knowledge
            - search_documents: For local document content (legacy tool, prefer consult_document_agent)
            - consult_document_agent: DELEGATE identifying/reading files to the specialist agent
            """
            gather_msgs = state.get("gather_messages", [])
            iterations = state.get("gather_iterations", 0) + 1
            
            # Loop guard
            if iterations > 5:
                print_and_log(f"[{self.name}] ⚠️ Max gather iterations reached, moving to synthesize")
                return Command(
                    update={"gather_iterations": iterations},
                    goto="synthesize"
                )
            
            print_and_log(f"[{self.name}] 🔄 Gather iteration {iterations}...")
            
            # Dynamic Model Selection
            model_name = config.get("configurable", {}).get("model_name")
            if model_name:
                # Must bind tools again!
                llm = self._create_llm(model_name)
                # Use self.tools which includes consult_document_agent
                llm_node = llm.bind_tools(self.tools)
            else:
                llm_node = llm_with_tools
            
            # Call LLM with tools
            response = await llm_node.ainvoke(gather_msgs)
            
            # Check if LLM wants to call tools
            if hasattr(response, "tool_calls") and response.tool_calls:
                tool_names = [tc["name"] for tc in response.tool_calls]
                print_and_log(f"[{self.name}] 🔧 Tool calls: {tool_names}")
                
                return Command(
                    update={
                        "gather_messages": [response],
                        "gather_iterations": iterations
                    },
                    goto="tools"
                )
            else:
                # No more tool calls - extract any gathered info and proceed
                print_and_log(f"[{self.name}] ✅ Gathering complete, moving to synthesize")
                return Command(
                    update={
                        "gather_messages": [response],
                        "gather_iterations": iterations
                    },
                    goto="synthesize"
                )

        async def tools_node(state: ResearchState) -> Command[Literal["gather_info"]]:
            """
            Execute tools and store results.
            """
            gather_msgs = state.get("gather_messages", [])
            legal_refs = state.get("legal_references", "")
            found_docs = state.get("found_documents", "")
            
            # Get the last AI message with tool calls
            last_msg = gather_msgs[-1] if gather_msgs else None
            if not last_msg or not hasattr(last_msg, "tool_calls"):
                return Command(update={}, goto="gather_info")
            
            # Execute each tool call
            tool_results = []
            for tool_call in last_msg.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                
                print_and_log(f"[{self.name}] ⚙️ Executing: {tool_name}")
                
                try:
                    if tool_name == "get_legal_references":
                        result = await get_legal_references.ainvoke(tool_args)
                        legal_refs += f"\n{result}"
                        result = await search_documents.ainvoke(tool_args)
                        found_docs += f"\n{result}"
                    elif tool_name == "compare_compliance":
                        result = await compare_compliance.ainvoke(tool_args)
                        # Append comparison result to legal_references or found_documents? 
                        # Comparison is a synthesis of both, so maybe legal_references fits better as "analysis"
                        # or just append to found_docs so it's part of the context for synthesis.
                        found_docs += f"\n=== COMPARISON REPORT ===\n{result}"
                    elif tool_name == "consult_document_agent":
                        # This tool is defined dynamically in __init__
                        # We need to find it in self.tools to invoke it, 
                        # OR since we bound it to the LLM, LangGraph's ToolNode can handle it 
                        # IF we used the standard ToolNode. 
                        # But here we are manually executing in 'tools_node'.
                        
                        # We need to execute the wrapper function we created.
                        # It's a local function closure in __init__, so we can't easily access it here 
                        # UNLESS we stored it or use the tool instance from self.tools.
                        
                        target_tool = next((t for t in self.tools if t.name == "consult_document_agent"), None)
                        if target_tool:
                             # Tool invocation
                             result = await target_tool.ainvoke(tool_args)
                        else:
                             result = "Error: consult_document_agent tool not found."
                             
                        found_docs += f"\n{result}"

                    else:
                        result = f"Unknown tool: {tool_name}"
                    
                    tool_results.append(ToolMessage(
                        content=str(result),
                        tool_call_id=tool_call["id"],
                        name=tool_name
                    ))
                except Exception as e:
                    print_and_log(f"[{self.name}] ❌ Tool error: {e}")
                    tool_results.append(ToolMessage(
                        content=f"Error: {str(e)}",
                        tool_call_id=tool_call["id"],
                        name=tool_name
                    ))
            
            return Command(
                update={
                    "gather_messages": tool_results,
                    "legal_references": legal_refs,
                    "found_documents": found_docs
                },
                goto="gather_info"
            )

        async def synthesize_node(state: ResearchState, config) -> Command[Literal["__end__"]]:
            """
            Step 3: Synthesize findings into an answer.
            
            Combines legal references (from LLM knowledge) with
            local document content for comprehensive analysis.
            """
            legal_refs = state.get("legal_references", "")
            found_docs = state.get("found_documents", "")
            task = state.get("task_description", "")
            
            # Build context from gathered info
            context_str = ""
            
            if legal_refs:
                context_str += "\n=== ПРАВНА РАМКА (от законодателството) ===\n"
                context_str += legal_refs
            
            if found_docs:
                context_str += "\n=== ЛОКАЛНИ ДОКУМЕНТИ ===\n"
                context_str += found_docs
            
            if not context_str.strip():
                context_str = "Не беше намерена релевантна информация."
            
            print_and_log(f"[{self.name}] ✍️ Synthesizing answer...")
            
            # Dynamic Model Selection
            model_name = config.get("configurable", {}).get("model_name")
            if model_name:
                llm = self._create_llm(model_name)
            else:
                llm = self.llm

            # The LLM response will be streamed via astream_events in orchestrator
            response = await llm.ainvoke(
                [HumanMessage(content=SYNTHESIZE_PROMPT.format(task_description=task, context=context_str))]
            )
            
            final_answer = response.content
            
            return Command(
                update={
                    "draft_answer": final_answer,
                    "messages": [AIMessage(content=final_answer, name=self.name)]
                },
                goto=END
            )

        # --- Graph Construction ---
        graph = StateGraph(ResearchState)
        
        graph.add_node("analyze", analyze_node)
        graph.add_node("gather_info", gather_info_node)
        graph.add_node("tools", tools_node)
        graph.add_node("synthesize", synthesize_node)
        
        graph.add_edge(START, "analyze")
        # Conditional edges are handled by Command pattern
        
        return graph.compile(checkpointer=self.checkpointer)
