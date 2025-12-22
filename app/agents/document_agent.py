"""
Document Agent - Specialized for file system operations and document analysis.

Refactored to use "Thinking in LangGraph" Structured State.
Tracks read files in state variable to avoid infinite loops.

Enhanced with streaming "thinking" events so users can see agent reasoning.
"""
from typing import Annotated, Literal, TypedDict, List, Dict
import operator
import json
from pathlib import Path

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command

from app.agents.base_agent import BaseAgent, AgentConfig
from app.agents.prompts.document_prompt import (
    DOCUMENT_SYSTEM_PROMPT,
    ANALYZE_PROMPT
)
from app.tools import read_file, list_directory, glob, grep, read_document, search_indexed
from app.tools.context_tools import add_to_context

# --- Structured State ---
class DocumentState(TypedDict):
    """
    Domain state for Document Agent.
    Tracks what has been found and what has been fully read.
    """
    # Input/Context
    messages: Annotated[list[BaseMessage], operator.add]
    task_description: str
    
    # Working Memory
    search_intent: str             # What are we looking for?
    found_paths: list[str]         # Paths found by list/glob/search
    read_cache: dict[str, str]     # {virtual_path: content} - Prevents re-reading
    
    # Output
    final_response: str


class DocumentAgent(BaseAgent):
    """
    Document Agent - Explicit Workflow Implementation.
    
    Flow:
    1. Analyze -> Determine search strategy.
    2. Search -> Use tools (glob/grep/semantic) to find paths.
    3. Read -> Read content of found paths (skipping cache).
    4. Synthesize -> Answer based on read content.
    
    Enhanced with streaming thinking events - the analyze node now streams
    its reasoning as it determines the search strategy.
    """
    
    def __init__(self):
        config = AgentConfig(
            name="document_agent",
            # We keep these tools available for the nodes to call, 
            # or for the LLM to call if we used tool-calling nodes (but we'll use explicit nodes here)
            tools=[read_file, list_directory, glob, grep, search_indexed, read_document, add_to_context],
            system_prompt=DOCUMENT_SYSTEM_PROMPT,
        )
        super().__init__(config)
        self._graph = self._build_graph()

    def _build_graph(self):
        
        # Note: We use the LLM's astream for thinking events which are captured
        # by the parent graph's astream_events. The node names ("analyze", "search", etc.)
        # are used by the orchestrator to emit step events.
        
        async def analyze_node(state: DocumentState) -> Command[Literal["search"]]:
            """
            Step 1: Analyze request and determine search strategy.
            
            The LLM call here will stream "thinking" tokens to the UI.
            """
            messages = state.get("messages", [])
            task_desc = state.get("task_description")
            if not task_desc and messages:
                task_desc = messages[-1].content
            
            print(f"[{self.name}] 🔍 Analyzing request: {task_desc[:50]}...")
            
            # Use ainvoke - the streaming happens through astream_events in orchestrator
            response = await self.llm.ainvoke(
                [HumanMessage(content=ANALYZE_PROMPT.format(task_description=task_desc))]
            )
            
            try:
                content = response.content.replace("```json", "").replace("```", "").strip()
                data = json.loads(content)
                intent = data.get("intent", task_desc)
                strategy = data.get("strategy", "hybrid")
                query = data.get("query", task_desc)
            except:
                intent = task_desc
                strategy = "hybrid"
                query = task_desc
                
            return Command(
                update={
                    "task_description": task_desc,
                    "search_intent": json.dumps({"strategy": strategy, "query": query})
                },
                goto="search"
            )

        async def search_node(state: DocumentState) -> Command[Literal["read"]]:
            """
            Step 2: Execute search based on strategy.
            
            Tool calls here will be captured as tool-call/tool-result events.
            """
            search_data = json.loads(state.get("search_intent", "{}"))
            strategy = search_data.get("strategy", "hybrid")
            query = search_data.get("query", "")
            
            found = []
            print(f"[{self.name}] 🕵️ Strategy: {strategy}, Query: {query}")
            
            # Handle list queries (if LLM returns array)
            if isinstance(query, list):
                if not query:
                    query = ""
                else:
                    # For glob, we can only really search one pattern or loop. 
                    # Let's support looping for filename strategy, otherwise join.
                    if strategy == "filename":
                        # Iterative Glob handled inside filename block
                        pass 
                    else:
                        query = " ".join(str(q) for q in query)

            try:
                if strategy == "filename":
                    # glob
                    patterns = query if isinstance(query, list) else [query]
                    for pattern in patterns:
                        res = await glob.ainvoke({"pattern": str(pattern)})
                        found.extend([f for f in res.split("\n") if f.strip() and not f.startswith("No files") and not f.startswith("Error")])
                    
                elif strategy == "keyword":
                    # grep
                    res = await grep.ainvoke({"search_text": str(query)})
                    # Grep returns file:line:content. We just want unique files.
                    paths = set()
                    for line in res.split("\n"):
                        # Filter out error messages or "No matches"
                        if "No matches found" in line or line.startswith("Error"):
                            continue
                        if ":" in line:
                            path = line.split(":")[0].strip()
                            if path:
                                paths.add(path)
                    found = list(paths)
                    
                elif strategy == "list":
                     # list dir
                    res = await list_directory.ainvoke({"dir_path": "/project/"})
                    found = [line.split(" ")[1] for line in res.split("\n") if "📄" in line]
                    
                else: 
                    # hybrid (default) - combines semantic + keyword for best results
                    search_results = await search_indexed.ainvoke({
                        "query": str(query), 
                        "search_type": "hybrid",
                        "limit": 5  # Top 5 results only
                    })
                    
                    # Extract paths from the formatted search results
                    if isinstance(search_results, str):
                        for line in search_results.split("\n"):
                            if line.strip().startswith("📄"):
                                # Extract path after the emoji
                                path = line.strip().replace("📄", "", 1).strip()
                                found.append(path) 

            except Exception as e:
                print(f"[{self.name}] ❌ Search error: {e}")

            # Fallback/Normalization
            # If search_indexed was used, we might not have 'paths'.
            # If we used glob/grep we have paths.
            
            return Command(
                update={"found_paths": found},
                goto="read"
            )

        async def read_node(state: DocumentState, config: RunnableConfig) -> Command[Literal["synthesize"]]:
            """
            Step 3: Read document contents.
            
            Tool calls to read_document will stream as tool-call/tool-result events.
            """
            paths = state.get("found_paths", [])
            cache_read = state.get("read_cache", {}).copy()
            
            # Extract thread_id/conversation_id
            thread_id = config.get("configurable", {}).get("thread_id", "default")
            
            new_reads = 0
            for path in paths:
                path = path.strip()
                if path and path not in cache_read:
                    try:
                        print(f"[{self.name}] 📖 Reading: {path} (CID: {thread_id})")
                        
                        # ALways use read_document which now supports caching and all file types
                        content = await read_document.ainvoke({
                            "file_path": path,
                            "conversation_id": thread_id
                        })
                            
                        cache_read[path] = content
                        new_reads += 1
                    except Exception as e:
                        cache_read[path] = f"Error reading: {e}"
            
            if new_reads == 0 and not paths:
                 print(f"[{self.name}] ⚠️ No new files to read.")
            
            return Command(
                update={"read_cache": cache_read},
                goto="synthesize"
            )

        async def synthesize_node(state: DocumentState) -> Command[Literal["__end__"]]:
            """
            Step 4: Pass raw file content via add_to_context.
            
            Instead of synthesizing (which loses nuance for legal work),
            we pass the raw content to the conversation context so that
            ResearchAgent can analyze the full text with all legal details.
            """
            cache = state.get("read_cache", {})
            task = state.get("task_description")
            
            if not cache:
                print(f"[{self.name}] ⚠️ No files to add to context.")
                return Command(
                    update={
                        "final_response": "Не бяха намерени релевантни файлове.",
                        "messages": [AIMessage(content="Не бяха намерени релевантни файлове.", name=self.name)]
                    },
                    goto=END
                )
            
            # Get list of found file paths
            file_paths = list(cache.keys())
            
            # Build raw content for message (so ResearchAgent sees it in sub-state)
            raw_content = ""
            for path, content in cache.items():
                raw_content += f"\n=== {path} ===\n{content}\n"
            
            print(f"[{self.name}] 📎 Adding {len(file_paths)} files to context...")
            
            # Directly invoke add_to_context tool (no LLM decision needed)
            try:
                context_result = add_to_context.invoke({
                    "files": file_paths,
                    "summary": f"Намерени документи по заявка: {task[:100]}"
                })
                print(f"[{self.name}] ✅ Context updated: {context_result}")
            except Exception as e:
                print(f"[{self.name}] ❌ Failed to add to context: {e}")
            
            # Return the raw content as the response (ResearchAgent will receive this)
            summary_msg = f"Намерени {len(file_paths)} документа:\n"
            summary_msg += "\n".join(f"- {p}" for p in file_paths)
            summary_msg += f"\n\n---\n{raw_content}"
            
            print(f"[{self.name}] 📤 Returning {len(summary_msg)} chars to parent agent")
            
            return Command(
                update={
                    "final_response": summary_msg,
                    "messages": [AIMessage(content=summary_msg, name=self.name)]
                },
                goto=END
            )

        graph = StateGraph(DocumentState)
        graph.add_node("analyze", analyze_node)
        graph.add_node("search", search_node)
        graph.add_node("read", read_node)
        graph.add_node("synthesize", synthesize_node)
        graph.add_edge(START, "analyze")
        
        return graph.compile(checkpointer=self.checkpointer)
