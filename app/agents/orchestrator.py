"""
Orchestrator Agent - Supervisor Graph Pattern.

Replaces tool-based delegation with a StateGraph where:
1. Supervisor Node: Decides which agent acts next.
2. Worker Nodes: Specialized agents that execute tasks.
3. Cyclical: Supervisor -> Worker -> Supervisor -> ... -> FINISH

Enhanced with streaming "thinking" events so users can see agent reasoning in real-time.
"""
from typing import Annotated, Literal, Any, AsyncIterator
from typing_extensions import TypedDict
import operator
import json
import asyncio
from pydantic import BaseModel, Field

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode

from app.agents.base_agent import BaseAgent, AgentConfig
from app.agents.prompts.orchestrator_prompt import ORCHESTRATOR_SYSTEM_PROMPT
from app.api.websocket import manager
from app.config import get_settings
from app.services.event_emitter import EventEmitter, EventType
from app.services.conversation_logger import print_and_log
from app.tracing import get_run_callbacks, get_current_run_id

# --- State Definition ---
class AgentState(TypedDict):
    """The shared state of the supervisor graph."""
    messages: Annotated[list[BaseMessage], operator.add]
    next: str
    task_list: list[dict]  # List of {id, description, status}
    current_task_id: str | None  # Currently executing task
    guard_trigger_count: int  # Track Autonomy Guard triggers to prevent loops

# --- Agent Registry Helper ---
# We use this to retrieve specialized agents dynamically
_agent_registry: dict[str, BaseAgent] = {}

def register_agent(name: str, agent: BaseAgent) -> None:
    _agent_registry[name] = agent

def get_agent(name: str) -> BaseAgent | None:
    return _agent_registry.get(name)

def list_agent_names() -> list[str]:
    return list(_agent_registry.keys())

# --- Tool Definitions ---

class TaskItem(BaseModel):
    id: str = Field(description="Unique task ID, e.g., 't1'")
    description: str = Field(description="Clear task description")
    status: Literal["pending", "in_progress", "done"] = "pending"

class PlanTask(BaseModel):
    """Create or update the task plan."""
    goal: str = Field(description="The overall goal.")
    tasks: list[TaskItem] = Field(description="List of tasks to execute.")

class DelegateTask(BaseModel):
    """Delegate a specific task to a worker agent."""
    agent_name: str = Field(description="Name of the agent to delegate to.")
    task: str = Field(description="Specific instructions for the agent.")
    task_id: str = Field(description="ID of the task being executed.")

# --- Orchestrator Implementation ---
class OrchestratorAgent(BaseAgent):
    """
    Supervisor orchestrator that manages a team of agents.
    Does NOT use the standard BaseAgent graph. Builds its own Supervisor Graph.
    
    Enhanced with streaming thinking events.
    """
    def __init__(self):
        # Config is minimal as we build a custom graph
        super().__init__(AgentConfig(name="orchestrator"))
        self._supervisor_graph = None
        
    @property
    def graph(self):
        if self._supervisor_graph is None:
            self._supervisor_graph = self._build_supervisor_graph()
        return self._supervisor_graph

    def _build_supervisor_graph(self):
        """Build the Supervisor StateGraph."""
        all_agents = list_agent_names()
        # Filter out document_agent from top-level orchestration
        members = [name for name in all_agents if name != "document_agent"]
        
        # Ensure we have members
        if not members:
            # Fallback if accessed before initialization
            members = ["research_agent"] 


        # Use the imported system prompt which contains specific routing instructions (in Bulgarian)
        # Note: The prompt defines static roles, so we don't need to inject 'members' dynamically 
        # unless we want to enforce strictness, but the prompt is robust enough.
        system_prompt = ORCHESTRATOR_SYSTEM_PROMPT

        options = members + ["FINISH"]
        
        # 1. The Supervisor Node
        def supervisor_node(state: AgentState) -> dict:
            # --- Autonomy Guard & Finish Check ---
            # If the last message was from a worker and said "FINISH", we might want to stop.
            # But in the Command pattern, the supervisor decides based on the plan.
            
            # Prepare tools
            tools = [PlanTask, DelegateTask]
            
            # Simple router chain with tools
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                MessagesPlaceholder(variable_name="messages"),
            ])
            
            # Bind tools to LLM
            llm_with_tools = self.llm.bind_tools(tools)
            chain = prompt | llm_with_tools
            
            try:
                response = chain.invoke(state)
            except Exception as e:
                print_and_log(f"[Orchestrator] ❌ LLM Error: {e}")
                return {"next": "FINISH"}

            # Analyze response
            tool_calls = getattr(response, "tool_calls", [])
            
            # --- Autonomy Guard: Check for pending tasks ---
            task_list = state.get("task_list", [])
            pending_tasks = [t for t in task_list if t["status"] == "pending"]
            
            # Default: If no tool call, assume we are chatting or done.
            # But the prompt enforces tool use. 
            if not tool_calls:
                # Fallback: check text for FINISH
                content = str(response.content)
                
                # AUTONOMY GUARD: Block FINISH if pending tasks exist
                if pending_tasks:
                    # check loop counter
                    current_triggers = state.get("guard_trigger_count", 0)
                    if current_triggers >= 3:
                         print_and_log(f"[Orchestrator] 🚨 Autonomy Guard Loop Detected ({current_triggers} retries). Forcing FINISH to prevent crash.")
                         return {
                             "next": "FINISH", 
                             "messages": [
                                 response,
                                 AIMessage(content="Error: Unable to proceed. Orchestrator stuck in delegation loop. Stopping.")
                             ]
                         }

                    next_task = pending_tasks[0]  # Get first pending task
                    print_and_log(f"[Orchestrator] ⚠️ Autonomy Guard: {len(pending_tasks)} pending tasks, forcing delegation of {next_task['id']} (Attempt {current_triggers + 1}/3)")
                    
                    # Build explicit delegation instruction
                    delegation_instruction = (
                        f"INCOMPLETE WORK - You have {len(pending_tasks)} pending task(s).\n\n"
                        f"NEXT TASK TO DELEGATE:\n"
                        f"- Task ID: {next_task['id']}\n"
                        f"- Description: {next_task['description']}\n\n"
                        f"Use DelegateTask to assign this task to an appropriate agent NOW. "
                        f"Do NOT call PlanTask or try to FINISH."
                    )
                    
                    return {
                        "next": "supervisor",
                        "messages": [
                            response,
                            HumanMessage(content=delegation_instruction, name="system")
                        ],
                        "guard_trigger_count": current_triggers + 1
                    }
                
                if "FINISH" in content:
                    return {"next": "FINISH", "messages": [response]}
                # Otherwise, loop back to supervisor to try again
                return {"next": "FINISH", "messages": [response]}  # Route to END via FINISH


            # Handle Tool Calls
            # We only process the FIRST valid Delegation tool call effectively in one turn for now,
            # (Sequential), or PlanTask.
            
            # Reset guard trigger count on valid tool use
            extra_state_updates = {"guard_trigger_count": 0}

            next_step = "supervisor" # Default loop back if just planning
            messages_to_add = [response]
            
            for tc in tool_calls:
                tool_name = tc["name"]
                args = tc["args"]
                
                if tool_name == "PlanTask":
                    # GUARD: Skip if we already have an active task list
                    existing_tasks = state.get("task_list", [])
                    if existing_tasks:
                        print_and_log(f"[Orchestrator] ⏭️ Skipping PlanTask - already have {len(existing_tasks)} tasks")
                        # Check if all tasks are done
                        all_done = all(t.get("status") == "done" for t in existing_tasks)
                        if all_done:
                            print_and_log(f"[Orchestrator] ✅ All tasks complete, finishing")
                            messages_to_add.append(
                                ToolMessage(
                                    content="All tasks completed. Finishing.",
                                    tool_call_id=tc["id"]
                                )
                            )
                            return {"next": "FINISH", "messages": messages_to_add}
                        else:
                            # Some tasks remain, let supervisor delegate next one
                            messages_to_add.append(
                                ToolMessage(
                                    content=f"Plan already exists with {len(existing_tasks)} tasks. Continue delegating pending tasks.",
                                    tool_call_id=tc["id"]
                                )
                            )
                            return {"next": "supervisor", "messages": messages_to_add}
                    
                    # First time planning - store tasks in state
                    print_and_log(f"[Orchestrator] 📋 Plan Update: {args.get('goal')}")
                    new_tasks = [
                        {"id": t.get("id", f"t{i}"), "description": t.get("description", ""), "status": "pending"}
                        for i, t in enumerate(args.get("tasks", []))
                    ]
                    print_and_log(f"[Orchestrator] 📝 Created {len(new_tasks)} tasks: {[t['id'] for t in new_tasks]}")
                    messages_to_add.append(
                        ToolMessage(
                            content=f"Plan created. Tasks: {len(new_tasks)}",
                            tool_call_id=tc["id"]
                        )
                    )
                    next_step = "supervisor"
                    return {"next": next_step, "messages": messages_to_add, "task_list": new_tasks, **extra_state_updates}
                    
                elif tool_name == "DelegateTask":
                    agent_name = args.get("agent_name")
                    task_desc = args.get("task")
                    task_id = args.get("task_id")
                    
                    if agent_name not in members:
                        print_and_log(f"[Orchestrator] ⚠️ Invalid agent: {agent_name}")
                        messages_to_add.append(
                            ToolMessage(
                                content=f"Error: Agent '{agent_name}' not found. Available: {members}",
                                tool_call_id=tc["id"]
                            )
                        )
                        next_step = "supervisor"
                    else:
                        print_and_log(f"[Orchestrator] 👉 Delegating to {agent_name}: {task_desc} (task: {task_id})")
                        # Mark task as in_progress in task_list
                        updated_tasks = []
                        for t in state.get("task_list", []):
                            if t["id"] == task_id:
                                updated_tasks.append({**t, "status": "in_progress"})
                            else:
                                updated_tasks.append(t)
                        
                        messages_to_add.append(
                            ToolMessage(
                                content=f"Delegated to {agent_name}.",
                                tool_call_id=tc["id"]
                            )
                        )
                        messages_to_add.append(
                            HumanMessage(content=task_desc, name="supervisor")
                        )
                        next_step = agent_name
                        return {
                            "next": next_step,
                            "messages": messages_to_add,
                            "task_list": updated_tasks,
                            "current_task_id": task_id,
                            **extra_state_updates
                        } 
            
            return {"next": next_step, "messages": messages_to_add, **extra_state_updates}

        # 2. Worker Node Builder
        def make_worker_node(agent_name: str):
            async def worker_node(state: AgentState, config) -> dict:
                agent = get_agent(agent_name)
                # Use ainvoke instead of invoke_with_messages to allow event streaming
                # The parent graph's astream_events will capture the child graph's events
                # if they share the same trace context (which they should in LangGraph).
                # Add recursion_limit to prevent infinite loops in sub-agents
                worker_config = {**config, "recursion_limit": 50}
                result_state = await agent.graph.ainvoke(state, worker_config)
                
                # Extract the final response from the agent
                # We assume the agent's last message is the result
                last_message = result_state["messages"][-1]
                
                content = last_message.content if hasattr(last_message, "content") else str(last_message)
                
                messages_to_return = [AIMessage(content=content, name=agent_name)]
                
                # Mark current task as done
                current_task_id = state.get("current_task_id")
                updated_tasks = []
                for t in state.get("task_list", []):
                    if t["id"] == current_task_id:
                        updated_tasks.append({**t, "status": "done"})
                        print_and_log(f"[Orchestrator] ✅ Task {current_task_id} completed by {agent_name}")
                    else:
                        updated_tasks.append(t)
                     
                return {
                    "messages": messages_to_return,
                    "task_list": updated_tasks,
                    "current_task_id": None  # Clear current task
                }
            return worker_node


        # 3. Build Graph
        workflow = StateGraph(AgentState)
        
        workflow.add_node("supervisor", supervisor_node)
        
        for member in members:
            workflow.add_node(member, make_worker_node(member))
            
        # 4. Edges
        # Start -> Supervisor
        workflow.add_edge(START, "supervisor")
        
        # Supervisor -> Conditional (Worker or End)
        conditional_map = {k: k for k in members}
        conditional_map["FINISH"] = END
        conditional_map["supervisor"] = "supervisor"
        conditional_map[END] = END  # Safety: handle if END constant is returned directly
        
        workflow.add_conditional_edges(
            "supervisor", 
            lambda x: x["next"], 
            conditional_map
        )
        
        # Worker -> Supervisor (Always report back)
        for member in members:
            workflow.add_edge(member, "supervisor")
            
        return workflow.compile(checkpointer=self.checkpointer)

    async def stream(
        self,
        message: str,
        thread_id: str = "default",
        context_files: list[str] | None = None
    ) -> AsyncIterator[dict[str, Any]]:
        """
        Streaming handler for the Supervisor Graph.
        
        Enhanced to emit thinking events as agents process requests.
        """
        # Import here to avoid circular import
        from app.services.conversation_logger import set_current_thread
        
        # Set thread context for logging - all print_and_log calls will use this
        set_current_thread(thread_id)
        
        # Prepare initial state
        final_message = message
        if context_files:
            final_message = f"[Active Context Files]\n{chr(10).join(context_files)}\n\n[User Message]\n{message}"
        
        inputs = {"messages": [HumanMessage(content=final_message)]}
        
        # Build config with LangSmith tracing callbacks
        callbacks, trace_metadata = get_run_callbacks(
            thread_id=thread_id,
            agent_name="orchestrator",
            extra_tags=["supervisor", "multi-agent"],
            extra_metadata={"has_context_files": bool(context_files)}
        )
        
        config = {
            "configurable": {"thread_id": thread_id},
            "callbacks": callbacks,
            "metadata": trace_metadata,
            "tags": ["orchestrator", "lawyers-dashboard"],
        }
        
        # Track current agent for context
        current_agent = "orchestrator"
        current_step = None
        current_run_id = None  # Will be populated for feedback
        current_task_id = None  # Track task for task-complete events
        # Track task statuses across PlanTask calls to preserve done/in_progress
        known_task_statuses: dict[str, str] = {}  # {task_id: status}
        
        try:
            # Stream events from the graph
            # Higher recursion limit for orchestrator (cycles through multiple agents)
            async for event in self.graph.astream_events(inputs, config={**config, "recursion_limit": 100}, version="v2"):
                kind = event.get("event")
                tags = event.get("tags", [])
                name = event.get("name", "")
                
                # Determine which agent is producing this event
                agent_name = self._extract_agent_from_event(event, tags, current_agent)
                
                # --- Node/Step transitions ---
                if kind == "on_chain_start":
                    # Detect node transitions for step events
                    node_name = event.get("name", "")
                    if node_name in ["analyze", "search", "read", "synthesize", "retrieve"]:
                        step_descriptions = {
                            "analyze": "Analyzing request...",
                            "search": "Searching for documents...",
                            "read": "Reading document contents...",
                            "synthesize": "Synthesizing response...",
                            "retrieve": "Retrieving information..."
                        }
                        current_step = node_name
                        yield {
                            "type": "step",
                            "agent": agent_name,
                            "step": node_name,
                            "description": step_descriptions.get(node_name, f"Executing {node_name}...")
                        }
                    elif node_name in list_agent_names():
                        # Worker agent starting
                        current_agent = node_name
                        yield {
                            "type": "agent-start",
                            "agent": node_name,
                            "task": "Processing delegated task"
                        }
                
                elif kind == "on_chain_end":
                    node_name = event.get("name", "")
                    if node_name in list_agent_names():
                        yield {
                            "type": "agent-end",
                            "agent": node_name
                        }
                        # Emit task-complete event if we have a tracked task
                        if current_task_id:
                            yield {
                                "type": "task-complete",
                                "task_id": current_task_id,
                                "agent": node_name
                            }
                            
                            # Emit updated task-plan with current statuses
                            # Get the updated task_list from the chain output
                            output_data = event.get("data", {}).get("output", {})
                            updated_task_list = output_data.get("task_list", [])
                            if updated_task_list:
                                # Update our status tracking
                                for t in updated_task_list:
                                    tid = t.get("id", "")
                                    if tid:
                                        known_task_statuses[tid] = t.get("status", "pending")
                                
                                # Find the goal from existing plan (or use default)
                                yield {
                                    "type": "task-plan",
                                    "goal": "",  # Frontend should keep existing goal
                                    "tasks": [
                                        {"id": t.get("id", ""), "description": t.get("description", ""), "status": t.get("status", "pending")}
                                        for t in updated_task_list
                                    ]
                                }
                            
                            current_task_id = None  # Clear after completion
                
                # --- Streaming LLM content (thinking) ---
                elif kind == "on_chat_model_stream":
                    content = event.get("data", {}).get("chunk", {})
                    if hasattr(content, "content") and content.content:
                        text_content = content.content
                        if isinstance(text_content, list):
                            # Handle multimodal/list content - Flatten to string
                            text_content = "".join([
                                c if isinstance(c, str) else 
                                (c.get("text", "") if isinstance(c, dict) and c.get("type") == "text" else str(c))
                                for c in text_content
                            ])
                        
                        # Determine event type based on context
                        # If we're in an "analyze" or similar step, it's "thinking"
                        # Otherwise it's regular "content"
                        event_type = "thinking" if current_step in ["analyze", "retrieve"] else "content"
                        
                        # Skip streaming mostly-empty chunks (common with garbled PDF extraction)
                        if text_content.strip() == "" or len(text_content.strip()) < 3:
                            continue
                        
                        chunk = {
                            "type": event_type, 
                            "text": text_content,
                            "agent": agent_name,
                        }
                        if current_step:
                            chunk["step"] = current_step
                        yield chunk
                
                # --- LLM call complete ---
                elif kind == "on_chat_model_end":
                    # If we were in a thinking step, signal completion
                    if current_step in ["analyze", "retrieve"]:
                        yield {
                            "type": "thinking-complete",
                            "agent": agent_name,
                            "step": current_step
                        }
                    
                    # --- Task Events: Detect PlanTask/DelegateTask from LLM response ---
                    output = event.get("data", {}).get("output", {})
                    if hasattr(output, "tool_calls"):
                        for tc in output.tool_calls:
                            tool_name = tc.get("name", "")
                            args = tc.get("args", {})
                            
                            if tool_name == "PlanTask":
                                # Emit task-plan event, merging with known statuses
                                # This preserves done/in_progress for existing tasks
                                # while allowing new tasks to be added
                                tasks = args.get("tasks", [])
                                merged_tasks = []
                                for i, t in enumerate(tasks):
                                    tid = t.get("id", f"t{i}")
                                    desc = t.get("description", "")
                                    # Preserve existing status if known, otherwise pending
                                    status = known_task_statuses.get(tid, "pending")
                                    merged_tasks.append({"id": tid, "description": desc, "status": status})
                                    # Track new tasks as pending
                                    if tid not in known_task_statuses:
                                        known_task_statuses[tid] = "pending"
                                
                                yield {
                                    "type": "task-plan",
                                    "goal": args.get("goal", ""),
                                    "tasks": merged_tasks
                                }
                            
                            elif tool_name == "DelegateTask":
                                # Emit task-start event and track for completion
                                task_id = args.get("task_id", "")
                                current_task_id = task_id  # Track for task-complete
                                yield {
                                    "type": "task-start",
                                    "task_id": task_id,
                                    "description": args.get("task", ""),
                                    "agent": args.get("agent_name", "")
                                }
                
                # --- Tool invocation ---
                elif kind == "on_tool_start":
                    tool_name = event.get("name", "unknown")
                    tool_input = event.get("data", {}).get("input", {})
                    
                    yield {
                        "type": "tool-call",
                        "tool": tool_name,
                        "args": tool_input,
                        "agent": agent_name,
                        "step": current_step
                    }
                
                elif kind == "on_tool_end":
                    tool_name = event.get("name", "unknown")
                    output = event.get("data", {}).get("output", "")
                    
                    # Check for context updates
                    if tool_name == "add_to_context":
                        try:
                            if isinstance(output, str):
                                data = json.loads(output)
                                if data.get("action") == "add_to_context":
                                    yield {
                                        "type": "context-update",
                                        "files": data.get("files", []),
                                        "agent": agent_name
                                    }
                        except Exception:
                            pass

                    # Truncate large outputs
                    if isinstance(output, str) and len(output) > 200:
                        output = output[:200] + "..."
                    
                    yield {
                        "type": "tool-result",
                        "tool": tool_name,
                        "result": output,
                        "agent": agent_name,
                        "step": current_step
                    }


            # Include run_id in done event for feedback submission
            yield {
                "type": "done", 
                "agent": "orchestrator",
                "run_id": get_current_run_id()  # For feedback attachment
            }

        except Exception as e:
            yield {"type": "error", "error": str(e), "agent": "orchestrator"}
    
    def _extract_agent_from_event(self, event: dict, tags: list, current_agent: str = "orchestrator") -> str:
        """Extract agent name from event metadata, falling back to current_agent."""
        # Try to find agent name in tags
        known_agents = list_agent_names()
        for tag in tags:
            if tag in known_agents:
                return tag
        
        # Check event name
        name = event.get("name", "")
        if name in known_agents:
            return name
        
        # Check metadata
        metadata = event.get("metadata", {})
        if metadata.get("langgraph_node") in known_agents:
            return metadata.get("langgraph_node")
        
        return current_agent

# --- Registry Initialization ---
def initialize_agent_registry() -> None:
    from app.agents.document_agent import DocumentAgent
    from app.agents.research_agent import ResearchAgent
    from app.agents.drafting_agent import DraftingAgent
    from app.agents.template_agent import TemplateAgent
    # Note: AnalysisAgent is NOT registered here - it's only used by the indexer service
    
    register_agent("document_agent", DocumentAgent())
    register_agent("research_agent", ResearchAgent())
    register_agent("drafting_agent", DraftingAgent())
    register_agent("template_agent", TemplateAgent())
    
    print_and_log(f"[Orchestrator] Registered agents: {list_agent_names()}")
