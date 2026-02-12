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
from app.services.telemetry import track_token_usage, track_event
from app.tracing import get_run_callbacks, get_current_run_id

# --- State Definition ---
class AgentState(TypedDict):
    """The shared state of the supervisor graph."""
    messages: Annotated[list[BaseMessage], operator.add]
    next: str
    task_list: list[dict]  # List of {id, description, status}
    current_task_id: str | None  # Currently executing task
    guard_trigger_count: int  # Track Autonomy Guard triggers to prevent loops
    project_context: Annotated[list[dict], operator.add]  # Proactive information about project files

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
        self._members = None
        
    @property
    def graph(self):
        if self._supervisor_graph is None:
            self._supervisor_graph = self._build_supervisor_graph()
        return self._supervisor_graph

    @property
    def members(self) -> list[str]:
        """Lazy load members list."""
        if self._members is None:
            all_agents = list_agent_names()
            # Filter out document_agent from top-level orchestration
            self._members = [name for name in all_agents if name != "document_agent"]
            
            # Ensure we have members (fallback)
            if not self._members:
                self._members = ["research_agent"] 
        return self._members

    def _build_supervisor_graph(self):
        """Build the Supervisor StateGraph."""
        # Ensure members are initialized
        members = self.members

        # 1. Build Graph
        workflow = StateGraph(AgentState)
        
        workflow.add_node("supervisor", self._supervisor_node)
        
        for member in members:
            workflow.add_node(member, self._make_worker_node(member))
            
        # 2. Edges
        # Start -> Supervisor
        workflow.add_edge(START, "supervisor")
        
        # Supervisor -> Conditional (Worker or End)
        conditional_map = {k: k for k in members}
        conditional_map["FINISH"] = END
        conditional_map["supervisor"] = "supervisor"
        conditional_map[END] = END  # Safety
        
        workflow.add_conditional_edges(
            "supervisor", 
            lambda x: x["next"], 
            conditional_map
        )
        
        # Worker -> Supervisor (Always report back)
        for member in members:
            workflow.add_edge(member, "supervisor")
            
        return workflow.compile(checkpointer=self.checkpointer)

    async def _supervisor_node(self, state: AgentState, config) -> dict:
        """
        The main supervisor node logic.
        Decides next steps based on LLM response, autonomy guards, and tools.
        """
        # 1. Prepare Chain
        chain = self._prepare_supervisor_chain(config)
        
        # 2. Manage Context
        messages = await self.context_manager.manage_context(state["messages"])
        chain_input = {**state, "messages": messages}
        
        # 3. Invoke LLM
        try:
            response = await chain.ainvoke(chain_input, config)
        except Exception as e:
            print_and_log(f"[Orchestrator] ❌ LLM Error: {e}")
            return {"next": "FINISH"}

        # 4. Access State & Response info
        tool_calls = getattr(response, "tool_calls", [])
        is_user_turn = self._is_user_turn(state)
        
        # 5. Check Autonomy Guard (prevents loops / missed tasks)
        # Returns a dict if guard prevents normal execution
        guard_result = self._check_autonomy_guard(state, response, tool_calls, is_user_turn)
        if guard_result:
            return guard_result

        # 6. Process Response (Tools or Text)
        if not tool_calls:
            # Fallback: check text for FINISH
            content = str(response.content)
            if "FINISH" in content:
                return {"next": "FINISH", "messages": [response]}
            # Otherwise, loop back to supervisor (or FINISH if we want to be strict, but keeping orig behavior)
            return {"next": "FINISH", "messages": [response]}

        # 7. Process Valid Tool Calls
        return self._process_tool_calls(state, response, tool_calls, is_user_turn)

    def _prepare_supervisor_chain(self, config):
        """Creates the LLM chain with tools and system prompt."""
        system_prompt = ORCHESTRATOR_SYSTEM_PROMPT
        tools = [PlanTask, DelegateTask]
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="messages"),
        ])
        
        # Dynamic LLM Selection
        model_name = config.get("configurable", {}).get("model_name")
        if model_name:
             supervisor_llm = self._create_llm(model_name)
             print_and_log(f"[Orchestrator] 🔄 Supervisor using model: {model_name}")
        else:
             supervisor_llm = self.llm

        llm_with_tools = supervisor_llm.bind_tools(tools)
        return prompt | llm_with_tools

    def _is_user_turn(self, state: AgentState) -> bool:
        """Check if the last message was from the user."""
        messages_list = state.get("messages", [])
        if messages_list and isinstance(messages_list[-1], HumanMessage):
            return True
        return False

    def _check_autonomy_guard(self, state: AgentState, response, tool_calls: list, is_user_turn: bool) -> dict | None:
        """
        AUTONOMY GUARD: Block FINISH if pending tasks exist (unless user asked to stop/intervene).
        Returns a state dict if the guard is triggered, None otherwise.
        """
        task_list = state.get("task_list", [])
        pending_tasks = [t for t in task_list if t["status"] == "pending"]
        
        # If there are NO tool calls, the LLM might be trying to chat or finish prematurely.
        if not tool_calls and pending_tasks and not is_user_turn:
            current_triggers = state.get("guard_trigger_count", 0)
            
            # 1. Loop Detection
            if current_triggers >= 3:
                 print_and_log(f"[Orchestrator] 🚨 Autonomy Guard Loop Detected ({current_triggers} retries). Forcing FINISH to prevent crash.")
                 return {
                     "next": "FINISH", 
                     "messages": [
                         response,
                         AIMessage(content="Error: Unable to proceed. Orchestrator stuck in delegation loop. Stopping.")
                     ]
                 }

            # 2. Force Delegation
            next_task = pending_tasks[0]
            print_and_log(f"[Orchestrator] ⚠️ Autonomy Guard: {len(pending_tasks)} pending tasks, forcing delegation of {next_task['id']} (Attempt {current_triggers + 1}/3)")
            
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
        
        return None

    def _process_tool_calls(self, state: AgentState, response, tool_calls: list, is_user_turn: bool) -> dict:
        """Handle PlanTask and DelegateTask."""
        # Reset guard trigger count on valid tool use
        extra_state_updates = {"guard_trigger_count": 0}

        next_step = "supervisor" 
        messages_to_add = [response]
        
        # We only process the FIRST valid tool call effectively in one turn for now (Sequential)
        # BUG FIX: Prioritize DelegateTask if present. 
        # Sometimes LLM sends [PlanTask, DelegateTask]. If we process PlanTask first, 
        # it might trigger "Plan Already Exists" guard and block the valid delegation.
        delegate_tc = next((t for t in tool_calls if t["name"] == "DelegateTask"), None)
        if delegate_tc:
            tc = delegate_tc
        else:
            tc = tool_calls[0]
        tool_name = tc["name"]
        args = tc["args"]
        
        if tool_name == "PlanTask":
            return self._handle_plan_task(state, tc, args, is_user_turn, extra_state_updates)
            
        elif tool_name == "DelegateTask":
            return self._handle_delegate_task(state, tc, args, extra_state_updates)
            
        # Fallback for unknown tools
        return {"next": next_step, "messages": messages_to_add, **extra_state_updates}

    def _handle_plan_task(self, state: AgentState, tc, args, is_user_turn: bool, extra_state_updates: dict) -> dict:
        """Handle execution of PlanTask."""
        existing_tasks = state.get("task_list", [])
        pending_tasks = [t for t in existing_tasks if t["status"] == "pending"]
        
        # GUARD: Skip if we already have an active task list
        # This prevents looping when user says "ok" and LLM tries to plan again instead of delegated
        if pending_tasks:
            # If user explicitly asks to change plan, we might allow it, but for simple "ok" we block
            if is_user_turn:
                 print_and_log(f"[Orchestrator] 🛑 Blocked Redundant PlanTask - Interpreting user input as approval to execute existing tasks.")
                 return {
                    "next": "supervisor", 
                    "messages": [
                        ToolMessage(
                            content=f"PLAN ALREADY EXISTS. Do not re-plan. Immediately call DelegateTask for the first pending task.",
                            tool_call_id=tc["id"]
                        )
                    ]
                 }

            print_and_log(f"[Orchestrator] ⏭️ Skipping PlanTask - {len(pending_tasks)} tasks still pending, force-delegating")
            # Check if all tasks are done
            all_done = all(t.get("status") == "done" for t in existing_tasks)
            if all_done:
                print_and_log(f"[Orchestrator] ✅ All tasks complete, finishing")
                return {
                    "next": "FINISH", 
                    "messages": [
                        ToolMessage(content="All tasks completed. Finishing.", tool_call_id=tc["id"])
                    ]
                }
            else:
                # Force-delegate the first pending task instead of looping back to supervisor
                next_task = pending_tasks[0]
                agent_name = self._pick_agent_for_task(next_task["description"])
                
                # Mark task as in_progress
                updated_tasks = []
                for t in existing_tasks:
                    if t["id"] == next_task["id"]:
                        updated_tasks.append({**t, "status": "in_progress"})
                    else:
                        updated_tasks.append(t)
                
                print_and_log(f"[Orchestrator] 🔀 Force-delegating '{next_task['id']}' to {agent_name}")
                return {
                    "next": agent_name, 
                    "messages": [
                        ToolMessage(
                            content=f"Plan exists. Force-delegating task {next_task['id']} to {agent_name}.",
                            tool_call_id=tc["id"]
                        ),
                        HumanMessage(content=next_task["description"], name="supervisor")
                    ],
                    "task_list": updated_tasks,
                    "current_task_id": next_task["id"],
                }
        
        # First time planning or Re-planning requested by user
        print_and_log(f"[Orchestrator] 📋 Plan Update: {args.get('goal')}")
        
        # AUTO-CLEAR: If all previous tasks are done, treat this as a fresh start ONLY if user initiated
        # This prevents "Task already done" collisions when user starts a new request, 
        # but preserves history if agent is just updating/restating plan.
        if is_user_turn and existing_tasks and all(t.get("status") == "done" for t in existing_tasks):
            print_and_log(f"[Orchestrator] 🧹 Clearing {len(existing_tasks)} completed tasks to start fresh plan (User Triggered).")
            existing_tasks = []
        
        # Merge with existing status to prevent loops
        existing_map = {t["id"]: t for t in existing_tasks}
        new_tasks = []
        for i, t in enumerate(args.get("tasks", [])):
            t_id = t.get("id", f"t{i}")
            # Preserve status if task already exists
            status = existing_map.get(t_id, {}).get("status", "pending")
            new_tasks.append({
                "id": t_id, 
                "description": t.get("description", ""), 
                "status": status
            })
            
        print_and_log(f"[Orchestrator] 📝 Created {len(new_tasks)} tasks: {[t['id'] for t in new_tasks]}")
        
        # FINAL GUARD: If all tasks in the NEW plan are *already* done (e.g. agent effectively just re-stated finished tasks),
        # do NOT pause for review. Look for 'done' status in new_tasks.
        if all(t.get("status") == "done" for t in new_tasks):
             print_and_log(f"[Orchestrator] ✅ Plan updated but all tasks are already complete. Auto-finishing.")
             return {
                 "next": "FINISH", 
                 "messages": [
                     ToolMessage(content="Plan updated. All tasks are complete.", tool_call_id=tc["id"])
                 ],
                 "task_list": new_tasks, 
                 **extra_state_updates
             }

        # Auto-delegate the first task immediately (bypass supervisor LLM to prevent PlanTask loop)
        first_task = new_tasks[0]
        agent_name = self._pick_agent_for_task(first_task["description"])
        
        # Mark first task as in_progress
        for t in new_tasks:
            if t["id"] == first_task["id"]:
                t["status"] = "in_progress"
        
        print_and_log(f"[Orchestrator] 🚀 Plan created, auto-delegating '{first_task['id']}' to {agent_name}")
        
        messages_to_add = [
            ToolMessage(
                content=f"Plan created with {len(new_tasks)} tasks. Auto-delegating task {first_task['id']} to {agent_name}.",
                tool_call_id=tc["id"]
            ),
            HumanMessage(content=first_task["description"], name="supervisor")
        ]
        
        return {
            "next": agent_name,
            "messages": messages_to_add,
            "task_list": new_tasks,
            "current_task_id": first_task["id"],
            **extra_state_updates
        }

    def _pick_agent_for_task(self, description: str) -> str:
        """Pick the best agent for a task based on keyword matching."""
        desc_lower = description.lower()
        
        drafting_keywords = ["изготви", "генерирай", "създай документ", "напиши", "draft", "generate", "write"]
        template_keywords = ["шаблон", "template"]
        
        if any(kw in desc_lower for kw in template_keywords):
            return "template_agent"
        if any(kw in desc_lower for kw in drafting_keywords):
            return "drafting_agent"
        
        # Default to research_agent for analysis, search, read, etc.
        return "research_agent"

    def _handle_delegate_task(self, state: AgentState, tc, args, extra_state_updates: dict) -> dict:
        """Handle execution of DelegateTask."""
        agent_name = args.get("agent_name")
        task_desc = args.get("task")
        task_id = args.get("task_id")
        
        # GUARD: Prevent re-delegation of completed tasks
        current_tasks = state.get("task_list", [])
        target_task = next((t for t in current_tasks if t["id"] == task_id), None)
        if target_task and target_task.get("status") == "done":
             # Check if there are ANY pending tasks left
             pending_count = sum(1 for t in current_tasks if t.get("status") in ["pending", "in_progress"])
             
             if pending_count == 0:
                 print_and_log(f"[Orchestrator] 🛑 Task {task_id} is done & NO pending tasks. Forcing FINISH to prevent loop.")
                 return {
                     "next": "FINISH", 
                     "messages": [
                         ToolMessage(content="All tasks are complete. Terminating workflow.", tool_call_id=tc["id"])
                     ],
                     **extra_state_updates
                 }
             else:
                 print_and_log(f"[Orchestrator] 🛑 Prevented Re-delegation: Task {task_id} is already done.")
                 messages_to_add = [
                     ToolMessage(
                         content=f"TASK ALREADY COMPLETE. Task {task_id} is marked as 'done'. Do not re-delegate it. Choose the next pending task or finish.",
                         tool_call_id=tc["id"]
                     )
                 ]
                 return {"next": "supervisor", "messages": messages_to_add, **extra_state_updates}
        
        if agent_name not in self.members:
            print_and_log(f"[Orchestrator] ⚠️ Invalid agent: {agent_name}")
            messages_to_add = [
                ToolMessage(
                    content=f"Error: Agent '{agent_name}' not found. Available: {self.members}",
                    tool_call_id=tc["id"]
                )
            ]
            return {"next": "supervisor", "messages": messages_to_add, **extra_state_updates}
        
        print_and_log(f"[Orchestrator] 👉 Delegating to {agent_name}: {task_desc} (task: {task_id})")
        
        # Mark task as in_progress in task_list
        updated_tasks = []
        for t in state.get("task_list", []):
            if t["id"] == task_id:
                updated_tasks.append({**t, "status": "in_progress"})
            else:
                updated_tasks.append(t)
        
        messages_to_add = [
            ToolMessage(content=f"Delegated to {agent_name}.", tool_call_id=tc["id"]),
            HumanMessage(content=task_desc, name="supervisor")
        ]
        
        return {
            "next": agent_name,
            "messages": messages_to_add,
            "task_list": updated_tasks,
            "current_task_id": task_id,
            **extra_state_updates
        }

    def _make_worker_node(self, agent_name: str):
        """Creates a worker node wrapper that invokes the sub-agent."""
        async def worker_node(state: AgentState, config) -> dict:
            agent = get_agent(agent_name)
            
            # Config propagation
            parent_configurable = config.get("configurable", {})
            model_name = parent_configurable.get("model_name")
            worker_configurable = {"model_name": model_name} if model_name else {}
            
            worker_config = {
                **config, 
                "recursion_limit": 50,
                "configurable": {
                    **parent_configurable,
                    **worker_configurable
                }
            }
            
            # Execute sub-agent
            result_state = await agent.graph.ainvoke(state, worker_config)
            
            # Extract Result
            last_message = result_state["messages"][-1]
            content = last_message.content if hasattr(last_message, "content") else str(last_message)
            messages_to_return = [AIMessage(content=content, name=agent_name)]
            
            # Capture Context Updates from Sub-Agent Tools
            new_context = []
            for msg in result_state.get("messages", []):
                if isinstance(msg, ToolMessage) and msg.name == "add_to_context":
                    try:
                        # The tool output is likely a JSON string
                        data = json.loads(msg.content)
                        if data.get("action") == "add_to_context":
                             files = data.get("files", [])
                             summary = data.get("summary", "")
                             
                             # Add to extracted context
                             for f in files:
                                 new_context.append({
                                     "path": f,
                                     "summary": summary,
                                     "documentType": f.split('.')[-1] if '.' in f else "unknown"
                                 })
                    except Exception as e:
                        print_and_log(f"[Orchestrator] ⚠️ Failed to parse add_to_context output: {e}")

            if new_context:
                print_and_log(f"[Orchestrator] 📎 Captured {len(new_context)} new context files from {agent_name}")
            
            # Update Task Status
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
                "current_task_id": None,  # Clear current task
                "project_context": new_context  # Update global state (via operator.add)
            }
        return worker_node

    async def stream(
        self,
        message: str,
        thread_id: str = "default",
        context_files: list[str] | None = None,
        project_context: list[dict] | None = None,
        model_name: str | None = None
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
        
        inputs = {
            "messages": [HumanMessage(content=final_message)],
        }
        
        # Only include project_context if provided to avoid overwriting existing state with None
        if project_context is not None:
             inputs["project_context"] = project_context
        
        # Build config with LangSmith tracing callbacks
        has_context = bool(context_files) or bool(project_context)
        callbacks, trace_metadata = get_run_callbacks(
            thread_id=thread_id,
            agent_name="orchestrator",
            extra_tags=["supervisor", "multi-agent"],
            extra_metadata={"has_context_files": has_context, "project_context_count": len(project_context) if project_context else 0}
        )
        
        config = {
            "configurable": {"thread_id": thread_id, "model_name": model_name},
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
        
        # Track chat_message event
        await track_event("chat_message", metadata={
            "thread_id": thread_id,
            "model": model_name or "gemini-2.5-flash",
            "has_context_files": has_context,
            "project_context_count": len(project_context) if project_context else 0,
        })
        
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
                    
                    # --- Track token usage ---
                    output = event.get("data", {}).get("output", {})
                    if hasattr(output, "usage_metadata") and output.usage_metadata:
                        usage = output.usage_metadata
                        model_name = config.get("configurable", {}).get("model_name") or "gemini-2.5-flash"
                        await track_token_usage(
                            thread_id=thread_id,
                            agent_name=agent_name,
                            model_name=model_name,
                            input_tokens=usage.get("input_tokens", 0),
                            output_tokens=usage.get("output_tokens", 0),
                        )
                    
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
                    
                    # Skip tool-call event for remote tools (they already send local_tool_call)
                    remote_tool_names = ["read_file", "write_file", "list_directory", "glob", "grep", 
                                        "execute_ocr", "search_indexed", "read_document", "get_project_overview"]
                    if tool_name in remote_tool_names:
                        continue  # Don't emit duplicate tool-call for remote tools
                    
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
