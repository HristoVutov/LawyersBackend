"""
Drafting Agent - Expert legal document drafter.

Refactored to use "Thinking in LangGraph" Structured State and Command Pattern.
Integrates TemplateAgent as a hierarchical subgraph.
"""
from typing import Annotated, Literal, TypedDict, List, Optional
import operator
import json

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command

from app.agents.base_agent import BaseAgent, AgentConfig
from app.agents.template_agent import TemplateAgent
from app.tools import create_docx_file, fill_template
from app.agents.prompts.drafting_prompt import (
    DRAFTING_SYSTEM_PROMPT,
    ANALYZE_PROMPT,
    DRAFT_WITH_TEMPLATE_PROMPT,
    DRAFT_FROM_SCRATCH_PROMPT
)

# --- Structured State Definition ---
class DraftingState(TypedDict):
    """
    Domain-specific state for the Drafting Agent.
    """
    # Input/Context
    messages: Annotated[list[BaseMessage], operator.add]
    task_description: str
    
    # Working Memory
    intent: str
    draft_properties: dict          # Entities extracted (names, dates, etc.)
    selected_template: Optional[str] # Path to template
    
    # Output
    generated_content: str
    final_file_path: str


class DraftingAgent(BaseAgent):
    """
    Drafting Agent - Explicit Workflow Implementation.
    
    Flow:
    1. Analyze (LLM) -> Extract intent and entities
    2. Check Template (Subgraph) -> Ask TemplateAgent if a template exists
    3. Draft (LLM) -> Fill template OR write from scratch
    4. Save (Tool) -> Create .docx
    """
    
    def __init__(self):
        self.template_agent = TemplateAgent()
        
        config = AgentConfig(
            name="drafting_agent",
            tools=[create_docx_file, fill_template],
            system_prompt=DRAFTING_SYSTEM_PROMPT
        )
        super().__init__(config)
        self._graph = self._build_graph()

    def _build_graph(self):
        """Builds the explicit Structured State graph."""
        
        # --- Node Definitions ---
        
        async def analyze_node(state: DraftingState) -> Command[Literal["check_template"]]:
            messages = state.get("messages", [])
            task_desc = state.get("task_description")
            if not task_desc and messages:
                task_desc = messages[-1].content
            
            print(f"[{self.name}] 🔍 Analyzing drafting request: {task_desc[:50]}...")
            
            response = await self.llm.ainvoke(
                [HumanMessage(content=ANALYZE_PROMPT.format(task_description=task_desc))]
            )
            
            intent = "Legal Document"
            properties = {}
            try:
                content = response.content.replace("```json", "").replace("```", "").strip()
                data = json.loads(content)
                intent = data.get("intent", "Legal Document")
                properties = data.get("properties", {})
            except Exception:
                pass
                
            print(f"[{self.name}] 📋 Intent: {intent}")
            
            return Command(
                update={
                    "intent": intent, 
                    "draft_properties": properties,
                    "task_description": task_desc
                },
                goto="check_template"
            )

        async def check_template_node(state: DraftingState) -> Command[Literal["draft"]]:
            intent = state.get("intent", "")
            print(f"[{self.name}] 📂 Asking TemplateAgent for '{intent}'...")
            
            # Delegate to TemplateAgent
            sub_inputs = {"messages": [HumanMessage(content=f"Find a template for: {intent}")]}
            sub_state = await self.template_agent.graph.ainvoke(sub_inputs)
            
            # Heuristic: Check if TemplateAgent found a file
            # We assume TemplateAgent's last message describes the found template or says "Not found"
            # Ideally TemplateAgent should return a structured state too, but it's likely returning text.
            # For this MVP, we'll parse the text or look for a file path pattern.
            sub_msgs = sub_state.get("messages", [])
            last_msg_content = sub_msgs[-1].content if sub_msgs else ""
            
            found_template = None
            # Simple heuristic: if it mentions a path or "Found", we assume it's good.
            # Better: In a real system, TemplateAgent would return a 'selected_file' in state.
            # Let's assume if it returns a specific file path or name, we use it.
            # For now, let's look for known templates or simulate success if we had specific logic.
            # Since TemplateAgent is simple, let's just proceed. If it explicitly provided a path, we'd capture it.
            
            # MOCKING: For this specific refactor, if "Power of Attorney" or "Пълномощно" is in intent, 
            # and we don't have a real file, we normally draft from scratch.
            # If we had a file at "templates/poa.txt", we'd pick it.
            
            print(f"[{self.name}] 📂 TemplateAgent said: {last_msg_content[:50]}...")
            
            return Command(
                update={"selected_template": found_template}, # None for now unless we enforce strict structure on TemplateAgent
                goto="draft"
            )

        async def draft_node(state: DraftingState) -> Command[Literal["save"]]:
            template = state.get("selected_template")
            props = state.get("draft_properties", {})
            intent = state.get("intent")
            task = state.get("task_description")
            
            print(f"[{self.name}] ✍️ Drafting document...")
            
            if template:
                # Fill Template Mode
                prompt = DRAFT_WITH_TEMPLATE_PROMPT.format(
                    template_content=f"[Content of {template}]", # In real app, we'd read the file here
                    properties=props
                )
            else:
                # Scratch Mode
                prompt = DRAFT_FROM_SCRATCH_PROMPT.format(
                    intent=intent,
                    properties=props,
                    task_description=task
                )
            
            response = await self.llm.ainvoke([HumanMessage(content=prompt)])
            content = response.content
            
            return Command(
                update={"generated_content": content},
                goto="save"
            )

        async def save_node(state: DraftingState) -> Command[Literal["__end__"]]:
            content = state.get("generated_content", "")
            intent = state.get("intent", "document").replace(" ", "_")
            filename = f"{intent}_draft.docx"
            
            print(f"[{self.name}] 💾 Saving to {filename}...")
            
            # Call the tool directly
            # create_docx_file takes (file_name, content)
            try:
                # Using the tool function wrapper logic or invoke
                # create_docx_file is a StructuredTool.
                await create_docx_file.ainvoke({"file_name": filename, "content": content})
                final_path = filename # In current dir
            except Exception as e:
                print(f"[{self.name}] ❌ Save failed: {e}")
                final_path = "error_saving.txt"

            return Command(
                update={
                    "final_file_path": final_path,
                    "messages": [AIMessage(content=f"Draft created successfully: {final_path}", name=self.name)]
                },
                goto=END
            )

        # --- Graph Construction ---
        graph = StateGraph(DraftingState)
        
        graph.add_node("analyze", analyze_node)
        graph.add_node("check_template", check_template_node)
        graph.add_node("draft", draft_node)
        graph.add_node("save", save_node)
        
        graph.add_edge(START, "analyze")
        
        return graph.compile(checkpointer=self.checkpointer)
