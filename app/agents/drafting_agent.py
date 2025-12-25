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
from app.services.conversation_logger import print_and_log
# Direct imports to avoid circular import
from app.tools.file_tools import create_docx_file
from app.tools.legal_tools import fill_template
from app.agents.prompts.drafting_prompt import (
    DRAFTING_SYSTEM_PROMPT,
    ANALYZE_PROMPT,
    DRAFT_WITH_TEMPLATE_PROMPT,
    DRAFT_FROM_SCRATCH_PROMPT,
    DRAFT_ANALYSIS_PROMPT,
    SUPPORTED_HTML_FORMATTING
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
    document_type: str          # e.g. "Checklist", "Strategy", "Motion"
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

    def _extract_text_content(self, content: str | list | dict) -> str:
        """Helper to robustly extract text from LLM response content."""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            # Join text parts from blocks
            text_parts = []
            for part in content:
                if isinstance(part, str):
                    text_parts.append(part)
                elif isinstance(part, dict) and "text" in part:
                    text_parts.append(part["text"])
            return "".join(text_parts)
        return str(content)

    def _build_graph(self):
        """Builds the explicit Structured State graph."""
        
        # --- Node Definitions ---
        
        async def analyze_node(state: DraftingState, config) -> Command[Literal["check_template"]]:
            messages = state.get("messages", [])
            task_desc = state.get("task_description")
            if not task_desc and messages:
                task_desc = messages[-1].content
            
            print_and_log(f"[{self.name}] 🔍 Analyzing drafting request: {task_desc[:50]}...")
            
            model_name = config.get("configurable", {}).get("model_name")
            if model_name:
                llm = self._create_llm(model_name)
            else:
                llm = self.llm

            response = await llm.ainvoke(
                [HumanMessage(content=ANALYZE_PROMPT.format(task_description=task_desc))]
            )
            
            intent = "Legal Document"
            properties = {}
            try:
                raw_content = self._extract_text_content(response.content)
                content = raw_content.replace("```json", "").replace("```", "").strip()
                data = json.loads(content)
                intent = data.get("intent", "Legal Document")
                document_type = data.get("document_type", "Document")
                properties = data.get("properties", {})
            except Exception:
                pass
                
            print_and_log(f"[{self.name}] 📋 Intent: {intent} (Type: {document_type})")
            
            return Command(
                update={
                    "intent": intent, 
                    "document_type": document_type,
                    "draft_properties": properties,
                    "task_description": task_desc
                },
                goto="check_template"
            )

        async def check_template_node(state: DraftingState, config) -> Command[Literal["draft"]]:
            intent = state.get("intent", "")
            print_and_log(f"[{self.name}] 📂 Asking TemplateAgent for '{intent}'...")
            
            # Delegate to TemplateAgent
            sub_inputs = {"messages": [HumanMessage(content=f"Find a template for: {intent}")]}
            
            # Pass config down to TemplateAgent!
            sub_state = await self.template_agent.graph.ainvoke(sub_inputs, config)
            
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
            
            print_and_log(f"[{self.name}] 📂 TemplateAgent said: {str(last_msg_content)[:50]}...")
            
            return Command(
                update={"selected_template": found_template}, # None for now unless we enforce strict structure on TemplateAgent
                goto="draft"
            )

        async def draft_node(state: DraftingState, config) -> Command[Literal["save"]]:
            template = state.get("selected_template")
            props = state.get("draft_properties", {})
            intent = state.get("intent")
            task = state.get("task_description")
            
            print_and_log(f"[{self.name}] ✍️ Drafting document...")
            
            if template:
                # Fill Template Mode
                prompt = DRAFT_WITH_TEMPLATE_PROMPT.format(
                    template_content=f"[Content of {template}]", # In real app, we'd read the file here
                    properties=props,
                    formatting_instructions=SUPPORTED_HTML_FORMATTING
                )
            else:
                # Scratch Mode - Check for Analysis vs Contract
                doc_type = state.get("document_type", "Document")
                
                # List of types that should use the Analysis prompt
                analysis_types = ["Analysis", "Memo", "Opinion", "Strategy", "Checklist", "Report", "Review"]
                
                # Check if doc_type matches or if intent suggests analysis
                is_analysis = doc_type in analysis_types or "analysis" in intent.lower() or "memo" in intent.lower()
                
                if is_analysis:
                    print_and_log(f"[{self.name}] 🧠 Using Analysis Prompt for '{doc_type}'")
                    prompt = DRAFT_ANALYSIS_PROMPT.format(
                        intent=intent,
                        properties=props,
                        task_description=task,
                        formatting_instructions=SUPPORTED_HTML_FORMATTING
                    )
                else:
                    print_and_log(f"[{self.name}] 📝 Using Standard Drafting Prompt for '{doc_type}'")
                    prompt = DRAFT_FROM_SCRATCH_PROMPT.format(
                        intent=intent,
                        properties=props,
                        task_description=task,
                        formatting_instructions=SUPPORTED_HTML_FORMATTING
                    )
            
            model_name = config.get("configurable", {}).get("model_name")
            if model_name:
                llm = self._create_llm(model_name)
            else:
                llm = self.llm

            response = await llm.ainvoke([HumanMessage(content=prompt)])
            content = self._extract_text_content(response.content)
            
            return Command(
                update={"generated_content": content},
                goto="save"
            )

        async def save_node(state: DraftingState, config) -> Command[Literal["__end__"]]:
            content = state.get("generated_content", "")
            intent = state.get("intent", "document")
            # Clean intent for filename - keep it safe
            # Replace spaces and non-ascii chars
            safe_intent = "".join([c if c.isalnum() else "_" for c in intent if c.isascii()])
            if not safe_intent:
                safe_intent = "document"
            
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{safe_intent}_{timestamp}.docx"
            
            # Get thread_id for organization
            thread_id = config.get("configurable", {}).get("thread_id", "default")
            
            print_and_log(f"[{self.name}] 💾 Saving to /output/{thread_id}/{filename}...")
            
            final_path = "error_saving.txt"
            
            try:
                # 1. Save Markdown for internal context
                from app.tools.file_tools import write_file
                md_path = f"/output/{thread_id}/{safe_intent}_{timestamp}.md"
                await write_file.ainvoke({"file_path": md_path, "content": content})
                
                # 2. Save DOCX for client delivery
                result = await create_docx_file.ainvoke({
                    "file_name": filename, 
                    "content": content,
                    "subdir": thread_id
                })
                
                # Check result string for success
                if "✅" in result:
                    # Extract path from result if needed, or reconstruct
                    final_path = f"/output/{thread_id}/{filename}"
                else:
                    final_path = result # Return error message
                    
            except Exception as e:
                print_and_log(f"[{self.name}] ❌ Save failed: {e}")
                final_path = f"Error: {e}"

            return Command(
                update={
                    "final_file_path": final_path,
                    "messages": [AIMessage(content=f"Draft created successfully.\nMarkdown (Internal): {md_path}\nDOCX (Client): {final_path}", name=self.name)]
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
