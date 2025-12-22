from typing import List, Optional
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from app.services.conversation_logger import print_and_log
from app.agents.prompts.summarization_prompt import SUMMARIZATION_SYSTEM_PROMPT

class ConversationSummary(BaseModel):
    """Structured summary of the conversation history."""
    summary_content: str = Field(description="Narrative summary of the conversation history.")
    active_goal: str = Field(description=" The current high-level objective of the user.")
    pending_tasks: List[str] = Field(default_factory=list, description="List of tasks explicitly mentioned as pending or next steps.")
    key_entities: List[str] = Field(default_factory=list, description="Important names, dates, case numbers, or legal references.")

class ContextManager:
    """
    Manages the context window by summarizing older messages when a threshold is reached.
    """
    def __init__(
        self, 
        model: ChatGoogleGenerativeAI, 
        max_messages: int = 40, 
        summary_limit: int = 10, # Number of messages to keep raw after summary
        system_prompt: str = SUMMARIZATION_SYSTEM_PROMPT
    ):
        self.model = model
        self.max_messages = max_messages
        self.summary_limit = summary_limit
        self.tokenizer = model  # Gemini model has count_tokens support, but we rely on message count for now for speed/simplicity
        self.system_prompt = system_prompt
        self.structured_model = model.with_structured_output(ConversationSummary)

    async def manage_context(self, messages: List[BaseMessage]) -> List[BaseMessage]:
        """
        Compresses the message history if it exceeds the max_messages threshold.
        
        Strategy:
        1. If len(messages) <= max_messages, return as is.
        2. Identify the "messages to summarize" (older ones) and "preserved messages" (recent ones).
        3. Send "messages to summarize" to the LLM to generate a structured summary.
        4. Construct a new context: [SystemMessage(Summary)] + [Preserved Messages].
        """
        
        # 0. Quick check
        if len(messages) <= self.max_messages:
            return messages

        print_and_log(f"[ContextManager] 🧹 Context size ({len(messages)}) exceeded limit ({self.max_messages}). Summarizing...")

        # 1. Split messages
        # We always want to keep the System Prompt if it's the first message
        start_index = 0
        original_system_message = None
        if messages and isinstance(messages[0], SystemMessage):
            original_system_message = messages[0]
            start_index = 1
        
        # The messages eligible for summarization are those before the trailing window
        # We want to keep the last 'summary_limit' messages intact
        
        # Safe slice
        end_index = len(messages) - self.summary_limit
        
        # If the cut-off is before the start (logic specific check), just return
        if end_index <= start_index:
             return messages
             
        to_summarize = messages[start_index:end_index]
        recent_messages = messages[end_index:]
        
        # 2. Invoke Summarization
        try:
            summary: ConversationSummary = await self.structured_model.ainvoke(
                [SystemMessage(content=self.system_prompt)] + to_summarize
            )
            
            print_and_log(f"[ContextManager] 📝 Summary generated: {summary.active_goal}")
            
            # 3. Format Summary as a System Message (or injected context)
            # We explicitly format it so the next agent understands this is HISTORY.
            
            summary_text = (
                f"### PREVIOUS CONVERSATION SUMMARY ###\n"
                f"Narrative: {summary.summary_content}\n"
                f"Current Goal: {summary.active_goal}\n"
                f"Pending Tasks: {', '.join(summary.pending_tasks)}\n"
                f"Key Entities: {', '.join(summary.key_entities)}\n"
                f"#######################################"
            )
            
            new_context = []
            if original_system_message:
                new_context.append(original_system_message)
            
            new_context.append(SystemMessage(content=summary_text))
            new_context.extend(recent_messages)
            
            print_and_log(f"[ContextManager] ✅ Context reduced from {len(messages)} to {len(new_context)} messages.")
            return new_context

        except Exception as e:
            print_and_log(f"[ContextManager] ❌ Summarization failed: {e}. Returning original messages.")
            return messages
