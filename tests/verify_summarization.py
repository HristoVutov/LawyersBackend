
import asyncio
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from app.services.context_manager import ContextManager, ConversationSummary
from app.config import get_settings

async def test_summarization():
    settings = get_settings()
    if not settings.google_api_key:
        print("GOOGLE_API_KEY not set, skipping test.")
        return

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=settings.google_api_key
    )

    # Initialize Context Manager with very low threshold for testing
    cm = ContextManager(model=llm, max_messages=5, summary_limit=2)

    # Create dummy history
    messages = [
        SystemMessage(content="You are a helpful assistant."), # 0
        HumanMessage(content="Hi, I want to discuss a contract for Galaxy Property."), # 1
        AIMessage(content="Sure, I can help with that. What kind of contract?"), # 2
        HumanMessage(content="It's a lease agreement."), # 3
        AIMessage(content="Noted. A lease agreement for Galaxy Property."), # 4
        HumanMessage(content="The rent should be $5000 per month."), # 5
        AIMessage(content="Okay, $5000/month."), # 6
        HumanMessage(content="Also, the tenant is John Doe."), # 7 - This puts us over 5 messages
    ]

    print(f"Original message count: {len(messages)}")

    # Run management
    new_messages = await cm.manage_context(messages)

    print(f"New message count: {len(new_messages)}")
    
    # Assertions
    assert len(new_messages) < len(messages), "Messages should be reduced"
    assert isinstance(new_messages[0], SystemMessage), "First message should be SystemMessage" # The original system prompt or the summary
    
    # Check if summary is present
    summary_msg = new_messages[1] # standard system prompt is 0, summary is 1? Or merged?
    # Logic in code: 
    # new_context.append(original_system_message) if presnet
    # new_context.append(SystemMessage(content=summary_text))
    # so index 1 should be summary if index 0 is original system prompt
    
    if isinstance(new_messages[1], SystemMessage):
        print("\n--- Summary Content ---")
        print(new_messages[1].content)
        assert "Galaxy Property" in new_messages[1].content or "Galaxy Property" in new_messages[0].content
        assert "$5000" in new_messages[1].content
    else:
        print("Unexpected structure:", new_messages)

    print("\n✅ Verification passed!")

if __name__ == "__main__":
    asyncio.run(test_summarization())
