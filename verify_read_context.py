import asyncio
import sys
import json
from langchain_core.messages import HumanMessage
from app.agents.document_agent import DocumentAgent
from app.middleware.virtual_fs import get_vfs
from app.tools.context_tools import add_to_context

async def main():
    print("Initializing DocumentAgent...")
    
    # Initialize VFS with a valid project path
    vfs = get_vfs()
    vfs.set_project("C:/Users/vutov/Documents/LawyersProjects/Test1")
    
    agent = DocumentAgent()
    
    # Query for a binary file to test READ + CONTEXT
    query = "Find the invoice file (InvoiceBG.docx) and add it to context."
    print(f"Query: {query}\n")
    
    try:
        # Use invoke to get final result, but capturing streaming is better to see content
        async for event in agent.stream(query):
            event_type = event.get("type")
            
            if event_type == "tool-call":
                print(f"\n🛠️ [Tool Call] {event.get('agent')}")
                for tc in event.get("toolCalls", []):
                    # print(f"   Tool: {tc['name']}")
                    # print(f"   Args: {tc['args']}\n")
                    pass
                    
            elif event_type == "content":
                text = event.get("text", "")
                if text.strip():
                     print(f"\n💬 [Content]: {text}")
                    
            elif event_type == "tool-result":
                res = str(event.get('result'))
                print(f"\n✅ [Tool Result] {event.get('toolName')}: {res[:200]}...")
            
    except Exception as e:
        print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
