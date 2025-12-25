import asyncio
import os
import shutil
from pathlib import Path
from app.agents.drafting_agent import DraftingAgent
from langchain_core.messages import HumanMessage
from app.api.websocket import manager

# Mock websocket manager to prevent errors
class MockManager:
    async def send_json_to_thread(self, *args, **kwargs):
        pass

manager.send_json_to_thread = MockManager().send_json_to_thread

async def verify_document_generation():
    print(">>> Starting Document Generation Verification...")
    
    
    # 1. Setup
    from app.middleware.virtual_fs import set_project_path
    cwd = os.getcwd()
    set_project_path(cwd)
    print(f">>> VFS Initialized with: {cwd}")
    
    agent = DraftingAgent()
    thread_id = "test_thread_123"
    output_dir = Path(".generatedFiles")
    thread_dir = output_dir / thread_id
    
    # Clean up previous run
    if thread_dir.exists():
        shutil.rmtree(thread_dir)
        print(f">>> Cleaned up {thread_dir}")
    
    # 2. Run Agent
    task = "Analyze the difference between the provided checklist and the local documents regarding property X. Create a comparison memo."
    
    config = {
        "configurable": {
            "thread_id": thread_id,
            "model_name": "gemini-3-pro-preview" 
        }
    }
    
    inputs = {
        "messages": [HumanMessage(content=task)],
        "task_description": task
    }
    
    print(f">>> invoking agent with task: '{task}'...")
    final_state = await agent.graph.ainvoke(inputs, config)
    
    # 3. Assertions
    print("\n>>> Verifying Results...")
    
    # Check Intent
    intent = final_state.get("intent")
    doc_type = final_state.get("document_type")
    print(f"   Intent: {intent}")
    print(f"   Type: {doc_type}")
    
    if not doc_type:
        print("FAILED: document_type not extracted.")
    else:
        print("PASSED: Intent extraction passed.")

    # Check Files
    files = list(thread_dir.glob("*"))
    print(f"   Files found in {thread_dir}: {[f.name for f in files]}")
    
    md_files = list(thread_dir.glob("*.md"))
    docx_files = list(thread_dir.glob("*.docx"))
    
    if not md_files:
        print("FAILED: No Markdown file generated.")
    else:
        print(f"PASSED: Markdown file found: {md_files[0].name}")
        
    if not docx_files:
        print("FAILED: No DOCX file generated.")
    else:
        print(f"PASSED: DOCX file found: {docx_files[0].name}")

    if md_files and docx_files:
        print("\n>>> VERIFICATION SUCCESSFUL! Both formats generated.")
    else:
        print("\n>>> VERIFICATION FAILED.")

if __name__ == "__main__":
    asyncio.run(verify_document_generation())
