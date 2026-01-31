import asyncio
import json
import operator
from typing import Annotated, TypedDict
# Adjust path to allow imports from app
import sys
import os
sys.path.append(os.getcwd())

from langchain_core.messages import AIMessage, ToolMessage, BaseMessage
# Mocking get_agent before importing Orchestrator if possible, 
# but python imports execute immediately. 
# Better to patch it after import.

from app.agents.orchestrator import OrchestratorAgent

# Mock Agent for testing
class MockAgent:
    class Graph:
        async def ainvoke(self, state, config):
            print(" [Mock] Sub-agent execution simulating add_to_context...")
            return {
                "messages": [
                    AIMessage(content="I will add a file to context.", name="mock_agent"),
                    ToolMessage(
                        content=json.dumps({
                            "action": "add_to_context",
                            "files": ["/mock/file.txt"],
                            "summary": "A mock file for testing"
                        }),
                        name="add_to_context",
                        tool_call_id="call_test_123"
                    ),
                    AIMessage(content="Done adding context.", name="mock_agent")
                ]
            }
    graph = Graph()

async def main():
    print("--- Starting Verification: Context Persistence ---")
    
    # 1. Patch get_agent to return our MockAgent
    import app.agents.orchestrator as orch_module
    original_get_agent = orch_module.get_agent
    orch_module.get_agent = lambda name: MockAgent()
    
    try:
        # 2. Instantiate Orchestrator
        orch = OrchestratorAgent()
        
        # 3. Create the worker node function
        # We use "mock_agent" as the name
        worker_node_func = orch._make_worker_node("mock_agent")
        
        # 4. Prepare Mock State
        # Note: We must respect the TypedDict structure roughly
        initial_state = {
            "messages": [],
            "task_list": [{"id": "t1", "description": "test task", "status": "in_progress"}],
            "current_task_id": "t1",
            "project_context": [] 
        }
        
        # 5. Run the node
        print("--- Executing Worker Node ---")
        result = await worker_node_func(initial_state, {"configurable": {}})
        
        # 6. Verify Result
        print("\n--- Verification Results ---")
        if "project_context" not in result:
             print("❌ FAILED: 'project_context' key missing from result.")
        else:
             ctx = result["project_context"]
             print(f"Captured Context: {ctx}")
             
             if len(ctx) == 1 and ctx[0]["path"] == "/mock/file.txt":
                 print("✅ SUCCESS: Context correctly captured from tool output.")
             else:
                 print("❌ FAILED: Context content verification failed.")

    except Exception as e:
        print(f"❌ ERROR details: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Restore (good practice)
        orch_module.get_agent = original_get_agent

if __name__ == "__main__":
    asyncio.run(main())
