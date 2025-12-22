"""
Test script for Virtual File System with real project.

Sets up the VFS with Test1 project and runs a query through the orchestrator.
"""
import asyncio
import os
import sys

# Ensure the app package is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


async def main():
    """Run test with Test1 project."""
    from app.middleware.virtual_fs import set_project_path, get_vfs
    from app.agents import OrchestratorAgent, initialize_agent_registry
    from app.agents.base_agent import AgentConfig
    
    # --- Setup ---
    PROJECT_PATH = r"C:\Users\vutov\Documents\LawyersProjects\Test1"
    
    print("=" * 60)
    print("🧪 Virtual File System Test")
    print("=" * 60)
    
    # Set project path
    print(f"\n📁 Setting project path: {PROJECT_PATH}")
    set_project_path(PROJECT_PATH)
    
    # Show mounts
    vfs = get_vfs()
    print("\n📂 Virtual Mounts:")
    for vpath, info in vfs.list_mounts().items():
        perm = "RW" if info["permission"] == "read_write" else "R"
        exists = "✓" if info["exists"] else "✗"
        print(f"   {vpath} -> {info['real_path']} [{perm}] {exists}")
    
    # --- Initialize Agents ---
    print("\n🤖 Initializing agents...")
    initialize_agent_registry()
    
    # Create orchestrator with Gemini 3 Pro Preview
    orchestrator = OrchestratorAgent()
    # Override model to gemini-3-pro-preview (same as JS agents)
    orchestrator.model_name = "gemini-3-pro-preview"
    orchestrator._llm = None  # Reset to force re-init with new model
    
    print(f"   Model: {orchestrator.model_name}")
    
    # --- Run Query ---
    query = """намери имотите на галакси, анализирай за всички под имоти който имаме,
направи план за отдаване под наем хотелските стай в royal garden и апартаментите в аирбнб и боокинг платформите"""
    
    print("\n" + "=" * 60)
    print("📨 Query:")
    print(query)
    print("=" * 60)
    
    # Stream response
    print("\n🔄 Processing...\n")
    
    # Collect all events for analysis
    all_events = []
    tool_calls_count = 0
    delegate_calls = []
    content_chunks = []
    
    async for chunk in orchestrator.stream(query, thread_id="test-vfs-1"):
        chunk_type = chunk.get("type", "")
        agent_name = chunk.get("agent", "unknown")
        all_events.append(chunk)
        
        if chunk_type == "content":
            text = chunk.get("text", "")
            # Normalize text to string
            if isinstance(text, list):
                text = " ".join(str(t) for t in text)
            elif not isinstance(text, str):
                text = str(text)
            content_chunks.append(text)
            # Print content inline (limited)
            if len(text) < 200:
                print(text, end="", flush=True)
        
        elif chunk_type == "tool-call":
            tool_calls_count += 1
            for tc in chunk.get("toolCalls", []):
                tool_name = tc['name']
                args = tc.get("args", {})
                
                print(f"\n\n🔧 [{agent_name}] Tool #{tool_calls_count}: {tool_name}")
                
                # Track delegate_task calls
                if tool_name == "delegate_task":
                    target = args.get("agent_name", "?")
                    task_desc = str(args.get("task", ""))[:100]
                    delegate_calls.append({"agent": target, "task": task_desc})
                    print(f"   → Delegating to: {target}")
                    print(f"   → Task: {task_desc}...")
                elif tool_name == "plan_task":
                    goal = args.get("goal", "")[:80]
                    tasks = args.get("tasks", [])
                    print(f"   → Goal: {goal}")
                    print(f"   → Tasks: {len(tasks)}")
                else:
                    args_str = str(args)[:150]
                    print(f"   → Args: {args_str}...")
        
        elif chunk_type == "tool-result":
            result = str(chunk.get("result", ""))[:200]
            print(f"\n   📋 Result: {result}...")
        
        elif chunk_type == "done":
            print(f"\n\n✅ Done [{agent_name}]")
        
        elif chunk_type == "error":
            print(f"\n❌ Error: {chunk.get('error', 'Unknown error')}")
    
    # Analysis summary
    print("\n" + "=" * 60)
    print("📊 ANALYSIS SUMMARY")
    print("=" * 60)
    print(f"Total tool calls: {tool_calls_count}")
    print(f"Delegate calls: {len(delegate_calls)}")
    
    if delegate_calls:
        print("\n🔀 Delegations made:")
        for i, d in enumerate(delegate_calls, 1):
            print(f"   {i}. → {d['agent']}: {d['task'][:60]}...")
    else:
        print("\n⚠️  NO DELEGATE_TASK CALLS - Agent returned text directly!")
    
    # Check if final response was just text
    final_text = "".join(content_chunks)
    if len(final_text) > 500 and tool_calls_count == 0:
        print("\n⚠️  Agent returned long text response without using tools!")
    
    print("\n" + "=" * 60)
    print("🏁 Test Complete")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
