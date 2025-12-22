"""
Test script to verify streaming thinking events.

Run with: python test_streaming_events.py
"""
import asyncio
import json


async def main():
    from app.agents.orchestrator import OrchestratorAgent, initialize_agent_registry
    
    print("=" * 60)
    print("Initializing agent registry...")
    initialize_agent_registry()
    
    print("Creating orchestrator...")
    orchestrator = OrchestratorAgent()
    
    print("=" * 60)
    print("Testing streaming events with a simple query...")
    print("=" * 60)
    
    # Count events by type
    event_counts = {}
    all_events = []
    
    message = "What documents do we have about Galaxy?"
    
    async for event in orchestrator.stream(message, thread_id="test_streaming"):
        event_type = event.get("type", "unknown")
        event_counts[event_type] = event_counts.get(event_type, 0) + 1
        all_events.append(event)
        
        # Pretty print each event
        agent = event.get("agent", "?")
        step = event.get("step", "")
        
        if event_type == "step":
            print(f"\n🔄 [{agent}] Step: {event.get('step')} - {event.get('description')}")
        elif event_type == "thinking":
            text = event.get("text", "")[:50]
            print(f"💭 [{agent}] {text}", end="", flush=True)
        elif event_type == "thinking-complete":
            print(f"\n✅ [{agent}] Thinking complete")
        elif event_type == "content":
            text = event.get("text", "")[:50]
            print(f"📝 [{agent}] {text}", end="", flush=True)
        elif event_type == "tool-call":
            tool = event.get("tool", "unknown")
            print(f"\n🔧 [{agent}] Tool call: {tool}")
        elif event_type == "tool-result":
            tool = event.get("tool", "unknown")
            result = str(event.get("result", ""))[:100]
            print(f"📦 [{agent}] Tool result ({tool}): {result[:50]}...")
        elif event_type == "agent-start":
            print(f"\n🚀 [{agent}] Agent starting: {event.get('task', '')}")
        elif event_type == "agent-end":
            print(f"\n🏁 [{agent}] Agent completed")
        elif event_type == "done":
            print(f"\n\n✨ Done!")
        elif event_type == "error":
            print(f"\n❌ Error: {event.get('error')}")
    
    print("\n" + "=" * 60)
    print("Event Summary:")
    print("=" * 60)
    for event_type, count in sorted(event_counts.items()):
        print(f"  {event_type}: {count}")
    
    print(f"\nTotal events: {len(all_events)}")
    

if __name__ == "__main__":
    asyncio.run(main())
