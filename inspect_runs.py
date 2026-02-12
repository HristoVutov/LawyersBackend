import argparse
import os
from langsmith import Client
from dotenv import load_dotenv

load_dotenv()

def inspect_run(project_name: str):
    print(f"--- Inspecting runs for project: '{project_name}' ---")
    
    client = Client()
    runs = list(client.list_runs(project_name=project_name, execution_order=1))
    
    if not runs:
        print(f"No runs found.")
        return

    print(f"Found {len(runs)} runs.")
    
    # Find first run with outputs
    completed_runs = [r for r in runs if r.outputs]
    if not completed_runs:
        print("No completed runs found yet.")
        return

    first_run = completed_runs[0]
    print(f"Inspecting Run ID: {first_run.id}")
    outputs = first_run.outputs or {}
    print(f"Outputs keys: {list(outputs.keys())}")
    if "tool_trace" in outputs:
        print(f"Tool Trace found: {len(outputs['tool_trace'])} items")
        print(outputs['tool_trace'])
    else:
        print("❌ Tool Trace NOT found in outputs.")

    print(f"Feedback Stats: {first_run.feedback_stats}")
    
    if first_run.feedback_stats and "search_quality" in first_run.feedback_stats:
        print(f"✅ 'search_quality' metric found: {first_run.feedback_stats['search_quality']}")
    else:
        print("❌ 'search_quality' metric NOT found.")

    # Check if there are any feedback stats in any run
    count_with_feedback = sum(1 for r in runs if r.feedback_stats)
    print(f"Runs with feedback: {count_with_feedback}/{len(runs)}")

if __name__ == "__main__":
    inspect_run("lawyers-comprehensive-eval-71b6af92")



