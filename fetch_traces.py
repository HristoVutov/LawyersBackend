import os
import json
from langsmith import Client
from dotenv import load_dotenv

load_dotenv()

client = Client()
project_name = "lawyers-dashboard"

print(f"Fetching last 2 root runs from project '{project_name}'...")
# 1. Fetch root runs to identify the traces
runs = list(client.list_runs(
    project_name=project_name, 
    limit=2, 
    is_root=True,
    filter='neq(status, "pending")' # Optional: only finished runs
))

if not runs:
    print("No runs found.")
else:
    print(f"Found {len(runs)} root runs.")
    for i, root_run in enumerate(runs):
        trace_id = root_run.trace_id
        print(f"Fetching full trace for Trace ID: {trace_id}...")
        
        # 2. Fetch all runs sharing this trace_id to get the full tree
        # This returns a flat list of all runs (root + children + grandchildren)
        all_runs_in_trace = list(client.list_runs(trace_id=trace_id))
        
        # Convert to list of dicts
        trace_data_flat = []
        for r in all_runs_in_trace:
            trace_data_flat.append({
                "id": str(r.id),
                "name": r.name,
                "parent_run_id": str(r.parent_run_id) if r.parent_run_id else None,
                "type": r.run_type,
                "status": r.status,
                "error": r.error,
                "inputs": r.inputs,
                "outputs": r.outputs,
                "start_time": str(r.start_time),
                "end_time": str(r.end_time),
                "events": r.events,
                "metadata": r.extra.get("metadata") if r.extra else {},
                "tags": r.tags
            })
            
        full_trace_export = {
            "root_run_id": str(root_run.id),
            "trace_id": str(trace_id),
            "runs": trace_data_flat,
            "summary": "Full trace export containing all child runs."
        }
        
        filename = f"traces/fetched_trace_{i+1}_full.json"
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(full_trace_export, f, indent=2, default=str)
        print(f"Saved full trace {i+1} to {filename} ({len(trace_data_flat)} total runs in trace)")
