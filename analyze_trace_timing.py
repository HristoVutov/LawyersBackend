import json
import sys
from datetime import datetime
from typing import Dict, List, Any

def parse_time(time_str: str) -> datetime:
    # Handle potentially different formats or timezone info
    # 2026-01-31 12:25:08.708258
    # 2026-01-31T12:25:08.708258+00:00
    try:
        if 'T' in time_str:
            return datetime.fromisoformat(time_str)
        else:
            return datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S.%f")
    except ValueError:
        # Fallback for truncated microseconds or other formats
        try: 
             return datetime.strptime(time_str.split('+')[0], "%Y-%m-%d %H:%M:%S.%f")
        except:
             return datetime.strptime(time_str.split('+')[0], "%Y-%m-%d %H:%M:%S")

def calculate_duration(start: str, end: str) -> float:
    if not start or not end:
        return 0.0
    s = parse_time(start)
    e = parse_time(end)
    return (e - s).total_seconds()

def analyze_run(run: Dict[str, Any], depth=0):
    start_time = run.get('start_time')
    end_time = run.get('end_time')
    name = run.get('name')
    run_type = run.get('type')
    metadata = run.get('metadata', {})
    node = metadata.get('langgraph_node')
    
    # Context Analysis
    usage = run.get('usage_metadata', {})
    input_tokens = usage.get('input_tokens', 0)
    output_tokens = usage.get('output_tokens', 0)
    
    # Fallback if no token usage (estimate from inputs)
    if not input_tokens and run.get('inputs'):
        input_len = len(str(run['inputs']))
        input_tokens = f"~{input_len // 4} chars" # Rough proxy or just chars
    
    duration = calculate_duration(start_time, end_time)
    
    # Categorize
    category = "UNKNOWN"
    if node:
        category = f"NODE: {node}"
    elif run_type == "llm":
        category = "LLM"
    elif run_type == "tool":
        category = "TOOL"
    elif run_type == "chain":
        category = "CHAIN"
    
    # Print if interesting
    prefix = "  " * depth
    # Only print if significant duration OR significant context
    if duration > 0.05 or (isinstance(input_tokens, int) and input_tokens > 100): 
        token_info = f" | In: {input_tokens} / Out: {output_tokens}"
        print(f"{prefix}[{category}] {name}: {duration:.2f}s{token_info}")

    # Recurse if children exist (structure depends on trace format, flat list in file?)
    # The file seems to be a list of runs. We need to reconstruct hierarchy if possible, 
    # but for now let's just print the list flattened or grouped by parent_run_id if we want smarts.
    # Given the previous context, the file is a flat list of ALL runs in the trace.
    
    return {
        "id": run.get("id"),
        "parent_id": run.get("parent_run_id"),
        "name": name,
        "category": category,
        "duration": duration,
        "start": start_time,
        "depth": depth
    }

def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_trace_timing.py <trace_file>")
        sys.exit(1)
        
    trace_file = sys.argv[1]
    
    try:
        with open(trace_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        if isinstance(data, dict) and "runs" in data:
            runs = data["runs"]
        elif isinstance(data, list):
            runs = data
        else:
            print("Unknown JSON format. Expected list or dict with 'runs' key.")
            return
            
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    print(f"loaded {len(runs)} runs.")
    
    # Reconstruct Hierarchy to pretty print
    runs_by_id = {r['id']: r for r in runs}
    children_map = {}
    
    for r in runs:
        parent = r.get('parent_run_id')
        if parent:
            if parent not in children_map:
                children_map[parent] = []
            children_map[parent].append(r)
            
    # Find roots (runs with no parent IN THIS TRACE)
    # Note: 'fetched_trace_1_full.json' might contain one top level trace or multiple.
    
    roots = [r for r in runs if not r.get('parent_run_id')]
    
    if not roots:
        # If no roots found (maybe partial trace?), fallback to processing all
        roots = runs 
        # But usually there IS a root. Let's find nodes that have parents NOT in the dataset?
        ids = set(runs_by_id.keys())
        roots = [r for r in runs if r.get('parent_run_id') not in ids]

    def print_tree(nodes, depth=0):
        # Sort by start time
        nodes.sort(key=lambda x: x.get('start_time') or "")
        
        for node in nodes:
            info = analyze_run(node, depth)
            
            # Recurse
            children = children_map.get(node['id'], [])
            print_tree(children, depth + 1)

    print("\n--- Execution Timing Analysis ---\n")
    print_tree(roots)

if __name__ == "__main__":
    main()
