import argparse
import os
from collections import defaultdict
from langsmith import Client
from dotenv import load_dotenv

# Load environment variables (LANGCHAIN_API_KEY, etc.)
load_dotenv()

def list_experiments(filter_name: str = "lawyers-comprehensive-eval"):
    print(f"--- Searching for experiments matching '{filter_name}' ---")
    client = Client()
    try:
        projects = list(client.list_projects())
    except Exception as e:
        print(f"Error listing projects: {e}")
        return []

    experiments = [p for p in projects if filter_name in p.name]
    
    if not experiments:
        print(f"No experiments found matching '{filter_name}'.")
        return []
    
    # Sort by start_time descending (newest first)
    # Note: Attribute is likely start_time based on debug output
    experiments.sort(key=lambda x: getattr(x, 'start_time', getattr(x, 'created_at', 0)), reverse=True)
    
    print(f"Found {len(experiments)} matching experiments:")
    for i, exp in enumerate(experiments):
        # timestamp = getattr(exp, 'start_time', getattr(exp, 'created_at', 'N/A'))
        print(f"{i+1}. {exp.name}") 
        
    return experiments

def get_experiment_results(project_name: str):
    print(f"\n--- Fetching results for project/experiment: '{project_name}' ---")
    
    client = Client()
    
    # Fetch runs for the project
    runs = list(client.list_runs(project_name=project_name, execution_order=1))
    
    if not runs:
        print(f"⚠️ No runs found for project '{project_name}'.")
        return

    print(f"Found {len(runs)} runs.")
    
    # Aggregate metrics
    metrics = defaultdict(list)
    
    for run in runs:
        if run.feedback_stats:
            for key, stats in run.feedback_stats.items():
                if isinstance(stats, dict) and "avg" in stats:
                     metrics[key].append(stats["avg"])
    
    print("\n--- AGGREGATE METRICS ---")
    if not metrics:
        print("No feedback metrics found on these runs.")
    else:
        # Print header
        print(f"{'Metric':<20} | {'Average':<10} | {'Count':<10}")
        print("-" * 46)
        
        for key, values in metrics.items():
            avg_score = sum(values) / len(values)
            print(f"{key:<20} | {avg_score:<10.2f} | {len(values):<10}")

    print(f"\nView detailed trace: https://smith.langchain.com/o/{os.environ.get('LANGCHAIN_PROJECT_ID')}/projects/{project_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch LangSmith experiment results via CLI.")
    parser.add_argument("project_name", type=str, nargs="?", help="The name of the project/experiment in LangSmith. If omitted, lists available experiments.")
    parser.add_argument("--latest", action="store_true", help="Automatically fetch results for the latest experiment found.")
    
    args = parser.parse_args()
    
    if args.project_name:
        get_experiment_results(args.project_name)
    else:
        experiments = list_experiments()
        if args.latest and experiments:
            latest_exp = experiments[0]
            get_experiment_results(latest_exp.name)
        elif not experiments:
             pass 
        else:
            print("\nUse `python get_experiment_results.py <experiment_name>` or `python get_experiment_results.py --latest` to see results.")
