import argparse
from collections import defaultdict
from langsmith import Client
from dotenv import load_dotenv

load_dotenv()

EXPERIMENTS = [
    "lawyers-comprehensive-eval-c1f15663", # New retrieval fix
    "lawyers-comprehensive-eval-71b6af92", # Previous (High Search, Low Faithfulness)
    "lawyers-comprehensive-eval-4c7f6ffc"  # Original Baseline
]

def analyze_experiments():
    client = Client()
    
    results = []
    
    print(f"{'Experiment':<40} | {'Faithful':<8} | {'Relevance':<9} | {'Coherence':<9} | {'SearchQual':<10} | {'Runs':<5}")
    print("-" * 100)

    for exp_name in EXPERIMENTS:
        runs = list(client.list_runs(project_name=exp_name, execution_order=1))
        
        if not runs:
            print(f"{exp_name:<40} | {'N/A':<8} | {'N/A':<9} | {'N/A':<9} | {'N/A':<10} | 0")
            continue
            
        metrics = defaultdict(list)
        for run in runs:
            if run.feedback_stats:
                for key, stats in run.feedback_stats.items():
                    if isinstance(stats, dict) and "avg" in stats:
                        metrics[key].append(stats["avg"])
        
        # Calculate averages
        def get_avg(key):
            if metrics[key]:
                return f"{sum(metrics[key]) / len(metrics[key]):.2f}"
            return "N/A"
            
        avg_faith = get_avg("faithfulness")
        avg_rel = get_avg("relevance")
        avg_coh = get_avg("coherence")
        avg_search = get_avg("search_quality")
            
        line = f"{exp_name:<40} | {avg_faith:<8} | {avg_rel:<9} | {avg_coh:<9} | {avg_search:<10} | {len(runs)}"
        print(line)
        results.append(line)

    with open("latest_comparison.txt", "w", encoding="utf-8") as f:
        f.write(f"{'Experiment':<40} | {'Faithful':<8} | {'Relevance':<9} | {'Coherence':<9} | {'SearchQual':<10} | {'Runs':<5}\n")
        f.write("-" * 100 + "\n")
        for line in results:
            f.write(line + "\n")

if __name__ == "__main__":
    analyze_experiments()
