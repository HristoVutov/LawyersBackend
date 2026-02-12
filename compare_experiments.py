import argparse
from collections import defaultdict
from langsmith import Client
from dotenv import load_dotenv


load_dotenv()

EXPERIMENTS = [
    "lawyers-comprehensive-eval-4c7f6ffc", # Latest
    "lawyers-comprehensive-eval-51188a4c",
    "lawyers-comprehensive-eval-c5d4a1af",
    "lawyers-comprehensive-eval-211f615e"
]

def analyze_experiments():
    client = Client()
    
    results = []
    
    print(f"{'Experiment':<40} | {'Faithfulness':<12} | {'Relevance':<12} | {'Coherence':<12} | {'Runs':<5}")
    print("-" * 90)

    for exp_name in EXPERIMENTS:
        runs = list(client.list_runs(project_name=exp_name, execution_order=1))
        
        if not runs:
            print(f"{exp_name:<40} | {'N/A':<12} | {'N/A':<12} | {'N/A':<12} | 0")
            continue
            
        metrics = defaultdict(list)
        for run in runs:
            if run.feedback_stats:
                for key, stats in run.feedback_stats.items():
                    if isinstance(stats, dict) and "avg" in stats:
                        metrics[key].append(stats["avg"])
        
        # Calculate averages
        avg_faith = "N/A"
        avg_rel = "N/A"
        avg_coh = "N/A"
        
        if metrics["faithfulness"]:
            avg_faith = f"{sum(metrics['faithfulness']) / len(metrics['faithfulness']):.2f}"
            
        if metrics["relevance"]:
            avg_rel = f"{sum(metrics['relevance']) / len(metrics['relevance']):.2f}"
            
        if metrics["coherence"]:
            avg_coh = f"{sum(metrics['coherence']) / len(metrics['coherence']):.2f}"
            

        line = f"{exp_name:<40} | {avg_faith:<12} | {avg_rel:<12} | {avg_coh:<12} | {len(runs)}"
        print(line)
        results.append(line)

    with open("comparison_results.txt", "w", encoding="utf-8") as f:
        f.write(f"{'Experiment':<40} | {'Faithfulness':<12} | {'Relevance':<12} | {'Coherence':<12} | {'Runs':<5}\n")
        f.write("-" * 90 + "\n")
        for line in results:
            f.write(line + "\n")


if __name__ == "__main__":
    analyze_experiments()
