from langsmith import Client
import os
from dotenv import load_dotenv

load_dotenv()

EXPERIMENT = "lawyers-comprehensive-eval-c1f15663"

def check():
    client = Client()
    runs = list(client.list_runs(project_name=EXPERIMENT, execution_order=1))
    
    total_faith = 0
    count = 0
    
    for run in runs:
        if run.feedback_stats and "faithfulness" in run.feedback_stats:
            score = run.feedback_stats["faithfulness"].get("avg")
            if score is not None:
                total_faith += score
                count += 1
                
    if count > 0:
        avg_faith = total_faith / count
        print(f"Experiment: {EXPERIMENT}")
        print(f"Runs with Feedback: {count}")
        print(f"Average Faithfulness: {avg_faith:.4f}")
    else:
        print("No faithfulness scores found yet.")

if __name__ == "__main__":
    check()
