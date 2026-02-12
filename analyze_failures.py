import argparse
from langsmith import Client
from dotenv import load_dotenv

load_dotenv()

EXPERIMENT_NAME = "lawyers-comprehensive-eval-4c7f6ffc"

def analyze_failures():
    print(f"--- Analyzing failures for '{EXPERIMENT_NAME}' ---")
    client = Client()
    runs = list(client.list_runs(project_name=EXPERIMENT_NAME, execution_order=1))
    
    failures = []
    
    for run in runs:
        if not run.feedback_stats:
            continue
            
        # Check for low scores directly from feedback_stats if possible, 
        # but feedback_stats aggregates. We need the actual feedback entries.
        # So we use client.list_feedback
        
        feedbacks = list(client.list_feedback(run_ids=[run.id]))
        
        run_issues = []
        for fb in feedbacks:
            if fb.score is not None and fb.score < 0.6: # Threshold for "failure"
                run_issues.append(f"{fb.key}: {fb.score} - {fb.comment}")
        
        if run_issues:
            # Get input/output for context
            inputs = run.inputs.get("input", "N/A")
            outputs = run.outputs.get("output", "N/A")
            failures.append({
                "input": inputs,
                "output": outputs,
                "issues": run_issues
            })
            

    # Report to file
    with open("failure_analysis.txt", "w", encoding="utf-8") as f:
        if not failures:
            f.write("No specific low-scoring examples found (checked < 0.6).\n")
        else:
            f.write(f"Found {len(failures)} problematic examples:\n\n")
            for i, fail in enumerate(failures):
                f.write(f"Example {i+1}:\n")
                f.write(f"  Input: {fail['input']}\n")
                f.write(f"  Output: {fail['output']}\n")
                f.write("  Issues:\n")
                for issue in fail['issues']:
                    f.write(f"    - {issue}\n")
                f.write("-" * 40 + "\n")


if __name__ == "__main__":
    analyze_failures()
