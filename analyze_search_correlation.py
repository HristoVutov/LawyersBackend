import argparse
from langsmith import Client
from dotenv import load_dotenv

load_dotenv()

EXPERIMENT_NAME = "lawyers-comprehensive-eval-71b6af92"

def analyze_correlation():
    print(f"--- Analyzing '{EXPERIMENT_NAME}' for Search Quality vs Faithfulness ---")
    client = Client()
    runs = list(client.list_runs(project_name=EXPERIMENT_NAME, execution_order=1))
    
    analysis_data = []
    
    for run in runs:
        if not run.feedback_stats:
            continue
            
        feedbacks = list(client.list_feedback(run_ids=[run.id]))
        
        search_score = None
        faith_score = None
        rel_score = None
        
        search_comment = ""
        faith_comment = ""
        
        for fb in feedbacks:
            if fb.key == "search_quality":
                search_score = fb.score
                search_comment = fb.comment
            elif fb.key == "faithfulness":
                faith_score = fb.score
                faith_comment = fb.comment
            elif fb.key == "relevance":
                rel_score = fb.score
        
        if search_score is not None:
            analysis_data.append({
                "input": run.inputs.get("input", "N/A"),
                "search": search_score,
                "faith": faith_score,
                "relevance": rel_score,
                "search_comment": search_comment,
                "faith_comment": faith_comment
            })
            
    # Sort by Search Quality (High to Low)
    analysis_data.sort(key=lambda x: x["search"], reverse=True)
    
    print(f"Found {len(analysis_data)} runs with search quality data.")
    
    with open("search_analysis.txt", "w", encoding="utf-8") as f:
        f.write(f"Analysis of {len(analysis_data)} runs with search quality scores.\n\n")
        
        f.write("=== HIGH Search Quality (> 0.8) ===\n")
        for d in analysis_data:
            if d["search"] > 0.8:
                f.write(f"Input: {d['input']}\n")
                f.write(f"Scores -> Search: {d['search']}, Faith: {d['faith']}, Rel: {d['relevance']}\n")
                f.write(f"Search Comment: {d['search_comment']}\n")
                f.write(f"Faith Comment: {d['faith_comment']}\n")
                f.write("-" * 40 + "\n")

        f.write("\n=== LOW Search Quality (< 0.5) ===\n")
        for d in analysis_data:
            if d["search"] < 0.5:
                f.write(f"Input: {d['input']}\n")
                f.write(f"Scores -> Search: {d['search']}, Faith: {d['faith']}, Rel: {d['relevance']}\n")
                f.write(f"Search Comment: {d['search_comment']}\n")
                f.write("-" * 40 + "\n")
    print("Analysis written to search_analysis.txt")

if __name__ == "__main__":
    analyze_correlation()

