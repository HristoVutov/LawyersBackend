import argparse
from langsmith import Client
from dotenv import load_dotenv

load_dotenv()

EXPERIMENT_NAME = "lawyers-comprehensive-eval-71b6af92"

def verify_traces():
    print(f"--- Verifying Traces for '{EXPERIMENT_NAME}' ---")
    client = Client()
    runs = list(client.list_runs(project_name=EXPERIMENT_NAME, execution_order=1))
    
    with open("trace_verification.txt", "w", encoding="utf-8") as f:
        f.write(f"Trace Verification for {len(runs)} runs.\n")
        f.write("Hypothesis Check: Good Query + Bad Context = Low Faithfulness\n\n")
        
        for run in runs:
            if not run.outputs:
                continue
                
            # Inputs/Outputs
            question = run.inputs.get("input", "N/A")
            answer = run.outputs.get("output", "N/A")
            context = run.outputs.get("retrieved_context", [])
            tool_trace = run.outputs.get("tool_trace", [])
            
            # Scores
            scores = {}
            if run.feedback_stats:
                for k, v in run.feedback_stats.items():
                    if isinstance(v, dict) and "avg" in v:
                        scores[k] = v["avg"]
                        
            # Format Context (Critical!)
            context_str = "No context retrieved."
            if context:
                # Join context items and truncate
                full_context = "\n".join([str(c) for c in context])
                if len(full_context) > 500:
                    context_str = full_context[:500] + "... [TRUNCATED]"
                else:
                    context_str = full_context
            
            # Format Tool Trace
            trace_str = "No tool calls."
            if tool_trace:
                trace_lines = []
                for t in tool_trace:
                    args = t.get("args")
                    query = args.get("query") if isinstance(args, dict) else str(args)
                    trace_lines.append(f"Tool: {t['tool']} | Query: {query}")
                trace_str = "\n".join(trace_lines)

            # Write to file
            f.write(f"Question: {question}\n")
            f.write(f"Scores: Faith={scores.get('faithfulness', 'N/A')}, Search={scores.get('search_quality', 'N/A')}, Rel={scores.get('relevance', 'N/A')}\n")
            f.write("-" * 20 + " TRACE " + "-" * 20 + "\n")
            f.write(f"[Search Action]:\n{trace_str}\n\n")
            f.write(f"[Retrieved Context] (What the LLM actually saw):\n{context_str}\n\n")
            f.write("-" * 20 + " OUTPUT " + "-" * 20 + "\n")
            f.write(f"[Generated Answer]:\n{answer[:300]}... [TRUNCATED]\n")
            f.write("=" * 80 + "\n\n")

    print(f"Verification report written to `trace_verification.txt`")

if __name__ == "__main__":
    verify_traces()
