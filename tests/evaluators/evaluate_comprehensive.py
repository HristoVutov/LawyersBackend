import asyncio
import os
import argparse
import json
from typing import Optional, Any
from datetime import datetime

from langsmith import Client, evaluate, aevaluate
from langsmith.schemas import Run, Example
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from dotenv import load_dotenv
from app.tracing import setup_tracing

load_dotenv()
setup_tracing()

# Import your agent
from app.agents.research_agent import ResearchAgent
from app.config import get_settings

# --- Configuration ---
DATASET_NAME = "D1"
EXPERIMENT_PREFIX = "lawyers-comprehensive-eval"

# Initialize LLM for Judges (Evaluators)
def get_evaluator_llm():
    settings = get_settings()
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=settings.google_api_key,
        temperature=0
    )


# --- 1. Define the Target (System under Test) ---
async def target(inputs: dict) -> dict:
    """
    Wraps the ResearchAgent to accept dataset inputs and return outputs.
    Adapts 'input' from dataset to the agent's expected input.
    """
    question = inputs.get("input") or inputs.get("question") or inputs.get("query")
    
    # Handle 'messages' list (common in chat datasets)
    if not question and "messages" in inputs:
        messages = inputs["messages"]
        if isinstance(messages, list) and messages:
            # Try to find the last user message
            for msg in reversed(messages):
                if isinstance(msg, dict):
                    role = msg.get("role") or msg.get("type")
                    if role in ["user", "human"]:
                        question = msg.get("content")
                        break
                elif hasattr(msg, "content"): # LangChain Message object
                    if getattr(msg, "type", "") in ["human", "user"] or isinstance(msg, HumanMessage):
                         question = msg.content
                         break
            
            # Fallback to last message content if no user message identified
            if not question:
                last_msg = messages[-1]
                if isinstance(last_msg, dict):
                    question = last_msg.get("content")
                elif hasattr(last_msg, "content"):
                    question = last_msg.content

    if not question:
        # Fallback: use the first string value found
        for k, v in inputs.items():
            if isinstance(v, str):
                question = v
                break
    
    if not question:
        raise ValueError(f"Dataset input key missing. Available keys: {list(inputs.keys())}")

    agent = ResearchAgent()
    
    # Run the agent
    # We use a distinct thread_id to avoid polluting production threads
    thread_id = f"eval_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    
    try:
        # Assuming agent.graph.ainvoke or similar entry point
        # ResearchAgent expects 'messages' or 'task_description'
        result = await agent.graph.ainvoke(
            {
                "messages": [HumanMessage(content=question)], 
                "task_description": question
            },
            config={"configurable": {"thread_id": thread_id}}
        )
        
        # Extract the final answer and potentially retrieved documents
        # Adjust based on your agent's actual output schema
        final_answer = result.get("draft_answer") or result.get("final_response") or "No answer generated."
        

        # Extract legal references and found documents
        legal_refs = result.get("legal_references", "")
        found_docs = result.get("found_documents", "")
        
        # Extract tool usage traces for evaluation
        gather_msgs = result.get("gather_messages", [])
        tool_trace = []
        for msg in gather_msgs:
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    tool_trace.append({
                        "tool": tc["name"],
                        "args": tc["args"]
                    })
        
        # Combine into a list of context strings
        context = []
        if legal_refs:
            context.append(f"Legal References:\n{legal_refs}")
        if found_docs:
            context.append(f"Found Documents:\n{found_docs}")
        
        return {
            "output": final_answer,
            "retrieved_context": context,
            "tool_trace": tool_trace # Passed to evaluators
        }
    except Exception as e:
        return {
            "output": f"Error executing agent: {str(e)}",
            "retrieved_context": [],
            "tool_trace": []
        }


# --- 2. Define Evaluators (LLM-as-a-Judge) ---

async def search_quality_evaluator(run: Run, example: Example) -> dict:
    """
    Evaluates if the agent used search tools effectively.
    Checks if:
    1. Search tools were called.
    2. Search queries were relevant to the question.
    """
    input_question = example.inputs.get("input", "")
    tool_trace = run.outputs.get("tool_trace", [])
    
    # Filter for search-related tools
    search_calls = [t for t in tool_trace if t["tool"] in ["get_legal_references", "search_documents", "consult_document_agent"]]
    
    if not search_calls:
        return {
            "key": "search_quality",
            "score": 0.0,
            "comment": "No search tools were called. The agent attempted to answer without gathering information."
        }
        
    # Extract queries
    queries = []
    for call in search_calls:
        args = call["args"]
        if isinstance(args, dict):
            # Handle different query argument names
            q = args.get("query") or args.get("questions") or args.get("topic") or str(args)
            queries.append(f"- Tool: {call['tool']}, Query: {q}")
        else:
            queries.append(f"- Tool: {call['tool']}, Args: {str(args)}")
            
    queries_str = "\n".join(queries)

    prompt = f"""You are a Search Quality Evaluator.
    
    User Question:
    {input_question}
    
    Executed Search Queries:
    {queries_str}
    
    Instruction:
    - Rate the quality of the search queries on a scale of 0.0 to 1.0.
    - 1.0: Queries are highly relevant, specific, and likely to yield the correct answer.
    - 0.5: Queries are somewhat relevant but too broad or slightly off-topic.
    - 0.0: Queries are irrelevant, nonsensical, or completely missing the point.
    
    JSON Output Format: {{ "score": float, "reasoning": "..." }}
    """
    
    llm = get_evaluator_llm()
    response = await llm.ainvoke([HumanMessage(content=prompt)])
    
    try:
        content = response.content.replace("```json", "").replace("```", "").strip()
        result = json.loads(content)
        return {
            "key": "search_quality",
            "score": result.get("score", 0),
            "comment": result.get("reasoning", "")
        }
    except:
        return {"key": "search_quality", "score": 0, "comment": "Error parsing evaluator response."}


async def faithfulness_evaluator(run: Run, example: Example) -> dict:
    """
    Measures if the answer is derived ONLY from the retrieved context (if available).
    Prevents hallucinations.
    """
    prediction = run.outputs.get("output", "")
    retrieved_context = run.outputs.get("retrieved_context", [])
    
    # If no context was retrieved, faithfulness might not be applicable or is 1.0 (if relying on internal knowledge)
    # But for RAG, we usually want to punish answers not in context. 
    # Let's check if we have context.
    
    context_str = "\n".join([str(doc) for doc in retrieved_context]) if retrieved_context else "No context retrieved."
    
    prompt = f"""You are a Faithfulness Evaluator.
    
    Task: specificy if the generated answer is faithful to the retrieved context.
    
    Retrieved Context:
    {context_str}
    
    Generated Answer:
    {prediction}
    
    Instruction:
    - Return Score 1 if the answer is fully supported by the context.
    - Return Score 0 if the answer contains information NOT present in the context (hallucination).
    - If the context is empty and the answer contains specific facts, it is a hallucination (Score 0).
    - Provide a brief reasoning.
    
    JSON Output Format: {{ "score": 0 or 1, "reasoning": "..." }}
    """
    
    llm = get_evaluator_llm()
    response = await llm.ainvoke([HumanMessage(content=prompt)])
    
    try:
        content = response.content.replace("```json", "").replace("```", "").strip()
        result = json.loads(content)
        return {
            "key": "faithfulness",
            "score": result.get("score", 0),
            "comment": result.get("reasoning", "")
        }
    except:
        return {"key": "faithfulness", "score": 0, "comment": "Failed to parse evaluator response."}


async def relevance_evaluator(run: Run, example: Example) -> dict:
    """
    Measures if the answer directly addresses the user's question.
    """
    input_question = example.inputs.get("input", "")
    prediction = run.outputs.get("output", "")
    
    prompt = f"""You are a Relevance Evaluator.
    
    User Question:
    {input_question}
    
    Generated Answer:
    {prediction}
    
    Instruction:
    - Rate relevance on a scale of 0.0 to 1.0.
    - 1.0: Directly answers the question completely.
    - 0.5: Generic or partial answer.
    - 0.0: Irrelevant or refuses to answer without cause.
    
    JSON Output Format: {{ "score": float, "reasoning": "..." }}
    """
    
    llm = get_evaluator_llm()
    response = await llm.ainvoke([HumanMessage(content=prompt)])
    
    try:
        content = response.content.replace("```json", "").replace("```", "").strip()
        result = json.loads(content)
        return {
            "key": "relevance",
            "score": result.get("score", 0),
            "comment": result.get("reasoning", "")
        }
    except:
        return {"key": "relevance", "score": 0, "comment": "Error parsing."}


async def coherence_evaluator(run: Run, example: Example) -> dict:
    """
    Measures if the answer is well-structured and logical.
    """
    prediction = run.outputs.get("output", "")
    
    prompt = f"""You are a Coherence Evaluator.
    
    Generated Answer:
    {prediction}
    
    Instruction:
    - Rate coherence on a scale of 0.0 to 1.0.
    - Is the text grammatically correct?
    - Does it flow logically?
    - Is it easy to read?
    
    JSON Output Format: {{ "score": float, "reasoning": "..." }}
    """
    
    llm = get_evaluator_llm()
    response = await llm.ainvoke([HumanMessage(content=prompt)])
    
    try:
        content = response.content.replace("```json", "").replace("```", "").strip()
        result = json.loads(content)
        return {
            "key": "coherence",
            "score": result.get("score", 0),
            "comment": result.get("reasoning", "")
        }
    except:
        return {"key": "coherence", "score": 0, "comment": "Error parsing."}


# --- 3. Run Evaluation ---
async def run_comprehensive_eval(dataset_name: str = DATASET_NAME):
    print(f"--- Starting Comprehensive Evaluation on Dataset '{dataset_name}' ---")
    
    client = Client()
    
    # Check if dataset exists
    if not client.has_dataset(dataset_name=dataset_name):
        print(f"❌ Dataset '{dataset_name}' not found in LangSmith project.")
        print(f"Available datasets: {[d.name for d in client.list_datasets()]}")
        return

    # Run evaluation
    results = await aevaluate(
        target,
        data=dataset_name,
        evaluators=[
            faithfulness_evaluator,
            relevance_evaluator,
            coherence_evaluator,
            search_quality_evaluator
        ],
        experiment_prefix=EXPERIMENT_PREFIX,
        metadata={
            "version": "1.1",
            "agent": "ResearchAgent",
            "changes": "Added search_quality_evaluator"
        },
        max_concurrency=4  # Adjust based on rate limits
    )

    
    print("\n✅ Evaluation Complete.")
    # Wait for results if it's a future or just iterate
    # AsyncExperimentResults is an async iterator of Result objects
    
    print("\n--- RESULTS SUMMARY ---")
    
    metrics = {"faithfulness": [], "relevance": [], "coherence": []}
    
    i = 0
    async for result in results:
        i += 1
        print(f"\nExample {i}:")
        print(f"  Input: {result.run.inputs.get('input', '')[:50]}...")
        print(f"  Output: {result.run.outputs.get('output', '')[:50]}...")
        
        for key in metrics.keys():
            score = result.evaluation_results.get("key", {}).get("score") # This structure might vary
            # Actually result.evaluation_results is a dict of results wrapper
            # Let's look at the structure: result.evaluation_results['faithfulness'].score
            
            val = None
            if key in result.evaluation_results:
                 val = result.evaluation_results[key].score
                 if val is not None:
                     metrics[key].append(val)
                     print(f"  - {key}: {val}")
    
    print("\n--- AGGREGATE METRICS ---")
    for key, scores in metrics.items():
        if scores:
            avg = sum(scores) / len(scores)
            print(f"{key.capitalize()}: {avg:.2f} ({len(scores)} samples)")
        else:
             print(f"{key.capitalize()}: N/A")

    # Construct URL manually if needed or print experiment name
    # We can get project ID from client but let's just print name
    print(f"\nUse LangSmith UI to see detailed traces for experiment: {EXPERIMENT_PREFIX}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=DATASET_NAME, help="LangSmith dataset name")
    args = parser.parse_args()
    
    asyncio.run(run_comprehensive_eval(args.dataset))
