
import asyncio
import os
import argparse
from typing import Optional

from langsmith import Client, evaluate
from langsmith.schemas import Run, Example
from langchain_core.messages import HumanMessage
# from langchain.evaluation import load_evaluator

# Import your agent (ensure app is in pythonpath)
from app.agents.research_agent import ResearchAgent

# --- Configuration ---
LANGSMITH_PROJECT = os.environ.get("LANGCHAIN_PROJECT", "Lawyers-Research-Eval")
DATASET_NAME = "Lawyers-Research-Benchmark"

# --- 1. Define Dataset ---
EXAMPLE_INPUTS = [
    {
        "input": "What are the legal requirements for a valid contract in Bulgaria?",
        "output": "A valid contract in Bulgaria requires: 1. Agreement between parties. 2. Legal capacity. 3. Consent free from vice. 4. Lawful object. 5. Proper form (if required). 6. Cause."
    },
    {
        "input": "How is a limited liability company (OOD) incorporated?",
        "output": "Incorporating an OOD involves: constituent assembly, appointing a manager, depositing capital, submitting to Commercial Register."
    }
]

async def create_dataset_if_not_exists():
    client = Client()
    datasets = list(client.list_datasets(dataset_name=DATASET_NAME))
    if datasets:
        print(f"Dataset '{DATASET_NAME}' already exists.")
        return datasets[0]
    
    print(f"Creating dataset '{DATASET_NAME}'...")
    dataset = client.create_dataset(dataset_name=DATASET_NAME, description="Legal research benchmark questions")
    
    for example in EXAMPLE_INPUTS:
        client.create_example(
            inputs={"input": example["input"]},
            outputs={"output": example["output"]},
            dataset_id=dataset.id
        )
    return dataset

# --- 2. Define the Target ---
async def target(inputs: dict) -> dict:
    """
    Wraps the ResearchAgent to accept {'input': 'question'} and return {'output': 'answer'}
    """
    question = inputs["input"]
    agent = ResearchAgent()
    
    # Run the agent
    # Provide a thread_id to satisfy checkpointer requirements
    result = await agent.graph.ainvoke(
        {
            "messages": [HumanMessage(content=question)], 
            "task_description": question
        },
        config={"configurable": {"thread_id": "eval_test_run"}}
    )
    
    final_answer = result.get("draft_answer", "No answer generated.")
    return {"output": final_answer}

# --- 3. Define Evaluators ---

# Helper function for evaluation logic
async def evaluate_answer(question: str, prediction: str, reference: str) -> dict:
    """Evaluates answer correctness using the agent's LLM."""
    agent = ResearchAgent()
    
    prompt = f"""You are an expert legal evaluator.
    
    Question: {question}
    
    Ground Truth Answer:
    {reference}
    
    Agent's Predicted Answer:
    {prediction}
    
    Task: Rate the correctness of the Agent's answer based on the Ground Truth.
    - If the agent covers the key points from the ground truth, mark as Correct.
    - If it contradicts or misses critical info, mark as Incorrect.
    - Ignore style or length differences if the facts are right.
    
    Respond in JSON format: {{ "score": 0 to 1, "reasoning": "explanation" }}
    """
    
    messages = [HumanMessage(content=prompt)]
    response = await agent.llm.ainvoke(messages)
    
    import json
    try:
        content = response.content.replace("```json", "").replace("```", "").strip()
        result = json.loads(content)
        score = result.get("score", 0)
        reasoning = result.get("reasoning", "No reasoning provided")
    except:
        score = 0
        reasoning = "Failed to parse evaluator response: " + str(response.content)
        
    return {"score": score, "reasoning": reasoning}

# Evaluator for LangSmith runs
async def correctness_evaluator(run: Run, example: Example) -> dict:
    prediction = run.outputs.get("output")
    reference = example.outputs.get("output")
    input_text = example.inputs.get("input")
    
    result = await evaluate_answer(input_text, prediction, reference)
    
    return {
        "key": "correctness",
        "score": result["score"],
        "comment": result["reasoning"]
    }

# --- 4. Run Evaluation ---
async def run_evaluation(custom_question: Optional[str] = None, custom_ground_truth: Optional[str] = None):
    
    client = Client()
    
    # Custom single run
    if custom_question:
        print(f"--- Running Custom Evaluation ---")
        print(f"Question: {custom_question}")
        
        # 1. Run Agent
        inputs = {"input": custom_question}
        result = await target(inputs)
        prediction = result["output"]
        print(f"\nAgent Answer:\n{prediction}\n")
        
        # 2. Evaluate if ground truth exists
        if custom_ground_truth:
            print(f"Evaluating against Ground Truth...")
            eval_result = await evaluate_answer(custom_question, prediction, custom_ground_truth)
            print(f"Correctness: {eval_result['score']}")
            print(f"Reasoning: {eval_result['reasoning']}")
        return

    # Batch Run
    await create_dataset_if_not_exists()
    
    print(f"--- Running Batch Evaluation on '{DATASET_NAME}' ---")
    
    results = await evaluate(
        target,
        data=DATASET_NAME,
        evaluators=[correctness_evaluator],
        experiment_prefix="lawyers-research",
        metadata={"version": "1.0"}
    )
    
    print("\nEvaluation Complete.")
    print(results)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", type=str, help="Custom question to ask")
    parser.add_argument("--truth", type=str, help="Expected answer for custom question")
    args = parser.parse_args()
    
    asyncio.run(run_evaluation(args.question, args.truth))
