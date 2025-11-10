#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
A command-line tool to benchmark AI chatbot models using the OpenRouter API.

This script executes a series of prompts against a list of target AI models
and uses a secondary, low-cost model as an "LLM-as-a-Judge" to score the
responses.

Execution:
    python bench_llm.py --models <model1>,<model2> prompts.jsonl
"""

import os
import sys
import json
import time
import math
import argparse
import requests

# --- Constants ---
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
# Floor value to prevent math errors with log(0) for time or cost
LOG_FLOOR = 1e-6
# Your website or app URL to identify requests
# See: https://openrouter.ai/docs#referrer-header
REFERRER_URL = "https://github.com/allyourdatas/jules-eng-test"

# --- API Communication ---

def call_openrouter_api(model_id, messages, max_tokens):
    """
    Calls the OpenRouter API for a given model and prompt.

    Args:
        model_id (str): The ID of the target model.
        messages (list): A list of message objects for the chat API.
        max_tokens (int): The maximum number of tokens for the response.

    Returns:
        A tuple containing:
        - str: The response text from the model.
        - float: The elapsed time for the API call in seconds.
        - float: The total cost of the API call in USD.
    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("Error: OPENROUTER_API_KEY environment variable not set.", file=sys.stderr)
        sys.exit(1)

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "X-Request-Referrer": REFERRER_URL,
    }

    body = {
        "model": model_id,
        "messages": messages,
        "max_tokens": max_tokens,
        "stream": False # Ensure we get usage data
    }

    start_time = time.time()
    try:
        response = requests.post(OPENROUTER_API_URL, headers=headers, json=body, timeout=120)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"Error calling OpenRouter API for model {model_id}: {e}", file=sys.stderr)
        return "API_ERROR", 0.0, 0.0
    finally:
        end_time = time.time()

    elapsed_time = end_time - start_time
    data = response.json()

    response_text = ""
    if data.get("choices") and len(data["choices"]) > 0:
        response_text = data["choices"][0]["message"]["content"].strip()

    # Handle inconsistent 'usage' field from the OpenRouter API, which can be
    # a dictionary (e.g., {"cost": 0.1}) or a numeric value (e.g., 0.1).
    cost = 0.0
    usage_field = data.get("usage")

    if isinstance(usage_field, dict):
        # Case 1: usage is a dictionary, e.g., {"cost": 0.001, "completion_tokens": 100}
        cost = usage_field.get("cost", 0.0)
        # Fallback check for older/alternate API versions
        if cost == 0.0 and "total_cost" in usage_field:
            cost = usage_field.get("total_cost", 0.0)
    elif isinstance(usage_field, (int, float)):
        # Case 2: usage is a direct numeric value for the cost
        cost = usage_field

    # Log a warning if cost is 0 but tokens were used.
    # Note: token counts can appear at the top level or inside the usage dict.
    completion_tokens = data.get("tokens_completion", 0)
    if completion_tokens == 0 and isinstance(usage_field, dict):
        completion_tokens = usage_field.get("completion_tokens", 0)

    if cost == 0.0 and completion_tokens > 0:
        print(f"Warning: API returned 0 cost for model {model_id} despite generating {completion_tokens} tokens. Check API key status.", file=sys.stderr)

    # In case the model is truly a free model, the cost remains 0.0.

    return response_text, elapsed_time, cost

# --- Main Execution Logic ---

def main():
    """Main function to run the benchmarking tool."""
    parser = argparse.ArgumentParser(description="Benchmark LLM models via OpenRouter API.")
    parser.add_argument(
        "--models",
        type=str,
        required=True,
        help="A comma-separated list of target model IDs to benchmark."
    )
    parser.add_argument(
        "prompts_file",
        type=str,
        help="Path to the JSONL file containing the prompts."
    )
    parser.add_argument(
        "--scoring-model",
        type=str,
        default="openai/gpt-oss-120b",
        help="The model ID to use for scoring the responses."
    )
    args = parser.parse_args()
    target_models = [model.strip() for model in args.models.split(',')]
    scoring_model = args.scoring_model

    print("Starting LLM Benchmark...", file=sys.stderr)
    print(f"Target Models: {target_models}", file=sys.stderr)
    print(f"Scoring Model: {scoring_model}", file=sys.stderr)
    print("-" * 20, file=sys.stderr)

    all_results = []
    raw_responses_by_prompt = {}

    # 1. Iterate through prompts from the specified file
    try:
        with open(args.prompts_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    prompt_data = json.loads(line)
                    prompt_id = prompt_data["id"]
                    prompt_text = prompt_data["prompt"]
                    scoring_rubric = prompt_data["scoring_rubric"]
                    max_tokens = prompt_data.get("max_tokens", 2048)
                except json.JSONDecodeError:
                    print(f"Warning: Skipping malformed JSON line: {line.strip()}", file=sys.stderr)
                    continue

                print(f"Processing Prompt ID: {prompt_id}", file=sys.stderr)
                raw_responses_by_prompt[prompt_id] = []
                model_responses = []

                # 2. Benchmark Phase: Get responses from target models
                for model in target_models:
                    print(f"  - Benchmarking model: {model}", file=sys.stderr)
                    messages = [{"role": "user", "content": prompt_text}]
                    response_text, t_model, c_model = call_openrouter_api(model, messages, max_tokens)
                    model_responses.append({
                        "model": model,
                        "response_text": response_text,
                        "t_model": t_model,
                        "c_model": c_model
                    })
                    raw_responses_by_prompt[prompt_id].append(
                        f"[MODEL: {model}]\n{response_text}\n"
                    )

                # 3. Scoring Phase: Use LLM-as-a-Judge
                for response in model_responses:
                    s_model = 0
                    c_score = 0.0

                    # Only score if the target model's response is valid
                    if response["response_text"] and response["response_text"] != "API_ERROR":
                        print(f"  - Scoring response from: {response['model']}", file=sys.stderr)

                        # Construct the detailed prompt for the scoring model
                        scoring_prompt_messages = [
                            {
                                "role": "system",
                                "content": (
                                    "You are an expert evaluator. Your task is to score a model's "
                                    "response based on a given rubric. You must return ONLY the "
                                    "final numerical score and nothing else."
                                )
                            },
                            {
                                "role": "user",
                                "content": (
                                    f"**Original Prompt:**\n{prompt_text}\n\n"
                                    f"**Scoring Rubric:**\n{scoring_rubric}\n\n"
                                    f"**Model's Response to Evaluate:**\n{response['response_text']}\n\n"
                                    "------\n"
                                    "Based on the rubric, what is the numerical score for the model's response?"
                                )
                            }
                        ]

                        score_text, _, c_score = call_openrouter_api(scoring_model, scoring_prompt_messages, 10)

                        # Check for API errors from the scoring model
                        if score_text == "API_ERROR":
                            print(f"    Warning: Scoring model '{scoring_model}' failed. Defaulting score to 0.", file=sys.stderr)
                            s_model = 0
                        else:
                            # Extract and validate the score
                            try:
                                s_model = int(float(score_text))
                            except (ValueError, TypeError):
                                print(f"    Warning: Could not parse score '{score_text}'. Defaulting to 0.", file=sys.stderr)
                                s_model = 0
                    else:
                        print(f"  - Skipping scoring for failed model: {response['model']}", file=sys.stderr)


                    # 4. Calculate Final Metrics
                    t_model = response["t_model"]
                    c_total = response["c_model"] + c_score

                    # Use floor values to prevent log(<=0)
                    safe_t_model = max(t_model, LOG_FLOOR)
                    safe_c_total = max(c_total, LOG_FLOOR)

                    adjusted_score = (100 - s_model) * math.log(safe_t_model)
                    adjusted_cost = (100 - s_model) * math.log(safe_c_total)

                    all_results.append({
                        "Prompt ID": prompt_id,
                        "Target Model": response["model"],
                        "Score (S_model)": s_model,
                        "Time (T_model) [s]": round(t_model, 4),
                        "Total Cost (C_total) [USD]": f"{c_total:.6f}",
                        "Adjusted Score": round(adjusted_score, 4),
                        "Adjusted Cost": round(adjusted_cost, 4),
                    })
    except FileNotFoundError:
        print(f"Error: Prompts file not found at '{args.prompts_file}'", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
        sys.exit(1)


    # 5. Output Results
    print("\n" + "="*40)
    print("      RAW RESPONSES")
    print("="*40 + "\n")

    for prompt_id, responses in raw_responses_by_prompt.items():
        print(f"--- PROMPT ID: {prompt_id} ---")
        print("\n".join(responses))

    print("\n" + "="*40)
    print("      SUMMARY TABLE")
    print("="*40 + "\n")

    if all_results:
        # Print Markdown table header
        headers = all_results[0].keys()
        print(f"| {' | '.join(headers)} |")
        print(f"|{'|'.join([':---' for _ in headers])}|")
        # Print rows
        for result in all_results:
            print(f"| {' | '.join(map(str, result.values()))} |")

if __name__ == "__main__":
    main()
