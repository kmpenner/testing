# Jules's Guide to Running the AI Chatbot Benchmarking Tool

This document provides instructions on how to use the `bench_llm.py` script to benchmark AI models.

## Prerequisites

1.  **Python 3:** Ensure you have Python 3 installed.
2.  **`requests` library:** Install the required library using pip:
    ```bash
    pip install requests
    ```

## Setup

1.  **Set the OpenRouter API Key:**
    The script requires an OpenRouter API key to function. You must set this key as an environment variable.

    ```bash
    export OPENROUTER_API_KEY="YOUR_API_KEY_HERE"
    ```
    Replace `"YOUR_API_KEY_HERE"` with your actual OpenRouter API key.

## Execution

The script is designed to be run from the command line, with model names passed as arguments and prompts piped from a `.jsonl` file via `stdin`.

### Command Structure

```bash
python bench_llm.py --models <model_id_1>,<model_id_2>,... < <prompts_file>.jsonl
```

### Example

Using the provided `prompts.jsonl` file and benchmarking the `google/gemini-flash-1.5` and `anthropic/claude-3.5-sonnet` models, the command would be:

```bash
python bench_llm.py --models "google/gemini-flash-1.5,anthropic/claude-3.5-sonnet" < prompts.jsonl
```

### Notes

*   Ensure the `prompts.jsonl` file is in the same directory as the script or provide the correct path to it.
*   The model IDs should be valid OpenRouter model IDs.
*   The output will be printed to the console in two parts: first the raw responses from each model, and then a summary table in Markdown format.
