# CSE 151B Competition

Final code submission for the CSE 151B Kaggle competition.

## Contents

| File / Directory               | Description                                                                                            |
| ------------------------------ | ------------------------------------------------------------------------------------------------------ |
| `script.py`                     | Main inference script. Defines `run_inference()` and runs the full prediction pipeline.                |
| `data/private.jsonl`           | Private evaluation dataset. Place the private dataset here before running inference.                   |
| `results/`                     | Output directory created by the script. Contains intermediate pass CSVs and the final submission file. |
| `results/submission_final.csv` | Final CSV produced by `run_inference()`.                                                               |
| `README.md`                    | Setup, model, and reproduction instructions.                                                           |

## Model

This submission uses the base model:

```text
Qwen/Qwen3-4B-Thinking-2507
```

No fine-tuned checkpoint is required for this version. The model is downloaded automatically by vLLM from Hugging Face when the vLLM server starts.

## Hardware Used

GPU / accelerator used:

```text
Apple M4 Pro via Metal / vLLM Metal
```

Approximate total generation / inference time:

```text
Approximately 60 hours for the full private set.
```

## Python Environment

The code expects a Python environment with vLLM Metal, OpenAI client compatibility, and tqdm installed.

Required Python packages include:

```bash
pip install openai tqdm huggingface_hub
```

The vLLM Metal environment used for generation was:

```bash
source ~/.venv-vllm-metal/bin/activate
```

## Hugging Face Authentication

Add a Hugging Face token before starting vLLM if the model download requires authentication or if the model is not already cached:

```bash
export HF_TOKEN="hf_your_token_here"
```

## Dataset Setup

Place the private dataset at:

```text
data/private.jsonl
```

The inference script expects this exact path:

```python
DATA_PATH = "data/private.jsonl"
```

## vLLM Metal Server Setup

Start the vLLM server before running inference.

Use the following settings for Apple Silicon / Mac:

```bash
source ~/.venv-vllm-metal/bin/activate

export HF_TOKEN="hf_your_token_here"

VLLM_METAL_MEMORY_FRACTION=0.88 \
VLLM_METAL_USE_PAGED_ATTENTION=1 \
VLLM_METAL_PREFIX_CACHE=1 \
VLLM_METAL_PREFIX_CACHE_FRACTION=0.03 \
vllm serve Qwen/Qwen3-4B-Thinking-2507 \
  --dtype bfloat16 \
  --trust-remote-code \
  --max-model-len 16384 \
  --max-num-seqs 8 \
  --max-num-batched-tokens 8192 \
  --enable-chunked-prefill \
  --enable-prefix-caching
```

Leave this terminal running. The inference code connects to:

```text
http://localhost:8000/v1
```

You should see:

```text
Application startup complete.
```

To stop the server, press:

```bash
Control + C
```

## Running Inference

In a second terminal, activate the same environment:

```bash
source ~/.venv-vllm-metal/bin/activate
```

Then run:

```bash
python script.py
```

This calls `run_inference()` and produces:

```text
results/submission_final.csv
```

## Inference Pipeline

`run_inference()` performs the full end-to-end pipeline:

1. Loads the private dataset from `data/private.jsonl`.
2. Sends each problem to the local vLLM OpenAI-compatible server.
3. Uses different prompts for multiple-choice and non-multiple-choice questions.
4. Generates initial answers with the base Qwen model.
5. Retries responses that do not contain a valid `\boxed{}` answer after `</think>`.
6. Applies answer extraction and validity checks.
7. Writes intermediate CSVs after each pass.
8. Writes the final submission CSV to `results/submission_final.csv`.

## Final Hyperparameters

The final inference hyperparameters are stored directly in `script.py`.

### Pass 0

```python
max_tokens = 4096
temperature = 0.6
top_p = 0.95
top_k = 20
repetition_penalty = 1.00
prompt_style = "normal"
```

### Pass 1

```python
max_tokens = 8192
temperature = 0.6
top_p = 0.95
top_k = 20
repetition_penalty = 1.00
prompt_style = "normal"
```

### Pass 2

```python
max_tokens = 12288
temperature = 0.35
top_p = 0.90
top_k = 10
repetition_penalty = 1.05
prompt_style = "normal"
```

Other settings:

```python
CONCURRENCY = 8
ITERATIONS = 2
```

## Output Format

The final CSV has the columns:

```text
id,response
```

The final output file is:

```text
results/submission_final.csv
```

## Reproducibility Notes

The code saves intermediate CSV files during generation so that interrupted runs can resume. For a completely fresh reproduction run, delete the `results/` directory before running:

```bash
rm -rf results
python script.py
```