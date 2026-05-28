# CSE 151B Competition — Starter Code

Open **`starter_code_cse151b_comp.ipynb`** to get started.

The notebook covers environment setup, inference with Qwen3-4B-Thinking (INT8), and scoring against the public dataset.

## Contents

| File | Description |
|---|---|
| `starter_code_cse151b_comp.ipynb` | Main entry point |
| `judger.py` | Response scoring logic |
| `utils.py` | Utilities used by `judger.py` |
| `data/public.jsonl` | Public dataset with ground-truth answers |
| `results/` | Output JSONL files written at runtime |

# vLLM Metal Quick Start Guide

## 1. Start a new terminal

Activate the `vllm-metal` environment:

```bash
source ~/.venv-vllm-metal/bin/activate
````

## 2. Add your Hugging Face token

Replace `hf_your_token_here` with your actual token:

```bash
export HF_TOKEN="hf_your_token_here"
```

## 3. Configure vLLM Metal

Use these settings for Apple Silicon / Mac:

```bash
export VLLM_METAL_USE_MLX=1
export VLLM_MLX_DEVICE=gpu
export VLLM_METAL_USE_PAGED_ATTENTION=1
export VLLM_METAL_MEMORY_FRACTION=0.75
```

## 4. Start the vLLM server

Recommended stable config:

```bash
vllm serve Qwen/Qwen3-4B-Thinking-2507 \
  --dtype auto \
  --trust-remote-code \
  --max-model-len 16384 \
  --max-num-seqs 4 \
  --max-num-batched-tokens 8192 \
  --no-enable-prefix-caching
```

Leave this terminal running.

You should see something like:

```text
Application startup complete.
```

## 5. Test the server from a second terminal

```bash
curl http://localhost:8000/v1/models
```

Then test generation:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-4B-Thinking-2507",
    "messages": [
      {"role": "user", "content": "Answer in one sentence: what is 2+2?"}
    ],
    "max_tokens": 64,
    "temperature": 0
  }'
```

## 6. Run from Python

Install the OpenAI client if needed:

```bash
pip install openai
```

Python example:

```python
from openai import OpenAI

MODEL_ID = "Qwen/Qwen3-4B-Thinking-2507"

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="EMPTY",
    timeout=600,
)

result = client.chat.completions.create(
    model=MODEL_ID,
    messages=[
        {"role": "system", "content": "You are an expert mathematician."},
        {"role": "user", "content": "What is 2 + 2? Put the answer in \\boxed{}."},
    ],
    max_tokens=4096,
    temperature=0.2,
    top_p=0.95,
    presence_penalty=0.0,
    extra_body={
        "top_k": 20,
        "min_p": 0.0,
        "repetition_penalty": 1.0,
    },
)

print(result.choices[0].message.content)
```

## 7. Important token rule

Do not set:

```python
max_tokens = 16384
```

when the server uses:

```bash
--max-model-len 16384
```

The total must fit:

```text
prompt tokens + output tokens <= max model length
```

Recommended:

```python
max_tokens = 4096
```

Use `8192` only if responses are being cut off.

## 8. Stop the server

In the server terminal, press:

```bash
Control + C
```

If it does not stop, find the process:

```bash
lsof -i :8000
```

Then kill the PID:

```bash
kill <PID>
```

Force kill if needed:

```bash
kill -9 <PID>
```

## 9. Restart with a different config

Stop the old server first, then rerun:

```bash
vllm serve Qwen/Qwen3-4B-Thinking-2507 \
  --dtype auto \
  --trust-remote-code \
  --max-model-len 16384 \
  --max-num-seqs 4 \
  --max-num-batched-tokens 8192 \
  --no-enable-prefix-caching
```

