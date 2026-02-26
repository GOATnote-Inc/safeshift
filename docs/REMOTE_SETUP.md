# Remote GPU Setup

SafeShift executes inference on remote GPUs while grading locally.

## Architecture

```
Local (MacBook)              Remote (GPU Server)
┌──────────────┐             ┌──────────────────┐
│ safeshift    │   HTTP/API  │ vLLM / TGI       │
│ CLI          │ ──────────> │ serving model     │
│ grade local  │ <────────── │ OpenAI-compat API │
│ analyze      │   JSON      └──────────────────┘
└──────────────┘
```

## Option 1: vLLM Remote Server (Recommended for v0.1)

### On the GPU server:

```bash
# Install vLLM
pip install vllm

# Start server with your model
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.1-70B \
    --dtype float16 \
    --port 8000
```

### On your local machine:

```bash
# Run evaluation against remote vLLM
safeshift run \
    --matrix configs/matrices/default_matrix.yaml \
    --executor vllm \
    --model meta-llama/Llama-3.1-70B \
    --remote http://gpu-server:8000/v1

# Or set environment variable
export VLLM_BASE_URL=http://gpu-server:8000/v1
safeshift run --matrix configs/matrices/default_matrix.yaml --executor vllm
```

## Option 2: Cloud API

For testing cloud models (no GPU needed):

```bash
export OPENAI_API_KEY=sk-...
safeshift run --executor api --model gpt-4o --scenario SCN-C-001

export ANTHROPIC_API_KEY=sk-ant-...
safeshift run --executor api --model claude-sonnet-4-6 --scenario SCN-C-001
```

Note: Cloud APIs don't support optimization configs (quantization, etc.)
since optimization happens server-side. The API executor measures
end-to-end latency only.

## Option 3: SSH Dispatch (v0.2)

SSH-based dispatch to remote servers is planned for v0.2.

## Testing Your Setup

```bash
# Mock (no GPU needed)
safeshift run --matrix configs/matrices/quick_matrix.yaml --executor mock

# Test remote connection
safeshift run --scenario SCN-C-001 --executor vllm --model your-model
```
