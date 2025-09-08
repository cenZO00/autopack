<div align="center">
  <a href="https://granulavision.com">
    <img src="media/banner.jpg">
  </a>

<h3 align="center">autopack</h3>

  <p align="justify">
    autopack makes your Hugging Face models easy to run, share, and ship. It quantizes once and exports to multiple runtimes, with sensible defaults and an automatic flow that produces a readable summary. It supports HF, ONNX, and GGUF (llama.cpp) formats and can publish to the Hugging Face Hub in one shot.
  </p>
  <p align="center">
    <a href="#about-the-project">About</a>
    &middot;
    <a href="#requirements">Requirements</a>
    &middot;
    <a href="#setup">Setup</a>
    &middot;
    <a href="#building-instructions">Building Instructions</a>
    &middot;
    <a href="#running-the-application">Running</a>
    &middot;
    <a href="#detailed-usage">Detailed Usage</a>
    &middot;
    <a href="#q-and-a">Q&A</a>
  </p>
</div>

---

# About The Project

## What is autopack?

autopack is a CLI that helps you quantize and package Hugging Face models into multiple useful formats in a single pass, with an option to publish artifacts to the Hub. 

You have a 120b llm and want to optimise it so that people (not corpotations with clusters of b200s) can use it on their 8gb 2060? all you need to do is run 

```bash
autopack auto meta-llama/Llama-3-8B -o out/llama
```



### Why use it?

- Fast: generate multiple variants in one command.
- Practical: built on Transformers, bitsandbytes, ONNX, and llama.cpp.
- Portable: CPU- and GPU-friendly artifacts, good defaults.

# Requirements

## Core

- Python 3.9+
- PyTorch, Transformers, Hugging Face Hub
- Optional: bitsandbytes (4/8-bit), optimum[onnxruntime] (ONNX), llama.cpp (GGUF tools)

### Notes

- GGUF export requires a built llama.cpp and `llama-quantize` in PATH.
- Set `HUGGINGFACE_HUB_TOKEN` to publish, or pass `--token`.

# Setup

## Install

```bash
pip install -e .
```

### Optional extras

```bash
# ONNX export support
pip install -e '.[onnx]'

# GGUF export helpers (converter deps)
pip install -e '.[gguf]'

# llama.cpp runtime bindings (llama-cpp-python)
pip install -e '.[llama]'

# Everything for llama.cpp functionality (GGUF export + runtime)
pip install -e '.[gguf,llama]'
```

Note: for GGUF and llama.cpp functionality you also need the llama.cpp tools
(`llama-quantize`, `llama-cli`) available on your `PATH`. You can build the
vendored copy and export `PATH` as shown in
[Vendored llama.cpp quick build](#vendored-llamacpp-quick-build).

# Building Instructions

```bash
python -m build
```

# Running the Application

## Quickstart

```bash
autopack auto meta-llama/Llama-3-8B -o out/llama3 --output-format hf
```

Add ONNX and GGUF:
```bash
autopack auto meta-llama/Llama-3-8B -o out/llama3 --output-format hf onnx gguf
```

GGUF only (with default presets Q4_K_M, Q5_K_M, Q8_0):
```bash
autopack auto meta-llama/Llama-3-8B -o out/llama3-gguf --output-format gguf
```

Publish to Hub:
```bash
autopack publish out/llama3-4bit your-username/llama3-4bit --private \
  --commit-message "Add 4-bit quantized weights"
```

# Detailed Usage

## Commands Overview

### auto

Run common HF quantization variants and optional ONNX/GGUF exports in one go, with a summary table and generated README in the output folder.

```bash
autopack auto <model_id_or_path> -o <out_dir> \
  --output-format hf [onnx] [gguf] \
  [--eval-dataset <dataset>[::<config>]] \
  [--revision <rev>] [--trust-remote-code]
```

Key points:
- Default HF variants: bnb-4bit, bnb-8bit, int8-dynamic, bf16
- Add ONNX and/or GGUF via `--output-format`
- If `--eval-dataset` is provided, perplexity is computed for each HF variant

### quantize

Produce specific formats with a chosen quantization strategy.

```bash
autopack quantize <model_id_or_path> -o <out_dir> \
  --output-format hf [onnx] [gguf] \
  [--quantization bnb-4bit|bnb-8bit|int8-dynamic|none] \
  [--dtype auto|float16|bfloat16|float32] \
  [--device-map auto|cpu] [--prune <0..0.95>] \
  [--revision <rev>] [--trust-remote-code]
```

### publish

Upload an exported model folder to the Hugging Face Hub.

```bash
autopack publish <folder> <user_or_org/repo> \
  [--private] [--token $HUGGINGFACE_HUB_TOKEN] \
  [--branch <rev>] [--commit-message "..."] [--no-create]
```

## Common Options

- `--trust-remote-code`: enable loading custom modeling code from Hub repos
- `--revision`: branch/tag/commit to load
- `--device-map`: set to `cpu` to force CPU; defaults to `auto`
- `--dtype`: compute dtype for non-INT8 layers (applies to HF exports)
- `--prune`: global magnitude pruning ratio across Linear layers (0..0.95)

## Output Formats

- `hf`: Transformers checkpoint with tokenizer and config
- `onnx`: ONNX export using `optimum[onnxruntime]` for CausalLM
- `gguf`: llama.cpp GGUF via `convert_hf_to_gguf.py` and `llama-quantize`

## GGUF Details

- Converter resolution order:
  1) `--gguf-converter` if provided
  2) `$LLAMA_CPP_CONVERT` env var
  3) Vendored script: `third_party/llama.cpp/convert_hf_to_gguf.py`
  4) `~/llama.cpp/convert_hf_to_gguf.py` or `~/src/llama.cpp/convert_hf_to_gguf.py`
- Quant presets: uppercase (e.g., `Q4_K_M`). If omitted, autopack generates `Q4_K_M`, `Q5_K_M`, `Q8_0` by default.
- Isolation: by default, conversion runs in an isolated `.venv` inside the output dir. Disable with `--gguf-no-isolation`.
- Architecture checks: pass `--gguf-force` to bypass the basic architecture guard.
- Ensure `llama-quantize` is in `PATH` (typically in `third_party/llama.cpp/build/bin`).

## ONNX Details

- Requires: `pip install 'optimum[onnxruntime]'`
- Uses `ORTModelForCausalLM`; non-CausalLM models may not be supported in this version.

## Perplexity Evaluation

- `--eval-dataset` accepts `dataset` or `dataset:config` (e.g., `wikitext-2-raw-v1`)
- Device selection is automatic (`cuda` if available, else `cpu`)
- Only CausalLM architectures are supported for perplexity computation
- Uses a bounded sample count and expects a `text` field in the dataset

## More Examples

CPU-friendly int8 dynamic with pruning:
```bash
autopack quantize meta-llama/Llama-3-8B -o out/llama3-cpu \
  --output-format hf --quantization int8-dynamic --prune 0.2 --device-map cpu
```

BF16 only (no quantization):
```bash
autopack quantize meta-llama/Llama-3-8B -o out/llama3-bf16 \
  --output-format hf --quantization none --dtype bfloat16
```

Override GGUF presets:
```bash
autopack auto meta-llama/Llama-3-8B -o out/llama3-gguf \
  --output-format gguf --gguf-quant Q5_K_M Q8_0
```

Hello World (Transformers on CPU):
```bash
pip install -e .
autopack auto sshleifer/tiny-gpt2 -o out/tiny --output-format hf
python - <<'PY'
from transformers import AutoTokenizer, AutoModelForCausalLM
tok = AutoTokenizer.from_pretrained('out/tiny/bf16')
m   = AutoModelForCausalLM.from_pretrained('out/tiny/bf16', device_map='cpu')
ids = tok('Hello world', return_tensors='pt').input_ids
out = m.generate(ids, max_new_tokens=8)
print(tok.decode(out[0]))
PY
```

Hello World (GGUF with llama.cpp):
```bash
autopack auto sshleifer/tiny-gpt2 -o out/tiny-gguf --output-format gguf
./third_party/llama.cpp/build/bin/llama-cli -m out/tiny-gguf/gguf/model-Q4_K_M.gguf -p "Hello world" -n 16
```

## Vendored llama.cpp quick build

```bash
cd third_party/llama.cpp
cmake -S . -B build -DGGML_NATIVE=ON
cmake --build build -j
```

## Troubleshooting

- `llama-quantize` not found: build llama.cpp and ensure `build/bin` is in `PATH`.
- BitsAndBytes on Windows: currently not installed by default; prefer CPU/int8-dynamic flows.
- Custom code prompt: pass `--trust-remote-code` to avoid the interactive confirmation.

## Environment Variables

- `HUGGINGFACE_HUB_TOKEN`: token to publish to the Hub
- `LLAMA_CPP_CONVERT`: path to `convert_hf_to_gguf.py`
- `PATH`: should include the directory with `llama-quantize`

# Q&A

## FAQs

### What does “auto” do?

Generates HF variants (4-bit, 8-bit, int8-dynamic, bf16) and prints a summary; GGUF/ONNX are opt-in.

### What if I omit `--gguf-quant`?

autopack will create multiple useful presets by default (Q4_K_M, Q5_K_M, Q8_0).

---

License: Apache-2.0
