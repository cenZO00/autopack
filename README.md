<div align="center">
  <a href="https://granulavision.com">
    <img src="https://raw.githubusercontent.com/GranulaVision/autopack/refs/heads/main/media/banner.jpg">
  </a>

<h3 align="center">autopack</h3> 

[![PyPI Downloads](https://static.pepy.tech/personalized-badge/autopack-grn?period=total&units=INTERNATIONAL_SYSTEM&left_color=BLACK&right_color=GREEN&left_text=downloads)](https://pepy.tech/projects/autopack-grn)
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

You have a 120B LLM and want to optimize it so that people (not corporations with clusters of B200s) can use it on their 8GB 2060? All you need to do is run:

```bash
autopack sentence-transformers/all-MiniLM-L6-v2
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
pip install autopack-grn
```

### Optional extras

```bash
# ONNX export support
pip install 'autopack-grn[onnx]'

# GGUF export helpers (converter deps)
pip install 'autopack-grn[gguf]'

# llama.cpp runtime bindings (llama-cpp-python)
pip install 'autopack-grn[llama]'

# Everything for llama.cpp functionality (GGUF export + runtime)
pip install 'autopack-grn[gguf,llama]'
```

Note: for GGUF and llama.cpp functionality you also need the llama.cpp tools
(`llama-quantize`, `llama-cli`) available on your `PATH`. You can build the
vendored copy and export `PATH` as shown in
[Vendored llama.cpp quick build](#vendored-llamacpp-quick-build).

### From source (dev)

```bash
pip install -e .

# Optional extras while developing
pip install -e '.[onnx]'
pip install -e '.[gguf]'
pip install -e '.[llama]'
pip install -e '.[gguf,llama]'
```

# Building Instructions

```bash
python -m build
```

# Running the Application

## Quickstart

```bash
autopack meta-llama/Llama-3-8B --output-format hf
```

Add ONNX and GGUF:
```bash
autopack meta-llama/Llama-3-8B --output-format hf onnx gguf --summary-json --skip-existing
```

GGUF only (with default presets Q4_K_M, Q5_K_M, Q8_0):
```bash
autopack meta-llama/Llama-3-8B --output-format gguf --skip-existing
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
autopack [auto] <model_id_or_path> [-o <out_dir>] \
  --output-format hf [onnx] [gguf] \
  [--eval-dataset <dataset>[::<config>]] \
  [--revision <rev>] [--trust-remote-code] [--device auto|cpu|cuda] \
  [--no-bench] [--bench-prompt "..."] [--bench-max-new-tokens 16] \
  [--bench-warmup 0] [--bench-runs 1] \
  [--hf-variant bnb-4bit|bnb-8bit|int8-dynamic|bf16] \
  [--hf-variants bnb-4bit bnb-8bit int8-dynamic bf16]
```

Key points:
- Default HF variants: bnb-4bit, bnb-8bit, int8-dynamic, bf16
- Add ONNX and/or GGUF via `--output-format`
- If `-o/--output-dir` is omitted, the output folder defaults to the last path segment of the model id/path (e.g., `user/model` -> `model`).
- Benchmarking is enabled by default in `auto`; use `--no-bench` to disable.
- If `--eval-dataset` is provided, perplexity is computed for each HF variant
- If benchmarking is enabled, autopack measures actual Tokens/s per backend and replaces heuristic speedups with real Tokens/s and speedup vs bf16 in the summary and the generated README.
- For very large models, use `--hf-variant bf16` (single) or `--hf-variants bf16 int8-dynamic` (subset) to reduce loads.

### quantize

Produce specific formats with a chosen quantization strategy.

```bash
autopack quantize <model_id_or_path> [-o <out_dir>] \
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

### bench

Run standalone benchmarks on existing models/artifacts.

```bash
autopack bench <target> \
  --backend hf [onnx] [gguf] \
  [--prompt "Hello"] [--max-new-tokens 64] \
  [--device auto] [--num-warmup 1] [--num-runs 3] \
  [--trust-remote-code] [--llama-cli /path/to/llama-cli]
```

Notes:
- For HF, `target` can be a Hub id or local folder. For ONNX, pass the exported folder. For GGUF, pass a `.gguf` file or a folder containing one.
- ONNX benchmarking requires `optimum[onnxruntime]`. GGUF benchmarking requires `llama-cli`.

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
- `--eval-text-key` controls which dataset column is used for text (default: `text`)
- Device selection is automatic (`cuda` if available, else `cpu`)
- Only CausalLM architectures are supported for perplexity computation
- Uses a bounded sample count and expects a `text` field in the dataset

## More Examples

Single-variant run (bf16 only):
```bash
autopack meta-llama/Llama-3-8B --output-format hf --hf-variant bf16
```

Subset of variants:
```bash
autopack meta-llama/Llama-3-8B --output-format hf --hf-variants bf16 int8-dynamic
```

CPU-friendly int8 dynamic with pruning:
```bash
autopack quantize meta-llama/Llama-3-8B \
  --output-format hf --quantization int8-dynamic --prune 0.2 --device-map cpu
```

BF16 only (no quantization):
```bash
autopack quantize meta-llama/Llama-3-8B \
  --output-format hf --quantization none --dtype bfloat16
```

Override GGUF presets:
```bash
autopack meta-llama/Llama-3-8B \
  --output-format gguf --gguf-quant Q5_K_M Q8_0
```

Auto with benchmarking (reports Tokens/s and real speedup vs bf16):
```bash
autopack sshleifer/tiny-gpt2 --output-format hf
```

Hello World (Transformers on CPU):
```bash
pip install autopack-grn
autopack sshleifer/tiny-gpt2 --output-format hf
python - <<'PY'
from transformers import AutoTokenizer, AutoModelForCausalLM
tok = AutoTokenizer.from_pretrained('tiny-gpt2/bf16')
m   = AutoModelForCausalLM.from_pretrained('tiny-gpt2/bf16', device_map='cpu')
ids = tok('Hello world', return_tensors='pt').input_ids
out = m.generate(ids, max_new_tokens=8)
print(tok.decode(out[0]))
PY
```

Hello World (GGUF with llama.cpp):
```bash
autopack sshleifer/tiny-gpt2 --output-format gguf
./third_party/llama.cpp/build/bin/llama-cli -m tiny-gpt2/gguf/model-Q4_K_M.gguf -p "Hello world" -n 16
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

## Large models

For very large models (tens of GBs), prefer a minimal, resumable flow:

- **Single variant**: use `--hf-variant bf16` (or another) to avoid multiple loads
- **Avoid extra runs**: add `--no-bench`
- **Resume-friendly**: keep `--skip-existing`
- **CPU-safe**: add `--device cpu` to skip GPU-only paths

Examples:
```bash
# BF16 only, no benchmarking, resume if partial outputs exist
autopack user/model-giant --output-format hf \
  --hf-variant bf16 --no-bench --skip-existing

# CPU-focused subset
autopack user/model-giant --output-format hf \
  --hf-variants bf16 int8-dynamic --device cpu --no-bench --skip-existing
```

---

License: Apache-2.0
