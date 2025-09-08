<p align="center">
  <img src="media/banner.svg" alt="autopack banner" width="100%"/>
</p>

## autopack

Make your Hugging Face models easy to run, share, and ship — in one go.

- Quantize once, export to multiple runtimes
- Sensible defaults with an “auto” flow and a readable summary
- Opt-in formats: HF, ONNX, GGUF (llama.cpp), GGML (experimental)
- One-shot publish to the Hugging Face Hub

### Why use autopack?
- Fast: generate multiple useful variants in a single command
- Practical: built around widely-used stacks (Transformers, bitsandbytes, ONNX, llama.cpp)
- Portable: CPU- and GPU-friendly artifacts, good defaults, easy to try

---

### Install

```bash
# Local development install
pip install -e .
```



Notes:
- GGUF export requires `llama.cpp` built locally (converter script and `llama-quantize` available).
- Set `HUGGINGFACE_HUB_TOKEN` to publish to the Hub, or pass `--token`.
- `bitsandbytes` and dynamic INT8 are optional; CPU-only flows are supported.

---

### Quickstart

1) Auto-flow (HF variants only, default):
```bash
autopack auto meta-llama/Llama-3-8B -o out/llama3
```
This produces: `bnb-4bit`, `bnb-8bit`, `int8-dynamic`, and `bf16` under `out/llama3/`, plus a summary.

2) Add ONNX and GGUF to auto:
```bash
autopack auto meta-llama/Llama-3-8B -o out/llama3 --output-format hf onnx gguf
```

3) GGUF-only (default multi-quant when unspecified):
```bash
autopack auto meta-llama/Llama-3-8B -o out/llama3-gguf --output-format gguf
# Defaults: Q4_K_M, Q5_K_M, Q8_0
```

4) Override GGUF quant presets:
```bash
autopack auto meta-llama/Llama-3-8B -o out/llama3-gguf --output-format gguf \
  --gguf-quant Q5_K_M Q8_0
```

5) Manual quantize flows:
```bash
# HF only (4-bit by default)
autopack quantize meta-llama/Llama-3-8B -o out/llama3-4bit

# HF + ONNX
autopack quantize meta-llama/Llama-3-8B -o out/llama3 --output-format hf onnx

# CPU-friendly (int8 dynamic + pruning)
autopack quantize meta-llama/Llama-3-8B -o out/llama3-cpu \
  --quantization int8-dynamic --prune 0.2 --device-map cpu

# GGUF only
autopack quantize meta-llama/Llama-3-8B -o out/llama3 --output-format gguf
```

6) Publish to the Hub:
```bash
autopack publish out/llama3-4bit your-username/llama3-4bit --private \
  --commit-message "Add 4-bit quantized weights"
```

---

### Format notes

- HF: quantized Transformers checkpoints (bnb 4/8-bit, int8-dynamic, bf16)
- ONNX: exported with `optimum[onnxruntime]` (install optional extra)
- GGUF (llama.cpp):
  - Build `llama.cpp` locally or use the vendored copy under `third_party/llama.cpp`
  - Provide `--gguf-converter` or set `LLAMA_CPP_CONVERT` if autodetection fails
  - To quantize GGUF, ensure `llama-quantize` is in your PATH
  - If you don’t pass `--gguf-quant`, autopack will create multiple useful presets by default
- GGML (experimental): currently raises a guided error explaining how to supply a converter

---

### llama.cpp (vendored) quick build

```bash
cd third_party/llama.cpp
cmake -S . -B build -DGGML_NATIVE=ON
cmake --build build -j
```

After building, `autopack` will try these converter locations in order:
- `third_party/llama.cpp/convert_hf_to_gguf.py`
- `$LLAMA_CPP_CONVERT` (env var)
- `~/llama.cpp/convert_hf_to_gguf.py` or `~/src/llama.cpp/convert_hf_to_gguf.py`

Optional: Python bindings for llama.cpp
```bash
pip install .[llama]
```

---


### Hello World

Smallest end-to-end on CPU, using a tiny model:

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

Hello World with GGUF (requires built llama.cpp):

```bash
autopack auto sshleifer/tiny-gpt2 -o out/tiny-gguf --output-format gguf
./third_party/llama.cpp/build/bin/llama-cli -m out/tiny-gguf/gguf/model-Q4_K_M.gguf -p "Hello world" -n 16
```

### FAQ

- What does “auto” do by default?
  - Generates HF variants (4-bit, 8-bit, int8-dynamic, bf16) and prints a summary.
- Will it export GGUF automatically?
  - No. GGUF is opt-in: add `--output-format gguf` (or combine with `hf`/`onnx`).
- If I ask for GGUF and don’t specify a quant level, what happens?
  - It will create multiple useful presets by default (Q4_K_M, Q5_K_M, Q8_0).
- GGML support?
  - Experimental: you’ll get a clear guidance error until a converter is supplied.

---

### License

MIT
