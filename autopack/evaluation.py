import math
import os
from typing import Any, Dict, Optional, List, Tuple

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import time


def _measure_generation_latency(
    generate_fn,
    prompt_ids: torch.Tensor,
    num_warmup: int = 1,
    num_runs: int = 3,
) -> Tuple[float, float, int]:
    """Measure average latency (s) and tokens/s for a generation callable.

    The callable should accept `input_ids` and return generated ids.
    """
    # Warmup
    for _ in range(num_warmup):
        _ = generate_fn(prompt_ids)

    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    start = time.perf_counter()
    total_new_tokens = 0
    total_new_tokens = 0
    for _ in range(num_runs):
        out_ids = generate_fn(prompt_ids)
        total_new_tokens += max(0, out_ids.shape[1] - prompt_ids.shape[1])
    end = time.perf_counter()
    duration = (end - start) / max(1, num_runs)
    tps = (total_new_tokens / max(1, num_runs)) / max(duration, 1e-9)
    avg_new_tokens = int(round(total_new_tokens / max(1, num_runs)))
    return duration, tps, avg_new_tokens


def calculate_perplexity(
    model_id_or_path: str,
    dataset_id: str,
    dataset_config: str,
    dataset_split: str = "test",
    n_samples: int = 256,
    device: str = "auto",
    trust_remote_code: bool = False,
    text_key: str = "text",
) -> float:
    """Calculate perplexity of a model on a dataset.

    Note: this is a simplified implementation for general comparison.
    """
    config = AutoConfig.from_pretrained(model_id_or_path, trust_remote_code=trust_remote_code)
    archs = getattr(config, "architectures", []) or []
    if not any("CausalLM" in str(arch) for arch in archs):
        raise TypeError(
            f"Perplexity calculation is only supported for CausalLM models, but got {archs}"
        )

    # Select device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = AutoModelForCausalLM.from_pretrained(
        model_id_or_path, trust_remote_code=trust_remote_code
    ).to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_id_or_path, trust_remote_code=trust_remote_code)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    dataset_kwargs = {"split": dataset_split}
    if dataset_config:
        dataset_kwargs["name"] = dataset_config
    dataset = load_dataset(dataset_id, **dataset_kwargs)
    text_list = []
    for sample in dataset.select(range(min(n_samples, len(dataset)))):
        txt = sample.get(text_key)
        if txt:
            text_list.append(txt)

    max_ctx = getattr(model.config, "max_position_embeddings", None) or 2048
    encodings = tokenizer(
        text_list,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_ctx,
    ).to(device)

    max_length = encodings.input_ids.shape[1]
    seq_len = max_length

    nlls = []
    prev_end_loc = 0
    for begin_loc in range(0, seq_len, max_length):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.inference_mode():
            outputs = model(input_ids, labels=target_ids)
            neg_log_likelihood = outputs.loss

        nlls.append(neg_log_likelihood)
        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    ppl = torch.exp(torch.stack(nlls).mean())
    return ppl.item()


def benchmark_hf(
    model_id_or_path: str,
    prompt: str = "Hello world",
    max_new_tokens: int = 64,
    device: str = "auto",
    trust_remote_code: bool = False,
    num_warmup: int = 1,
    num_runs: int = 3,
) -> Dict[str, Any]:
    """Benchmark Transformers (HF) model generation latency and throughput."""
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_id_or_path, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id_or_path, trust_remote_code=trust_remote_code
    ).to(device)
    model.eval()

    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    def _gen(input_ids: torch.Tensor) -> torch.Tensor:
        with torch.inference_mode():
            out = model.generate(input_ids, max_new_tokens=max_new_tokens)
        return out

    latency_s, tokens_per_s, new_tokens = _measure_generation_latency(_gen, prompt_ids, num_warmup, num_runs)

    max_mem = None
    if device == "cuda":
        try:
            max_mem = torch.cuda.max_memory_allocated()
            torch.cuda.reset_max_memory_allocated()
        except Exception:
            pass

    return {
        "backend": "hf",
        "latency_s": latency_s,
        "tokens_per_s": tokens_per_s,
        "new_tokens": new_tokens,
        "device": device,
        "max_memory_bytes": max_mem,
    }


def benchmark_onnx(
    model_dir: str,
    prompt: str = "Hello world",
    max_new_tokens: int = 64,
    device: str = "auto",
    num_warmup: int = 1,
    num_runs: int = 3,
) -> Dict[str, Any]:
    """Benchmark ONNXRuntime model (exported via optimum) if available."""
    try:
        from optimum.onnxruntime import ORTModelForCausalLM
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("ONNX benchmarking requires 'optimum[onnxruntime]'.") from exc

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model = ORTModelForCausalLM.from_pretrained(model_dir)
    model.eval()

    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids

    def _gen(input_ids: torch.Tensor) -> torch.Tensor:
        # ORT model expects CPU tensors; generation API mirrors HF
        with torch.inference_mode():
            out = model.generate(input_ids, max_new_tokens=max_new_tokens)
        return out

    latency_s, tokens_per_s, new_tokens = _measure_generation_latency(_gen, prompt_ids, num_warmup, num_runs)
    return {
        "backend": "onnx",
        "latency_s": latency_s,
        "tokens_per_s": tokens_per_s,
        "new_tokens": new_tokens,
        "device": device,
        "max_memory_bytes": None,
    }


def benchmark_gguf(
    gguf_model_path: str,
    prompt: str = "Hello world",
    max_new_tokens: int = 64,
    llama_cli_path: Optional[str] = None,
    num_warmup: int = 1,
    num_runs: int = 3,
) -> Dict[str, Any]:
    """Benchmark llama.cpp via subprocess using llama-cli if available.

    We call llama-cli for generation and time the process wall-clock. Tokens/s are parsed from stdout when available,
    otherwise computed from elapsed time and token count.
    """
    import subprocess
    import shlex

    # Resolve llama-cli
    if llama_cli_path is None:
        # Try vendored build path first
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        candidate = os.path.join(repo_root, "third_party", "llama.cpp", "build", "bin", "llama-cli")
        llama_cli_path = candidate if os.path.isfile(candidate) else "llama-cli"

    def _run_once() -> Tuple[float, int, Optional[float]]:
        cmd = [
            llama_cli_path,
            "-m",
            gguf_model_path,
            "-p",
            prompt,
            "-n",
            str(max_new_tokens),
        ]
        t0 = time.perf_counter()
        try:
            proc = subprocess.run(cmd, capture_output=True, check=True, text=True)
            stdout = proc.stdout
        except Exception as exc:
            raise RuntimeError(f"Failed to run llama-cli: {exc}")
        dt = time.perf_counter() - t0
        # Try to parse tokens/s from stdout footer lines like: "eval time ... tok/s"
        parsed_tps = None
        for line in stdout.splitlines()[-10:]:
            if "tok/s" in line:
                # crude parse: take last float before 'tok/s'
                try:
                    parts = line.replace(",", " ").split()
                    for i, p in enumerate(parts):
                        if p.startswith("tok/s") or p.endswith("tok/s"):
                            # previous numeric token may be the tps
                            for j in range(i - 1, -1, -1):
                                try:
                                    parsed_tps = float(parts[j])
                                    break
                                except ValueError:
                                    continue
                            break
                except Exception:
                    pass
                if parsed_tps is not None:
                    break
        return dt, max_new_tokens, parsed_tps

    # Warmup
    for _ in range(num_warmup):
        _run_once()
    # Timed runs
    total_dt = 0.0
    total_tokens = 0
    parsed_tps_values: List[float] = []
    for _ in range(num_runs):
        dt, num_tokens, parsed_tps = _run_once()
        total_dt += dt
        total_tokens += num_tokens
        if parsed_tps is not None:
            parsed_tps_values.append(parsed_tps)

    avg_dt = total_dt / max(1, num_runs)
    computed_tps = (total_tokens / max(1, num_runs)) / max(avg_dt, 1e-9)
    tokens_per_s = parsed_tps_values[-1] if parsed_tps_values else computed_tps

    return {
        "backend": "gguf",
        "latency_s": avg_dt,
        "tokens_per_s": tokens_per_s,
        "new_tokens": max_new_tokens,
        "device": "cpu",
        "max_memory_bytes": None,
    }
