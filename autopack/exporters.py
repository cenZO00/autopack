import os
import shutil
import subprocess
from typing import Optional, Sequence
from pathlib import Path
import sys
import venv

from transformers import AutoTokenizer, AutoConfig
from huggingface_hub import snapshot_download


def export_onnx(
    model_id_or_path: str,
    output_dir: str,
    trust_remote_code: bool = False,
    revision: Optional[str] = None,
) -> str:
    """Export a model to ONNX using Optimum ORT wrappers."""
    try:
        from optimum.onnxruntime import ORTModelForCausalLM
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "ONNX export requires 'optimum[onnxruntime]' optional dependency."
        ) from exc

    os.makedirs(output_dir, exist_ok=True)
    model = ORTModelForCausalLM.from_pretrained(
        model_id_or_path,
        export=True,
        revision=revision,
        trust_remote_code=trust_remote_code,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_id_or_path, revision=revision, trust_remote_code=trust_remote_code
    )

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    return output_dir


def _find_gguf_converter(converter_path: Optional[str] = None) -> str:
    """Find the llama.cpp convert_hf_to_gguf.py script."""
    # Resolve converter script
    if converter_path is None:
        env_path = os.environ.get("LLAMA_CPP_CONVERT")
        # Try vendored llama.cpp under third_party relative to repo root
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        vendored_convert = os.path.join(repo_root, "third_party", "llama.cpp", "convert_hf_to_gguf.py")
        candidates = [
            env_path,
            vendored_convert,
            os.path.expanduser("~/llama.cpp/convert_hf_to_gguf.py"),
            os.path.expanduser("~/src/llama.cpp/convert_hf_to_gguf.py"),
        ]
        converter_path = next((p for p in candidates if p and os.path.isfile(p)), None)
    if not converter_path or not os.path.isfile(converter_path):
        raise RuntimeError(
            "Could not locate llama.cpp convert_hf_to_gguf.py. Provide --gguf-converter or set LLAMA_CPP_CONVERT."
        )
    return converter_path


def export_gguf(
    model_id_or_path: str,
    output_dir: str,
    quant: Optional[str] = None,
    converter_path: Optional[str] = None,
    trust_remote_code: bool = False,
    revision: Optional[str] = None,
    extra_args: Optional[Sequence[str]] = None,
    env: Optional[dict] = None,
    isolate_env: bool = True,
    force: bool = False,
) -> str:
    """Export a model to GGUF format via llama.cpp."""

    # --- Determine source and basic support ---
    # Fast architecture guard to avoid spinning up envs for unsupported models
    try:
        cfg = AutoConfig.from_pretrained(model_id_or_path, revision=revision, trust_remote_code=trust_remote_code)
        archs = cfg.architectures or []
    except Exception:
        archs = []

    unsupported_markers = ("Bert", "GPT2", "OPT", "T5", "Whisper", "Wav2Vec")
    if not force and any(marker in arch for arch in archs for marker in unsupported_markers):
        raise RuntimeError(
            f"GGUF export: architecture {archs} is not supported by llama.cpp converter. "
            "Try a LLaMA/Mistral/Gemma/Qwen/Phi family model or pass --gguf-force to bypass."
        )

    # --- Environment Isolation (optional) ---
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    requirements_path = os.path.join(
        repo_root, "third_party", "llama.cpp", "requirements", "requirements-convert_hf_to_gguf.txt"
    )

    python_executable = sys.executable
    venv_dir = None
    if isolate_env:
        try:
            venv_dir = os.path.join(output_dir, ".venv")
            venv.create(venv_dir, with_pip=True)
            # Handle Windows vs POSIX venv layout
            bin_dir = os.path.join(venv_dir, "Scripts") if os.name == "nt" else os.path.join(venv_dir, "bin")
            pip_executable = os.path.join(bin_dir, "pip")
            python_executable = os.path.join(bin_dir, "python")
            env_vars = dict(env or os.environ)
            env_vars.setdefault("PIP_DISABLE_PIP_VERSION_CHECK", "1")
            env_vars.setdefault("PIP_NO_INPUT", "1")
            subprocess.run(
                [pip_executable, "install", "-r", requirements_path], check=True, text=True, env=env_vars
            )
        except Exception as exc:
            print(
                f"GGUF: could not prepare isolated env ({exc}). Falling back to system Python."
            )
            python_executable = sys.executable
            venv_dir = None

    # Resolve model_id_or_path to a local directory; download from Hub if needed
    hf_dir = os.path.abspath(model_id_or_path)
    downloaded_temp_dir: Optional[str] = None
    if not os.path.isdir(hf_dir):
        # Download snapshot to a local temp folder under output_dir
        downloaded_temp_dir = os.path.join(output_dir, "_hf_src")
        os.makedirs(downloaded_temp_dir, exist_ok=True)
        hf_dir = snapshot_download(
            repo_id=model_id_or_path,
            revision=revision,
            local_dir=downloaded_temp_dir,
            local_dir_use_symlinks=False,
        )

    # Resolve converter script
    converter_path = _find_gguf_converter(converter_path)

    # Run llama.cpp converter
    gguf_unquantized = os.path.join(output_dir, "model-f16.gguf")
    convert_cmd = [
        python_executable,
        converter_path,
        hf_dir,
        "--outfile",
        gguf_unquantized,
        "--outtype",
        "f16",
    ]
    if extra_args:
        convert_cmd.extend(extra_args)
    subprocess.run(convert_cmd, check=True, text=True, env=env)

    if not quant:
        # Clean up venv and any temp snapshot, then return unquantized path
        if venv_dir and os.path.isdir(venv_dir):
            shutil.rmtree(venv_dir, ignore_errors=True)
        if downloaded_temp_dir and os.path.isdir(downloaded_temp_dir):
            shutil.rmtree(downloaded_temp_dir, ignore_errors=True)
        return gguf_unquantized

    # Run llama.cpp quantizer
    quantized_path = os.path.join(output_dir, f"model-{quant}.gguf")
    quantize_cmd = ["llama-quantize", gguf_unquantized, quantized_path, quant]
    try:
        subprocess.run(quantize_cmd, check=True, text=True, env=env)
    except FileNotFoundError as exc:
        raise RuntimeError("'llama-quantize' not found. Build llama.cpp and ensure build/bin is in PATH.") from exc

    # Clean up unquantized GGUF, venv, and any temp snapshot
    os.remove(gguf_unquantized)
    if venv_dir and os.path.isdir(venv_dir):
        shutil.rmtree(venv_dir, ignore_errors=True)
    if downloaded_temp_dir and os.path.isdir(downloaded_temp_dir):
        shutil.rmtree(downloaded_temp_dir, ignore_errors=True)

    return quantized_path



# No GGML export in this version

