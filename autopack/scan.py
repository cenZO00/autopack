import os
import json
from typing import Dict, Any, List, Optional, Tuple

from transformers import AutoConfig, AutoTokenizer, BitsAndBytesConfig
from huggingface_hub import HfApi, snapshot_download


def _dir_size_bytes(path: str) -> int:
    total = 0
    for root, _dirs, files in os.walk(path):
        for f in files:
            fp = os.path.join(root, f)
            try:
                total += os.path.getsize(fp)
            except OSError:
                pass
    return total


def _repo_sizes(repo_id: str, revision: Optional[str] = None) -> Tuple[int, int, int, bool]:
    """Return (total_bytes, weight_bytes, num_weight_files, safetensors_only) without downloading weights."""
    api = HfApi()
    info = api.repo_info(repo_id=repo_id, revision=revision, repo_type="model", files_metadata=True)
    total_bytes = 0
    weight_bytes = 0
    num_weight_files = 0
    safetensors_only = True
    siblings = getattr(info, "siblings", None) or getattr(info, "siblings_info", None) or []
    for s in siblings:
        # Try multiple ways to get filename and size for compatibility across hub versions
        name = getattr(s, "rfilename", None) or getattr(s, "path", None) or ""
        size = getattr(s, "size", None)
        if size is None:
            lfs = getattr(s, "lfs", None) or {}
            size = lfs.get("size", 0)
        size_int = int(size or 0)
        total_bytes += size_int
        if isinstance(name, str) and name.endswith(('.bin', '.pt', '.safetensors')):
            num_weight_files += 1
            weight_bytes += size_int
            if not name.endswith('.safetensors'):
                safetensors_only = False
    return total_bytes, weight_bytes, num_weight_files, safetensors_only


def _infer_export_targets(model_type: Optional[str]) -> Dict[str, bool]:
    llama_like = {"llama", "mistral", "mixtral", "qwen2", "qwen2_moe", "gemma", "phi3"}
    is_llama_family = (model_type or "").lower() in llama_like
    return {
        "hf": True,
        "onnx": True,
        "gguf": is_llama_family,
    }


def _detect_existing_quant(cfg: Any) -> Dict[str, Any]:
    qc = getattr(cfg, "quantization_config", None)
    if isinstance(qc, BitsAndBytesConfig):
        # Runtime config object; summarize key flags
        return {
            "type": "bitsandbytes",
            "bnb_4bit": bool(getattr(qc, "load_in_4bit", False)),
            "bnb_8bit": bool(getattr(qc, "load_in_8bit", False)),
        }
    if qc:
        # Likely a dict persisted in config
        kind = None
        if isinstance(qc, dict):
            # Heuristics for common formats
            if any(k in qc for k in ("desc_act", "group_size", "bits", "weight_dtype")):
                # Could be AWQ/GPTQ/etc.
                if "gptq" in json.dumps(qc).lower():
                    kind = "gptq"
                elif "awq" in json.dumps(qc).lower():
                    kind = "awq"
                else:
                    kind = "other"
        return {"type": kind or "custom", "config": qc}
    # Legacy flags
    if getattr(cfg, "load_in_8bit", False):
        return {"type": "bitsandbytes", "bnb_8bit": True}
    if getattr(cfg, "load_in_4bit", False):
        return {"type": "bitsandbytes", "bnb_4bit": True}
    return {}


def scan_model(
    model_id_or_path: str,
    revision: Optional[str] = None,
    trust_remote_code: bool = False,
    local_files_only: bool = False,
    resolve_cache: bool = False,
) -> Dict[str, Any]:
    """Collect lightweight metadata about a HF model or local folder."""
    is_local = os.path.isdir(model_id_or_path)
    resolved_path = model_id_or_path

    if not is_local and resolve_cache:
        try:
            resolved_path = snapshot_download(repo_id=model_id_or_path, revision=revision, local_files_only=local_files_only)
            is_local = True
        except Exception:
            # Fall back to remote scan
            resolved_path = None

    # Load config/tokenizer metadata (does not download large weights)
    cfg = AutoConfig.from_pretrained(
        model_id_or_path,
        revision=revision,
        trust_remote_code=trust_remote_code,
        local_files_only=local_files_only,
    )
    tok = None
    try:
        tok = AutoTokenizer.from_pretrained(
            model_id_or_path,
            revision=revision,
            trust_remote_code=trust_remote_code,
            local_files_only=local_files_only,
        )
    except Exception:
        pass

    model_type = getattr(cfg, "model_type", None)
    architectures = getattr(cfg, "architectures", None)
    max_positions = getattr(cfg, "max_position_embeddings", None) or getattr(cfg, "max_sequence_length", None)
    num_hidden_layers = getattr(cfg, "num_hidden_layers", None)
    hidden_size = getattr(cfg, "hidden_size", None) or getattr(cfg, "n_embd", None)
    num_attention_heads = getattr(cfg, "num_attention_heads", None) or getattr(cfg, "n_head", None)
    rope_scaling = getattr(cfg, "rope_scaling", None)

    quant_info = _detect_existing_quant(cfg)
    exports = _infer_export_targets(model_type)

    # Size estimation and file inventory
    total_bytes = 0
    weight_bytes = 0
    num_weight_files = 0
    safetensors_only = False
    total_files = 0
    tokenizer_required = {
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
        "special_tokens_map.json",
    }
    tokenizer_present: set[str] = set()
    has_config_json = False
    weight_safetensors_count = 0
    weight_other_count = 0
    top_entries: List[Tuple[str, int]] = []  # (path/name, size)
    if is_local and resolved_path:
        total_bytes = _dir_size_bytes(resolved_path)
        weight_files: List[str] = []
        for root, _dirs, files in os.walk(resolved_path):
            for name in files:
                total_files += 1
                rel = os.path.relpath(os.path.join(root, name), resolved_path)
                fp = os.path.join(root, name)
                try:
                    sz = os.path.getsize(fp)
                except OSError:
                    sz = 0
                top_entries.append((rel, int(sz)))
                if name == "config.json":
                    has_config_json = True
                if name in tokenizer_required:
                    tokenizer_present.add(name)
            # weights tally
            for name in files:
                if name.endswith(('.safetensors', '.bin', '.pt')):
                    fp = os.path.join(root, name)
                    weight_files.append(fp)
                    try:
                        weight_bytes += os.path.getsize(fp)
                    except OSError:
                        pass
                    if name.endswith('.safetensors'):
                        weight_safetensors_count += 1
                    else:
                        weight_other_count += 1
        num_weight_files = len(weight_files)
        safetensors_only = (num_weight_files > 0 and all(f.endswith('.safetensors') for f in weight_files))
    else:
        try:
            total_bytes, weight_bytes, num_weight_files, safetensors_only = _repo_sizes(model_id_or_path, revision)
            # Query file list for counts and top files using repo_info again
            api = HfApi()
            info = api.repo_info(repo_id=model_id_or_path, revision=revision, repo_type="model", files_metadata=True)
            siblings = getattr(info, "siblings", None) or getattr(info, "siblings_info", None) or []
            for s in siblings:
                name = getattr(s, "rfilename", None) or getattr(s, "path", None) or ""
                size = getattr(s, "size", None)
                if size is None:
                    lfs = getattr(s, "lfs", None) or {}
                    size = lfs.get("size", 0)
                size_int = int(size or 0)
                if not isinstance(name, str):
                    continue
                total_files += 1
                top_entries.append((name, size_int))
                if name == "config.json":
                    has_config_json = True
                base = os.path.basename(name)
                if base in tokenizer_required:
                    tokenizer_present.add(base)
                if base.endswith('.safetensors'):
                    weight_safetensors_count += 1
                elif base.endswith(('.bin', '.pt')):
                    weight_other_count += 1
        except Exception:
            pass

    tok_name = type(tok).__name__ if tok is not None else None
    has_chat_template = False
    try:
        has_chat_template = bool(getattr(tok, "chat_template", None))
    except Exception:
        has_chat_template = False

    # sort top entries by size desc
    top_entries.sort(key=lambda x: x[1], reverse=True)

    return {
        "model": model_id_or_path,
        "resolved_path": resolved_path if is_local else None,
        "is_local": is_local,
        "config": {
            "model_type": model_type,
            "architectures": architectures,
            "hidden_size": hidden_size,
            "num_hidden_layers": num_hidden_layers,
            "num_attention_heads": num_attention_heads,
            "max_position_embeddings": max_positions,
            "rope_scaling": rope_scaling,
        },
        "tokenizer": {
            "class": tok_name,
            "chat_template": has_chat_template,
        },
        "quantization": quant_info,
        "sizes": {
            "total_bytes": int(total_bytes),
            "weight_bytes": int(weight_bytes),
            "num_weight_files": int(num_weight_files),
            "safetensors_only": bool(safetensors_only),
        },
        "files": {
            "total": int(total_files),
            "config_present": bool(has_config_json),
            "tokenizer": {
                "present": int(len(tokenizer_present)),
                "required": int(len(tokenizer_required)),
                "missing": sorted(list(tokenizer_required - tokenizer_present)),
            },
            "weights": {
                "safetensors": int(weight_safetensors_count),
                "other": int(weight_other_count),
            },
            "top": [
                {"name": name, "size_bytes": int(size)} for name, size in top_entries
            ],
        },
        "export_support": exports,
        "suggestions": _suggest_next_steps(model_id_or_path, exports, quant_info),
    }


def _suggest_next_steps(model: str, exports: Dict[str, bool], quant_info: Dict[str, Any]) -> List[str]:
    recs: List[str] = []
    if not quant_info:
        recs.append(f"autopack quantize '{model}' -o {os.path.basename(model.rstrip('/'))} --output-format hf")
    if exports.get("onnx", False):
        recs.append(f"autopack quantize '{model}' -o out --output-format onnx")
    if exports.get("gguf", False):
        recs.append(f"autopack quantize '{model}' -o out --output-format gguf --gguf-quant Q4_K_M")
    return recs


def print_scan_report(report: Dict[str, Any], as_json: bool = False, show_files: bool = False, limit_files: int = 50) -> None:
    if as_json:
        print(json.dumps(report, indent=2))
        return
    def _human_size(num_bytes: int) -> str:
        units = ["B", "KB", "MB", "GB", "TB"]
        size = float(num_bytes)
        for unit in units:
            if size < 1024.0 or unit == units[-1]:
                return f"{size:.2f} {unit}"
            size /= 1024.0
    cfg = report.get("config", {})
    sizes = report.get("sizes", {})
    tok = report.get("tokenizer", {})
    files = report.get("files", {})
    print("Model:", report.get("model"))
    if report.get("resolved_path"):
        print("Resolved path:", report.get("resolved_path"))
    print("- model_type:", cfg.get("model_type"))
    print("- architectures:", cfg.get("architectures"))
    print("- hidden_size / layers / heads:", cfg.get("hidden_size"), cfg.get("num_hidden_layers"), cfg.get("num_attention_heads"))
    print("- max_position_embeddings:", cfg.get("max_position_embeddings"))
    if cfg.get("rope_scaling"):
        print("- rope_scaling:", cfg.get("rope_scaling"))
    q = report.get("quantization") or {}
    if q:
        print("Existing quantization:", q)
    print("Sizes:")
    print("  - total:", _human_size(int(sizes.get('total_bytes', 0))) )
    print("  - weights:", _human_size(int(sizes.get('weight_bytes', 0))), f"({sizes.get('num_weight_files', 0)} files)")
    print("  - safetensors_only:", sizes.get("safetensors_only"))
    if files:
        print("Files summary:")
        print("  - total:", files.get("total", 0))
        print("  - config.json present:", files.get("config_present", False))
        tk = files.get("tokenizer", {})
        print("  - tokenizer files:", f"{tk.get('present',0)}/{tk.get('required',0)} present")
        missing = tk.get("missing", [])
        if missing:
            print("    missing:", ", ".join(missing))
        w = files.get("weights", {})
        print("  - weight files:", f"{w.get('safetensors',0)} safetensors, {w.get('other',0)} other")
        top = files.get("top", [])
        if top:
            print("Top files:")
            for entry in top[:5]:
                print(f"  - {entry.get('name')}  {_human_size(int(entry.get('size_bytes',0)))}")
    print("Tokenizer:")
    print("  - class:", tok.get("class"))
    print("  - chat_template:", tok.get("chat_template"))
    print("Export support:")
    for k, v in (report.get("export_support") or {}).items():
        print(f"  - {k}: {v}")
    if show_files:
        top = (report.get("files", {}) or {}).get("top", [])
        if top:
            print("All (top N) files:")
            for entry in top[: max(1, limit_files)]:
                print(f"  - {entry.get('name')}  {_human_size(int(entry.get('size_bytes',0)))}")
    if report.get("suggestions"):
        print("\nNext steps:")
        for s in report["suggestions"]:
            print("  -", s)


