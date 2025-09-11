from dataclasses import dataclass
from typing import Optional, Type
import gc

import logging
import torch
import torch.quantization as tq
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from .prune import apply_global_magnitude_pruning

logger = logging.getLogger(__name__)


_DTYPE_MAP = {
    "auto": None,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def _get_auto_model_class(
    model_id_or_path: str,
    revision: Optional[str] = None,
    trust_remote_code: bool = False,
    local_files_only: bool = False,
) -> Type[AutoModel]:
    """Inspect model config to determine which AutoModel class to use."""
    config = AutoConfig.from_pretrained(
        model_id_or_path, revision=revision, trust_remote_code=trust_remote_code, local_files_only=local_files_only
    )
    archs = config.architectures
    if not archs:
        # Fallback for models with no architecture specified (rare)
        return AutoModelForCausalLM

    # Heuristic: search for a task-specific architecture
    for arch in archs:
        if "CausalLM" in arch:
            return AutoModelForCausalLM
        if "MaskedLM" in arch:
            return AutoModelForMaskedLM
    # Fallback for models that don't fit a clear task type (e.g., encoders)
    return AutoModel


@dataclass
class QuantizeArgs:
    model_id_or_path: str
    output_dir: str
    quantization: str = "bnb-4bit"  # ["bnb-4bit", "bnb-8bit", "none"]
    dtype: str = "bfloat16"  # ["auto", "float16", "bfloat16", "float32"]
    device_map: str = "auto"
    trust_remote_code: bool = False
    revision: Optional[str] = None
    prune: float = 0.0


def _build_bnb_config(quantization: str, dtype: str) -> Optional[BitsAndBytesConfig]:
    compute_dtype = _DTYPE_MAP.get(dtype)
    if quantization == "bnb-4bit":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype or torch.bfloat16,
        )
    if quantization == "bnb-8bit":
        return BitsAndBytesConfig(
            load_in_8bit=True,
        )
    return None


def quantize_to_hf(
    model_id_or_path: str,
    output_dir: str,
    quantization: str = "bnb-4bit",
    dtype: str = "bfloat16",
    device_map: str = "auto",
    trust_remote_code: bool = False,
    revision: Optional[str] = None,
    prune: float = 0.0,
    local_files_only: bool = False,
) -> str:
    """Load a model with bitsandbytes quantization and save in HF format.

    Returns the output_dir.
    """
    if quantization not in {"bnb-4bit", "bnb-8bit", "int8-dynamic", "none"}:
        raise ValueError("quantization must be one of: 'bnb-4bit', 'bnb-8bit', 'int8-dynamic', 'none'")

    # Detect if the source model is already pre-quantized with a non-BitsAndBytes
    # quantizer (e.g., MxFP4). If so, avoid passing a BitsAndBytesConfig which
    # would conflict with the existing quantization config.
    try:
        src_config = AutoConfig.from_pretrained(
            model_id_or_path,
            revision=revision,
            trust_remote_code=trust_remote_code,
            local_files_only=local_files_only,
        )
        existing_qc = getattr(src_config, "quantization_config", None)
        is_pre_quantized = bool(existing_qc) and not isinstance(existing_qc, BitsAndBytesConfig)
    except Exception:
        src_config = None
        is_pre_quantized = False

    quant_config = _build_bnb_config(quantization, dtype)
    if is_pre_quantized and quant_config is not None:
        logger.warning(
            "Detected existing non-BitsAndBytes quantization in source model; "
            "skipping BitsAndBytes quantization and loading as-is."
        )
        quant_config = None

    tokenizer = AutoTokenizer.from_pretrained(
        model_id_or_path,
        revision=revision,
        use_fast=True,
        trust_remote_code=trust_remote_code,
        local_files_only=local_files_only,
    )

    if quantization == "int8-dynamic":
        AutoModelClass = _get_auto_model_class(
            model_id_or_path, revision, trust_remote_code=trust_remote_code, local_files_only=local_files_only
        )
        # Load in float on CPU, then apply PyTorch dynamic quantization to Linear layers
        model = AutoModelClass.from_pretrained(
            model_id_or_path,
            revision=revision,
            trust_remote_code=trust_remote_code,
            device_map="cpu",
            dtype=torch.float32,
            local_files_only=local_files_only,
        )
        if prune and prune > 0.0:
            apply_global_magnitude_pruning(model, prune)
        model = tq.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
    elif quant_config is not None:
        AutoModelClass = _get_auto_model_class(
            model_id_or_path, revision, trust_remote_code=trust_remote_code, local_files_only=local_files_only
        )
        # If user requested CPU device map, force CPU to avoid CUDA allocations from BnB/MxFP4 integrations
        effective_device_map = device_map
        if isinstance(effective_device_map, str) and effective_device_map.lower() == "cpu":
            effective_device_map = "cpu"
        model = AutoModelClass.from_pretrained(
            model_id_or_path,
            revision=revision,
            trust_remote_code=trust_remote_code,
            device_map=effective_device_map,
            quantization_config=quant_config,
            local_files_only=local_files_only,
        )
    else:
        torch_dtype = _DTYPE_MAP.get(dtype)
        AutoModelClass = _get_auto_model_class(
            model_id_or_path, revision, trust_remote_code=trust_remote_code, local_files_only=local_files_only
        )
        # Respect explicit CPU request; avoids CUDA allocations during dequant of pre-quantized models
        effective_device_map = device_map
        if isinstance(effective_device_map, str) and effective_device_map.lower() == "cpu":
            effective_device_map = "cpu"
        model = AutoModelClass.from_pretrained(
            model_id_or_path,
            revision=revision,
            trust_remote_code=trust_remote_code,
            device_map=effective_device_map,
            dtype=torch_dtype,
            local_files_only=local_files_only,
        )
        if prune and prune > 0.0:
            apply_global_magnitude_pruning(model, prune)
    # For bnb and none paths, optionally prune after load above. For int8-dynamic we already pruned before quant.
    if quantization in {"bnb-4bit", "bnb-8bit"} and prune and prune > 0.0:
        apply_global_magnitude_pruning(model, prune)

    # Ensure inference mode prior to saving
    model.eval()
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    # Robust save: try safetensors, fallback to PyTorch if shared tensors error
    try:
        model.save_pretrained(output_dir, safe_serialization=True)
    except Exception as e:
        logger.debug(f"Safe serialization failed: {e}")
        try:
            # Fallback when tensors share storage (e.g., some BERT heads)
            model.save_pretrained(output_dir, safe_serialization=False)
        except Exception as e2:
            logger.debug(f"Standard serialization also failed: {e2}")
            # Last resort: save state dict manually
            import os
            os.makedirs(output_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))
            # Save config
            model.config.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    # Proactively free memory to avoid OOM across sequential variants
    try:
        del model
    except Exception:
        pass
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    gc.collect()
    return output_dir


