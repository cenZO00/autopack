from dataclasses import dataclass
from typing import Optional, Type

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


def _get_auto_model_class(model_id_or_path: str, revision: Optional[str] = None) -> Type[AutoModel]:
    """Inspect model config to determine which AutoModel class to use."""
    config = AutoConfig.from_pretrained(model_id_or_path, revision=revision)
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
) -> str:
    """Load a model with bitsandbytes quantization and save in HF format.

    Returns the output_dir.
    """
    if quantization not in {"bnb-4bit", "bnb-8bit", "int8-dynamic", "none"}:
        raise ValueError("quantization must be one of: 'bnb-4bit', 'bnb-8bit', 'int8-dynamic', 'none'")

    quant_config = _build_bnb_config(quantization, dtype)

    tokenizer = AutoTokenizer.from_pretrained(
        model_id_or_path,
        revision=revision,
        use_fast=True,
        trust_remote_code=trust_remote_code,
    )

    if quantization == "int8-dynamic":
        AutoModelClass = _get_auto_model_class(model_id_or_path, revision)
        # Load in float on CPU, then apply PyTorch dynamic quantization to Linear layers
        model = AutoModelClass.from_pretrained(
            model_id_or_path,
            revision=revision,
            trust_remote_code=trust_remote_code,
            device_map="cpu",
            dtype=torch.float32,
        )
        if prune and prune > 0.0:
            apply_global_magnitude_pruning(model, prune)
        model = tq.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
    elif quant_config is not None:
        AutoModelClass = _get_auto_model_class(model_id_or_path, revision)
        model = AutoModelClass.from_pretrained(
            model_id_or_path,
            revision=revision,
            trust_remote_code=trust_remote_code,
            device_map=device_map,
            quantization_config=quant_config,
        )
    else:
        torch_dtype = _DTYPE_MAP.get(dtype)
        AutoModelClass = _get_auto_model_class(model_id_or_path, revision)
        model = AutoModelClass.from_pretrained(
            model_id_or_path,
            revision=revision,
            trust_remote_code=trust_remote_code,
            device_map=device_map,
            dtype=torch_dtype,
        )
        if prune and prune > 0.0:
            apply_global_magnitude_pruning(model, prune)
    # For bnb and none paths, optionally prune after load above. For int8-dynamic we already pruned before quant.
    if quantization in {"bnb-4bit", "bnb-8bit"} and prune and prune > 0.0:
        apply_global_magnitude_pruning(model, prune)

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
    return output_dir


