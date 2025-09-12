from dataclasses import dataclass
from typing import Optional, Type
import gc
import psutil
import logging
import torch
import torch.quantization as tq
from tqdm import tqdm
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


def _suppress_transformers_progress():
    """Suppress transformers progress bars for cleaner output."""
    try:
        from transformers.utils import logging as hf_logging
        hf_logging.set_verbosity_error()
        # Disable all transformers progress bars
        import os
        os.environ["TRANSFORMERS_VERBOSITY"] = "error"
        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
        os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
        # Try dedicated library controls if available
        try:
            # Some versions expose explicit progress bar toggles
            from transformers.utils.logging import disable_progress_bar as _hf_disable_pb  # type: ignore
            try:
                _hf_disable_pb()
            except Exception:
                pass
        except Exception:
            pass

        try:
            from huggingface_hub.utils import disable_progress_bars as _hub_disable_pbs  # type: ignore
            try:
                _hub_disable_pbs()
            except Exception:
                pass
        except Exception:
            pass

        # As a last resort, patch tqdm.__init__ safely to auto-disable when called
        # from transformers or huggingface_hub. We avoid calling str(self) before init.
        try:
            import inspect
            import tqdm as _tqdm

            original_init = _tqdm.tqdm.__init__

            def _patched_init(self, *args, **kwargs):
                disable_kwarg = kwargs.get("disable", None)
                # If the caller did not explicitly set disable, decide based on call stack
                if disable_kwarg is None:
                    try:
                        stack = inspect.stack()
                        called_from_hf = any(
                            (frame.filename and ("transformers" in frame.filename or "huggingface_hub" in frame.filename))
                            for frame in stack
                        )
                    except Exception:
                        called_from_hf = False
                    if called_from_hf:
                        kwargs["disable"] = True
                return original_init(self, *args, **kwargs)

            _tqdm.tqdm.__init__ = _patched_init  # type: ignore[method-assign]
        except Exception:
            pass
    except Exception:
        pass


class CleanProgressBar:
    """A single, clean progress bar for all operations."""
    
    def __init__(self, total: int, desc: str = "Processing"):
        self.total = total
        self.current = 0
        self.desc = desc
        self.current_step = ""
        self.start_time = None
        self.pbar = None
        
    def start(self):
        """Start the progress bar."""
        import time
        self.start_time = time.time()
        self.pbar = tqdm(
            total=self.total,
            desc=self.desc,
            unit="variant",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
            ncols=120,
            leave=True,
            dynamic_ncols=False
        )
        
    def update_step(self, step: str):
        """Update the current step description."""
        self.current_step = step
        if self.pbar:
            self.pbar.set_postfix_str(step)
            
    def update(self, n: int = 1):
        """Update progress."""
        self.current += n
        if self.pbar:
            self.pbar.update(n)
            
    def close(self):
        """Close the progress bar."""
        if self.pbar:
            self.pbar.close()
            self.pbar = None


def _check_memory_availability(required_gb: float = 8.0) -> bool:
    """Check if there's enough available memory to load a model."""
    try:
        available_memory = psutil.virtual_memory().available / (1024**3)  # Convert to GB
        return available_memory >= required_gb
    except Exception:
        return True  # If we can't check, assume it's okay


def _get_model_size_estimate(model_id_or_path: str, local_files_only: bool = False, quantization: str = "none") -> float:
    """Estimate model size in GB based on config and quantization type."""
    try:
        config = AutoConfig.from_pretrained(model_id_or_path, local_files_only=local_files_only)
        # Rough estimation: num_layers * hidden_size * hidden_size * 4 bytes (float32)
        # This is a very rough estimate, actual size may vary
        if hasattr(config, 'num_hidden_layers') and hasattr(config, 'hidden_size'):
            num_layers = config.num_hidden_layers
            hidden_size = config.hidden_size
            # Estimate for transformer layers (attention + MLP)
            params_per_layer = hidden_size * hidden_size * 4  # 4 matrices per layer
            total_params = num_layers * params_per_layer
            # Add embedding and output layers
            total_params += hidden_size * config.vocab_size * 2  # input and output embeddings
            size_gb = (total_params * 4) / (1024**3)  # 4 bytes per parameter
            
            # Adjust for quantization type
            if quantization == "int8-dynamic":
                # int8-dynamic loads full float32 model first, then quantizes
                return size_gb * 1.2  # 20% overhead for processing
            elif quantization == "bnb-4bit":
                return size_gb * 0.25  # ~4x compression
            elif quantization == "bnb-8bit":
                return size_gb * 0.5   # ~2x compression
            else:
                return size_gb
    except Exception:
        pass
    
    # Default estimates for 7B models
    if quantization == "int8-dynamic":
        return 8.5  # Higher estimate for int8-dynamic
    elif quantization == "bnb-4bit":
        return 1.8
    elif quantization == "bnb-8bit":
        return 3.5
    else:
        return 7.0


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
    # Guard against unsupported compute dtypes for BnB (expects fp16/bf16)
    if compute_dtype not in (torch.float16, torch.bfloat16, None):
        compute_dtype = None

    if quantization == "bnb-4bit":
        # Prefer fp16 compute on CUDA for better kernel throughput on consumer GPUs.
        default_compute = torch.float16 if torch.cuda.is_available() else torch.bfloat16
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=False,
            bnb_4bit_compute_dtype=compute_dtype or default_compute,
        )
    if quantization == "bnb-8bit":
        # Set sane defaults to avoid CPU offload and keep thresholds explicit
        return BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
            llm_int8_enable_fp32_cpu_offload=False,
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

    # Check memory availability before loading
    estimated_size = _get_model_size_estimate(model_id_or_path, local_files_only, quantization)
    required_memory = estimated_size * 1.5  # Add 50% buffer for processing
    
    # Special warning for int8-dynamic on large models
    if quantization == "int8-dynamic" and estimated_size > 6.0:
        logger.warning(
            f"int8-dynamic quantization requires loading the full model in float32 format, "
            f"which may require {estimated_size:.1f}GB+ of memory. Consider using bnb-4bit or bnb-8bit instead."
        )
    
    if not _check_memory_availability(required_memory):
        available_gb = psutil.virtual_memory().available / (1024**3)
        if quantization == "int8-dynamic":
            suggestion = "Consider using --hf-variants bnb-4bit bnb-8bit instead of int8-dynamic"
        else:
            suggestion = "Consider using a smaller model or adding swap space"
        raise RuntimeError(
            f"Insufficient memory: estimated {estimated_size:.1f}GB model requires ~{required_memory:.1f}GB, "
            f"but only {available_gb:.1f}GB available. {suggestion}."
        )
    
    # Suppress transformers progress bars
    _suppress_transformers_progress()
    
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

    # Load tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_id_or_path,
            revision=revision,
            use_fast=True,
            trust_remote_code=trust_remote_code,
            local_files_only=local_files_only,
        )
    except Exception as e:
        raise RuntimeError(f"Failed to load tokenizer: {e}")

    # Load model
    model = None
    try:
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
    except Exception as e:
        # Clean up any partially loaded model
        if model is not None:
            try:
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
            except Exception:
                pass
        raise RuntimeError(f"Failed to load model: {e}")

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


