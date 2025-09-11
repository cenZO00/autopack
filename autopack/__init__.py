"""Autopack: Quantize and publish LLMs.

Provides a CLI and Python API to:
- Load models from Hugging Face Hub or local paths
- Quantize with bitsandbytes 4-bit or 8-bit
- Export in multiple formats (HF, ONNX; GGUF optional)
- Publish artifacts to Hugging Face Hub
"""

__all__ = [
    "__version__",
]

__version__ = "0.1.3.2"


