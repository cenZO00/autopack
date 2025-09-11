# Changelog

## [0.1.3.2] - 2025-09-10

### Added
- `--hf-variant` flag for single-variant runs (convenience alias for `--hf-variants <one>`).
- `--hf-variants` flag to limit which HF variants to produce in `auto`.
- Auto-detection of pre-quantized models (e.g., MxFP4) to skip conflicting BitsAndBytes variants.
- Model resolution to local cache once to avoid repeated downloads for large models.
- Memory cleanup between variants to free RAM/VRAM after each save.

### Changed
- `auto` now resolves the model to a local snapshot and reuses it for all variants.
- Pre-quantized models automatically skip bnb-4bit and bnb-8bit variants to avoid conflicts.

### Fixed
- Prevent quantization config conflicts when loading pre-quantized models.

[0.1.3.2]: https://github.com/GranulaVision/autopack/releases/tag/v0.1.3.2

## [0.1.3.1] - 2025-09-10

### Added
- Benchmarking is enabled by default for `auto`; new `--no-bench` flag to disable.
- Default subcommand: running `autopack <model>` now implies `auto`.
- Default output directory: if `-o/--output-dir` is omitted for `auto`/`quantize`, it uses the last segment of the model id/path (e.g., `user/model` -> `model`).

### Changed
- Tuned default benchmark settings for fast runs (max_new_tokens=16, warmup=0, runs=1).
- CLI help/version handling no longer gets hijacked by default-to-auto logic.
- README updated to highlight simplified CLI defaults and default benchmarking.

### Fixed
- Replaced heuristic speedup table in `auto` with real Tokens/s and speedup vs bf16 by default.
- Prevent conflict when loading pre-quantized models (e.g., MxFP4): skip passing BitsAndBytes config if a non-BNB quantization is detected in config.

[0.1.3.1]: https://github.com/GranulaVision/autopack/releases/tag/v0.1.3.1

All notable changes to this project will be documented in this file.

## [0.1.3] - 2025-09-09

### Added
- CLI flags: `--skip-existing`, `--summary-json`, and `--eval-text-key`.
- Machine-readable `summary.json` generation alongside the auto README.

### Changed
- Generated README usage now recommends `accelerate` and `safetensors` and uses `AutoModelForCausalLM`.
- Perplexity evaluation runs with `model.eval()` and `torch.inference_mode()`; text column is configurable.
- Quantization ensures `model.eval()` before save and sets `pad_token` if missing.

### Removed
- `ggml` output option from CLI; GGML export is not supported in this version.

### Fixed
- GGUF exporter now cleans temporary Hub snapshots and supports Windows virtualenv paths.
- Version synchronized between package and project metadata (`0.1.3`).

[0.1.3]: https://github.com/GranulaVision/autopack/releases/tag/v0.1.3

