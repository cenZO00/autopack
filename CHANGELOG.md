# Changelog

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

