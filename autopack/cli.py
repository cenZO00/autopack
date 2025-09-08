import argparse
import os
import sys
from typing import List, Optional, Tuple, Dict

from . import __version__
from .quantize import quantize_to_hf
from .exporters import export_onnx, export_gguf, export_ggml
from .publish import publish_folder_to_hub
from .evaluation import calculate_perplexity
from transformers.utils import logging as hf_logging
from tqdm import tqdm


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="autopack",
        description="Quantize and publish Hugging Face models in multiple formats.",
    )

    # Global flags
    parser.add_argument("--version", action="version", version=f"autopack {__version__}")

    subparsers = parser.add_subparsers(dest="command", required=False)

    # auto command (default): run 4 quantization variants and summarize sizes/speedups
    a = subparsers.add_parser("auto", help="Run 4 quant variants and print a summary table")
    a.add_argument("--verbose", action="store_true", help="Enable verbose logs")
    a.add_argument("model", help="Hugging Face repo id (user/model) or local path")
    a.add_argument(
        "-o",
        "--output-dir",
        required=True,
        help="Base directory to write the quantized model variants",
    )
    a.add_argument(
        "--output-format",
        nargs="+",
        choices=["hf", "onnx", "gguf", "ggml"],
        default=["hf"],
        help="One or more output formats to produce (gguf/ggml are opt-in)",
    )
    a.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allow custom code from model repos",
    )
    a.add_argument(
        "--revision",
        default=None,
        help="Model revision/branch/tag to load (for Hub models)",
    )
    a.add_argument(
        "--eval-dataset",
        default=None,
        help="Optional Hugging Face dataset to run perplexity evaluation on (e.g., wikitext-2-raw-v1)",
    )
    a.add_argument(
        "--gguf-converter",
        default=None,
        help="Path to llama.cpp convert.py (or set LLAMA_CPP_CONVERT)",
    )
    a.add_argument(
        "--gguf-quant",
        nargs="+",
        default=None,
        help="One or more llama.cpp quant presets (e.g., Q4_K_M Q5_K_M)",
    )
    a.add_argument(
        "--gguf-extra-arg",
        dest="gguf_extra_args",
        action="append",
        default=None,
        help="Additional arguments for convert.py (repeatable)",
    )
    a.add_argument(
        "--gguf-no-isolation",
        action="store_true",
        help="Do not use an isolated virtualenv for GGUF conversion",
    )
    a.add_argument(
        "--gguf-force",
        action="store_true",
        help="Bypass architecture support check for GGUF conversion",
    )

    # quantize command
    q = subparsers.add_parser("quantize", help="Quantize a model and export in chosen formats")
    q.add_argument("--verbose", action="store_true", help="Enable verbose logs")
    q.add_argument("model", help="Hugging Face repo id (user/model) or local path")
    q.add_argument(
        "--revision",
        default=None,
        help="Model revision/branch/tag to load (for Hub models)",
    )
    q.add_argument(
        "-o",
        "--output-dir",
        required=True,
        help="Directory to write the exported model(s)",
    )
    q.add_argument(
        "--quantization",
        choices=["bnb-4bit", "bnb-8bit", "int8-dynamic", "none"],
        default="bnb-4bit",
        help="Quantization strategy (bnb 4/8-bit or PyTorch int8-dynamic)",
    )
    q.add_argument(
        "--dtype",
        choices=["auto", "float16", "bfloat16", "float32"],
        default="bfloat16",
        help="Compute dtype for 4/8-bit layers (where applicable)",
    )
    q.add_argument(
        "--device-map",
        default="auto",
        help="Device map to use when loading the model (e.g., 'auto', 'cpu')",
    )
    q.add_argument(
        "--prune",
        type=float,
        default=0.0,
        help="Global magnitude pruning ratio (0..0.95)",
    )
    q.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allow custom code from model repos",
    )
    q.add_argument(
        "--output-format",
        nargs="+",
        choices=["hf", "onnx", "gguf", "ggml"],
        default=["hf"],
        help="One or more output formats to produce",
    )
    q.add_argument(
        "--gguf-converter",
        default=None,
        help="Path to llama.cpp convert.py (or set LLAMA_CPP_CONVERT)",
    )
    q.add_argument(
        "--gguf-quant",
        default=None,
        help="Optional llama.cpp quant preset (e.g., Q4_K_M, Q5_K_M)",
    )
    q.add_argument(
        "--gguf-extra-arg",
        dest="gguf_extra_args",
        action="append",
        default=None,
        help="Additional arguments for convert.py (repeatable)",
    )
    q.add_argument(
        "--gguf-no-isolation",
        action="store_true",
        help="Do not use an isolated virtualenv for GGUF conversion",
    )
    q.add_argument(
        "--gguf-force",
        action="store_true",
        help="Bypass architecture support check for GGUF conversion",
    )

    # publish command
    p = subparsers.add_parser("publish", help="Publish an exported model folder to the Hugging Face Hub")
    p.add_argument("--verbose", action="store_true", help="Enable verbose logs")
    p.add_argument("folder", help="Local folder with model files to publish")
    p.add_argument("repo", help="Destination repo id, e.g., user/model")
    p.add_argument("--token", default=os.environ.get("HUGGINGFACE_HUB_TOKEN"), help="HF token (or set HUGGINGFACE_HUB_TOKEN)")
    p.add_argument("--private", action="store_true", help="Create/use a private repository")
    p.add_argument("--branch", default=None, help="Target branch (revision)")
    p.add_argument("--commit-message", default="Add model artifacts via autopack", help="Commit message")
    p.add_argument("--no-create", action="store_true", help="Do not attempt to create the repo if missing")

    # If invoked with no arguments, show help and exit
    if argv is None and len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)
    return parser.parse_args(argv)


def _generate_readme(
    base_model_id: str,
    output_dir: str,
    results: List[Tuple[str, str, int]],
    baseline_size: int,
    est_speed: dict,
    est_quality_drop: dict,
    perplexities: Dict[str, float],
):
    """Generate a README.md file in the output directory with a summary and code snippets."""
    readme_path = os.path.join(output_dir, "README.md")
    lines = [
        f"# Quantized Variants of `{base_model_id}`\n\n",
        "This directory contains multiple quantized versions of the base model, generated by the `autopack` tool.\n\n",
        "## Summary\n",
    ]

    if perplexities:
        lines.append(
            "| Variant | Output Path | Size | Rel Size | Est Speedup | Est. Quality Drop | Perplexity |\n"
        )
        lines.append("|---|---|---:|---:|---:|---:|---:|\n")
    else:
        lines.append(
            "| Variant | Output Path | Size | Rel Size | Est Speedup | Est. Quality Drop |\n"
        )
        lines.append("|---|---|---:|---:|---:|---:|\n")

    for name, out_dir, sz in results:
        rel = sz / baseline_size if baseline_size else 1.0
        size_h = _human_size(sz)
        speed = est_speed.get(name, 1.0)
        quality = est_quality_drop.get(name, "N/A")
        # Use relative paths for the README
        relative_path = os.path.relpath(out_dir, output_dir)
        if perplexities:
            ppl = perplexities.get(name)
            ppl_str = f"{ppl:.4f}" if ppl else "N/A"
            lines.append(
                f"| {name} | `{relative_path}` | {size_h} | {rel:.2f} | {speed:.2f}x | {quality} | {ppl_str} |\n"
            )
        else:
            lines.append(
                f"| {name} | `{relative_path}` | {size_h} | {rel:.2f} | {speed:.2f}x | {quality} |\n"
            )

    lines.append("\n## Usage\n")
    lines.append(
        "To load and use these models, you will need the `transformers` library and, for some variants, `bitsandbytes`.\n"
    )
    lines.append("```bash\npip install transformers bitsandbytes\n```\n\n")

    for name, out_dir, _ in results:
        relative_path = os.path.relpath(out_dir, output_dir)
        lines.append(f"### {name}\n")
        if name.startswith("gguf"):
            lines.append(
                "This model is in GGUF format and is designed to be used with `llama.cpp`.\n\n"
            )
            lines.append(
                "**Note**: You must have `llama.cpp` compiled and available on your system to run this model.\n\n"
            )
            lines.append("Example usage:\n")
            lines.append("```bash\n")
            lines.append(f'./main -m ./{relative_path} -n 128 -p "Once upon a time"\n')
            lines.append("```\n\n")
        else:
            lines.append("```python\n")
            lines.append("from transformers import AutoTokenizer, AutoModel\n\n")
            lines.append(f'model_path = "./{relative_path}"\n\n')
            lines.append("tokenizer = AutoTokenizer.from_pretrained(model_path)\n")
            # The `trust_remote_code` is important for quanto-quantized models
            lines.append(
                'model = AutoModel.from_pretrained(model_path, device_map="auto", trust_remote_code=True)\n'
            )
            lines.append("```\n\n")

    with open(readme_path, "w") as f:
        f.writelines(lines)
    print(f"Generated summary README at: {readme_path}")


def _human_size(num_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(num_bytes)
    for unit in units:
        if size < 1024.0 or unit == units[-1]:
            return f"{size:.2f} {unit}"
        size /= 1024.0


def _dir_size(path: str) -> int:
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            fp = os.path.join(root, f)
            try:
                total += os.path.getsize(fp)
            except OSError:
                pass
    return total


def run_auto(args: argparse.Namespace) -> int:
    os.makedirs(args.output_dir, exist_ok=True)
    variants: List[Tuple[str, dict]] = [
        ("bnb-4bit", {"quantization": "bnb-4bit", "dtype": "bfloat16", "device_map": "auto"}),
        ("bnb-8bit", {"quantization": "bnb-8bit", "dtype": "auto", "device_map": "auto"}),
        ("int8-dynamic", {"quantization": "int8-dynamic", "dtype": "float32", "device_map": "cpu"}),
        ("bf16", {"quantization": "none", "dtype": "bfloat16", "device_map": "auto"}),
    ]

    results = []
    perplexities: Dict[str, float] = {}

    # Progress total across requested outputs
    total_steps = 0
    if "hf" in args.output_format:
        total_steps += len(variants)
    if "onnx" in args.output_format:
        total_steps += 1
    if "gguf" in args.output_format:
        # Determine quant list early for progress accounting
        quant_list = args.gguf_quant if args.gguf_quant else ["Q4_K_M", "Q5_K_M", "Q8_0"]
        total_steps += len(quant_list)

    pbar = tqdm(total=total_steps, desc="autopack", unit="step", disable=(total_steps == 0))

    # --- Run HF variants (opt-in for auto) ---
    if "hf" in args.output_format:
        for name, params in variants:
            pbar.set_description(f"HF {name}")
            out_dir = os.path.join(args.output_dir, name)
            os.makedirs(out_dir, exist_ok=True)
            quantize_to_hf(
                model_id_or_path=args.model,
                output_dir=out_dir,
                quantization=params["quantization"],
                dtype=params["dtype"],
                device_map=params["device_map"],
                trust_remote_code=args.trust_remote_code,
                revision=args.revision,
            )
            size_bytes = _dir_size(out_dir)
            results.append((name, out_dir, size_bytes))
            pbar.update(1)

            # Run perplexity evaluation if requested
            if args.eval_dataset:
                try:
                    if ":" in args.eval_dataset:
                        dataset_id, dataset_config = args.eval_dataset.split(":", 1)
                    else:
                        dataset_id, dataset_config = args.eval_dataset, None
                    print(f"Running perplexity evaluation for {name} on {dataset_id} ({dataset_config})...")
                    ppl = calculate_perplexity(
                        out_dir,
                        dataset_id,
                        dataset_config or "",
                        trust_remote_code=args.trust_remote_code,
                    )
                    perplexities[name] = ppl
                    print(f"  - Perplexity: {ppl:.4f}")
                except Exception as e:
                    print(f"  - Could not calculate perplexity: {e}")

    # --- Optional ONNX export for auto ---
    if "onnx" in args.output_format:
        pbar.set_description("ONNX export")
        onnx_dir = os.path.join(args.output_dir, "onnx")
        os.makedirs(onnx_dir, exist_ok=True)
        try:
            export_onnx(
                model_id_or_path=args.model,
                output_dir=onnx_dir,
                trust_remote_code=args.trust_remote_code,
                revision=args.revision,
            )
            size_bytes = _dir_size(onnx_dir)
            results.append(("onnx", onnx_dir, size_bytes))
        except Exception as e:
            print(f"Skipping ONNX export due to an error: {e}")
        finally:
            pbar.update(1)

    # --- Optional GGUF variants for auto (opt-in) ---
    if "gguf" in args.output_format:
        gguf_out_dir = os.path.join(args.output_dir, "gguf")
        os.makedirs(gguf_out_dir, exist_ok=True)
        try:
            # For auto command, explicitly point to the vendored llama.cpp relative to this script's location
            cli_dir = os.path.dirname(os.path.abspath(__file__))
            repo_root = os.path.dirname(cli_dir)
            default_converter_path = os.path.join(repo_root, "third_party", "llama.cpp", "convert_hf_to_gguf.py")
            llama_cpp_bin = os.path.join(repo_root, "third_party", "llama.cpp", "build", "bin")
            env = os.environ.copy()
            env["PATH"] = f"{llama_cpp_bin}:{env['PATH']}"

            # Prefer bf16 local export if it exists; otherwise, use original model path
            bf16_out_dir = next((r[1] for r in results if r[0] == "bf16"), None)
            source_model_path = bf16_out_dir if bf16_out_dir else args.model

            # Determine quant list (normalize to uppercase)
            quant_list = args.gguf_quant if args.gguf_quant else ["Q4_K_M", "Q5_K_M", "Q8_0"]
            quant_list = [q.upper() for q in quant_list]

            for quant in quant_list:
                try:
                    pbar.set_description(f"GGUF {quant}")
                    gguf_path = export_gguf(
                        model_id_or_path=source_model_path,
                        output_dir=gguf_out_dir,
                        quant=quant,
                        converter_path=(args.gguf_converter or default_converter_path),
                        trust_remote_code=args.trust_remote_code,
                        revision=args.revision,
                        extra_args=args.gguf_extra_args,
                        env=env,
                        isolate_env=not args.gguf_no_isolation,
                        force=args.gguf_force,
                    )
                    # Use directory size to avoid errors if path is None or if multiple files are produced
                    size_bytes = _dir_size(gguf_out_dir)
                    results.append((f"gguf-{quant}", gguf_out_dir, size_bytes))
                except Exception as e:
                    print(f"Skipping GGUF quant {quant} due to an error: {e}")
                finally:
                    pbar.update(1)
        except Exception as e:
            print(f"Skipping GGUF export due to an error: {e}")

    pbar.close()

    # --- Optional GGML export for auto (opt-in) ---
    if "ggml" in args.output_format:
        ggml_dir = os.path.join(args.output_dir, "ggml")
        os.makedirs(ggml_dir, exist_ok=True)
        try:
            export_ggml(
                model_id_or_path=args.model,
                output_dir=ggml_dir,
                trust_remote_code=args.trust_remote_code,
                revision=args.revision,
            )
        except Exception as e:
            print(f"Skipping GGML export due to an error: {e}")

    # Establish baseline size (bf16)
    baseline = next((r for r in results if r[0] == "bf16"), None)
    baseline_size = baseline[2] if baseline else max((r[2] for r in results if r[2] > 0), default=1)

    # Estimated speedups vs bf16 baseline (very rough heuristics; actual depends on HW)
    est_speed = {
        "bf16": 1.00,
        "bnb-8bit": 1.50,
        "bnb-4bit": 2.50,
        "int8-dynamic": 1.20,
        "gguf": 2.80,  # GGUF on CPU can be very fast
    }

    # Estimated quality drop (lower is better, very rough heuristics)
    est_quality_drop = {
        "bf16": "0.0%",
        "bnb-8bit": "~0.1-0.5%",
        "bnb-4bit": "~0.5-2.0%",
        "int8-dynamic": "~0.5-3.0%",
        "gguf": "~0.5-1.5%",  # For Q4_K_M
    }

    # Print table
    if perplexities:
        headers = ("Variant", "Output Path", "Size", "Rel Size", "Est Speedup", "Est. Quality Drop", "Perplexity")
        print("\nSummary of quantized variants:\n")
        print(f"{headers[0]:<14}  {headers[1]:<40}  {headers[2]:>12}  {headers[3]:>9}  {headers[4]:>12}  {headers[5]:>18}  {headers[6]:>12}")
        print("-" * 130)
    else:
        headers = ("Variant", "Output Path", "Size", "Rel Size", "Est Speedup", "Est. Quality Drop")
        print("\nSummary of quantized variants:\n")
        print(f"{headers[0]:<14}  {headers[1]:<40}  {headers[2]:>12}  {headers[3]:>9}  {headers[4]:>12}  {headers[5]:>18}")
        print("-" * 115)

    for name, out_dir, sz in results:
        rel = sz / baseline_size if baseline_size else 1.0
        size_h = _human_size(sz)
        speed = est_speed.get(name, 1.0)
        quality = est_quality_drop.get(name, "N/A")
        if perplexities:
            ppl = perplexities.get(name)
            ppl_str = f"{ppl:.4f}" if ppl else "N/A"
            print(f"{name:<14}  {out_dir:<40}  {size_h:>12}  {rel:>9.2f}  {speed:>12.2f}x  {quality:>18}  {ppl_str:>12}")
        else:
            print(f"{name:<14}  {out_dir:<40}  {size_h:>12}  {rel:>9.2f}  {speed:>12.2f}x  {quality:>18}")
    print()

    # Generate README.md
    _generate_readme(
        args.model, args.output_dir, results, baseline_size, est_speed, est_quality_drop, perplexities
    )

    return 0


def run_quantize(args: argparse.Namespace) -> int:
    os.makedirs(args.output_dir, exist_ok=True)
    total_steps = ("hf" in args.output_format) + ("onnx" in args.output_format) + ("gguf" in args.output_format)
    pbar = tqdm(total=total_steps, desc="autopack", unit="step", disable=(total_steps == 0))

    # Always produce HF format first when requested
    if "hf" in args.output_format:
        pbar.set_description("HF export")
        quantize_to_hf(
            model_id_or_path=args.model,
            output_dir=args.output_dir,
            quantization=args.quantization,
            dtype=args.dtype,
            device_map=args.device_map,
            trust_remote_code=args.trust_remote_code,
            revision=args.revision,
            prune=args.prune,
        )
        pbar.update(1)

    # Produce ONNX by exporting from the original model id/path (fresh load)
    if "onnx" in args.output_format:
        pbar.set_description("ONNX export")
        onnx_dir = os.path.join(args.output_dir, "onnx")
        os.makedirs(onnx_dir, exist_ok=True)
        export_onnx(
            model_id_or_path=args.model,
            output_dir=onnx_dir,
            trust_remote_code=args.trust_remote_code,
            revision=args.revision,
        )
        pbar.update(1)

    # GGUF export (optional)
    if "gguf" in args.output_format:
        pbar.set_description("GGUF export")
        gguf_dir = os.path.join(args.output_dir, "gguf")
        os.makedirs(gguf_dir, exist_ok=True)
        export_gguf(
            model_id_or_path=args.model,
            output_dir=gguf_dir,
            trust_remote_code=args.trust_remote_code,
            revision=args.revision,
            converter_path=args.gguf_converter,
            quant=(args.gguf_quant.upper() if isinstance(args.gguf_quant, str) else args.gguf_quant),
            extra_args=args.gguf_extra_args,
            isolate_env=not args.gguf_no_isolation,
            force=args.gguf_force,
        )
        pbar.update(1)

    pbar.close()

    # GGML export (optional)
    if "ggml" in args.output_format:
        ggml_dir = os.path.join(args.output_dir, "ggml")
        os.makedirs(ggml_dir, exist_ok=True)
        export_ggml(
            model_id_or_path=args.model,
            output_dir=ggml_dir,
            trust_remote_code=args.trust_remote_code,
            revision=args.revision,
        )

    return 0


def run_publish(args: argparse.Namespace) -> int:
    create = not args.no_create
    publish_folder_to_hub(
        folder=args.folder,
        repo_id=args.repo,
        token=args.token,
        private=args.private,
        commit_message=args.commit_message,
        revision=args.branch,
        create=create,
    )
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    # Reduce noisy logs by default; allow opt-in via --verbose on any subcommand
    hf_logging.set_verbosity_error()
    # If user provided args without a subcommand, default to 'auto'
    if argv is None and len(sys.argv) > 1 and sys.argv[1] not in {"quantize", "publish", "auto"}:
        argv = ["auto", *sys.argv[1:]]
    elif argv is not None and (len(argv) > 0 and argv[0] not in {"quantize", "publish", "auto"}):
        argv = ["auto", *argv]

    # Parse arguments
    args = parse_args(argv)
    # Enable verbose warnings from Transformers when requested
    if getattr(args, "verbose", False):
        hf_logging.set_verbosity_warning()
    if args.command == "auto":
        return run_auto(args)
    if args.command == "quantize":
        return run_quantize(args)
    if args.command == "publish":
        return run_publish(args)
    raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    sys.exit(main())


