import math
from typing import Any, Dict, Optional

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig


def calculate_perplexity(
    model_id_or_path: str,
    dataset_id: str,
    dataset_config: str,
    dataset_split: str = "test",
    n_samples: int = 256,
    device: str = "auto",
    trust_remote_code: bool = False,
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
    tokenizer = AutoTokenizer.from_pretrained(model_id_or_path, trust_remote_code=trust_remote_code)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    dataset_kwargs = {"split": dataset_split}
    if dataset_config:
        dataset_kwargs["name"] = dataset_config
    dataset = load_dataset(dataset_id, **dataset_kwargs)
    text_list = []
    for sample in dataset.select(range(min(n_samples, len(dataset)))):
        txt = sample.get("text")
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

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            neg_log_likelihood = outputs.loss

        nlls.append(neg_log_likelihood)
        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    ppl = torch.exp(torch.stack(nlls).mean())
    return ppl.item()
