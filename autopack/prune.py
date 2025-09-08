from typing import Iterable

import torch
from torch import nn


def _iter_linear_modules(model: nn.Module) -> Iterable[nn.Linear]:
    for module in model.modules():
        if isinstance(module, nn.Linear):
            yield module


def apply_global_magnitude_pruning(model: nn.Module, sparsity: float) -> None:
    """In-place global magnitude pruning across all Linear weights.

    Keeps the largest weights by absolute value and zeroes-out the rest.
    """
    sparsity = float(max(0.0, min(sparsity, 0.95)))
    if sparsity <= 0.0:
        return

    parameters = []
    for linear in _iter_linear_modules(model):
        parameters.append((linear, "weight"))

    if not parameters:
        return

    with torch.no_grad():
        all_weights = torch.cat(
            [torch.flatten(torch.abs(getattr(m, n))) for m, n in parameters]
        )
        k = int(sparsity * all_weights.numel())
        if k <= 0:
            return
        threshold, _ = torch.kthvalue(all_weights, k)

        for module, name in parameters:
            weight = getattr(module, name)
            mask = torch.abs(weight) > threshold
            weight.mul_(mask)


