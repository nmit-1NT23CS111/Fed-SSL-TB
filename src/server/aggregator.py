"""Federated aggregation primitives for encoder-only updates."""

import copy
from typing import Any, Dict, List

import torch


def _is_tensor_like(value: Any) -> bool:
    return isinstance(value, torch.Tensor)


def fedavg(
    encoder_weights_list: List[Dict[str, Any]],
    sample_counts: List[int],
) -> Dict[str, Any]:
    """Weighted average of local encoder states using the configured sample weights."""
    if len(encoder_weights_list) != len(sample_counts):
        raise ValueError("Number of weight dicts must match number of sample counts.")
    if not encoder_weights_list:
        raise ValueError("No weights to aggregate.")

    total_samples = sum(sample_counts)
    if total_samples <= 0:
        raise ValueError("Total sample count must be positive.")

    weights = [count / total_samples for count in sample_counts]

    first = encoder_weights_list[0]
    if isinstance(first, dict) and "encoder" in first and not any(_is_tensor_like(v) for v in first.values()):
        return {
            key: fedavg([state[key] for state in encoder_weights_list], sample_counts)
            for key in first
        }

    if isinstance(first, dict):
        agg = copy.deepcopy(first)
        for key in agg:
            if not _is_tensor_like(agg[key]):
                continue
            agg[key] = agg[key].float() * weights[0]
        for i in range(1, len(encoder_weights_list)):
            for key in agg:
                if _is_tensor_like(agg[key]) and _is_tensor_like(encoder_weights_list[i][key]):
                    agg[key] = agg[key] + encoder_weights_list[i][key].float() * weights[i]
        return agg

    raise TypeError("Unsupported weight structure for aggregation.")


def fedprox(
    global_weights: Dict[str, Any],
    local_weights_list: List[Dict[str, Any]],
    sample_counts: List[int],
    mu: float = 0.01,
) -> Dict[str, Any]:
    """FedProx uses the same weighted server aggregation as FedAvg; the proximal term is client-side only."""
    _ = global_weights, mu
    return fedavg(local_weights_list, sample_counts)
