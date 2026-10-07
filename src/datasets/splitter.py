"""Robust NIH-to-hospital splitting for federated SSL."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np
from torch.utils.data import Dataset


def split_nih_to_hospitals(
    dataset: Dataset,
    num_hospitals: int = 10,
    strategy: str = "non_iid",
    alpha: float = 1.0,
    save_dir: str = "data/processed",
    seed: int = 42,
    min_client_samples: int = 500,
) -> List[List[int]]:
    """Split the NIH dataset into hospital-specific index lists with controlled non-IID imbalance."""
    n = len(dataset)
    if num_hospitals < 1:
        raise ValueError("num_hospitals must be at least 1.")
    if n < num_hospitals:
        raise ValueError(f"Cannot create {num_hospitals} non-empty hospital splits from {n} samples.")
    if min_client_samples < 1:
        raise ValueError("min_client_samples must be >= 1.")

    if strategy == "iid":
        hospital_indices = _split_iid(np.arange(n), num_hospitals, seed)
    elif strategy == "non_iid":
        hospital_indices = _split_non_iid(np.arange(n), num_hospitals, alpha, seed, min_client_samples)
    else:
        raise ValueError(f"Unknown split strategy '{strategy}'. Use 'iid' or 'non_iid'.")

    _validate_hospital_split(hospital_indices, n, min_client_samples)
    _save_indices(hospital_indices, save_dir)
    _print_distribution(hospital_indices)
    return [idx.tolist() for idx in hospital_indices]


def _split_iid(indices: np.ndarray, num_hospitals: int, seed: int) -> List[np.ndarray]:
    rng = np.random.default_rng(seed)
    shuffled = indices.copy()
    rng.shuffle(shuffled)
    return np.array_split(shuffled, num_hospitals)


def _split_non_iid(
    indices: np.ndarray,
    num_hospitals: int,
    alpha: float,
    seed: int,
    min_client_samples: int,
) -> List[np.ndarray]:
    rng = np.random.default_rng(seed)
    shuffled = indices.copy()
    rng.shuffle(shuffled)

    if num_hospitals * min_client_samples > len(shuffled):
        raise ValueError(
            f"Requested minimum client size {min_client_samples} is too large: "
            f"{num_hospitals * min_client_samples} samples required for {num_hospitals} hospitals, but only {len(shuffled)} total images are available."
        )

    counts = np.full(num_hospitals, min_client_samples, dtype=int)
    remaining = len(shuffled) - counts.sum()
    if remaining < 0:
        raise ValueError("Remaining sample count became negative; reduce min_client_samples or hospital count.")
    if remaining > 0:
        proportions = np.full(num_hospitals, alpha, dtype=float)
        extra = rng.multinomial(remaining, proportions / proportions.sum())
        counts += extra

    # Ensure no zeros, allow some mild imbalance but keep the splits realistic.
    if np.any(counts <= 0):
        raise ValueError(f"Hospital counts collapsed to zero or negative values: {counts.tolist()}")

    hospital_indices: List[np.ndarray] = []
    start = 0
    for count in counts:
        hospital_indices.append(shuffled[start : start + count])
        start += count

    if start != len(shuffled):
        raise ValueError(f"Index assignment mismatch: expected {len(shuffled)}, assigned {start}.")
    return hospital_indices


def _validate_hospital_split(hospital_indices: List[np.ndarray], total_count: int, min_client_samples: int) -> None:
    flat = np.concatenate(hospital_indices) if hospital_indices else np.array([], dtype=int)
    if len(flat) != total_count:
        raise ValueError(f"Hospital assignment total mismatch: expected {total_count}, got {len(flat)}.")
    if len(np.unique(flat)) != total_count:
        raise ValueError("Duplicate NIH indices detected across hospitals.")
    counts = [len(x) for x in hospital_indices]
    if min(counts) < min_client_samples:
        raise ValueError(f"One or more hospitals fall below the configured minimum ({min_client_samples}): {counts}")


def _save_indices(hospital_indices: List[np.ndarray], save_dir: str) -> None:
    save_root = Path(save_dir)
    for i, idx in enumerate(hospital_indices):
        hospital_dir = save_root / f"hospital_{i + 1}"
        hospital_dir.mkdir(parents=True, exist_ok=True)
        np.save(str(hospital_dir / "indices.npy"), idx)


def _print_distribution(hospital_indices: List[np.ndarray]) -> None:
    counts = [len(idx) for idx in hospital_indices]
    total = sum(counts)
    mean = float(np.mean(counts))
    std = float(np.std(counts))
    print("\nHospital distribution")
    print("-" * 72)
    for i, count in enumerate(counts, start=1):
        print(f"Hospital {i}: {count:5d}")
    print(f"Total: {total}")
    print(f"Min: {min(counts)}")
    print(f"Max: {max(counts)}")
    print(f"Mean: {mean:.2f}")
    print(f"Std: {std:.2f}")
    print("-" * 72)


def load_hospital_indices(hospital_id: int, save_dir: str = "data/processed") -> List[int]:
    path = Path(save_dir) / f"hospital_{hospital_id}" / "indices.npy"
    if not path.exists():
        raise FileNotFoundError(f"No saved indices found at {path}. Run split_nih_to_hospitals() first.")
    return np.load(str(path)).tolist()
