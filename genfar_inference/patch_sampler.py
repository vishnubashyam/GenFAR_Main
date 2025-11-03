from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

PATCH_SIZE = 128
EDGE_MARGIN = 20
EPS = 1e-6


@dataclass(frozen=True)
class PatchMetadata:
    origin: Tuple[int, int, int]
    shape: Tuple[int, int, int]
    strategy: str


def sample_patch(
    volume: np.ndarray,
    rng: np.random.Generator,
    *,
    patch_size: int = PATCH_SIZE,
    margin: int = EDGE_MARGIN,
) -> Tuple[np.ndarray, PatchMetadata]:
    if volume.ndim != 3:
        raise ValueError(f"Expected 3D volume, got shape {volume.shape}")

    dims = volume.shape
    min_dim = min(dims)
    required = patch_size + 2 * margin
    strategy = "random"

    if min_dim >= required:
        starts = [
            rng.integers(margin, dims[axis] - patch_size - margin + 1)
            for axis in range(3)
        ]
    else:
        strategy = "center"
        starts = [max(0, (dims[axis] - patch_size) // 2) for axis in range(3)]

    slices = tuple(slice(start, start + patch_size) for start in starts)
    patch = volume[slices]

    if patch.shape != (patch_size, patch_size, patch_size):
        padding = []
        for axis_len in patch.shape:
            pad_total = patch_size - axis_len
            pad_before = pad_total // 2
            pad_after = pad_total - pad_before
            padding.append((pad_before, pad_after))
        patch = np.pad(patch, padding, mode="constant", constant_values=0.0)

    metadata = PatchMetadata(
        origin=(int(starts[0]), int(starts[1]), int(starts[2])),
        shape=tuple(int(dim) for dim in patch.shape),
        strategy=strategy,
    )
    return patch.astype(np.float32), metadata


def normalize_patch(patch: np.ndarray) -> np.ndarray:
    mean = float(patch.mean())
    std = float(patch.std())
    return (patch - mean) / (std + EPS)
