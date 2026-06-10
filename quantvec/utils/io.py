"""I/O utilities for vector data."""

import os
from typing import Union, Dict, Any

import numpy as np


def save_vectors(path: str, vectors: np.ndarray) -> None:
    """Save vectors to a binary file.

    Args:
        path: Output file path.
        vectors: Numpy array of vectors.
    """
    np.save(path, vectors)


def load_vectors(path: str) -> np.ndarray:
    """Load vectors from a binary file.

    Args:
        path: Input file path.

    Returns:
        Numpy array of vectors.
    """
    return np.load(path)


def save_codes(path: str, codes: Dict[str, Any]) -> None:
    """Save quantized codes to an NPZ archive.

    Args:
        path: Output file path (.npz).
        codes: Dictionary containing quantized codes and metadata.
    """
    np.savez_compressed(path, **codes)


def load_codes(path: str) -> Dict[str, Any]:
    """Load quantized codes from an NPZ archive.

    Args:
        path: Input file path (.npz).

    Returns:
        Dictionary containing quantized codes and metadata.
    """
    data = np.load(path)
    return {key: data[key] for key in data.files}


def format_bytes(size_bytes: float) -> str:
    """Format byte size to human-readable string.

    Args:
        size_bytes: Size in bytes.

    Returns:
        Human-readable string like "1.5 MB".
    """
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if abs(size_bytes) < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"


def estimate_memory(
    n_vectors: int,
    dim: int,
    dtype: Union[str, np.dtype] = "float32",
) -> str:
    """Estimate memory usage for a vector dataset.

    Args:
        n_vectors: Number of vectors.
        dim: Vector dimensionality.
        dtype: Numpy data type.

    Returns:
        Human-readable memory estimate.
    """
    dtype = np.dtype(dtype)
    bytes_per_vector = dim * dtype.itemsize
    total_bytes = n_vectors * bytes_per_vector
    return format_bytes(total_bytes)
