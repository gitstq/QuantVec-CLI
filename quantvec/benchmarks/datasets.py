"""Benchmark datasets for vector quantization evaluation."""

import os
from typing import Tuple

import numpy as np


def generate_random_dataset(
    n_train: int = 10_000,
    n_test: int = 1_000,
    dim: int = 768,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a random synthetic dataset.

    Args:
        n_train: Number of training vectors.
        n_test: Number of test/query vectors.
        dim: Vector dimensionality.
        seed: Random seed.

    Returns:
        Tuple of (train_vectors, test_vectors).
    """
    rng = np.random.default_rng(seed)
    train = rng.standard_normal((n_train, dim), dtype=np.float32)
    test = rng.standard_normal((n_test, dim), dtype=np.float32)
    # Normalize to unit vectors
    train /= np.linalg.norm(train, axis=1, keepdims=True)
    test /= np.linalg.norm(test, axis=1, keepdims=True)
    return train, test


def generate_clustered_dataset(
    n_train: int = 10_000,
    n_test: int = 1_000,
    dim: int = 768,
    n_clusters: int = 100,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a clustered synthetic dataset.

    Args:
        n_train: Number of training vectors.
        n_test: Number of test vectors.
        dim: Vector dimensionality.
        n_clusters: Number of clusters.
        seed: Random seed.

    Returns:
        Tuple of (train_vectors, test_vectors).
    """
    rng = np.random.default_rng(seed)
    centroids = rng.standard_normal((n_clusters, dim), dtype=np.float32)
    centroids /= np.linalg.norm(centroids, axis=1, keepdims=True)

    train_labels = rng.integers(0, n_clusters, n_train)
    train = centroids[train_labels] + rng.standard_normal((n_train, dim), dtype=np.float32) * 0.1
    train /= np.linalg.norm(train, axis=1, keepdims=True)

    test_labels = rng.integers(0, n_clusters, n_test)
    test = centroids[test_labels] + rng.standard_normal((n_test, dim), dtype=np.float32) * 0.1
    test /= np.linalg.norm(test, axis=1, keepdims=True)

    return train, test


def generate_glove_like_dataset(
    n_train: int = 10_000,
    n_test: int = 1_000,
    dim: int = 200,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a GloVe-like dataset (lower dimension, sparse-ish).

    Args:
        n_train: Number of training vectors.
        n_test: Number of test vectors.
        dim: Vector dimensionality (default 200 like GloVe).
        seed: Random seed.

    Returns:
        Tuple of (train_vectors, test_vectors).
    """
    rng = np.random.default_rng(seed)
    # GloVe-like: mixture of positive values, some sparsity
    train = rng.lognormal(0, 1, (n_train, dim)).astype(np.float32)
    # Zero out some entries
    mask = rng.random((n_train, dim)) > 0.3
    train *= mask
    train /= np.linalg.norm(train, axis=1, keepdims=True)

    test = rng.lognormal(0, 1, (n_test, dim)).astype(np.float32)
    mask = rng.random((n_test, dim)) > 0.3
    test *= mask
    test /= np.linalg.norm(test, axis=1, keepdims=True)

    return train, test


def generate_openai_like_dataset(
    n_train: int = 10_000,
    n_test: int = 1_000,
    dim: int = 1536,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate an OpenAI embedding-like dataset.

    OpenAI embeddings are high-dimensional and approximately unit-normal.

    Args:
        n_train: Number of training vectors.
        n_test: Number of test vectors.
        dim: Vector dimensionality (default 1536 like text-embedding-3-small).
        seed: Random seed.

    Returns:
        Tuple of (train_vectors, test_vectors).
    """
    rng = np.random.default_rng(seed)
    train = rng.standard_normal((n_train, dim), dtype=np.float32)
    train /= np.linalg.norm(train, axis=1, keepdims=True)

    test = rng.standard_normal((n_test, dim), dtype=np.float32)
    test /= np.linalg.norm(test, axis=1, keepdims=True)

    return train, test


DATASET_REGISTRY = {
    "random": generate_random_dataset,
    "clustered": generate_clustered_dataset,
    "glove": generate_glove_like_dataset,
    "openai": generate_openai_like_dataset,
}
