"""Benchmarks module."""

from quantvec.benchmarks.datasets import (
    generate_random_dataset,
    generate_clustered_dataset,
    generate_glove_like_dataset,
    generate_openai_like_dataset,
    DATASET_REGISTRY,
)
from quantvec.benchmarks.metrics import (
    compute_recall_at_k,
    compute_mean_reciprocal_rank,
    compute_ndcg_at_k,
    brute_force_knn,
    run_benchmark,
)

__all__ = [
    "generate_random_dataset",
    "generate_clustered_dataset",
    "generate_glove_like_dataset",
    "generate_openai_like_dataset",
    "DATASET_REGISTRY",
    "compute_recall_at_k",
    "compute_mean_reciprocal_rank",
    "compute_ndcg_at_k",
    "brute_force_knn",
    "run_benchmark",
]
