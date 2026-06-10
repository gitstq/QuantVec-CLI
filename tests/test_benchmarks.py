"""Tests for benchmark utilities."""

import numpy as np
import pytest

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
    brute_force_knn,
    run_benchmark,
)
from quantvec.quantizers import TurboQuantQuantizer


class TestDatasets:
    """Tests for dataset generation."""

    def test_random_dataset(self):
        train, test = generate_random_dataset(n_train=100, n_test=10, dim=32)
        assert train.shape == (100, 32)
        assert test.shape == (10, 32)
        assert train.dtype == np.float32

    def test_clustered_dataset(self):
        train, test = generate_clustered_dataset(n_train=100, n_test=10, dim=32)
        assert train.shape == (100, 32)
        assert test.shape == (10, 32)

    def test_glove_dataset(self):
        train, test = generate_glove_like_dataset(n_train=100, n_test=10, dim=200)
        assert train.shape == (100, 200)

    def test_openai_dataset(self):
        train, test = generate_openai_like_dataset(n_train=100, n_test=10, dim=1536)
        assert train.shape == (100, 1536)

    def test_registry(self):
        assert "random" in DATASET_REGISTRY
        assert "clustered" in DATASET_REGISTRY
        assert "glove" in DATASET_REGISTRY
        assert "openai" in DATASET_REGISTRY


class TestMetrics:
    """Tests for benchmark metrics."""

    def test_compute_recall_at_k(self):
        gt = np.array([[0, 1, 2], [3, 4, 5]])
        ret = np.array([[0, 1, 5], [3, 6, 7]])
        recall = compute_recall_at_k(gt, ret, k=3)
        expected = (2 / 3 + 1 / 3) / 2  # 0.5
        assert abs(recall - expected) < 0.01

    def test_compute_mrr(self):
        gt = np.array([[0], [1]])
        ret = np.array([[0, 2], [3, 1]])
        mrr = compute_mean_reciprocal_rank(gt, ret)
        expected = (1.0 + 0.5) / 2  # 0.75
        assert abs(mrr - expected) < 0.01

    def test_brute_force_knn(self):
        queries = np.array([[1, 0], [0, 1]], dtype=np.float32)
        db = np.array([[1, 0], [0, 1], [0.5, 0.5]], dtype=np.float32)
        scores, indices = brute_force_knn(queries, db, k=2)
        assert scores.shape == (2, 2)
        assert indices.shape == (2, 2)
        # First query should match first db vector
        assert indices[0, 0] == 0

    def test_run_benchmark(self):
        train, test = generate_random_dataset(n_train=100, n_test=10, dim=32)
        q = TurboQuantQuantizer(dim=32, bit_width=4)
        q.fit(train)
        results = run_benchmark(q, train, test, k_values=[1, 5])
        assert "compression_ratio" in results
        assert "recall@1" in results
        assert "recall@5" in results
        assert results["compression_ratio"] > 1.0
