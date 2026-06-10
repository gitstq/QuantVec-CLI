"""Benchmark metrics for vector quantization evaluation."""

from typing import Dict, List, Tuple

import numpy as np


def compute_recall_at_k(
    ground_truth_indices: np.ndarray,
    retrieved_indices: np.ndarray,
    k: int = 10,
) -> float:
    """Compute recall@k.

    Args:
        ground_truth_indices: True nearest neighbor indices of shape (n_queries, k_gt).
        retrieved_indices: Retrieved indices of shape (n_queries, k).
        k: Number of retrieved items to consider.

    Returns:
        Recall@k score.
    """
    n_queries = ground_truth_indices.shape[0]
    recalls = []
    for i in range(n_queries):
        gt_set = set(ground_truth_indices[i])
        ret_set = set(retrieved_indices[i, :k])
        recalls.append(len(gt_set & ret_set) / len(gt_set))
    return float(np.mean(recalls))


def compute_mean_reciprocal_rank(
    ground_truth_indices: np.ndarray,
    retrieved_indices: np.ndarray,
) -> float:
    """Compute Mean Reciprocal Rank (MRR).

    Args:
        ground_truth_indices: True nearest neighbor indices of shape (n_queries, k_gt).
        retrieved_indices: Retrieved indices of shape (n_queries, k).

    Returns:
        MRR score.
    """
    n_queries = ground_truth_indices.shape[0]
    k = retrieved_indices.shape[1]
    rr_sum = 0.0

    for i in range(n_queries):
        gt = ground_truth_indices[i, 0]  # First NN
        for rank, idx in enumerate(retrieved_indices[i], start=1):
            if idx == gt:
                rr_sum += 1.0 / rank
                break

    return rr_sum / n_queries


def compute_ndcg_at_k(
    ground_truth_indices: np.ndarray,
    ground_truth_scores: np.ndarray,
    retrieved_indices: np.ndarray,
    k: int = 10,
) -> float:
    """Compute NDCG@k.

    Args:
        ground_truth_indices: True nearest neighbor indices.
        ground_truth_scores: True similarity scores.
        retrieved_indices: Retrieved indices.
        k: Number of items to consider.

    Returns:
        NDCG@k score.
    """
    n_queries = ground_truth_indices.shape[0]
    ndcg_sum = 0.0

    for i in range(n_queries):
        # Build relevance mapping
        relevance = {}
        for idx, score in zip(ground_truth_indices[i], ground_truth_scores[i]):
            relevance[idx] = score

        # DCG
        dcg = 0.0
        for rank, idx in enumerate(retrieved_indices[i, :k], start=1):
            rel = relevance.get(idx, 0.0)
            dcg += rel / np.log2(rank + 1)

        # Ideal DCG
        ideal_scores = sorted(ground_truth_scores[i], reverse=True)[:k]
        idcg = sum(s / np.log2(r + 1) for r, s in enumerate(ideal_scores, start=1))

        if idcg > 0:
            ndcg_sum += dcg / idcg

    return ndcg_sum / n_queries


def brute_force_knn(
    queries: np.ndarray,
    database: np.ndarray,
    k: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """Brute-force k-nearest neighbors search.

    Args:
        queries: Query vectors of shape (n_queries, dim).
        database: Database vectors of shape (n_db, dim).
        k: Number of neighbors.

    Returns:
        Tuple of (scores, indices) both of shape (n_queries, k).
    """
    # Compute cosine similarity
    scores = queries @ database.T
    # Get top-k
    indices = np.argpartition(-scores, kth=k, axis=1)[:, :k]
    # Sort within top-k
    top_scores = np.take_along_axis(scores, indices, axis=1)
    sort_order = np.argsort(-top_scores, axis=1)
    indices = np.take_along_axis(indices, sort_order, axis=1)
    scores = np.take_along_axis(top_scores, sort_order, axis=1)
    return scores, indices


def run_benchmark(
    quantizer,
    train_vectors: np.ndarray,
    test_vectors: np.ndarray,
    k_values: List[int] = None,
) -> Dict[str, float]:
    """Run a full benchmark on a quantizer.

    Args:
        quantizer: Fitted quantizer instance.
        train_vectors: Training/database vectors.
        test_vectors: Test/query vectors.
        k_values: List of k values for recall evaluation.

    Returns:
        Dictionary of benchmark metrics.
    """
    if k_values is None:
        k_values = [1, 5, 10, 50, 100]

    # Ground truth
    gt_scores, gt_indices = brute_force_knn(test_vectors, train_vectors, k=max(k_values))

    # Quantized search
    codes = quantizer.encode(train_vectors)
    reconstructed = quantizer.decode(codes)

    # Search on reconstructed vectors
    rec_scores, rec_indices = brute_force_knn(test_vectors, reconstructed, k=max(k_values))

    results = {
        "compression_ratio": quantizer.compression_ratio(),
        "original_size_mb": train_vectors.nbytes / (1024 * 1024),
    }

    # Estimate compressed size
    if isinstance(codes, dict):
        # Rough estimate for dict-based codes
        compressed_size = sum(
            v.nbytes if isinstance(v, np.ndarray) else 0
            for v in codes.values()
        )
    else:
        compressed_size = codes.nbytes
    results["compressed_size_mb"] = compressed_size / (1024 * 1024)

    # Recall metrics
    for k in k_values:
        recall = compute_recall_at_k(gt_indices[:, :k], rec_indices, k=k)
        results[f"recall@{k}"] = recall

    # MRR
    results["mrr"] = compute_mean_reciprocal_rank(gt_indices[:, :1], rec_indices[:, :10])

    return results
