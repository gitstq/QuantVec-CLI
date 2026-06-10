"""Product Quantization (PQ) implementation."""

from typing import List, Optional

import numpy as np

from quantvec.quantizers.base import BaseQuantizer


class ProductQuantizer(BaseQuantizer):
    """Product Quantization.

    Splits the vector into m sub-vectors and quantizes each sub-vector
    independently using k-means codebooks. Achieves high compression with
    good recall, but requires a training phase.
    """

    def __init__(
        self,
        dim: int,
        bit_width: int = 8,
        n_subvectors: Optional[int] = None,
        n_centroids: Optional[int] = None,
    ) -> None:
        """Initialize PQ quantizer.

        Args:
            dim: Vector dimensionality.
            bit_width: Bits per code (default 8, determines n_centroids).
            n_subvectors: Number of sub-vectors (default: dim // 8).
            n_centroids: Number of centroids per sub-vector (default: 2**bit_width).
        """
        super().__init__(dim, bit_width)
        if n_subvectors is None:
            n_subvectors = max(1, dim // 8)
        if n_centroids is None:
            n_centroids = 2 ** bit_width

        if dim % n_subvectors != 0:
            raise ValueError(f"dim ({dim}) must be divisible by n_subvectors ({n_subvectors})")

        self.n_subvectors = n_subvectors
        self.n_centroids = n_centroids
        self.sub_dim = dim // n_subvectors
        self._codebooks: Optional[List[np.ndarray]] = None

    def fit(self, vectors: np.ndarray) -> "ProductQuantizer":
        """Fit k-means codebooks on reference vectors."""
        vectors = self._validate_vectors(vectors)
        n = vectors.shape[0]

        self._codebooks = []
        for m in range(self.n_subvectors):
            start = m * self.sub_dim
            end = start + self.sub_dim
            sub_vecs = vectors[:, start:end]

            # K-means++ initialization
            centroids = self._kmeans_plus_plus(sub_vecs, self.n_centroids)
            # Lloyd iterations
            centroids = self._lloyd_iterations(sub_vecs, centroids, max_iter=20)
            self._codebooks.append(centroids)

        self._is_fitted = True
        return self

    def encode(self, vectors: np.ndarray) -> np.ndarray:
        """Encode vectors to PQ codes."""
        if not self._is_fitted:
            raise RuntimeError("Quantizer must be fitted before encoding")

        vectors = self._validate_vectors(vectors)
        n = vectors.shape[0]
        codes = np.zeros((n, self.n_subvectors), dtype=np.uint8)

        for m in range(self.n_subvectors):
            start = m * self.sub_dim
            end = start + self.sub_dim
            sub_vecs = vectors[:, start:end]

            # Assign to nearest centroid
            dists = self._compute_distances(sub_vecs, self._codebooks[m])
            codes[:, m] = np.argmin(dists, axis=1).astype(np.uint8)

        return {
            "codes": codes,
            "codebooks": self._codebooks,
            "n_subvectors": self.n_subvectors,
            "sub_dim": self.sub_dim,
            "dim": self.dim,
        }

    def decode(self, codes: np.ndarray) -> np.ndarray:
        """Decode PQ codes back to float vectors."""
        if isinstance(codes, dict):
            pq_codes = codes["codes"]
            codebooks = codes["codebooks"]
            n_subvectors = codes["n_subvectors"]
            sub_dim = codes["sub_dim"]
            dim = codes["dim"]
        else:
            raise ValueError("PQ decode requires full code dict")

        n = pq_codes.shape[0]
        reconstructed = np.zeros((n, dim), dtype=np.float32)

        for m in range(n_subvectors):
            start = m * sub_dim
            end = start + sub_dim
            reconstructed[:, start:end] = codebooks[m][pq_codes[:, m]]

        return reconstructed

    def compression_ratio(self) -> float:
        """Return compression ratio."""
        original_bits = self.dim * 32
        # codes + codebooks
        codebook_bits = self.n_subvectors * self.n_centroids * self.sub_dim * 32
        codes_bits = self.n_subvectors * self.bit_width
        # Assume 1M vectors for amortized codebook cost
        n_vectors = 1_000_000
        compressed_bits = codes_bits + (codebook_bits / n_vectors)
        return original_bits / compressed_bits

    @staticmethod
    def _kmeans_plus_plus(data: np.ndarray, k: int) -> np.ndarray:
        """K-means++ initialization."""
        n, d = data.shape
        centroids = np.zeros((k, d), dtype=np.float32)
        centroids[0] = data[np.random.randint(n)]

        for i in range(1, k):
            dists = np.min(
                np.sum((data[:, np.newaxis, :] - centroids[:i, :]) ** 2, axis=2),
                axis=1,
            )
            dists = np.maximum(dists, 0)
            total = np.sum(dists)
            if total == 0:
                # All distances are zero, pick randomly
                centroids[i] = data[np.random.randint(n)]
                continue
            probs = dists / total
            # Ensure no NaN in probs
            probs = np.nan_to_num(probs, nan=0.0)
            probs = probs / np.sum(probs)
            centroids[i] = data[np.random.choice(n, p=probs)]

        return centroids

    @staticmethod
    def _lloyd_iterations(
        data: np.ndarray, centroids: np.ndarray, max_iter: int = 20
    ) -> np.ndarray:
        """Run Lloyd iterations."""
        for _ in range(max_iter):
            # Assign
            dists = ProductQuantizer._compute_distances(data, centroids)
            labels = np.argmin(dists, axis=1)

            # Update
            new_centroids = np.zeros_like(centroids)
            for i in range(centroids.shape[0]):
                mask = labels == i
                if np.any(mask):
                    new_centroids[i] = np.mean(data[mask], axis=0)
                else:
                    new_centroids[i] = centroids[i]

            if np.allclose(centroids, new_centroids, atol=1e-5):
                break
            centroids = new_centroids

        return centroids

    @staticmethod
    def _compute_distances(data: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        """Compute squared Euclidean distances between data and centroids."""
        # (n, 1, d) - (1, k, d) -> (n, k, d) -> sum -> (n, k)
        return np.sum(
            (data[:, np.newaxis, :] - centroids[np.newaxis, :, :]) ** 2,
            axis=2,
        )
