"""Base quantizer interface."""

from abc import ABC, abstractmethod
from typing import Optional, Tuple

import numpy as np


class BaseQuantizer(ABC):
    """Abstract base class for vector quantizers.

    All quantizers must implement the `fit`, `encode`, and `decode` methods.
    """

    def __init__(self, dim: int, bit_width: int = 4) -> None:
        """Initialize the quantizer.

        Args:
            dim: Dimensionality of input vectors.
            bit_width: Bits per quantized coordinate (2 or 4).
        """
        if dim <= 0:
            raise ValueError("dim must be positive")
        if bit_width not in (2, 4, 8):
            raise ValueError("bit_width must be 2, 4, or 8")
        self.dim = dim
        self.bit_width = bit_width
        self._is_fitted = False

    @abstractmethod
    def fit(self, vectors: np.ndarray) -> "BaseQuantizer":
        """Fit the quantizer on a set of reference vectors.

        Args:
            vectors: Array of shape (n, dim) with dtype float32.

        Returns:
            Self for method chaining.
        """
        ...

    @abstractmethod
    def encode(self, vectors: np.ndarray) -> np.ndarray:
        """Encode vectors into quantized codes.

        Args:
            vectors: Array of shape (n, dim) with dtype float32.

        Returns:
            Quantized codes as a compact numpy array.
        """
        ...

    @abstractmethod
    def decode(self, codes: np.ndarray) -> np.ndarray:
        """Decode quantized codes back to float vectors.

        Args:
            codes: Quantized codes produced by `encode`.

        Returns:
            Reconstructed vectors of shape (n, dim) with dtype float32.
        """
        ...

    @abstractmethod
    def compression_ratio(self) -> float:
        """Return the achieved compression ratio."""
        ...

    def _validate_vectors(self, vectors: np.ndarray) -> np.ndarray:
        """Validate and normalize input vectors.

        Args:
            vectors: Input array.

        Returns:
            Validated float32 array of shape (n, dim).
        """
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim != 2:
            raise ValueError(f"Expected 2D array, got {vectors.ndim}D")
        if vectors.shape[1] != self.dim:
            raise ValueError(
                f"Expected dim={self.dim}, got {vectors.shape[1]}"
            )
        return vectors

    def search(
        self,
        query: np.ndarray,
        codes: np.ndarray,
        k: int = 10,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Brute-force search over quantized codes.

        Args:
            query: Single query vector of shape (dim,) or (1, dim).
            codes: Quantized database codes.
            k: Number of nearest neighbors to return.

        Returns:
            Tuple of (scores, indices) both of shape (k,).
        """
        query = self._validate_vectors(query.reshape(1, -1))
        reconstructed = self.decode(codes)
        scores = np.dot(reconstructed, query[0])
        top_k = np.argsort(-scores)[:k]
        return scores[top_k], top_k
