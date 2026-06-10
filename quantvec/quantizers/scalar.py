"""Scalar (uniform) quantizer implementation."""

import numpy as np

from quantvec.quantizers.base import BaseQuantizer


class ScalarQuantizer(BaseQuantizer):
    """Uniform scalar quantizer.

    Divides the value range into equal-width bins and maps each coordinate
    to the nearest bin center. Simple and fast, but less optimal than
    Lloyd-Max for non-uniform distributions.
    """

    def __init__(self, dim: int, bit_width: int = 4) -> None:
        super().__init__(dim, bit_width)
        self._min_val: Optional[float] = None
        self._max_val: Optional[float] = None
        self._scale: Optional[float] = None

    def fit(self, vectors: np.ndarray) -> "ScalarQuantizer":
        """Fit by computing min/max range from reference vectors."""
        vectors = self._validate_vectors(vectors)
        self._min_val = float(np.min(vectors))
        self._max_val = float(np.max(vectors))
        if self._max_val == self._min_val:
            self._scale = 1.0
        else:
            self._scale = (self._max_val - self._min_val) / ((2 ** self.bit_width) - 1)
        self._is_fitted = True
        return self

    def encode(self, vectors: np.ndarray) -> np.ndarray:
        """Encode vectors to scalar-quantized codes."""
        if not self._is_fitted:
            raise RuntimeError("Quantizer must be fitted before encoding")

        vectors = self._validate_vectors(vectors)
        n_levels = 2 ** self.bit_width

        # Normalize to [0, n_levels-1]
        normalized = (vectors - self._min_val) / self._scale
        normalized = np.clip(normalized, 0, n_levels - 1)
        indices = np.round(normalized).astype(np.uint8)

        # Pack bits
        packed = self._pack_bits(indices, self.bit_width)

        return {
            "codes": packed,
            "min_val": self._min_val,
            "max_val": self._max_val,
            "scale": self._scale,
            "bit_width": self.bit_width,
            "dim": self.dim,
        }

    def decode(self, codes: np.ndarray) -> np.ndarray:
        """Decode scalar-quantized codes back to float vectors."""
        if isinstance(codes, dict):
            packed = codes["codes"]
            min_val = codes["min_val"]
            scale = codes["scale"]
            bit_width = codes["bit_width"]
            dim = codes["dim"]
        else:
            raise ValueError("Scalar decode requires full code dict")

        # Unpack bits
        indices = self._unpack_bits(packed, bit_width, dim)

        # Dequantize
        reconstructed = indices.astype(np.float32) * scale + min_val
        return reconstructed

    def compression_ratio(self) -> float:
        """Return compression ratio."""
        original_bits = self.dim * 32
        # codes + min + max + scale
        compressed_bits = self.dim * self.bit_width + 32 * 3
        return original_bits / compressed_bits

    @staticmethod
    def _pack_bits(indices: np.ndarray, bit_width: int) -> np.ndarray:
        """Pack n-bit indices into bytes."""
        n, d = indices.shape
        if bit_width == 2:
            padded_d = ((d + 3) // 4) * 4
            padded = np.zeros((n, padded_d), dtype=np.uint8)
            padded[:, :d] = indices
            packed = np.zeros((n, padded_d // 4), dtype=np.uint8)
            for i in range(4):
                packed |= (padded[:, i::4] & 0x3) << (2 * i)
            return packed
        elif bit_width == 4:
            padded_d = ((d + 1) // 2) * 2
            padded = np.zeros((n, padded_d), dtype=np.uint8)
            padded[:, :d] = indices
            packed = np.zeros((n, padded_d // 2), dtype=np.uint8)
            packed = (padded[:, 0::2] & 0xF) | ((padded[:, 1::2] & 0xF) << 4)
            return packed
        elif bit_width == 8:
            return indices.astype(np.uint8)
        else:
            raise ValueError(f"Unsupported bit_width: {bit_width}")

    @staticmethod
    def _unpack_bits(packed: np.ndarray, bit_width: int, dim: int) -> np.ndarray:
        """Unpack bytes back to n-bit indices."""
        n = packed.shape[0]
        if bit_width == 2:
            indices = np.zeros((n, (packed.shape[1] * 4)), dtype=np.uint8)
            for i in range(4):
                indices[:, i::4] = (packed >> (2 * i)) & 0x3
            return indices[:, :dim]
        elif bit_width == 4:
            indices = np.zeros((n, (packed.shape[1] * 2)), dtype=np.uint8)
            indices[:, 0::2] = packed & 0xF
            indices[:, 1::2] = (packed >> 4) & 0xF
            return indices[:, :dim]
        elif bit_width == 8:
            return packed[:, :dim]
        else:
            raise ValueError(f"Unsupported bit_width: {bit_width}")
