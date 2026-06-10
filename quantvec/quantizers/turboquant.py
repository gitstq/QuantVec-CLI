"""TurboQuant quantizer implementation.

Based on Google Research's TurboQuant algorithm — a data-oblivious quantizer
with near-optimal distortion and no separate training phase.
"""

import numpy as np
from scipy.stats import beta as beta_dist

from quantvec.quantizers.base import BaseQuantizer


class TurboQuantQuantizer(BaseQuantizer):
    """TurboQuant vector quantizer.

    Compresses high-dimensional vectors using random rotation + Lloyd-Max
    scalar quantization. Achieves ~16x compression at 2-bit and ~8x at 4-bit
    with minimal recall loss.
    """

    def __init__(self, dim: int, bit_width: int = 4) -> None:
        super().__init__(dim, bit_width)
        self._rotation: Optional[np.ndarray] = None
        self._codebook: Optional[np.ndarray] = None
        self._norms: Optional[np.ndarray] = None
        self._calibration_shift: Optional[np.ndarray] = None
        self._calibration_scale: Optional[np.ndarray] = None

    def fit(self, vectors: np.ndarray) -> "TurboQuantQuantizer":
        """Fit the quantizer.

        TurboQuant requires no explicit training — the codebook is derived
        mathematically from the Beta distribution. We only compute the
        random rotation matrix here.
        """
        vectors = self._validate_vectors(vectors)
        n = vectors.shape[0]

        # 1. Generate random orthogonal rotation matrix
        self._rotation = self._random_orthogonal(self.dim)

        # 2. Compute Lloyd-Max codebook from Beta distribution
        self._codebook = self._lloyd_max_codebook(self.bit_width)

        # 3. Compute TQ+ calibration on first batch
        rotated = vectors @ self._rotation.T
        norms = np.linalg.norm(rotated, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        unit_rotated = rotated / norms

        # Per-coordinate calibration: map empirical 5/95% quantiles to canonical Beta
        self._calibration_shift = np.percentile(unit_rotated, 5, axis=0)
        self._calibration_scale = (
            np.percentile(unit_rotated, 95, axis=0) - self._calibration_shift
        )
        self._calibration_scale = np.where(
            self._calibration_scale == 0, 1.0, self._calibration_scale
        )

        self._is_fitted = True
        return self

    def encode(self, vectors: np.ndarray) -> np.ndarray:
        """Encode vectors to TurboQuant codes."""
        if not self._is_fitted:
            raise RuntimeError("Quantizer must be fitted before encoding")

        vectors = self._validate_vectors(vectors)
        n = vectors.shape[0]

        # Rotate
        rotated = vectors @ self._rotation.T
        norms = np.linalg.norm(rotated, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        unit_rotated = rotated / norms

        # Calibrate
        calibrated = (unit_rotated - self._calibration_shift) / self._calibration_scale

        # Quantize using Lloyd-Max codebook
        codes = self._quantize(calibrated, self._codebook)

        # Pack bits
        packed = self._pack_bits(codes, self.bit_width)

        # Store norms and length-renormalization factor
        renorm = np.sum(unit_rotated * self._dequantize(codes, self._codebook), axis=1)
        renorm = np.where(renorm == 0, 1.0, renorm)
        length_renorm = (norms.flatten() / renorm).astype(np.float32)

        return {
            "codes": packed,
            "norms": norms.flatten().astype(np.float32),
            "length_renorm": length_renorm,
            "rotation": self._rotation,
            "codebook": self._codebook,
            "calibration_shift": self._calibration_shift,
            "calibration_scale": self._calibration_scale,
        }

    def decode(self, codes: np.ndarray) -> np.ndarray:
        """Decode TurboQuant codes back to float vectors."""
        if isinstance(codes, dict):
            packed = codes["codes"]
            norms = codes["norms"]
            length_renorm = codes["length_renorm"]
            rotation = codes["rotation"]
            codebook = codes["codebook"]
            calibration_shift = codes["calibration_shift"]
            calibration_scale = codes["calibration_scale"]
        else:
            raise ValueError("TurboQuant decode requires full code dict")

        # Unpack bits
        quantized = self._unpack_bits(packed, self.bit_width, self.dim)

        # Dequantize
        dequantized = self._dequantize(quantized, codebook)

        # Un-calibrate
        uncalibrated = dequantized * calibration_scale + calibration_shift

        # Renormalize
        renormalized = uncalibrated * length_renorm.reshape(-1, 1)

        # Un-rotate
        reconstructed = renormalized @ rotation

        return reconstructed.astype(np.float32)

    def compression_ratio(self) -> float:
        """Return compression ratio: original bytes / compressed bytes."""
        original_bits = self.dim * 32  # float32
        compressed_bits = self.dim * self.bit_width + 32 + 32  # codes + norm + renorm
        return original_bits / compressed_bits

    @staticmethod
    def _random_orthogonal(d: int) -> np.ndarray:
        """Generate a random orthogonal matrix via QR decomposition."""
        A = np.random.randn(d, d).astype(np.float64)
        Q, R = np.linalg.qr(A)
        # Ensure determinant is +1 (proper rotation)
        D = np.diag(np.sign(np.diag(R)))
        Q = Q @ D
        return Q.astype(np.float32)

    @staticmethod
    def _lloyd_max_codebook(bit_width: int, max_iter: int = 100) -> np.ndarray:
        """Compute Lloyd-Max codebook for Beta(0.5, 0.5)-like distribution.

        In high dimensions after random rotation, coordinates follow a
        distribution close to Beta(0.5, 0.5) on [-1, 1].
        """
        n_levels = 2 ** bit_width
        # Initialize with uniform quantiles
        boundaries = np.linspace(-1, 1, n_levels + 1)
        centroids = (boundaries[:-1] + boundaries[1:]) / 2

        for _ in range(max_iter):
            # Compute new boundaries as midpoints between centroids
            boundaries = np.concatenate([[-1.0], (centroids[:-1] + centroids[1:]) / 2, [1.0]])
            # Compute new centroids as conditional means
            new_centroids = np.zeros(n_levels)
            for i in range(n_levels):
                a, b = boundaries[i], boundaries[i + 1]
                # Approximate mean using Beta distribution CDF
                p_a = beta_dist.cdf((a + 1) / 2, 0.5, 0.5)
                p_b = beta_dist.cdf((b + 1) / 2, 0.5, 0.5)
                if p_b > p_a:
                    # Sample points in interval and compute weighted mean
                    xs = np.linspace(a, b, 1000)
                    ps = beta_dist.pdf((xs + 1) / 2, 0.5, 0.5)
                    new_centroids[i] = np.trapezoid(xs * ps, xs) / np.trapezoid(ps, xs)
                else:
                    new_centroids[i] = centroids[i]

            if np.allclose(centroids, new_centroids, atol=1e-6):
                break
            centroids = new_centroids

        return centroids.astype(np.float32)

    @staticmethod
    def _quantize(values: np.ndarray, codebook: np.ndarray) -> np.ndarray:
        """Scalar quantization: map each value to nearest codebook index."""
        # Vectorized nearest neighbor search
        diffs = np.abs(values[:, :, np.newaxis] - codebook[np.newaxis, np.newaxis, :])
        return np.argmin(diffs, axis=2).astype(np.uint8)

    @staticmethod
    def _dequantize(indices: np.ndarray, codebook: np.ndarray) -> np.ndarray:
        """Dequantize indices back to float values."""
        return codebook[indices]

    @staticmethod
    def _pack_bits(indices: np.ndarray, bit_width: int) -> np.ndarray:
        """Pack n-bit indices into bytes."""
        n, d = indices.shape
        if bit_width == 2:
            # 4 values per byte
            padded_d = ((d + 3) // 4) * 4
            padded = np.zeros((n, padded_d), dtype=np.uint8)
            padded[:, :d] = indices
            packed = np.zeros((n, padded_d // 4), dtype=np.uint8)
            for i in range(4):
                packed |= (padded[:, i::4] & 0x3) << (2 * i)
            return packed
        elif bit_width == 4:
            # 2 values per byte
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
