"""Tests for vector quantizers."""

import numpy as np
import pytest

from quantvec.quantizers import TurboQuantQuantizer, ScalarQuantizer, ProductQuantizer


@pytest.fixture
def sample_vectors():
    """Generate sample unit vectors for testing."""
    rng = np.random.default_rng(42)
    vectors = rng.standard_normal((100, 64), dtype=np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors


class TestTurboQuantQuantizer:
    """Tests for TurboQuant quantizer."""

    def test_init(self):
        q = TurboQuantQuantizer(dim=128, bit_width=4)
        assert q.dim == 128
        assert q.bit_width == 4
        assert not q._is_fitted

    def test_fit(self, sample_vectors):
        q = TurboQuantQuantizer(dim=64, bit_width=4)
        result = q.fit(sample_vectors)
        assert result is q
        assert q._is_fitted
        assert q._rotation is not None
        assert q._codebook is not None

    def test_encode_decode(self, sample_vectors):
        q = TurboQuantQuantizer(dim=64, bit_width=4)
        q.fit(sample_vectors)
        codes = q.encode(sample_vectors)
        reconstructed = q.decode(codes)
        assert reconstructed.shape == sample_vectors.shape
        assert reconstructed.dtype == np.float32

    def test_compression_ratio(self, sample_vectors):
        q = TurboQuantQuantizer(dim=64, bit_width=2)
        q.fit(sample_vectors)
        cr = q.compression_ratio()
        assert cr > 1.0

    def test_search(self, sample_vectors):
        q = TurboQuantQuantizer(dim=64, bit_width=4)
        q.fit(sample_vectors)
        codes = q.encode(sample_vectors)
        query = sample_vectors[0]
        scores, indices = q.search(query, codes, k=5)
        assert len(scores) == 5
        assert len(indices) == 5

    def test_invalid_dim(self):
        with pytest.raises(ValueError):
            TurboQuantQuantizer(dim=0)

    def test_invalid_bit_width(self):
        with pytest.raises(ValueError):
            TurboQuantQuantizer(dim=64, bit_width=3)


class TestScalarQuantizer:
    """Tests for scalar quantizer."""

    def test_init(self):
        q = ScalarQuantizer(dim=128, bit_width=4)
        assert q.dim == 128
        assert q.bit_width == 4

    def test_fit_encode_decode(self, sample_vectors):
        q = ScalarQuantizer(dim=64, bit_width=4)
        q.fit(sample_vectors)
        codes = q.encode(sample_vectors)
        reconstructed = q.decode(codes)
        assert reconstructed.shape == sample_vectors.shape

    def test_compression_ratio(self, sample_vectors):
        q = ScalarQuantizer(dim=64, bit_width=2)
        q.fit(sample_vectors)
        cr = q.compression_ratio()
        assert cr > 1.0


class TestProductQuantizer:
    """Tests for product quantizer."""

    def test_init(self):
        q = ProductQuantizer(dim=64, bit_width=8, n_subvectors=8)
        assert q.dim == 64
        assert q.n_subvectors == 8
        assert q.sub_dim == 8

    def test_invalid_subvectors(self):
        with pytest.raises(ValueError):
            ProductQuantizer(dim=64, n_subvectors=7)

    def test_fit_encode_decode(self, sample_vectors):
        q = ProductQuantizer(dim=64, bit_width=8, n_subvectors=8)
        q.fit(sample_vectors)
        codes = q.encode(sample_vectors)
        reconstructed = q.decode(codes)
        assert reconstructed.shape == sample_vectors.shape

    def test_compression_ratio(self, sample_vectors):
        q = ProductQuantizer(dim=64, bit_width=8, n_subvectors=8)
        q.fit(sample_vectors)
        cr = q.compression_ratio()
        assert cr > 1.0
