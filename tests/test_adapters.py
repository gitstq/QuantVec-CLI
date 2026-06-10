"""Tests for RAG framework adapters."""

from quantvec.adapters import (
    generate_langchain_adapter,
    generate_llamaindex_adapter,
    generate_haystack_adapter,
)


class TestAdapters:
    """Tests for adapter generators."""

    def test_langchain_adapter(self):
        code = generate_langchain_adapter("turboquant", 4, 768)
        assert "class QuantVecStore" in code
        assert "TurboQuantQuantizer" in code
        assert "langchain" in code.lower()

    def test_llamaindex_adapter(self):
        code = generate_llamaindex_adapter("scalar", 2, 512)
        assert "class QuantVecIndex" in code
        assert "ScalarQuantizer" in code
        assert "llama" in code.lower()

    def test_haystack_adapter(self):
        code = generate_haystack_adapter("product", 8, 1024)
        assert "class QuantVecDocumentStore" in code
        assert "ProductQuantizer" in code
        assert "haystack" in code.lower()

    def test_adapter_contains_methods(self):
        code = generate_langchain_adapter("turboquant", 4, 768)
        assert "def add_texts" in code
        assert "def similarity_search" in code
        assert "def from_texts" in code
