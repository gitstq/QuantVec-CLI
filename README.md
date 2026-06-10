<p align="center">
  <img src="https://raw.githubusercontent.com/gitstq/QuantVec-CLI/main/docs/assets/logo.png" alt="QuantVec-CLI Logo" width="120">
</p>

<h1 align="center">QuantVec-CLI</h1>

<p align="center">
  <b>A CLI tool for vector quantization, compression analysis, and RAG framework adapter generation</b>
</p>

<p align="center">
  <a href="https://github.com/gitstq/QuantVec-CLI/releases"><img src="https://img.shields.io/github/v/release/gitstq/QuantVec-CLI?style=flat-square&color=blue" alt="Release"></a>
  <a href="https://github.com/gitstq/QuantVec-CLI/blob/main/LICENSE"><img src="https://img.shields.io/github/license/gitstq/QuantVec-CLI?style=flat-square&color=green" alt="License"></a>
  <a href="https://github.com/gitstq/QuantVec-CLI/actions"><img src="https://img.shields.io/github/actions/workflow/status/gitstq/QuantVec-CLI/ci.yml?style=flat-square" alt="CI"></a>
  <img src="https://img.shields.io/badge/Python-3.9%2B-blue?style=flat-square&logo=python" alt="Python">
</p>

---

## 🎉 Project Introduction

**QuantVec-CLI** is a powerful command-line tool designed for developers working with vector embeddings and Retrieval-Augmented Generation (RAG) systems. It provides:

- **Vector Quantization**: Compress high-dimensional embeddings with minimal quality loss
- **Compression Analysis**: Compare quantization strategies side-by-side
- **RAG Adapters**: Generate ready-to-use integration code for popular frameworks

Inspired by the trending [turbovec](https://github.com/RyanCodrai/turbovec) project, QuantVec-CLI brings vector quantization capabilities to Python developers through an intuitive CLI interface.

## ✨ Core Features

| Feature | Description |
|---------|-------------|
| 🚀 **TurboQuant** | Data-oblivious quantizer with random rotation + Lloyd-Max scalar quantization |
| 📊 **Scalar Quantization** | Uniform scalar quantization with min/max range fitting |
| 🧩 **Product Quantization** | K-means based sub-vector quantization for high compression |
| 📈 **Benchmark Suite** | Built-in recall@k, MRR, and NDCG metrics |
| 📉 **Visualizations** | Generate compression ratio and recall curve plots |
| 🔌 **RAG Adapters** | Auto-generate LangChain, LlamaIndex, and Haystack integration code |
| 🎯 **Memory Estimation** | Calculate memory usage for vector datasets |

## 🚀 Quick Start

### Installation

```bash
pip install quantvec-cli
```

With optional RAG framework support:

```bash
pip install quantvec-cli[langchain]      # LangChain support
pip install quantvec-cli[llamaindex]     # LlamaIndex support
pip install quantvec-cli[haystack]       # Haystack support
pip install quantvec-cli[all]            # All frameworks
```

### Basic Usage

```bash
# Run a benchmark with TurboQuant
quantvec benchmark --quantizer turboquant --dim 768 --bit-width 4

# Compare all quantizers
quantvec compare --dim 768 --dataset openai

# Generate a LangChain adapter
quantvec adapter --framework langchain --quantizer turboquant --dim 768

# Estimate memory usage
quantvec estimate --dim 1536 --n-vectors 1000000

# List available quantizers
quantvec list-quantizers
```

### Python API

```python
import numpy as np
from quantvec.quantizers import TurboQuantQuantizer

# Generate sample vectors
vectors = np.random.randn(1000, 128).astype(np.float32)

# Create and fit quantizer
q = TurboQuantQuantizer(dim=128, bit_width=4)
q.fit(vectors)

# Encode and decode
codes = q.encode(vectors)
reconstructed = q.decode(codes)

print(f"Compression ratio: {q.compression_ratio():.2f}x")
```

## 📖 Detailed Usage Guide

### Benchmark Command

```bash
quantvec benchmark \
  --quantizer turboquant \
  --dim 768 \
  --bit-width 4 \
  --dataset random \
  --n-train 10000 \
  --n-test 1000 \
  --output-dir ./results \
  --visualize
```

**Options:**
- `--quantizer`: Choose from `turboquant`, `scalar`, `product`
- `--bit-width`: 2, 4, or 8 bits per coordinate
- `--dataset`: `random`, `clustered`, `glove`, `openai`
- `--visualize`: Generate comparison plots

### Compare Command

```bash
quantvec compare \
  --dim 768 \
  --dataset random \
  --n-train 5000 \
  --n-test 500 \
  --output-dir ./comparison
```

This runs all quantizers at 2-bit, 4-bit, and 8-bit configurations and generates comparison tables and plots.

### Adapter Generation

```bash
quantvec adapter \
  --framework langchain \
  --quantizer turboquant \
  --dim 768 \
  --bit-width 4 \
  --output ./adapter.py
```

Supported frameworks:
- **LangChain**: Generates `QuantVecStore` class
- **LlamaIndex**: Generates `QuantVecIndex` class
- **Haystack**: Generates `QuantVecDocumentStore` class

## 💡 Design Philosophy & Roadmap

### Design Principles

1. **CLI-First**: Optimized for terminal workflows with Rich formatting
2. **Framework Agnostic**: Core quantizers work independently of RAG frameworks
3. **Extensible**: Easy to add new quantizers and adapters
4. **Research-Grade**: Built-in benchmarks for reproducible evaluation

### Roadmap

- [ ] Add more quantization algorithms (Additive Quantization, Optimized Product Quantization)
- [ ] Support GPU-accelerated quantization
- [ ] Add streaming/online quantization for large datasets
- [ ] Implement SIMD-optimized search kernels
- [ ] Add support for sparse vectors
- [ ] Web UI for interactive exploration

## 📦 Packaging & Deployment

### From Source

```bash
git clone https://github.com/gitstq/QuantVec-CLI.git
cd QuantVec-CLI
pip install -e ".[dev]"
pytest
```

### Running Tests

```bash
pytest                    # Run all tests
pytest --cov=quantvec     # With coverage
pytest -v                 # Verbose output
```

### Building Distribution

```bash
python -m build
```

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Commit your changes: `git commit -m 'feat: add amazing feature'`
4. Push to the branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<p align="center">
  Made with ❤️ by the QuantVec Team
</p>
