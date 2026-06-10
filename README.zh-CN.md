<p align="center">
  <img src="https://raw.githubusercontent.com/gitstq/QuantVec-CLI/main/docs/assets/logo.png" alt="QuantVec-CLI Logo" width="120">
</p>

<h1 align="center">QuantVec-CLI</h1>

<p align="center">
  <b>向量量化、压缩分析与 RAG 框架适配器生成 CLI 工具</b>
</p>

<p align="center">
  <a href="https://github.com/gitstq/QuantVec-CLI/releases"><img src="https://img.shields.io/github/v/release/gitstq/QuantVec-CLI?style=flat-square&color=blue" alt="Release"></a>
  <a href="https://github.com/gitstq/QuantVec-CLI/blob/main/LICENSE"><img src="https://img.shields.io/github/license/gitstq/QuantVec-CLI?style=flat-square&color=green" alt="License"></a>
  <a href="https://github.com/gitstq/QuantVec-CLI/actions"><img src="https://img.shields.io/github/actions/workflow/status/gitstq/QuantVec-CLI/ci.yml?style=flat-square" alt="CI"></a>
  <img src="https://img.shields.io/badge/Python-3.9%2B-blue?style=flat-square&logo=python" alt="Python">
</p>

---

## 🎉 项目介绍

**QuantVec-CLI** 是一款面向向量嵌入和检索增强生成（RAG）系统开发者的强大命令行工具。它提供：

- **向量量化**：以最小质量损失压缩高维嵌入
- **压缩分析**：并排比较量化策略
- **RAG 适配器**：为流行框架生成即用型集成代码

灵感来源于热门的 [turbovec](https://github.com/RyanCodrai/turbovec) 项目，QuantVec-CLI 通过直观的 CLI 界面为 Python 开发者带来向量量化能力。

## ✨ 核心特性

| 特性 | 描述 |
|---------|-------------|
| 🚀 **TurboQuant** | 基于随机旋转 + Lloyd-Max 标量量化的数据无关量化器 |
| 📊 **标量量化** | 基于最小/最大范围拟合的均匀标量量化 |
| 🧩 **乘积量化** | 基于 K-Means 的子向量量化，实现高压缩率 |
| 📈 **基准测试套件** | 内置 recall@k、MRR 和 NDCG 指标 |
| 📉 **可视化** | 生成压缩率和召回率曲线图 |
| 🔌 **RAG 适配器** | 自动生成 LangChain、LlamaIndex 和 Haystack 集成代码 |
| 🎯 **内存估算** | 计算向量数据集的内存使用量 |

## 🚀 快速开始

### 安装

```bash
pip install quantvec-cli
```

带可选 RAG 框架支持：

```bash
pip install quantvec-cli[langchain]      # LangChain 支持
pip install quantvec-cli[llamaindex]     # LlamaIndex 支持
pip install quantvec-cli[haystack]       # Haystack 支持
pip install quantvec-cli[all]            # 所有框架
```

### 基本用法

```bash
# 使用 TurboQuant 运行基准测试
quantvec benchmark --quantizer turboquant --dim 768 --bit-width 4

# 比较所有量化器
quantvec compare --dim 768 --dataset openai

# 生成 LangChain 适配器
quantvec adapter --framework langchain --quantizer turboquant --dim 768

# 估算内存使用
quantvec estimate --dim 1536 --n-vectors 1000000

# 列出可用量化器
quantvec list-quantizers
```

### Python API

```python
import numpy as np
from quantvec.quantizers import TurboQuantQuantizer

# 生成样本向量
vectors = np.random.randn(1000, 128).astype(np.float32)

# 创建并拟合量化器
q = TurboQuantQuantizer(dim=128, bit_width=4)
q.fit(vectors)

# 编码和解码
codes = q.encode(vectors)
reconstructed = q.decode(codes)

print(f"压缩率: {q.compression_ratio():.2f}x")
```

## 📖 详细使用指南

### 基准测试命令

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

**选项：**
- `--quantizer`: 选择 `turboquant`、`scalar`、`product`
- `--bit-width`: 每个坐标的位数：2、4 或 8
- `--dataset`: `random`、`clustered`、`glove`、`openai`
- `--visualize`: 生成对比图表

### 比较命令

```bash
quantvec compare \
  --dim 768 \
  --dataset random \
  --n-train 5000 \
  --n-test 500 \
  --output-dir ./comparison
```

此命令会以 2-bit、4-bit 和 8-bit 配置运行所有量化器，并生成比较表格和图表。

### 适配器生成

```bash
quantvec adapter \
  --framework langchain \
  --quantizer turboquant \
  --dim 768 \
  --bit-width 4 \
  --output ./adapter.py
```

支持的框架：
- **LangChain**: 生成 `QuantVecStore` 类
- **LlamaIndex**: 生成 `QuantVecIndex` 类
- **Haystack**: 生成 `QuantVecDocumentStore` 类

## 💡 设计思路与迭代规划

### 设计原则

1. **CLI 优先**: 针对终端工作流优化，使用 Rich 格式化
2. **框架无关**: 核心量化器独立于 RAG 框架工作
3. **可扩展**: 易于添加新的量化器和适配器
4. **研究级**: 内置可复现的评估基准

### 路线图

- [ ] 添加更多量化算法（加法量化、优化乘积量化）
- [ ] 支持 GPU 加速量化
- [ ] 为大型数据集添加流式/在线量化
- [ ] 实现 SIMD 优化搜索内核
- [ ] 添加稀疏向量支持
- [ ] 用于交互式探索的 Web UI

## 📦 打包与部署

### 从源码安装

```bash
git clone https://github.com/gitstq/QuantVec-CLI.git
cd QuantVec-CLI
pip install -e ".[dev]"
pytest
```

### 运行测试

```bash
pytest                    # 运行所有测试
pytest --cov=quantvec     # 带覆盖率
pytest -v                 # 详细输出
```

### 构建分发包

```bash
python -m build
```

## 🤝 贡献指南

欢迎贡献！请参阅 [CONTRIBUTING.md](CONTRIBUTING.md) 了解指南。

1. Fork 仓库
2. 创建功能分支：`git checkout -b feature/amazing-feature`
3. 提交更改：`git commit -m 'feat: add amazing feature'`
4. 推送到分支：`git push origin feature/amazing-feature`
5. 打开 Pull Request

## 📄 开源协议

本项目采用 MIT 许可证 - 详情请参阅 [LICENSE](LICENSE) 文件。

---

<p align="center">
  Made with ❤️ by the QuantVec Team
</p>
