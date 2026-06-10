<p align="center">
  <img src="https://raw.githubusercontent.com/gitstq/QuantVec-CLI/main/docs/assets/logo.png" alt="QuantVec-CLI Logo" width="120">
</p>

<h1 align="center">QuantVec-CLI</h1>

<p align="center">
  <b>向量量化、壓縮分析與 RAG 框架適配器生成 CLI 工具</b>
</p>

<p align="center">
  <a href="https://github.com/gitstq/QuantVec-CLI/releases"><img src="https://img.shields.io/github/v/release/gitstq/QuantVec-CLI?style=flat-square&color=blue" alt="Release"></a>
  <a href="https://github.com/gitstq/QuantVec-CLI/blob/main/LICENSE"><img src="https://img.shields.io/github/license/gitstq/QuantVec-CLI?style=flat-square&color=green" alt="License"></a>
  <a href="https://github.com/gitstq/QuantVec-CLI/actions"><img src="https://img.shields.io/github/actions/workflow/status/gitstq/QuantVec-CLI/ci.yml?style=flat-square" alt="CI"></a>
  <img src="https://img.shields.io/badge/Python-3.9%2B-blue?style=flat-square&logo=python" alt="Python">
</p>

---

## 🎉 專案介紹

**QuantVec-CLI** 是一款面向向量嵌入和檢索增強生成（RAG）系統開發者的強大命令列工具。它提供：

- **向量量化**：以最小品質損失壓縮高維嵌入
- **壓縮分析**：並排比較量化策略
- **RAG 適配器**：為流行框架生成即用型整合代碼

靈感來源於熱門的 [turbovec](https://github.com/RyanCodrai/turbovec) 專案，QuantVec-CLI 透過直觀的 CLI 介面為 Python 開發者帶來向量量化能力。

## ✨ 核心特性

| 特性 | 描述 |
|---------|-------------|
| 🚀 **TurboQuant** | 基於隨機旋轉 + Lloyd-Max 標量量化的資料無關量化器 |
| 📊 **標量量化** | 基於最小/最大範圍擬合的均勻標量量化 |
| 🧩 **乘積量化** | 基於 K-Means 的子向量量化，實現高壓縮率 |
| 📈 **基準測試套件** | 內建 recall@k、MRR 和 NDCG 指標 |
| 📉 **視覺化** | 生成壓縮率和召回率曲線圖 |
| 🔌 **RAG 適配器** | 自動生成 LangChain、LlamaIndex 和 Haystack 整合代碼 |
| 🎯 **記憶體估算** | 計算向量資料集的記憶體使用量 |

## 🚀 快速開始

### 安裝

```bash
pip install quantvec-cli
```

帶可選 RAG 框架支援：

```bash
pip install quantvec-cli[langchain]      # LangChain 支援
pip install quantvec-cli[llamaindex]     # LlamaIndex 支援
pip install quantvec-cli[haystack]       # Haystack 支援
pip install quantvec-cli[all]            # 所有框架
```

### 基本用法

```bash
# 使用 TurboQuant 執行基準測試
quantvec benchmark --quantizer turboquant --dim 768 --bit-width 4

# 比較所有量化器
quantvec compare --dim 768 --dataset openai

# 生成 LangChain 適配器
quantvec adapter --framework langchain --quantizer turboquant --dim 768

# 估算記憶體使用
quantvec estimate --dim 1536 --n-vectors 1000000

# 列出可用量化器
quantvec list-quantizers
```

### Python API

```python
import numpy as np
from quantvec.quantizers import TurboQuantQuantizer

# 生成樣本向量
vectors = np.random.randn(1000, 128).astype(np.float32)

# 建立並擬合量化器
q = TurboQuantQuantizer(dim=128, bit_width=4)
q.fit(vectors)

# 編碼和解碼
codes = q.encode(vectors)
reconstructed = q.decode(codes)

print(f"壓縮率: {q.compression_ratio():.2f}x")
```

## 📖 詳細使用指南

### 基準測試命令

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

**選項：**
- `--quantizer`: 選擇 `turboquant`、`scalar`、`product`
- `--bit-width`: 每個座標的位元數：2、4 或 8
- `--dataset`: `random`、`clustered`、`glove`、`openai`
- `--visualize`: 生成對比圖表

### 比較命令

```bash
quantvec compare \
  --dim 768 \
  --dataset random \
  --n-train 5000 \
  --n-test 500 \
  --output-dir ./comparison
```

此命令會以 2-bit、4-bit 和 8-bit 配置執行所有量化器，並生成比較表格和圖表。

### 適配器生成

```bash
quantvec adapter \
  --framework langchain \
  --quantizer turboquant \
  --dim 768 \
  --bit-width 4 \
  --output ./adapter.py
```

支援的框架：
- **LangChain**: 生成 `QuantVecStore` 類別
- **LlamaIndex**: 生成 `QuantVecIndex` 類別
- **Haystack**: 生成 `QuantVecDocumentStore` 類別

## 💡 設計思路與迭代規劃

### 設計原則

1. **CLI 優先**: 針對終端工作流最佳化，使用 Rich 格式化
2. **框架無關**: 核心量化器獨立於 RAG 框架工作
3. **可擴展**: 易於添加新的量化器和適配器
4. **研究級**: 內建可復現的評估基準

### 路線圖

- [ ] 添加更多量化演算法（加法量化、最佳化乘積量化）
- [ ] 支援 GPU 加速量化
- [ ] 為大型資料集添加串流/線上量化
- [ ] 實現 SIMD 最佳化搜尋核心
- [ ] 添加稀疏向量支援
- [ ] 用於互動式探索的 Web UI

## 📦 打包與部署

### 從原始碼安裝

```bash
git clone https://github.com/gitstq/QuantVec-CLI.git
cd QuantVec-CLI
pip install -e ".[dev]"
pytest
```

### 執行測試

```bash
pytest                    # 執行所有測試
pytest --cov=quantvec     # 帶覆蓋率
pytest -v                 # 詳細輸出
```

### 構建分發包

```bash
python -m build
```

## 🤝 貢獻指南

歡迎貢獻！請參閱 [CONTRIBUTING.md](CONTRIBUTING.md) 了解指南。

1. Fork 倉庫
2. 建立功能分支：`git checkout -b feature/amazing-feature`
3. 提交更改：`git commit -m 'feat: add amazing feature'`
4. 推送到分支：`git push origin feature/amazing-feature`
5. 開啟 Pull Request

## 📄 開源協議

本專案採用 MIT 授權條款 - 詳情請參閱 [LICENSE](LICENSE) 檔案。

---

<p align="center">
  Made with ❤️ by the QuantVec Team
</p>
