"""Visualization utilities for benchmark results."""

import os
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np


def plot_recall_vs_compression(
    results: Dict[str, Dict[str, float]],
    output_path: str,
    title: str = "Recall vs Compression Ratio",
) -> str:
    """Plot recall@k vs compression ratio for multiple quantizers.

    Args:
        results: Dict mapping quantizer name to metrics dict.
        output_path: Path to save the plot.
        title: Plot title.

    Returns:
        Path to saved plot.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))

    for idx, (name, metrics) in enumerate(results.items()):
        cr = metrics.get("compression_ratio", 1.0)

        # Collect recall@k values
        k_values = []
        recalls = []
        for key, val in sorted(metrics.items()):
            if key.startswith("recall@"):
                k = int(key.split("@")[1])
                k_values.append(k)
                recalls.append(val)

        if k_values:
            ax.plot(
                [cr] * len(k_values),
                recalls,
                "o-",
                label=name,
                color=colors[idx],
                markersize=8,
            )

    ax.set_xlabel("Compression Ratio", fontsize=12)
    ax.set_ylabel("Recall", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    return output_path


def plot_recall_curves(
    results: Dict[str, Dict[str, float]],
    output_path: str,
    title: str = "Recall@k Curves",
) -> str:
    """Plot recall@k curves for multiple quantizers.

    Args:
        results: Dict mapping quantizer name to metrics dict.
        output_path: Path to save the plot.
        title: Plot title.

    Returns:
        Path to saved plot.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))

    for idx, (name, metrics) in enumerate(results.items()):
        k_values = []
        recalls = []
        for key, val in sorted(metrics.items()):
            if key.startswith("recall@"):
                k = int(key.split("@")[1])
                k_values.append(k)
                recalls.append(val)

        if k_values:
            ax.plot(
                k_values,
                recalls,
                "o-",
                label=name,
                color=colors[idx],
                markersize=8,
                linewidth=2,
            )

    ax.set_xlabel("k", fontsize=12)
    ax.set_ylabel("Recall@k", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    return output_path


def plot_compression_comparison(
    results: Dict[str, Dict[str, float]],
    output_path: str,
    title: str = "Compression Ratio Comparison",
) -> str:
    """Plot bar chart comparing compression ratios.

    Args:
        results: Dict mapping quantizer name to metrics dict.
        output_path: Path to save the plot.
        title: Plot title.

    Returns:
        Path to saved plot.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    names = list(results.keys())
    ratios = [results[n].get("compression_ratio", 1.0) for n in names]

    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(names)))
    bars = ax.bar(names, ratios, color=colors, edgecolor="black", linewidth=1.2)

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.1f}x",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    ax.set_ylabel("Compression Ratio", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    return output_path


def plot_size_comparison(
    results: Dict[str, Dict[str, float]],
    output_path: str,
    title: str = "Memory Usage Comparison",
) -> str:
    """Plot bar chart comparing original vs compressed sizes.

    Args:
        results: Dict mapping quantizer name to metrics dict.
        output_path: Path to save the plot.
        title: Plot title.

    Returns:
        Path to saved plot.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    names = list(results.keys())
    original = [results[n].get("original_size_mb", 0) for n in names]
    compressed = [results[n].get("compressed_size_mb", 0) for n in names]

    x = np.arange(len(names))
    width = 0.35

    bars1 = ax.bar(x - width / 2, original, width, label="Original", color="#e74c3c")
    bars2 = ax.bar(x + width / 2, compressed, width, label="Compressed", color="#2ecc71")

    ax.set_ylabel("Size (MB)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.legend(fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    return output_path


def generate_benchmark_report(
    results: Dict[str, Dict[str, float]],
    output_dir: str,
    dataset_name: str = "benchmark",
) -> List[str]:
    """Generate a full set of benchmark visualization plots.

    Args:
        results: Dict mapping quantizer name to metrics dict.
        output_dir: Directory to save plots.
        dataset_name: Name of the dataset for plot titles.

    Returns:
        List of saved plot file paths.
    """
    os.makedirs(output_dir, exist_ok=True)
    paths = []

    paths.append(plot_recall_curves(
        results,
        os.path.join(output_dir, f"{dataset_name}_recall_curves.png"),
        title=f"Recall@k Curves - {dataset_name}",
    ))

    paths.append(plot_compression_comparison(
        results,
        os.path.join(output_dir, f"{dataset_name}_compression.png"),
        title=f"Compression Ratio - {dataset_name}",
    ))

    paths.append(plot_size_comparison(
        results,
        os.path.join(output_dir, f"{dataset_name}_size.png"),
        title=f"Memory Usage - {dataset_name}",
    ))

    return paths
