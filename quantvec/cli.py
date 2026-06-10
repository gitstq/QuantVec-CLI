"""QuantVec-CLI main entry point."""

import os
import sys
from typing import Optional

import click
import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.syntax import Syntax

from quantvec import __version__
from quantvec.quantizers import TurboQuantQuantizer, ScalarQuantizer, ProductQuantizer
from quantvec.benchmarks.datasets import DATASET_REGISTRY
from quantvec.benchmarks.metrics import run_benchmark
from quantvec.viz.plots import generate_benchmark_report
from quantvec.adapters import (
    generate_langchain_adapter,
    generate_llamaindex_adapter,
    generate_haystack_adapter,
)
from quantvec.utils.io import format_bytes, estimate_memory

console = Console()

QUANTIZER_MAP = {
    "turboquant": TurboQuantQuantizer,
    "scalar": ScalarQuantizer,
    "product": ProductQuantizer,
}

ADAPTER_MAP = {
    "langchain": generate_langchain_adapter,
    "llamaindex": generate_llamaindex_adapter,
    "haystack": generate_haystack_adapter,
}


@click.group()
@click.version_option(version=__version__, prog_name="quantvec")
def main() -> None:
    """QuantVec-CLI: Vector quantization, compression analysis, and RAG adapters."""
    pass


@main.command()
@click.option(
    "--quantizer",
    "-q",
    type=click.Choice(list(QUANTIZER_MAP.keys())),
    default="turboquant",
    help="Quantizer type to use.",
)
@click.option(
    "--dim",
    "-d",
    type=int,
    default=768,
    help="Vector dimensionality.",
)
@click.option(
    "--bit-width",
    "-b",
    type=int,
    default=4,
    help="Bits per quantized coordinate (2, 4, or 8).",
)
@click.option(
    "--dataset",
    "-ds",
    type=click.Choice(list(DATASET_REGISTRY.keys())),
    default="random",
    help="Benchmark dataset type.",
)
@click.option(
    "--n-train",
    type=int,
    default=10_000,
    help="Number of training vectors.",
)
@click.option(
    "--n-test",
    type=int,
    default=1_000,
    help="Number of test vectors.",
)
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(),
    default="./quantvec_results",
    help="Output directory for results.",
)
@click.option(
    "--visualize",
    "-v",
    is_flag=True,
    help="Generate visualization plots.",
)
def benchmark(
    quantizer: str,
    dim: int,
    bit_width: int,
    dataset: str,
    n_train: int,
    n_test: int,
    output_dir: str,
    visualize: bool,
) -> None:
    """Run benchmark on a quantizer."""
    console.print(
        Panel.fit(
            f"[bold cyan]QuantVec Benchmark[/bold cyan]\n"
            f"Quantizer: [green]{quantizer}[/green] | "
            f"Dim: [green]{dim}[/green] | "
            f"Bit-width: [green]{bit_width}[/green] | "
            f"Dataset: [green]{dataset}[/green]",
            title="Configuration",
        )
    )

    # Generate dataset
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Generating dataset...", total=None)
        dataset_fn = DATASET_REGISTRY[dataset]
        train_vectors, test_vectors = dataset_fn(
            n_train=n_train, n_test=n_test, dim=dim
        )
        progress.update(task, completed=True)

    console.print(f"Train vectors: [green]{train_vectors.shape}[/green]")
    console.print(f"Test vectors: [green]{test_vectors.shape}[/green]")
    console.print(
        f"Original memory: [yellow]{format_bytes(train_vectors.nbytes)}[/yellow]"
    )

    # Fit and benchmark
    quantizer_cls = QUANTIZER_MAP[quantizer]
    q = quantizer_cls(dim=dim, bit_width=bit_width)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Fitting quantizer...", total=None)
        q.fit(train_vectors)
        progress.update(task, completed=True)

        task = progress.add_task("Running benchmark...", total=None)
        results = run_benchmark(q, train_vectors, test_vectors)
        progress.update(task, completed=True)

    # Display results
    table = Table(title="Benchmark Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Compression Ratio", f"{results['compression_ratio']:.2f}x")
    table.add_row("Original Size", f"{results['original_size_mb']:.2f} MB")
    table.add_row("Compressed Size", f"{results['compressed_size_mb']:.4f} MB")

    for key in sorted(results.keys()):
        if key.startswith("recall@"):
            table.add_row(key.upper(), f"{results[key]:.4f}")

    if "mrr" in results:
        table.add_row("MRR", f"{results['mrr']:.4f}")

    console.print(table)

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    if visualize:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Generating visualizations...", total=None)
            plot_paths = generate_benchmark_report(
                {quantizer: results}, output_dir, dataset
            )
            progress.update(task, completed=True)

        console.print("[green]Visualizations saved:[/green]")
        for path in plot_paths:
            console.print(f"  - {path}")

    console.print(f"\n[bold green]Benchmark complete![/bold green]")


@main.command()
@click.option(
    "--quantizer",
    "-q",
    type=click.Choice(list(QUANTIZER_MAP.keys())),
    default="turboquant",
    help="Quantizer type.",
)
@click.option(
    "--dim",
    "-d",
    type=int,
    default=768,
    help="Vector dimensionality.",
)
@click.option(
    "--bit-width",
    "-b",
    type=int,
    default=4,
    help="Bits per coordinate.",
)
@click.option(
    "--framework",
    "-f",
    type=click.Choice(list(ADAPTER_MAP.keys())),
    required=True,
    help="Target RAG framework.",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    help="Output file path (default: stdout).",
)
def adapter(
    quantizer: str,
    dim: int,
    bit_width: int,
    framework: str,
    output: Optional[str],
) -> None:
    """Generate RAG framework adapter code."""
    generator = ADAPTER_MAP[framework]
    code = generator(quantizer, bit_width, dim)

    if output:
        with open(output, "w") as f:
            f.write(code)
        console.print(f"[green]Adapter saved to:[/green] {output}")
    else:
        syntax = Syntax(code, "python", theme="monokai", line_numbers=True)
        console.print(Panel(syntax, title=f"{framework.title()} Adapter"))


@main.command()
@click.option(
    "--dim",
    "-d",
    type=int,
    default=768,
    help="Vector dimensionality.",
)
@click.option(
    "--n-vectors",
    "-n",
    type=int,
    default=1_000_000,
    help="Number of vectors.",
)
@click.option(
    "--dtype",
    type=click.Choice(["float32", "float16", "int8"]),
    default="float32",
    help="Data type.",
)
def estimate(dim: int, n_vectors: int, dtype: str) -> None:
    """Estimate memory usage for a vector dataset."""
    memory = estimate_memory(n_vectors, dim, dtype)
    console.print(
        Panel.fit(
            f"[bold]{n_vectors:,}[/bold] vectors @ [bold]{dim}D[/bold] ({dtype})\n"
            f"Estimated memory: [green]{memory}[/green]",
            title="Memory Estimate",
        )
    )


@main.command()
def list_quantizers() -> None:
    """List available quantizers."""
    table = Table(title="Available Quantizers")
    table.add_column("Name", style="cyan")
    table.add_column("Description", style="green")
    table.add_column("Training Required", style="yellow")

    table.add_row(
        "turboquant",
        "Data-oblivious quantizer with random rotation + Lloyd-Max",
        "No (online)",
    )
    table.add_row(
        "scalar",
        "Uniform scalar quantization (min/max range)",
        "Yes (fit on range)",
    )
    table.add_row(
        "product",
        "Product quantization with k-means codebooks",
        "Yes (k-means training)",
    )

    console.print(table)


@main.command(name="compare")
@click.option(
    "--dim",
    "-d",
    type=int,
    default=768,
    help="Vector dimensionality.",
)
@click.option(
    "--dataset",
    "-ds",
    type=click.Choice(list(DATASET_REGISTRY.keys())),
    default="random",
    help="Benchmark dataset type.",
)
@click.option(
    "--n-train",
    type=int,
    default=5_000,
    help="Number of training vectors.",
)
@click.option(
    "--n-test",
    type=int,
    default=500,
    help="Number of test vectors.",
)
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(),
    default="./quantvec_compare",
    help="Output directory for comparison results.",
)
def compare_quantizers(
    dim: int,
    dataset: str,
    n_train: int,
    n_test: int,
    output_dir: str,
) -> None:
    """Compare all quantizers side-by-side."""
    console.print(
        Panel.fit(
            f"[bold cyan]Quantizer Comparison[/bold cyan]\n"
            f"Dim: [green]{dim}[/green] | Dataset: [green]{dataset}[/green]",
            title="Configuration",
        )
    )

    # Generate dataset
    dataset_fn = DATASET_REGISTRY[dataset]
    train_vectors, test_vectors = dataset_fn(
        n_train=n_train, n_test=n_test, dim=dim
    )

    all_results = {}

    for name, quantizer_cls in QUANTIZER_MAP.items():
        console.print(f"\n[bold]Testing {name}...[/bold]")

        for bit_width in [2, 4, 8]:
            try:
                q = quantizer_cls(dim=dim, bit_width=bit_width)
                q.fit(train_vectors)
                results = run_benchmark(q, train_vectors, test_vectors)
                key = f"{name}-{bit_width}bit"
                all_results[key] = results

                console.print(
                    f"  {name}@{bit_width}bit: "
                    f"CR={results['compression_ratio']:.1f}x, "
                    f"R@1={results.get('recall@1', 0):.3f}, "
                    f"R@10={results.get('recall@10', 0):.3f}"
                )
            except Exception as e:
                console.print(f"  [red]Error with {name}@{bit_width}bit: {e}[/red]")

    # Summary table
    table = Table(title="Quantizer Comparison Summary")
    table.add_column("Quantizer", style="cyan")
    table.add_column("CR", style="green", justify="right")
    table.add_column("R@1", style="yellow", justify="right")
    table.add_column("R@10", style="yellow", justify="right")
    table.add_column("Size", style="blue", justify="right")

    for name, results in sorted(all_results.items()):
        table.add_row(
            name,
            f"{results['compression_ratio']:.1f}x",
            f"{results.get('recall@1', 0):.3f}",
            f"{results.get('recall@10', 0):.3f}",
            f"{results['compressed_size_mb']:.3f}MB",
        )

    console.print(table)

    # Generate comparison plots
    os.makedirs(output_dir, exist_ok=True)
    plot_paths = generate_benchmark_report(all_results, output_dir, dataset)
    console.print(f"\n[green]Comparison plots saved to:[/green] {output_dir}")
    for path in plot_paths:
        console.print(f"  - {path}")


if __name__ == "__main__":
    main()
