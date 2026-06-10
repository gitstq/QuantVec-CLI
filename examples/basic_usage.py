"""Basic usage example for QuantVec-CLI."""

import numpy as np

from quantvec.quantizers import TurboQuantQuantizer, ScalarQuantizer, ProductQuantizer
from quantvec.benchmarks.datasets import generate_random_dataset
from quantvec.benchmarks.metrics import run_benchmark


def main():
    print("=" * 60)
    print("QuantVec-CLI Basic Usage Example")
    print("=" * 60)

    # Generate sample data
    print("\n1. Generating sample vectors...")
    train, test = generate_random_dataset(n_train=1000, n_test=100, dim=128)
    print(f"   Train: {train.shape}, Test: {test.shape}")

    # TurboQuant
    print("\n2. TurboQuant (4-bit)...")
    tq = TurboQuantQuantizer(dim=128, bit_width=4)
    tq.fit(train)
    codes = tq.encode(train)
    reconstructed = tq.decode(codes)
    print(f"   Compression ratio: {tq.compression_ratio():.2f}x")
    print(f"   Reconstruction MSE: {np.mean((train - reconstructed) ** 2):.6f}")

    # Scalar
    print("\n3. Scalar Quantization (4-bit)...")
    sq = ScalarQuantizer(dim=128, bit_width=4)
    sq.fit(train)
    codes = sq.encode(train)
    reconstructed = sq.decode(codes)
    print(f"   Compression ratio: {sq.compression_ratio():.2f}x")
    print(f"   Reconstruction MSE: {np.mean((train - reconstructed) ** 2):.6f}")

    # Product Quantization
    print("\n4. Product Quantization (8-bit, 16 sub-vectors)...")
    pq = ProductQuantizer(dim=128, bit_width=8, n_subvectors=16)
    pq.fit(train)
    codes = pq.encode(train)
    reconstructed = pq.decode(codes)
    print(f"   Compression ratio: {pq.compression_ratio():.2f}x")
    print(f"   Reconstruction MSE: {np.mean((train - reconstructed) ** 2):.6f}")

    # Benchmark
    print("\n5. Running benchmark...")
    results = run_benchmark(tq, train, test, k_values=[1, 5, 10])
    print(f"   Recall@1:  {results['recall@1']:.4f}")
    print(f"   Recall@5:  {results['recall@5']:.4f}")
    print(f"   Recall@10: {results['recall@10']:.4f}")

    print("\n" + "=" * 60)
    print("Example complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
