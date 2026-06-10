#!/bin/bash
# Run comprehensive benchmarks for all quantizers

set -e

echo "========================================"
echo "QuantVec-CLI Comprehensive Benchmark"
echo "========================================"

# Create output directory
mkdir -p benchmark_results

# Compare all quantizers
echo ""
echo "Running quantizer comparison..."
quantvec compare \
    --dim 768 \
    --dataset random \
    --n-train 5000 \
    --n-test 500 \
    --output-dir benchmark_results/comparison

# Individual benchmarks with different bit widths
echo ""
echo "Running TurboQuant benchmarks..."
for bw in 2 4 8; do
    echo "  Bit-width: $bw"
    quantvec benchmark \
        --quantizer turboquant \
        --dim 768 \
        --bit-width $bw \
        --dataset openai \
        --n-train 10000 \
        --n-test 1000 \
        --output-dir benchmark_results/turboquant_${bw}bit \
        --visualize
done

echo ""
echo "Benchmarks complete! Results saved to benchmark_results/"
