# NML Performance Benchmarks

## Overview

NML v0.7.0 runtime: ~2,400 lines of C99, compiles to a 67 KB portable binary (83 KB with BLAS). Supports 85 opcodes across 32 tensor registers.

All benchmarks: Apple M1 Pro, 32 GB RAM, macOS. 20 runs (median) unless noted. Both NML and Python use identical initial weights, data, hyperparameters, and mini-batch size.

## Self-Training: NML vs Python

NML is the first machine language that can train its own neural networks. The TNET fused opcode runs the full forward-backward-update loop in a tight C loop, bypassing VM instruction dispatch.

### Small Network: 1→4→1 ReLU (y = 2x + 1)

Task: Learn y = 2x + 1 from a single point (x=3, y=7). 2,000 epochs, lr=0.001. 20 runs, median time.

| Method | Result | Loss | Time | Speedup |
|--------|--------|------|------|---------|
| Python/NumPy SGD | 7.0000 | 0.000000 | 46.9 ms | baseline |
| Python/NumPy Adam | 7.0000 | 0.000000 | 97.4 ms | baseline |
| NML TNET SGD | 7.0000 | 0.000000 | 0.28 ms | **166x faster** |
| NML TNET Adam | 7.0000 | 0.000000 | 0.54 ms | **182x faster** |
| NML interpreted (BKWD/WUPD/LOSS) | 6.8878 | 0.0126 | 228.0 ms | 0.2x |

Both NML TNET and Python produce identical results (7.0000 exact). NML TNET is 166-182x faster.

The interpreted NML path (self_train.nml using BKWD/WUPD/LOSS in a LOOP) is slower than Python because it pays VM dispatch overhead per instruction. TNET eliminates this.

### Medium Network: 1→256→1 ReLU (FIT bracket approximation)

Task: Train a 256-neuron network on 396 real bracket data points (generated from the FIT cascade program). Adam optimizer, mini-batch=64, lr=0.001. 3 runs, median time.

| Method | 1,000 epochs | 5,000 epochs |
|--------|-------------|-------------|
| Python/NumPy Adam | 1.149 s | 5.929 s |
| NML TNET (portable) | 0.853 s (1.3x) | 4.303 s (1.4x) |
| NML TNET (BLAS/Accelerate) | 0.649 s (1.8x) | 3.520 s (1.7x) |

At 256 neurons, the matrix multiply cost dominates. Python/NumPy calls BLAS internally (Apple Accelerate on macOS), so the raw math speed is similar. NML's advantage narrows to 1.3-1.8x because both are ultimately calling the same BLAS routines. The small-network speedup (166x) comes from eliminating Python interpreter and NumPy object overhead, which is negligible at larger matrix sizes.

### Why The Speedup Varies By Network Size

| Factor | Small (4 neurons) | Medium (256 neurons) |
|--------|-------------------|---------------------|
| Matrix math time | ~1 µs/epoch | ~150 µs/epoch |
| Python overhead | ~23 µs/epoch | ~23 µs/epoch |
| NML TNET overhead | ~0.14 µs/epoch | ~0.5 µs/epoch |
| **Overhead / total** | **96% (Python)** | **13% (Python)** |

At 4 neurons, Python spends 96% of its time on interpreter overhead (function dispatch, object allocation, GIL). At 256 neurons, the actual BLAS math dominates and overhead drops to ~13%. NML TNET has near-zero overhead at any scale.

### Scale Test: Heap + Mini-Batch

TNET allocates working buffers on the heap (not stack) and processes data in mini-batches of 64 samples. No size limit.

| Samples | Neurons | Epochs | Time | Heap Used |
|---------|---------|--------|------|-----------|
| 5 | 4 | 2,000 | 0.28 ms | 0.7 KB |
| 396 | 256 | 5,000 | 4.3 s | 397 KB |
| 10,000 | 256 | 10 | 4.2 ms | 397 KB |
| 100,000 | 4 | 10 | 40.7 ms | 7.2 KB |

Previous versions used fixed stack arrays (5.5 MB, max N×H = 65,536) and segfaulted on large datasets.

## Inference Benchmarks

### Program Comparison

All programs compute the same progressive rate for input=100,000:

| Approach | Instructions | Cycles | Time | Result | Accuracy |
|----------|-------------|--------|------|--------|----------|
| Cascade (CMPF/JMPF) | 66 | 24 | 85 µs | 13,614.00 | Exact |
| Tensor table (GATH loop) | 31 | 61 | 200 µs | 13,614.00 | Exact |
| Neural 32 neurons | 12 | 12 | 194 µs | 13,241.49 | ~$373 off |
| Neural 256 neurons | 12 | 18 | 190 µs | 13,609.58 | ~$4 off |

### Other Programs

| Program | Instructions | Cycles | Time |
|---------|-------------|--------|------|
| Hello World (SYS) | 29 | 29 | ~50 µs |
| Fibonacci 20 numbers | 13 | 165 | ~89 µs |
| FizzBuzz 1-30 | 29 | 418 | ~120 µs |
| Primes 2-50 | 23 | 1,276 | ~200 µs |
| Anomaly detector (3-layer NN) | 18 | 18 | ~34 µs |

## Binary Size

| Build | Size | Dependencies |
|-------|------|-------------|
| Portable (make) | 67 KB | libc, libm |
| BLAS (make nml-fast) | 83 KB | + Accelerate/OpenBLAS |
| Python + NumPy | ~50 MB | Full Python runtime |

NML's portable binary is 750x smaller than Python's runtime for equivalent or better training performance.

## Platform Support

| Platform | Build Command | Notes |
|----------|--------------|-------|
| macOS (ARM64) | make nml-fast | Uses Apple Accelerate |
| macOS (portable) | make | Naive C, no dependencies |
| Linux (x86_64) | make nml-fast | Uses OpenBLAS |
| Linux (portable) | make | Naive C |
| Raspberry Pi | arm-gcc runtime/nml.c -lm | Cross-compile, portable |
| ESP32 / MCU | xtensa-gcc runtime/nml.c -lm | 67 KB fits in 4 MB flash |

## Methodology

All benchmarks run on Apple M1 Pro, 32 GB RAM, macOS. Small network: 20 runs, median time. Medium network: 3 runs, median time. NML times from the runtime's built-in cycle/timing instrumentation. Python times via `time.perf_counter()`. Both use identical initial weights, training data, learning rate, and mini-batch size to ensure exact 1-to-1 comparison.
