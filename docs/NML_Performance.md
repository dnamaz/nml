# NML Performance Benchmarks

## Overview

NML v0.7.0 runtime: 2,401 lines of C99, compiles to a 67 KB portable binary (83 KB with BLAS). Supports 71 opcodes across 32 tensor registers.

## Self-Training Performance

NML is the first machine language that can train its own neural networks. The TNET fused opcode runs the full forward-backward-update loop in a tight C loop, bypassing VM instruction dispatch.

### Small Network: 1→4→1 ReLU (y=2x+1)

Task: Train a 4-neuron hidden layer to learn y=2x+1, 2,000 epochs, lr=0.001

| Method | Result | Loss | Time | Speedup |
|--------|--------|------|------|---------|
| Python/NumPy | 7.0000 | 0.000000 | 20.5 ms | 1x (baseline) |
| NML interpreted (self_train.nml) | 6.8878 | 0.0126 | 188.0 ms | 0.1x |
| NML TNET (portable) | 7.0000 | 0.000000 | 1.9 ms | 10.8x |
| NML TNET (BLAS) | 7.0000 | 0.000000 | 1.9 ms | 10.8x |

At 4 neurons, BLAS provides no benefit (matrices too small for BLAS overhead to pay off).

### Medium Network: 1→256→1 ReLU (bracket approximation)

Task: Train a 256-neuron hidden layer, 5,000 epochs, lr=0.0000001

| Method | Time | vs Python |
|--------|------|-----------|
| Python/NumPy | 133.1 ms | 1x |
| NML TNET (portable, naive C) | 27.4 ms | 4.9x faster |
| NML TNET (BLAS/Accelerate) | 13.0 ms | 10.2x faster |

At 256 neurons, BLAS (Apple Accelerate) provides 2.1x speedup over naive C.

### Why NML Is Faster Than Python

Python/NumPy uses BLAS internally (Apple Accelerate on macOS), so the raw matrix math is the same speed. The difference is overhead:

- Python: 8 NumPy C function calls per epoch + Python loop overhead + object allocation
- NML TNET: Zero dispatch overhead — the entire training loop runs in a single C function call

For 2,000 epochs: Python makes ~16,000 C function calls with Python object creation/destruction between each. NML TNET makes 1 function call containing 2,000 iterations of pure C loops.

### Why Interpreted NML Is Slower

NML's interpreted path (self_train.nml with BKWD/WUPD/LOSS) spends ~80% of time on VM instruction dispatch — the switch(opcode) evaluation for each of 24,000 instructions. The actual tensor math is fast. TNET eliminates this overhead.

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

| Build | Size | Dependencies | Training Speed |
|-------|------|-------------|---------------|
| Portable (make) | 67 KB | libc, libm | 27.4 ms (256-neuron) |
| BLAS (make nml-fast) | 83 KB | + Accelerate/OpenBLAS | 13.0 ms (256-neuron) |
| Python + NumPy | 1.9 GB + 32 MB | Full Python runtime | 133.1 ms (256-neuron) |

NML's portable binary is 28,000x smaller than Python for 10x better training performance.

## Platform Support

| Platform | Build Command | Notes |
|----------|--------------|-------|
| macOS (ARM64) | make nml-fast | Uses Apple Accelerate |
| macOS (portable) | make | Naive C, no dependencies |
| Linux (x86_64) | make nml-fast | Uses OpenBLAS |
| Linux (portable) | make | Naive C |
| Raspberry Pi | arm-gcc runtime/nml.c -lm | Cross-compile, portable |
| ESP32 / MCU | xtensa-gcc runtime/nml.c -lm | 67 KB fits in 4 MB flash |

## Neural Network Scaling

Bracket approximation accuracy vs neuron count (256-neuron, 5,000-20,000 epochs):

| Neurons | Epochs | MAE | Max Error | Training Time | Data File |
|---------|--------|-----|-----------|--------------|-----------|
| 32 | 5,000 | $45 | $473 | ~5 sec (Python) | 4 KB |
| 128 | 5,000 | $18 | $326 | ~30 sec | 8 KB |
| 256 | 5,000 | $139 | $698 | ~98 sec | 16 KB |
| 256 | 10,000 | $10.82 | $126 | ~202 sec | 16 KB |
| 256 | 20,000 | $7.29 | $81 | ~400 sec | 16 KB |

Note: Training time is for the Python bracket_embedding_trainer.py (weight generation). Once weights are exported, NML inference is 18 cycles / ~190 µs regardless of neuron count.

## Methodology

All benchmarks run on Apple M1 Pro, 32 GB RAM, macOS. Times are wall-clock via the NML runtime's built-in cycle/timing instrumentation. Python times via time.perf_counter(). Each measurement is a single run (not averaged) — NML execution is deterministic so repeated runs produce identical results.
