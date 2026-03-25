# NML — Neural Machine Language

A minimal, deterministic machine language designed for AI workloads. 89 opcodes across 8 extensions. ~5,000 lines of C. Zero ambiguity. NML compiles to an 83 KB binary with no dependencies beyond libc and libm, supports neural network inference and training, 4D vision ops with NCHW layout, GPU acceleration (CUDA, SYCL, Metal, Hailo), ONNX model import, a clean C library API, and a Python binding — making it deployable on everything from microcontrollers to GPU servers.

NML supports neural network inference, decision tree execution, transformer attention, signal processing, and general-purpose computation within a single instruction set and runtime — compiled to an 83 KB binary.

## Why NML

| Property | NML | Python/PyTorch |
|----------|-----|---------------|
| Vocabulary | 89 opcodes | 50,000+ tokens |
| Grammar rules | ~10 | 100+ |
| Syntactic ambiguity | Zero | High |
| Ways to express same op | 1 | 5-10+ |
| Runtime size (stripped) | 83 KB | 30+ MB |
| Avg tokens per program | 20-50 | 100-500 |

NML programs are deterministic, verifiable, and machine-readable. Every program has exactly one valid interpretation.

## New in This Release (Phases 1–5)

| Phase | What Was Added |
|-------|----------------|
| **Phase 1** | NumPy-style broadcasting for MADD/MSUB/EMUL/EDIV; 4D NCHW layout for CONV/POOL/UPSC/PADZ; cross-entropy loss (LOSS #2); configurable GPU thresholds via env vars |
| **Phase 2** | CUDA + SYCL kernels for CONV (im2col + cuBLAS), ATTN, GELU, SOFTMAX, LAYERNORM; elementwise kernel fusion (CONV/MMUL + RELU/SIGM/TANH/GELU in one dispatch) |
| **Phase 3** | N-layer TNET (`TNET R_config #epochs`) for arbitrary-depth MLPs (1–8 layers); BN opcode (batch normalization); DROP opcode (inverted dropout); WDECAY fully implemented; total 87 → 89 opcodes |
| **Phase 4** | `transpilers/nml_from_onnx.py`: converts ONNX models to NML programs + `.nml.data` weight files (28 ONNX op types); `nml_to_hailo.py` updated for CONV4D |
| **Phase 5** | `runtime/nml_api.h`: clean C library API (10 functions); shared library builds (`libnml.so`, `libnml.dll`, `libnml.a`); `python/nml.py`: ctypes VM class + subprocess fallback |

## What Makes NML Different

### Tensor Registers, Not Scalar Registers

Traditional ISAs (x86, ARM, RISC-V) operate on scalar values — a matrix multiply requires hundreds of instructions in a loop nest. Graph-based runtimes (ONNX, TVM) operate on abstract nodes. NML's 32 registers each hold an entire tensor. A matrix multiply is one instruction: `MMUL R3 R0 R1`. A full neural network layer — load weights, multiply, add bias, activate, store — is 8 instructions. An anomaly detector is 18.

### One ISA for All Model Types

ONNX handles neural networks. XGBoost has its own runtime. Decision trees need a separate executor. Gradient boosted ensembles need yet another. NML handles all of them in a single instruction set:

- **Neural networks**: `MMUL`, `MADD`, `RELU`, `SIGM`, `SOFT`, `ATTN`, `NORM`, `GELU`
- **Decision trees**: `CMPF` (feature compare), `LEAF` (leaf value), `TACC` (accumulate)
- **Training**: `BKWD` (backprop), `WUPD` (weight update), `LOSS`, `TNET` (fused 2-layer training)
- **Backward ops**: `RELUBK`, `SIGMBK`, `TANHBK`, `GELUBK`, `SOFTBK`, `MMULBK`, `CONVBK`, `POOLBK`, `NORMBK`, `ATTNBK` (arbitrary-architecture training)
- **Deep training**: `TNDEEP` (fused N-layer dense training with Adam/SGD)
- **Signal processing**: `FFT`, `FILT`
- **Vision**: `CONV`, `POOL`, `UPSC`, `PADZ`
- **General computation**: `LOOP`, `CALL`/`RET`, `SYS`, `MOD`

All model types share the same registers, memory model, and control flow. A program can mix a neural network forward pass with a decision tree in the same instruction sequence.

### Programs as Communication Protocol

The M2M extensions turn NML from an execution runtime into a machine-to-machine communication protocol. Programs carry their own metadata (`META`), can be composed from named fragments (`FRAG`/`LINK`), signed cryptographically (`SIGN`/`VRFY`), and patched differentially (`PTCH` — send 5 lines instead of retransmitting 200). No other ISA has trust, composition, and versioning built into the instruction set itself.

### Consensus as an Opcode

`VOTE` is an instruction, not an external orchestration layer. Multi-agent agreement — median, mean, quorum, min, max — is part of the program itself. When 5 models produce results and you need the trustworthy answer, it's one instruction: `VOTE R1 R0 #0`.

### Tri-Syntax for Three Audiences

Classic (`MMUL`) for machines and compilers. Symbolic (`×`) for token-efficient LLM communication. Verbose (`MATRIX_MULTIPLY`) for human auditors and regulators. All three compile to identical bytecode and can be mixed freely in the same program. No other ISA offers interchangeable syntax tiers.

### Self-Training as an Opcode

Most ISAs separate inference from training — you train in Python, export weights, run inference elsewhere. NML collapses this: `TNET #2000 #0.01` trains a neural network for 2,000 epochs at learning rate 0.01, using the current register state as the network. A program can arrive at a new machine, train on local data, and immediately run inference — no external framework, no Python, no GPU driver. The entire training loop compiles to the same 83 KB binary.

### Designed for LLM Generation

NML's zero-ambiguity grammar exists specifically so that language models can learn to generate correct programs. With 85 opcodes, ~10 grammar rules, and exactly one way to express each operation, a 7B parameter model achieves 100% grammar validity with constrained decoding (Outlines CFG) and 95% with temperature-varied retries after training on 228K pairs. With optional Outlines CFG constrained decoding, every generated token is guaranteed to be valid NML — 100% syntactic correctness by construction.

## Use Cases

### LLM-to-LLM Communication

When AI models need to exchange executable computation — not just text — NML provides a shared, unambiguous language. Model A generates an NML program, Model B verifies the signature (`SIGN`/`VRFY`) and executes it. No parsing ambiguity, no version conflicts, no prompt injection. The M2M extensions (`META`, `FRAG`, `LINK`, `PTCH`) enable composable, self-describing programs that models can inspect, modify, and chain together.

### Edge Inference

The entire NML runtime compiles to an 83 KB binary with zero dependencies beyond libc and libm. This means neural network inference, decision tree execution, and signal processing can run on microcontrollers, embedded systems, and IoT devices — anywhere Python and ONNX can't go. A sensor anomaly detector is 18 NML instructions and runs in 34 microseconds.

### Auditable Computation

Compliance formulas, regulatory rules, rate schedules, and financial models can be expressed as NML programs that are deterministic, traceable, and human-readable. The verbose syntax mode (`SET_VALUE`, `ACCUMULATE`, `COMPARE_FEATURE`) makes programs self-documenting for auditors. Every computation step is explicit — there is no hidden state, no implicit type coercion, no undefined behavior.

### Multi-Agent Consensus

Multiple AI models independently generate NML programs for the same task. Each program runs through the C runtime, producing a result. The `VOTE` instruction finds consensus across results — median, mean, quorum, min, or max. If 4 out of 5 models agree, the answer is trustworthy. This eliminates outlier hallucinations without requiring human review.

### Deterministic AI Pipelines

Replace opaque model inference with verifiable instruction sequences. If two systems run the same NML program on the same data, they produce identical results — bit-for-bit reproducible across platforms. This matters for regulated industries (finance, healthcare, insurance) where "the model said so" is not an acceptable audit trail, but "here is the 18-instruction program and its output" is.

### Self-Training at the Edge

An NML program can train its own neural network, then immediately run inference — on the same device, in the same runtime, with zero external dependencies. A sensor node receives a `TNET` program, trains on local calibration data, and starts classifying. An agent in a hive collective receives training data from the coordinator, runs gradient descent via `BKWD`/`WUPD`, and reports back trained weights. No Python. No GPU driver. No network round-trip to a training server.

### Training LLMs to Generate Structured Code

NML's zero-ambiguity grammar makes it dramatically easier to train models to write correct code. With only 85 opcodes, ~10 grammar rules, and exactly one way to express each operation, a 7B parameter model achieves 100% grammar accuracy (with Outlines CFG constrained decoding) or 95% (with 2 retries and temperature escalation) after training on 228K pairs. The additive-alias design philosophy — when models generate semantically valid alternative forms, accept them rather than fighting them — drove grammar pass rates from 85% to 100% without retraining.

## Quick Start

```bash
# Build the runtime
make

# Run Hello World
./nml programs/hello_world.nml

# Run anomaly detector (neural network)
./nml programs/anomaly_detector.nml programs/anomaly_weights.nml.data

# Run all tests
make test

# Run phase-specific tests
make test-phase1  # broadcasting, 4D vision, cross-entropy
make test-phase2  # GPU kernels + fusion (CPU validation path)
make test-phase3  # N-layer TNET, BN, DROP, WDECAY
make test-phase4  # ONNX import (requires: pip install onnx numpy)
make test-phase5  # C library + Python binding

# Build the shared library (for C/Python embedding)
make libnml.so    # Linux
make libnml.dll   # Windows
make libnml.a     # Static

# Import an ONNX model
python3 transpilers/nml_from_onnx.py model.onnx

# Build with crypto (SIGN/VRFY)
make nml-crypto

# Start the NML server (LLM chat + execution)
bash serve/start_agents.sh
```

## Tri-Syntax

Every NML instruction can be written in three interchangeable syntaxes — classic, symbolic, and verbose. All three compile to identical bytecode and can be mixed freely in the same program.

**Classic:**
```
LEAF  RC #176100.0
SCLR  RC RC #0.062
TACC  RA RA RC
ST    RA @result
HALT
```

**Symbolic:**
```
∎  γ  #176100.0
∗  γ  γ  #0.062
∑  α  α  γ
↑  α  @result
◼
```

**Verbose:**
```
SET_VALUE   SCRATCH #176100.0
SCALE       SCRATCH SCRATCH #0.062
ACCUMULATE  ACCUMULATOR ACCUMULATOR SCRATCH
STORE       ACCUMULATOR @result
STOP
```

## Instruction Set (89 Total)

| Extension | Count | Opcodes |
|-----------|-------|---------|
| Core | 35 | `MMUL` `MADD` `MSUB` `EMUL` `EDIV` `SDOT` `SCLR` `SDIV` `RELU` `SIGM` `TANH` `SOFT` `LD` `ST` `MOV` `ALLC` `RSHP` `TRNS` `SPLT` `MERG` `CMPF` `CMP` `CMPI` `JMPT` `JMPF` `JUMP` `LOOP` `ENDP` `CALL` `RET` `LEAF` `TACC` `SYNC` `HALT` `TRAP` |
| NML-V: Vision | 4 | `CONV` `POOL` `UPSC` `PADZ` |
| NML-T: Transformer | 4 | `ATTN` `NORM` `EMBD` `GELU` |
| NML-R: Reduction | 4 | `RDUC` `WHER` `CLMP` `CMPR` |
| NML-S: Signal | 2 | `FFT` `FILT` |
| NML-M2M | 12 | `META` `FRAG` `ENDF` `LINK` `PTCH` `SIGN` `VRFY` `VOTE` `PROJ` `DIST` `GATH` `SCAT` |
| NML-TR: Training | 21 | `BKWD` `WUPD` `LOSS` `TNET` `RELUBK` `SIGMBK` `TANHBK` `GELUBK` `SOFTBK` `MMULBK` `CONVBK` `POOLBK` `NORMBK` `ATTNBK` `TNDEEP` `TLOG` `TRAIN` `INFER` `WDECAY` `BN` `DROP` |
| NML-G: General | 5 | `SYS` `MOD` `ITOF` `FTOI` `BNOT` |

**Phase 3 additions:** `BN` (batch normalization, symbolic `⊞`, verbose `BATCH_NORM`) and `DROP` (inverted dropout, symbolic `≋`, verbose `DROPOUT`).

**4D vision ops:** CONV/POOL/UPSC/PADZ accept `[N,C,H,W]` input and `[C_out,C_in,KH,KW]` kernels (NCHW layout).

**Broadcasting:** MADD/MSUB/EMUL/EDIV follow NumPy right-align rules — shapes are right-aligned and dimensions broadcast when one of them is 1.

## Python Usage

```python
import sys
sys.path.insert(0, 'python')
import nml

# Library mode — requires libnml.so / libnml.dll
vm = nml.VM()
vm.load('programs/anomaly_detector.nml', 'programs/anomaly_weights.nml.data')
vm.set_input('sensor_data', [[0.9, 0.1, 0.95, 0.3]])
vm.run()
result = vm.get_output('anomaly_score')   # numpy array

# One-shot helper
result = nml.infer('programs/anomaly_detector.nml',
                   'programs/anomaly_weights.nml.data',
                   input_data=[[0.9, 0.1, 0.95, 0.3]])

# Subprocess fallback (no shared library needed)
stdout = nml.run_program('programs/anomaly_detector.nml',
                         'programs/anomaly_weights.nml.data')
```

Build the shared library first: `make libnml.so` (Linux) or `make libnml.dll` (Windows). See [docs/NML_API_Reference.md](docs/NML_API_Reference.md).

## C API Usage

```c
#include "runtime/nml_api.h"

nml_vm_t *vm = nml_vm_create();
nml_vm_load(vm, "model.nml", "model.nml.data");

float input[] = {0.9f, 0.1f, 0.95f, 0.3f};
int shape[]   = {1, 4};
nml_vm_set_input(vm, "sensor_data", input, shape, 2);

if (nml_vm_run(vm) != 0) {
    fprintf(stderr, "error: %s\n", nml_vm_last_error(vm));
}

NMLTensor out = nml_vm_get_output(vm, "anomaly_score");
printf("score: %.4f\n", out.data[0]);

nml_vm_destroy(vm);
```

Link with `-lnml` (shared) or `libnml.a` (static). Full reference: [docs/NML_API_Reference.md](docs/NML_API_Reference.md).

## ONNX Import

Convert any ONNX model to an NML program and weight data file:

```bash
# Install dependencies
pip install onnx numpy

# Convert (writes model.nml + model.nml.data)
python3 transpilers/nml_from_onnx.py model.onnx

# With explicit output paths
python3 transpilers/nml_from_onnx.py model.onnx \
    --output-nml out.nml --output-data out.nml.data

# Dry run — print program without writing files
python3 transpilers/nml_from_onnx.py model.onnx --dry-run

# Run the converted model
./nml out.nml out.nml.data
```

Handles 28 ONNX op types including Conv, Gemm, MatMul, BatchNormalization, Relu, Sigmoid, Tanh, MaxPool, AveragePool, Add, Mul, Reshape, Transpose, Flatten, Dropout, and more.
Register allocation: R0 = primary input, R1–R28 = intermediate activations, weight tensors in named memory slots.

## Example Programs

### Single Dense Layer (Neural Network)

```
LD    R0 @input
LD    R1 @weights
LD    R2 @bias
MMUL  R3 R0 R1
MADD  R3 R3 R2
RELU  R3 R3
ST    R3 @output
HALT
```

### Sum 1 to N (Loop)

```
LEAF  R0 #10
ALLC  R1 #[1]
LEAF  R2 #1
LOOP  R0
TACC  R1 R1 R2
ENDP
ST    R1 @result
HALT
```

### Subroutine (CALL/RET)

```
LEAF  R0 #7.0
CALL  #2
ST    R0 @result
HALT
SCLR  R0 R0 #2.0
RET
```

### Self-Training Network (TNET)

```
LD    R0 @training_inputs
LD    R9 @training_targets
LD    R1 @w1
LD    R2 @b1
LD    R3 @w2
LD    R4 @b2
TNET  #2000 #0.01
ST    RA @predictions
ST    R1 @trained_w1
ST    R3 @trained_w2
HALT
```

### Symbolic Anomaly Detector

```
↓  ι  @sensor_data
↓  κ  @w1
↓  λ  @b1
×  μ  ι  κ
⊕  μ  μ  λ
⌐  μ  μ
↓  ν  @w2
↓  ξ  @b2
×  ο  μ  ν
⊕  ο  ο  ξ
⌐  ο  ο
↓  π  @w3
↓  ρ  @b3
×  ς  ο  π
⊕  ς  ς  ρ
σ  ς  ς
↑  ς  @anomaly_score
◼
```

### Piecewise-Linear as ReLU Network

Any progressive rate schedule IS a ReLU network — each tier is `max(0, input - threshold) × rate_delta`. No training needed. Exact results.

```
LD    R0 @input_value
LD    R1 @deduction
MSUB  R2 R0 R1               ; adjusted = input - deduction
RELU  R2 R2                   ; clamp to 0
LD    R3 @thresholds          ; [0, 11925, 48475, ...]
LD    R5 @rate_deltas         ; [0.10, 0.02, 0.10, ...]
ALLC  RA #[1]                 ; result = 0
ALLC  RD #[1]                 ; tier = 0
LEAF  R9 #1.0
LOOP  #7
GATH  RC R3 RD                ; threshold = thresholds[tier]
MSUB  RC R2 RC                ; excess = adjusted - threshold
RELU  RC RC                   ; the "neuron": clip if not in tier
GATH  R8 R5 RD                ; rate_delta = rate_deltas[tier]
EMUL  RC RC R8                ; contribution = excess × rate_delta
TACC  RA RA RC                ; result += contribution
TACC  RD RD R9                ; tier++
ENDP
ST    RA @result
HALT
```

With data file (only 2 tensors — no pre-computed base amounts needed):

```
@input_value shape=1 data=100000.0
@deduction shape=1 data=15000.0
@thresholds shape=7 data=0,11925,48475,103350,197300,250525,609350
@rate_deltas shape=7 data=0.10,0.02,0.10,0.02,0.08,0.03,0.02
```

Result: 13,614.00 (exact). To update for a new rate schedule, change only the data file.

### Self-Contained Train + Infer

An NML program that trains its own neural network and immediately runs inference — no Python, no external tools:

```
; Load training data + initial weights
LD    R1 @w1
LD    R2 @b1
LD    R3 @w2
LD    R4 @b2
LD    R0 @training_inputs
LD    R9 @training_targets
; Train 5000 epochs (single instruction)
TNET  #5000 #0.001 #0
; Inference with trained weights (R1-R4 now updated)
LD    R0 @test_input
MMUL  R5 R0 R1
MADD  R5 R5 R2
RELU  R5 R5
MMUL  R6 R5 R3
MADD  RA R6 R4
ST    RA @prediction
HALT
```

The `.nml.data` file contains the training dataset, initial weights, and inference input. One binary, two files, complete ML system.

## Four Ways to Compute

NML supports multiple approaches to the same computation. These four programs all compute a progressive rate schedule:

### Approach 1: Cascade (66 instructions, 24 cycles)

Cascading threshold comparisons — the traditional approach. Each tier checks if the input exceeds a threshold and computes the marginal amount. Exact results.

```
CMPI  RE R7 #54875.0       ; input < 54875?
JMPF  #18                  ; no → check next tier
...
LEAF  RA #5578.50           ; base amount at this tier
LEAF  RC #54875.0           ; tier threshold
MSUB  R8 R7 RC              ; marginal = input - threshold
SCLR  R8 R8 #0.22           ; marginal * rate
TACC  RA RA R8              ; total = base + marginal
```

### Approach 2: Tensor Table Lookup (31 instructions, 61 cycles)

Rate table stored as three tensors (thresholds, rates, base amounts). GATH looks up the correct tier in a loop. Same exact result, but the program never changes — only the data tensors do.

```
LD    R3 @thresholds        ; [6400, 18325, 54875, ...]
LD    R4 @rates             ; [0.10, 0.12, 0.22, ...]
LD    R5 @base_amounts      ; [0, 1192.5, 5578.5, ...]
LOOP  #7
GATH  RC R3 RB              ; threshold[i]
CMP   RC R2                 ; input < threshold?
...
GATH  R8 R4 R7              ; rate for matching tier
GATH  R9 R5 R7              ; base for matching tier
MSUB  RB R2 RC              ; marginal = input - threshold
EMUL  RA RB R8              ; marginal * rate
TACC  RA RA R9              ; total = base + marginal
```

### Approach 3: Neural Network (12 instructions, 12 cycles)

A 32-neuron ReLU network trained to approximate the rate function. Smallest program, fewest cycles, but the result is approximate (within ~$373 of exact).

```
LD    R0 @input_value
LD    R1 @w1                ; 1×32 weights
LD    R2 @b1                ; 1×32 bias
MMUL  R3 R0 R1              ; forward pass
MADD  R3 R3 R2              ; add bias
RELU  R3 R3                 ; activate
LD    R4 @w2                ; 32×1 weights
LD    R5 @b2                ; 1×1 bias
MMUL  R6 R3 R4              ; output layer
MADD  RA R6 R5
ST    RA @result
HALT
```

### Comparison (input = 100,000)

| | Cascade | Tensor Table | Bracket-ReLU | Neural Network |
|-|---------|-------------|--------------|----------------|
| **Result** | 13,614.00 | 13,614.00 | 13,614.00 | ~13,241 |
| **Instructions** | 66 | 31 | 20 | 12 |
| **Data tensors** | 2 values | 5 tensors | 2 tensors | 5 tensors (weights) |
| **To update rates** | Rewrite program | Change 3 tensors | Change 2 tensors | Retrain network |
| **Accuracy** | Exact | Exact | Exact | ~$373 error |
| **Training needed** | No | No | No | Yes |

The Bracket-ReLU approach discovers that any progressive rate schedule IS a ReLU network — each tier is `max(0, input - threshold) × rate_delta`. It needs only 2 data tensors (thresholds + rate deltas) and no pre-computed base amounts.

Programs are in `programs/` with matching `.nml.data` files:

```bash
./nml programs/rate_cascade.nml programs/rate_cascade.nml.data
./nml programs/rate_tensor_table.nml programs/rate_tensor_table.nml.data
./nml programs/rate_neural.nml programs/rate_neural.nml.data
```

## Performance

### Self-Training: 166x Faster Than Python

NML can train its own neural networks using the TNET fused opcode. Same weights, same data, same hyperparameters — 1-to-1 comparison (20 runs, median time).

**Small network** — 1→4→1 ReLU, y=2x+1, 2,000 epochs, lr=0.001:

| Method | Result | Time | Speedup |
|--------|--------|------|---------|
| Python/NumPy SGD | 7.0000 (exact) | 46.9 ms | baseline |
| NML TNET SGD | 7.0000 (exact) | 0.28 ms | **166x faster** |
| Python/NumPy Adam | 7.0000 (exact) | 97.4 ms | baseline |
| NML TNET Adam | 7.0000 (exact) | 0.54 ms | **182x faster** |

**Medium network** — 1→256→1 ReLU, 396 piecewise-linear samples, Adam, mini-batch=64:

| Method | 1K epochs | 5K epochs |
|--------|-----------|-----------|
| Python/NumPy | 1.15 s | 5.93 s |
| NML TNET (portable) | 0.85 s (1.3x) | 4.30 s (1.4x) |
| NML TNET (BLAS) | 0.65 s (1.8x) | 3.52 s (1.7x) |

67 KB binary. Zero dependencies. 166x faster on small networks, 1.3-1.8x on large networks (where BLAS math dominates). Scales to 100K+ training samples with heap-allocated mini-batching.

### Inference Benchmarks

| Program | Instructions | Cycles | Time |
|---------|-------------|--------|------|
| Anomaly detector (3-layer NN) | 18 | 18 | 34 µs |
| Rate cascade (7 tiers) | 66 | 24 | 85 µs |
| Rate tensor table (GATH lookup) | 31 | 61 | 200 µs |
| Rate neural (32 neurons) | 12 | 12 | 194 µs |
| Fibonacci (20 numbers) | 13 | 165 | 89 µs |

## M2M: Signed Program Distribution

NML programs can be signed, verified, and distributed to fleets of agents. Build with `make nml-crypto` to enable HMAC-SHA256 signing.

```bash
# Sign a program
./nml-crypto --sign programs/fraud_detection.nml --key deadbeef01020304 --agent authority_v1 > signed.nml

# Agents verify before executing (tampered programs are rejected)
./nml-crypto signed.nml data.nml.data
# → [NML] Signature verified — signed by agent 'authority_v1'

# Patch a program (update threshold from 0.5 to 0.6)
echo 'PTCH @set 49 CMPI  RE RA #0.6' > update.ptch
echo 'PTCH @end' >> update.ptch
./nml-crypto programs/fraud_detection.nml --patch update.ptch > patched.nml

```

## Architecture

```
runtime/
  nml.c                  Single-file C99 runtime (~5,000 lines, 83 KB binary)
  nml_api.h              Public C library API (10 functions — Phase 5)
  nml_crypto.h           SHA-256 + HMAC-SHA256 for SIGN/VRFY (standalone, no deps)
  nmld.c                 NML daemon — pre-fork worker pool, binary cache
  nml_backend_cuda.cu    NVIDIA CUDA GPU backend (CONV im2col+cuBLAS, ATTN, GELU, ...)
  nml_backend_sycl.cpp   Intel SYCL GPU backend (CONV, ATTN, GELU, SOFTMAX, LAYERNORM, ...)
  nml_backend_metal.m    Apple Metal GPU backend
  nml_backend_hailo.cpp  Hailo NPU backend (HailoRT 4.x, file + embedded HEF)
  nml_backend_hailo.h    Hailo backend C interface

python/
  nml.py                 Python binding: ctypes VM class + subprocess fallback (Phase 5)

transpilers/
  nml_from_onnx.py   ONNX → NML program + .nml.data weights (28 op types — Phase 4)
  nml_to_hailo.py    NML → ONNX → HEF compiler (Hailo Dataflow Compiler; CONV4D support)
  nml_grammar.py     Formal grammar validator (all syntax variants)
  nml_executor.py    Program retriever + executor + tracer

serve/
  nml_server.py      HTTP server: LLM chat, NML execution, M2M distribution
  start_agents.sh    Starts NML server + optional RAG gateway + chat UI

terminal/
  nml_chat.jsx       LLM chat UI for NML code generation
  nml_terminal.jsx   Interactive NML emulator

programs/            Example NML programs (fraud detection, anomaly detector,
                     fibonacci, fizzbuzz, primes, rate calculations)

tests/               Opcode coverage tests, ONNX import tests, Python binding tests

docs/                Full specification, architecture documents, usage guide
```

## Registers

32 tensor registers with three naming conventions:

| Index | Classic | Greek | Purpose |
|-------|---------|-------|---------|
| 0-9 | R0-R9 | ι κ λ μ ν ξ ο π ρ ς | General purpose |
| 10 | RA | α | Accumulator |
| 11 | RB | β | General |
| 12 | RC | γ | Scratch |
| 13 | RD | δ | Counter |
| 14 | RE | φ | Condition flag |
| 15 | RF | ψ | Stack pointer |
| 16-22 | RG-RM | η θ ζ ω χ υ ε | Training / extended |
| 23-31 | RN-RV | — | Extended registers |

## Data Types

| Type | Size | Use Case |
|------|------|----------|
| f32 | 4 bytes | ML inference (default) |
| f64 | 8 bytes | Double-precision computation |
| i32 | 4 bytes | Counters, indices |

## Build Targets

| Target | Description |
|--------|-------------|
| `make nml` | Standard build (89 opcodes, portable C) |
| `make nml-fast` | BLAS acceleration (OpenBLAS on Linux, Accelerate on macOS) |
| `make nml-cuda` | NVIDIA CUDA + cuBLAS GPU backend |
| `make nml-sycl` | Intel SYCL GPU backend (requires `icpx` / Intel oneAPI) |
| `make nml-metal` | Apple Metal GPU (macOS only) |
| `make nml-mkl-cpu` | Intel oneMKL on CPU (AVX-512) |
| `make nml-aocl` | AMD AOCL/BLIS CPU BLAS (Zen-optimised) |
| `make nml-rpi` | Raspberry Pi 5 (Cortex-A76 + OpenBLAS) |
| `make nml-rpi-hailo` | Raspberry Pi + Hailo NPU (AI HAT / AI HAT+ 2) |
| `make nml-crypto` | HMAC-SHA256 + Ed25519 signing support |
| `make nml-wasm` | WebAssembly (requires `emcc`) |
| `make nmld` | Pre-fork worker pool daemon |
| `make libnml.so` | Shared library for C/Python embedding (Linux) |
| `make libnml.dll` | Shared library (Windows) |
| `make libnml.a` | Static library |
| `make release` | Build + strip |
| `make test` | All core tests |
| `make test-phase1` | Broadcasting, 4D vision, cross-entropy |
| `make test-phase2` | GPU kernels + kernel fusion (CPU path) |
| `make test-phase3` | N-layer TNET, BN, DROP, WDECAY |
| `make test-phase4` | ONNX import (requires `pip install onnx numpy`) |
| `make test-phase5` | C library + Python binding |

```bash
# Minimal build
gcc -O2 -o nml runtime/nml.c -lm

# Core only (35 opcodes, no extensions)
gcc -O2 -o nml-core runtime/nml.c -lm -DNML_NO_DEFAULT_EXTENSIONS

# Shared library
gcc -O2 -shared -fPIC -DNML_BUILD_LIB -o libnml.so runtime/nml.c -lm
```

## Running

```bash
./nml <program.nml> [data.nml.data] [--trace] [--max-cycles N] [--fragment NAME] [--describe]

# With crypto build:
./nml-crypto --sign <program.nml> --key <hex> [--agent <name>]   # Sign
./nml-crypto <program.nml> --patch <patch.ptch>                  # Patch
./nml-crypto <signed.nml> [data.nml.data]                        # Verify + Execute
```

- `--trace` prints each instruction as it executes
- `--max-cycles N` overrides the default 1M cycle limit
- `--fragment NAME` runs only the named fragment (for FRAG/LINK programs)
- `--describe` prints program metadata (META fields) without executing
- `--sign` signs a program with HMAC-SHA256 (requires `nml-crypto`)
- `--patch` applies differential patch directives (@set/@del/@ins)

## Data Files (.nml.data)

Memory contents are loaded from simple text files:

```
@sensor_data shape=1,4 data=0.9,0.1,0.95,0.3
@weights shape=4,3 data=0.2,-0.1,0.4,0.5,0.3,-0.2,...
@income shape=1 dtype=f64 data=185000.00
```

## Documentation

### Start Here

1. **[Getting Started](docs/GETTING_STARTED.md)** — 5-minute quick reference: registers, opcode cheat sheet, your first program, common patterns
2. **[NML Specification](docs/NML_SPEC.md)** — full instruction set reference with encoding format, error codes, and runtime limits

### Go Deeper

3. **[Usage Guide](docs/NML_Usage_Guide.md)** — all 89 instructions with detailed examples and edge cases
4. **[NML-G Specification](docs/NML_G_Spec.md)** — general-purpose extensions (SYS, MOD, ITOF, FTOI, BNOT)
5. **[NML-M2M Specification](docs/NML_M2M_Spec.md)** — machine-to-machine extensions (META, FRAG, SIGN, VOTE, PROJ, DIST)

### Architecture

6. **[Architecture Document](docs/NML_Architecture_Document.md)** — system design, philosophy, and the case for a machine-first language
7. **[Implementation Document](docs/NML_Implementation_Document.md)** — what was built, validation results, and performance benchmarks
8. **[Phases Roadmap](docs/NML_Phases_Roadmap.md)** — 5-phase capability expansion: broadcasting, GPU, training depth, ONNX import, C API

### API and Embedding

9. **[API Reference](docs/NML_API_Reference.md)** — C library API (nml_api.h) and Python binding (python/nml.py)

### Advanced

10. **[Multi-Agent Architecture](docs/NML_Multi_Agent_Architecture.md)** — distributed LLM communication using NML as a formal protocol; Part 8 covers 4D vision in agent pipelines
11. **[Backend Reference](docs/NML_Backends.md)** — all hardware acceleration backends: BLAS, CUDA, SYCL, Metal, Hailo NPU (build flags, thresholds, deployment modes)
12. **[Performance Guide](docs/NML_Performance.md)** — benchmarks, backend throughput comparison, platform support matrix

## Hardware Acceleration

NML has a layered dispatch chain for matrix operations. Each layer falls back to the next if unavailable:

```
SYCL GPU → Metal GPU → BLAS (CPU) → Scalar C
```

For Hailo NPU, the entire program is dispatched at the `main()` level — the NML interpreter never runs.

### Backend Summary

| Backend | Define | Platform | Accelerates |
|---------|--------|----------|-------------|
| Scalar C | _(default)_ | Any | All ops (baseline) |
| OpenBLAS | `NML_USE_OPENBLAS` | Linux / Raspberry Pi | MMUL, GEMV |
| Apple Accelerate | `NML_USE_ACCELERATE` | macOS / iOS | MMUL, GEMV (AMX) |
| Intel oneMKL CPU | `NML_USE_ONEMKL_CPU` | Intel Linux / Windows | MMUL, GEMV (AVX-512) |
| AMD AOCL | `NML_USE_AOCL` | AMD Linux | MMUL, GEMV (Zen-tuned) |
| Apple Metal | `NML_USE_METAL` | macOS (Apple Silicon) | MMUL + element-wise ops |
| NVIDIA CUDA | `NML_USE_CUDA` | NVIDIA GPU | CONV (im2col+cuBLAS), ATTN, GELU, SOFTMAX, LAYERNORM, MMUL |
| Intel SYCL | `NML_USE_SYCL` | Intel GPU (Arc, Iris Xe) | CONV, ATTN, GELU, SOFTMAX, LAYERNORM, MMUL, GEMV |
| Intel SYCL + oneMKL | `NML_USE_SYCL` + `NML_USE_ONEMKL` | Intel GPU | Peak Intel GPU perf |
| Hailo NPU | `NML_USE_HAILO` | Raspberry Pi + AI HAT/+ 2 | Whole-model inference (CONV4D supported) |

**GPU thresholds** (configurable via environment variables at runtime — no recompile needed):

| Variable | Default | Controls |
|----------|---------|----------|
| `NML_CUDA_MMUL_THRESHOLD` | 4096 | Min elements to use CUDA for MMUL |
| `NML_CUDA_EW_THRESHOLD` | 1024 | Min elements to use CUDA for elementwise ops |
| `NML_SYCL_MMUL_THRESHOLD` | 16384 | Min elements to use SYCL for MMUL |
| `NML_SYCL_EW_THRESHOLD` | 4096 | Min elements to use SYCL for elementwise ops |

**Kernel fusion (Phase 2):** A CONV or MMUL followed immediately by RELU, SIGM, TANH, or GELU on the same register executes in-place in a single GPU dispatch — saving one kernel launch per fused pair.

### Hailo NPU Deployment

NML supports the Raspberry Pi AI HAT and AI HAT+ 2 (Hailo-8L / Hailo-10H). Two deployment modes:

**Sidecar HEF** — `.hef` file lives alongside the `.nml` program. The runtime auto-selects the chip-specific file (e.g. `program.hailo8l.hef`) with a generic fallback. Use `$HAILO_ARCH` to pin the architecture.

```bash
# Build (requires HailoRT installed on Pi)
make nml-rpi-hailo

# Compile NML → HEF on dev machine (requires Hailo Dataflow Compiler)
python3 transpilers/nml_to_hailo.py program.nml weights.nml.data --arch hailo8l

# Run — Hailo NPU handles inference, NML interpreter never executes
./nml-rpi-hailo program.nml weights.nml.data
```

**Embedded HEF** — HEF compiled into the binary as a C byte array. No sidecar files at runtime. Fully self-contained deployment.

```bash
# One command: compile HEF + embed + link
make nml-rpi-hailo-embed PROGRAM=program.nml DATA=weights.nml.data ARCH=hailo8l

# Self-contained binary — no .hef file needed
./nml-rpi-hailo-embed program.nml weights.nml.data
```

See [docs/NML_Backends.md](docs/NML_Backends.md) for full backend reference, build flags, and threshold tuning.

## Limitations and Runtime Bounds

NML is intentionally constrained — these limits are design choices for determinism, small binary size, and edge deployability, not bugs.

### Tensor Limits

| Limit | Default | Override | Impact |
|-------|---------|----------|--------|
| Max elements per tensor | 16,777,216 (16M) | `-DNML_MAX_TENSOR_SIZE=N` | Memory-bounded by default. Embedded/RPi targets use `-DNML_MAX_TENSOR_SIZE=65536` (256×256). |
| Max dimensions | 4 | Not overridable | Supports up to 4D tensors (batch x channel x height x width). |
| Storage per tensor | ~512 KB | Scales with tensor size | Static allocation in union (f32/f64/i32 share space). 32 registers = ~16 MB register file. |

### Program Limits

| Limit | Default | Override |
|-------|---------|----------|
| Max instructions | 8,192 | `-DNML_MAX_INSTRUCTIONS=N` |
| Max memory slots | 64 | `-DNML_MAX_MEMORY_SLOTS=N` |
| Max loop nesting | 8 | `-DNML_MAX_LOOP_DEPTH=N` |
| Max call depth | 32 | `-DNML_MAX_CALL_DEPTH=N` |
| Max tokens per line | 8 | Not overridable |
| Max line length | 256 chars | Not overridable |
| Execution cycle limit | 1,000,000 | `--max-cycles N` |

### Training Limits

| Limit | Value | Notes |
|-------|-------|-------|
| TNET architecture | 2-layer ReLU only | Fixed: input→hidden→output. Use backward opcodes for other architectures. |
| TNET precision | f32 internally | Weights stored as register dtype, but forward/backward computed in float. |
| TNET mini-batch | 64 max | `B = min(N, 64)` — hardcoded in the fused loop. |
| TNDEEP max layers | 10 | Architecture descriptor in RV register. |
| TNDEEP max layer width | 4,096 | Backward gradient buffers are heap-allocated; runtime rejects `max_dim > 4096`. |
| TNDEEP optimizers | SGD (0) or Adam (1) | No learning rate schedules, no weight decay. |

### Precision

| Context | Dtype | Notes |
|---------|-------|-------|
| General computation | f32 or f64 | Per-tensor, set at creation. BLAS accelerates both. |
| TNET fused training | f32 only | All internal computation in float. |
| TNDEEP training | f64 activations, f32 Adam | Mixed — activation buffers are double, optimizer uses float constants. |
| Metal GPU | f32 only | MPS dispatch converts to float. BLAS fallback handles f64. |
| LEAF immediates | f64 | Parsed as double, stored in register's dtype. |

### GPU / Acceleration

| Constraint | Details |
|------------|---------|
| Metal GPU crossover | ≥1,048,576 elements (`NML_METAL_MMUL_THRESHOLD`, ~1024×1024). MPS framework launch cost is high; BLAS outperforms Metal below this. Override: `-DNML_METAL_MMUL_THRESHOLD=N`. |
| SYCL GPU crossover | ≥16,384 elements (`NML_SYCL_MMUL_THRESHOLD`, 128×128). Conservative default safe for both iGPU (break-even ~64×64) and dGPU with PCIe (break-even ~256×256). Override: `-DNML_SYCL_MMUL_THRESHOLD=N`. |
| BLAS acceleration | MMUL and GEMV. CONV, ATTN, FFT, POOL use scalar C loops regardless of build. |
| No GPU for TNET/TNDEEP | Fused training loops use scalar C code. BLAS-accelerated MMUL helps only the non-fused inference path. |
| Hailo NPU | Whole-model dispatch only — bypasses the NML interpreter entirely. No per-op dispatch. Requires a pre-compiled HEF alongside the `.nml` file (or embedded in the binary). |

### Not Supported

- **Automatic differentiation** — backward ops must be written explicitly (no computation graph).
- **Dynamic tensor shapes** — all tensors have fixed max allocation at compile time.
- **Multi-threading** — single-threaded execution; `nmld` uses pre-fork processes instead.
- **Sparse tensors** — all tensors are dense.
- **Complex numbers** — f32, f64, i32 only.
- **String/text data** — NML operates on numeric tensors only.
- **GPU training** — TNET/TNDEEP are CPU-only. Metal dispatch applies only to individual MMUL ops.

## Version History

| Version | Instructions | Key Changes |
|---------|-------------|-------------|
| v0.2 | 28 | Initial spec (NN + tree models) |
| v0.3 | 42 | NML-V, NML-T, NML-R, NML-S extensions |
| v0.4 | 49 | CALL/RET, backward jumps, error codes |
| v0.4.1 | 49 | Symbolic dual-syntax (Unicode opcodes + Greek registers) |
| v0.4.2 | 49 | Verbose human-readable aliases (tri-syntax) |
| v0.5 | 49 | Per-tensor data types (f32, f64, i32) |
| v0.6 | 60 | M2M extensions (META, FRAG, SIGN, VOTE, PROJ, DIST) |
| v0.6.2 | 67 | NML-G general-purpose (SYS, MOD, ITOF, FTOI, BNOT) |
| v0.6.3 | 67 | Compact form (pilcrow delimiter), MCP toolchain server |
| v0.6.4 | 67 | Alternative aliases for LLM trainability, bare number tolerance |
| v0.7.0 | 71 | NML-TR training extensions (BKWD, WUPD, LOSS, TNET), BLAS acceleration |
| v0.7.1 | 71 | NML daemon (nmld) with pre-fork workers and binary cache, constrained decoding (Outlines CFG), additive alias tolerance |
| v0.8.0 | 82 | Backward opcodes (RELUBK, SIGMBK, TANHBK, GELUBK, SOFTBK, MMULBK, CONVBK, POOLBK, NORMBK, ATTNBK), TNDEEP N-layer training |
| v0.8.1 | 82 | FRAG/LINK assembly-time resolution, SIGN/VRFY with HMAC-SHA256 (`-DNML_CRYPTO`), PTCH differential patching, M2M agent distribution protocol, fraud detection example |
| v0.9.0 | 85 | Config-driven training (TRAIN, INFER, WDECAY, TLOG) — separates training config from network architecture, enables forward-only inference |
| v0.10.0 | 89 | Broadcasting (MADD/MSUB/EMUL/EDIV); 4D NCHW vision ops; cross-entropy loss (`LOSS #2`); CUDA + SYCL GPU kernels (CONV/ATTN/GELU/SOFTMAX/LAYERNORM) with elementwise fusion; N-layer TNET (`TNET R_config #epochs`); BN (batch normalization) + DROP (inverted dropout) opcodes; WDECAY (AdamW L2); ONNX import transpiler (28 op types); C library API (`nml_api.h`); Python ctypes binding |
