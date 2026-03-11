# NML — Neural Machine Language

A minimal, deterministic machine language designed for AI workloads. 71 instructions. ~2,100 lines of C. Zero ambiguity.

NML supports neural network inference, decision tree execution, transformer attention, signal processing, and general-purpose computation within a single instruction set and runtime — compiled to an 83 KB binary.

## Why NML

| Property | NML | Python/PyTorch |
|----------|-----|---------------|
| Vocabulary | 71 opcodes | 50,000+ tokens |
| Grammar rules | ~10 | 100+ |
| Syntactic ambiguity | Zero | High |
| Ways to express same op | 1 | 5-10+ |
| Runtime size (stripped) | 83 KB | 30+ MB |
| Avg tokens per program | 20-50 | 100-500 |

NML programs are deterministic, verifiable, and machine-readable. Every program has exactly one valid interpretation.

## What Makes NML Different

### Tensor Registers, Not Scalar Registers

Traditional ISAs (x86, ARM, RISC-V) operate on scalar values — a matrix multiply requires hundreds of instructions in a loop nest. Graph-based runtimes (ONNX, TVM) operate on abstract nodes. NML's 32 registers each hold an entire tensor. A matrix multiply is one instruction: `MMUL R3 R0 R1`. A full neural network layer — load weights, multiply, add bias, activate, store — is 8 instructions. An anomaly detector is 18.

### One ISA for All Model Types

ONNX handles neural networks. XGBoost has its own runtime. Decision trees need a separate executor. Gradient boosted ensembles need yet another. NML handles all of them in a single instruction set:

- **Neural networks**: `MMUL`, `MADD`, `RELU`, `SIGM`, `SOFT`, `ATTN`, `NORM`, `GELU`
- **Decision trees**: `CMPF` (feature compare), `LEAF` (leaf value), `TACC` (accumulate)
- **Training**: `BKWD` (backprop), `WUPD` (weight update), `LOSS`, `TNET` (fused training loop)
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

NML's zero-ambiguity grammar exists specifically so that language models can learn to generate correct programs. With 71 opcodes, ~10 grammar rules, and exactly one way to express each operation, a 7-9B parameter model achieves 93% valid code generation after training on 436K pairs. The design decision — one way to do everything, no syntactic sugar, no implicit behavior — is driven by learnability, not programmer convenience.

## Use Cases

### LLM-to-LLM Communication

When AI models need to exchange executable computation — not just text — NML provides a shared, unambiguous language. Model A generates an NML program, Model B verifies the signature (`SIGN`/`VRFY`) and executes it. No parsing ambiguity, no version conflicts, no prompt injection. The M2M extensions (`META`, `FRAG`, `LINK`, `PTCH`) enable composable, self-describing programs that models can inspect, modify, and chain together.

### Edge Inference

The entire NML runtime compiles to an 83 KB binary with zero dependencies beyond libc and libm. This means neural network inference, decision tree execution, and signal processing can run on microcontrollers, embedded systems, and IoT devices — anywhere Python and ONNX can't go. A sensor anomaly detector is 18 NML instructions and runs in 34 microseconds.

### Auditable Financial and Regulatory Computation

Tax calculations, insurance rate tables, compliance formulas, and regulatory rules can be expressed as NML programs that are deterministic, traceable, and human-readable. The verbose syntax mode (`SET_VALUE`, `ACCUMULATE`, `COMPARE_FEATURE`) makes programs self-documenting for auditors. Every computation step is explicit — there is no hidden state, no implicit type coercion, no undefined behavior.

### Multi-Agent Consensus

Multiple AI models independently generate NML programs for the same task. Each program runs through the C runtime, producing a result. The `VOTE` instruction finds consensus across results — median, mean, quorum, min, or max. If 4 out of 5 models agree, the answer is trustworthy. This eliminates outlier hallucinations without requiring human review.

### Deterministic AI Pipelines

Replace opaque model inference with verifiable instruction sequences. If two systems run the same NML program on the same data, they produce identical results — bit-for-bit reproducible across platforms. This matters for regulated industries (finance, healthcare, insurance) where "the model said so" is not an acceptable audit trail, but "here is the 18-instruction program and its output" is.

### Self-Training at the Edge

An NML program can train its own neural network, then immediately run inference — on the same device, in the same runtime, with zero external dependencies. A sensor node receives a `TNET` program, trains on local calibration data, and starts classifying. An agent in a hive collective receives training data from the coordinator, runs gradient descent via `BKWD`/`WUPD`, and reports back trained weights. No Python. No GPU driver. No network round-trip to a training server.

### Training LLMs to Generate Structured Code

NML's zero-ambiguity grammar makes it dramatically easier to train models to write correct code. With only 71 opcodes, ~10 grammar rules, and exactly one way to express each operation, a 7-9B parameter model can learn to generate valid NML programs with 93% accuracy after training on 436K pairs. Compare this to Python, where the same model must learn thousands of library APIs, multiple coding styles, and ambiguous syntax.

## Quick Start

```bash
# Build
make

# Run Hello World
./nml programs/hello_world.nml

# Run anomaly detector (neural network)
./nml programs/anomaly_detector.nml programs/anomaly_weights.nml.data

# Run Fibonacci
./nml programs/fibonacci.nml

# Run all tests
make test
```

## Tri-Syntax

Every NML instruction can be written in three interchangeable syntaxes — classic, symbolic, and verbose. All three compile to identical bytecode and can be mixed freely in the same program.

**Classic:**
```
LEAF  RC #176100.0
SCLR  RC RC #0.062
TACC  RA RA RC
ST    RA @tax_amount
HALT
```

**Symbolic:**
```
∎  γ  #176100.0
∗  γ  γ  #0.062
∑  α  α  γ
↑  α  @tax_amount
◼
```

**Verbose:**
```
SET_VALUE   SCRATCH #176100.0
SCALE       SCRATCH SCRATCH #0.062
ACCUMULATE  ACCUMULATOR ACCUMULATOR SCRATCH
STORE       ACCUMULATOR @tax_amount
STOP
```

## Instruction Set (71 Total)

### Core (35 Instructions)

| Category | Instructions |
|----------|-------------|
| Arithmetic | `MMUL` `MADD` `MSUB` `EMUL` `EDIV` `SDOT` `SCLR` `SDIV` |
| Activation | `RELU` `SIGM` `TANH` `SOFT` |
| Memory | `LD` `ST` `MOV` `ALLC` |
| Data Flow | `RSHP` `TRNS` `SPLT` `MERG` |
| Comparison | `CMPF` `CMP` `CMPI` |
| Control | `JMPT` `JMPF` `JUMP` `LOOP` `ENDP` |
| Subroutine | `CALL` `RET` |
| Tree | `LEAF` `TACC` |
| System | `SYNC` `HALT` `TRAP` |

### NML-V: Vision (4)
`CONV` `POOL` `UPSC` `PADZ`

### NML-T: Transformer (4)
`ATTN` `NORM` `EMBD` `GELU`

### NML-R: Reduction (4)
`RDUC` `WHER` `CLMP` `CMPR`

### NML-S: Signal (2)
`FFT` `FILT`

### NML-M2M: Machine-to-Machine (13)
`META` `FRAG` `ENDF` `LINK` `PTCH` `SIGN` `VRFY` `VOTE` `PROJ` `DIST` `GATH` `SCAT`

### NML-TR: Training (4)
`BKWD` `WUPD` `LOSS` `TNET`

### NML-G: General Purpose (5)
`SYS` `MOD` `ITOF` `FTOI` `BNOT`

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

## Three Ways to Compute

NML supports multiple approaches to the same computation. These three programs all compute a progressive rate on an input of 100,000 with a deduction of 8,600:

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

| | Cascade | Tensor Table | Neural Network |
|-|---------|-------------|----------------|
| **Result** | 13,614.00 (exact) | 13,614.00 (exact) | 13,241.49 (approx) |
| **Instructions** | 66 | 31 | 12 |
| **Cycles** | 24 | 61 | 12 |
| **Time** | 202 µs | 200 µs | 194 µs |
| **Data file** | 2 values | 5 tensors | 5 tensors (weights) |
| **To update rates** | Rewrite program | Change data tensors | Retrain network |
| **Accuracy** | Exact | Exact | ~$373 error |

All three programs are in `programs/` with matching `.nml.data` files:

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

**Medium network** — 1→256→1 ReLU, 396 real FIT bracket samples, Adam, mini-batch=64:

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

## Architecture

```
runtime/nml.c        Single-file C99 runtime (~2,100 lines, 83 KB binary)
                     Assembler + validator + VM in one file
                     Parses all three syntax forms natively

transpilers/         Grammar validator, training data generators (core, equalize,
                     self-training, tensor table, cascade, library), formatting

serve/               MCP server, protocol definitions, cryptographic signing,
                     provenance tracking, and multi-agent orchestration

programs/            Example NML programs (anomaly detector, fibonacci,
                     fizzbuzz, primes, calculator, hello world)

tests/               Test programs covering core features, symbolic syntax,
                     verbose syntax, M2M extensions

terminal/            React + Bun JSX apps: NML emulator terminal and chat UI

docs/                Full specification, architecture documents, usage guide,
                     M2M spec, and integration plans
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
| f64 | 8 bytes | Financial calculations |
| i32 | 4 bytes | Counters, indices |

## Building

```bash
# Full build (71 instructions, all extensions)
make

# With BLAS acceleration (10x faster training)
make nml-fast

# Or directly with gcc
gcc -O2 -o nml runtime/nml.c -lm

# Core only (35 instructions, no extensions)
gcc -O2 -o nml-core runtime/nml.c -lm -DNML_NO_DEFAULT_EXTENSIONS

# With expanded limits for large programs
gcc -O2 -o nml-gp runtime/nml.c -lm \
    -DNML_MAX_INSTRUCTIONS=65536 \
    -DNML_MAX_MEMORY_SLOTS=256 \
    -DNML_MAX_CALL_DEPTH=128
```

## Running

```bash
./nml <program.nml> [data.nml.data] [--trace] [--max-cycles N]
```

- `--trace` prints each instruction as it executes
- `--max-cycles N` overrides the default 1M cycle limit

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

3. **[Usage Guide](docs/NML_Usage_Guide.md)** — all 71 instructions with detailed examples and edge cases
4. **[NML-G Specification](docs/NML_G_Spec.md)** — general-purpose extensions (SYS, MOD, ITOF, FTOI, BNOT)
5. **[NML-M2M Specification](docs/NML_M2M_Spec.md)** — machine-to-machine extensions (META, FRAG, SIGN, VOTE, PROJ, DIST)

### Architecture

6. **[Architecture Document](docs/NML_Architecture_Document.md)** — system design, philosophy, and the case for a machine-first language
7. **[Implementation Document](docs/NML_Implementation_Document.md)** — what was built, validation results, and performance benchmarks

### Advanced

8. **[Multi-Agent Architecture](docs/NML_Multi_Agent_Architecture.md)** — distributed LLM communication using NML as a formal protocol

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
