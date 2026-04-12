# NML — Neural Machine Language Specification v0.10.0

## Overview

NML is a minimal machine language designed for AI workloads. It supports neural network inference, decision tree model execution, and general-purpose computation within a single instruction set and runtime.

## Design Principles

1. **One way to do everything** — no syntactic ambiguity
2. **Fixed-width opcodes** — every opcode is exactly 4 characters (or fewer)
3. **Register-based** — 32 tensor registers (R0–RV)
4. **Typed operands** — registers, immediates, addresses
5. **No implicit behavior** — every side effect is explicit
6. **Model-agnostic** — supports neural networks, decision trees, and gradient boosted ensembles

## Registers

| Register | Name | Purpose |
|----------|------|---------|
| R0–R9 | General | General-purpose tensor registers (10) |
| RA | Accumulator | Tree prediction accumulation |
| RB | General | General-purpose |
| RC | Scratch | Leaf values, intermediate results |
| RD | Counter | Loop counter |
| RE | Flag | Condition flag (set by CMPF, CMP, CMPI) |
| RF | Stack | Stack pointer |
| RG | Gradient1 | Gradient tensor (backpropagation) |
| RH | Gradient2 | Gradient tensor (backpropagation) |
| RI | Gradient3 | Gradient tensor (backpropagation) |
| RJ | LearningRate | Learning rate scalar |
| RK–RV | Training/Hive | Training workspace and hive collective registers (12) |

## Instruction Set (35 Core + 14 Extensions + 12 M2M + 5 General + 21 Training = 89 Total)

Note: SADD and SSUB are counted within Arithmetic (10 opcodes). Training breakdown: 4 core ops (BKWD, WUPD, LOSS, TNET) + 1 deep trainer (TNDEEP) + 10 backward passes (RELU_BK through ATTN_BK) + 4 config-driven (TLOG, TRAIN, INFER, WDECAY) + 2 normalization/regularization (BN, DROP)

### Arithmetic (10 instructions)
```
MMUL  Rd Rs1 Rs2     — Matrix multiply: Rd = Rs1 @ Rs2
MADD  Rd Rs1 Rs2     — Element-wise add: Rd = Rs1 + Rs2
MSUB  Rd Rs1 Rs2     — Element-wise subtract: Rd = Rs1 - Rs2
EMUL  Rd Rs1 Rs2     — Element-wise multiply: Rd = Rs1 * Rs2
EDIV  Rd Rs1 Rs2     — Element-wise divide: Rd = Rs1 / Rs2
SDOT  Rd Rs1 Rs2     — Scalar dot product: Rd = dot(Rs1, Rs2)
SCLR  Rd Rs1 #imm    — Scalar multiply: Rd = Rs1 * imm
SDIV  Rd Rs1 #imm    — Scalar divide: Rd = Rs1 / imm
SADD  Rd Rs1 #imm    — Scalar add: Rd[i] = Rs1[i] + imm
SSUB  Rd Rs1 #imm    — Scalar subtract: Rd[i] = Rs1[i] - imm
```

### Activation (4 instructions)
```
RELU  Rd Rs           — ReLU activation: Rd = max(0, Rs)
SIGM  Rd Rs           — Sigmoid: Rd = 1/(1+exp(-Rs))
TANH  Rd Rs           — Tanh activation
SOFT  Rd Rs           — Softmax across last dimension
```

### Memory (4 instructions)
```
LD    Rd @addr        — Load tensor from memory address
ST    Rs @addr        — Store tensor to memory address
MOV   Rd Rs           — Copy register: Rd = Rs
ALLC  Rd #shape       — Allocate zero tensor with shape (e.g. #[1] #[3,4])
```

### Data Flow (4 instructions)
```
RSHP  Rd Rs #shape    — Reshape tensor (preserves data, changes dimensions)
TRNS  Rd Rs           — Transpose (swap last two dims, requires 2D)
SPLT  Rd Re Rs #dim [#split_at] — Split Rs along dim into Rd (lower) and Re (upper)
MERG  Rd Rs1 Rs2 #dim — Concatenate Rs1 and Rs2 along dim into Rd
```

### Comparison (3 instructions)
```
CMPF  Rd Rs #feat #thresh  — Tree feature compare: flag = (Rs[feat] < thresh)
CMP   Rs1 Rs2              — General compare: flag = (Rs1[0] < Rs2[0])
CMPI  Rd Rs #imm           — Immediate compare: flag = (Rs[0] < imm)
```

All comparison instructions set the global condition flag used by JMPT/JMPF.

### Control Flow (5 instructions)
```
JMPT  #offset         — Jump by offset if flag is true
JMPF  #offset         — Jump by offset if flag is false
JUMP  #offset         — Unconditional jump by offset
LOOP  #count          — Begin counted loop
ENDP                  — End loop (branch back if iterations remain)
```

Jump offsets can be positive (forward) or negative (backward). After the jump is applied, PC is incremented by 1 as usual, so `JMPT #-5` at PC 30 lands on PC 26 (30 + (-5) + 1 = 26).

### Subroutines (2 instructions)
```
CALL  #offset         — Push return address, jump by offset
RET                   — Pop return address, resume after CALL
```

CALL pushes PC+1 onto the call stack (max depth 32) and jumps by offset. RET pops the return address and resumes execution at the instruction after the corresponding CALL.

### Tree Model (3 instructions)
```
LEAF  Rd #value       — Store scalar leaf value in register
TACC  Rd Rs1 Rs2      — Tree accumulate: Rd = Rs1 + Rs2 (scalar addition)
```

(CMPF is listed under Comparison above.)

### System (3 instructions)
```
SYNC                  — Barrier / synchronization point (no-op in single-threaded)
HALT                  — Stop execution (required for clean termination)
TRAP  #code           — Trigger program fault with error code
```

### Extension: NML-V — Vision (4 instructions)
```
CONV  Rd Rs1 Rs2 [#stride] [#pad]  — 2D/4D convolution; accepts [N,C,H,W] input + [Cout,Cin,KH,KW] kernel (NCHW)
POOL  Rd Rs [#size] [#stride]      — Max pooling (2D and 4D NCHW)
UPSC  Rd Rs [#scale]               — Nearest-neighbor upscale
PADZ  Rd Rs [#pad]                 — Zero-padding
```

4D NCHW tensors accepted from v0.10.0. Prior 2D behaviour is preserved.

### Extension: NML-T — Transformer (4 instructions)
```
ATTN  Rd Rq Rk [Rv]               — Scaled dot-product attention
NORM  Rd Rs [Rgamma] [Rbeta]      — Layer normalization
EMBD  Rd Rtable Rindices           — Embedding table lookup
GELU  Rd Rs                        — GELU activation
```

### Extension: NML-R — Reduction (4 instructions)
```
RDUC  Rd Rs #op [#dim]            — Reduce (0=sum, 1=mean, 2=max, 3=min)
WHER  Rd Rcond Ra [Rb]            — Conditional select: Rd = cond ? a : b
CLMP  Rd Rs [#lo] [#hi]           — Clamp values to [lo, hi]
CMPR  Rd Rs [#threshold] [#op]    — Element-wise comparison → 0/1 mask
```

### Extension: NML-S — Signal (2 instructions)
```
FFT   Rd_real Rd_imag Rs           — Discrete Fourier Transform
FILT  Rd Rs Rs_coeffs              — FIR filter (1D convolution)
```

### Extension: NML-TR — Training (23 instructions)
```
BKWD   Rd Rs Rtarget              — Backpropagation: compute gradient of loss w.r.t. Rs into Rd
WUPD   Rd Rs Rgrad [Rlr]          — Weight update: Rd = Rs - lr * Rgrad (default lr from RJ)
LOSS   Rd Rs Rtarget [#mode]      — Loss: mode 0=MSE (default), 1=MAE, 2=cross-entropy
TNET   R_config #epochs           — N-layer MLP training; config tensor shape [L,3] rows=[in,hidden,out]
TNDEEP R_config #epochs           — Deep N-layer training with momentum and gradient clipping
BN     Rd Rs Rgamma Rbeta         — Batch normalization with learnable scale (γ) and shift (β)
DROP   Rd Rs #rate                — Inverted dropout at rate; pass #0.0 at inference to disable
WDECAY Rd #lambda                 — L2 weight decay (in-place): Rd[i] *= (1 - lambda)

; Backward passes (use underscore-form; MMULBK aliases are accepted)
RELU_BK  Rd Rs Rgrad              — ReLU backward
SIGM_BK  Rd Rs Rgrad              — Sigmoid backward
TANH_BK  Rd Rs Rgrad              — Tanh backward
GELU_BK  Rd Rs Rgrad              — GELU backward
SOFT_BK  Rd Rs Rtarget            — Softmax backward
MMUL_BK  Rd_dA Rd_dB Rgrad Ra Rb  — MMUL backward (two output gradients)
CONV_BK  Rd Rs Rgrad              — CONV backward
POOL_BK  Rd Rs Rgrad              — POOL backward
NORM_BK  Rd Rs Rgrad              — NORM backward
ATTN_BK  Rd Rs Rgrad              — Attention backward
LOSS_BK  Rd Rs Rtarget [#mode]    — Loss backward (same modes as LOSS)

; Config-driven training
TLOG   Rs [#interval]             — Log scalar value from Rs every N steps
TRAIN  R_config [@data [@labels]]  — Config-driven training (default input=R0, target=R9)
INFER  [Rd] [Rs]                  — Forward-only inference (default output=RA, input=R0)
```

TNET / TNDEEP config tensor shape `[L,3]` where each row is `[input_size, hidden_size, activation]`.
Supports 1–8 layers. R0 = input, R9 = targets at call time.
TNDEEP / TRAIN require RV (register 31) to contain the architecture descriptor.
TRAIN R_config: 6-element tensor `[epochs, lr, optimizer, print_every, patience, min_delta]`.
Optimizer codes: 0=SGD, 1=Adam, 2=AdamW. Patience=0 disables early stopping.

BN / DROP are Phase 3 additions (v0.10.0):
- BN: symbolic `⊞`, verbose `BATCH_NORM`
- DROP: symbolic `≋`, verbose `DROPOUT`

## Encoding Format

Each instruction encodes to a fixed 32-bit word:

```
[OPCODE: 7 bits][Rd: 5 bits][Rs1: 5 bits][Rs2/imm: 15 bits]
```

- 7-bit opcode → supports up to 128 instructions
- 5-bit register fields → 32 registers
- 15-bit immediate → shapes, addresses, scalar values

## Data File Format (.nml.data)

Memory contents are loaded from simple text files:

```
# Comments start with #
@label shape=dim1,dim2 data=val1,val2,val3,...
```

Example:
```
@sensor_data shape=1,4 data=0.9,0.1,0.95,0.3
@weights shape=4,3 data=0.2,-0.1,0.4,0.5,0.3,-0.2,...
```

## Runtime Limits

| Resource | Default | Configurable | Compile Flag |
|----------|---------|-------------|--------------|
| Registers | 32 | No | — |
| Max tensor elements | 16,777,216 (16M) | Yes (v0.6.2+) | `-DNML_MAX_TENSOR_SIZE=N` — embedded/RPi targets use 65,536 |
| Max instructions | 8,192 | Yes (v0.6.2+) | `-DNML_MAX_INSTRUCTIONS=N` |
| Max memory slots | 256 | Yes (v0.6.2+) | `-DNML_MAX_MEMORY_SLOTS=N` |
| Max loop depth | 8 | Yes (v0.6.2+) | `-DNML_MAX_LOOP_DEPTH=N` |
| Max call depth | 32 | Yes (v0.6.2+) | `-DNML_MAX_CALL_DEPTH=N` |
| Max cycles | 1,000,000 | Yes | `--max-cycles N` |

Example — build with expanded limits for general-purpose programs:
```bash
gcc -O2 -o nml-gp nml.c -lm \
    -DNML_MAX_INSTRUCTIONS=65536 \
    -DNML_MAX_MEMORY_SLOTS=256 \
    -DNML_MAX_CALL_DEPTH=128
```

## Error Handling

The v0.4 runtime returns structured error codes instead of crashing:

| Code | Name | Meaning |
|------|------|---------|
| 0 | NML_OK | Success |
| -1 | NML_ERR_SHAPE | Tensor shape mismatch |
| -2 | NML_ERR_OOB | Out-of-bounds access |
| -3 | NML_ERR_UNINIT | Uninitialized memory read |
| -4 | NML_ERR_OVERFLOW | Tensor size exceeds limit |
| -5 | NML_ERR_DIVZERO | Division by zero (SDIV/EDIV) |
| -6 | NML_ERR_OPCODE | Unknown opcode |
| -7 | NML_ERR_MEMORY | Out of memory slots |
| -9 | NML_ERR_CYCLE_LIMIT | Cycle limit exceeded |
| -10 | NML_ERR_TRAP | TRAP instruction executed |
| -11 | NML_ERR_CALL_DEPTH | Call stack overflow |
| -12 | NML_ERR_RET_EMPTY | RET with empty call stack |

## Pre-Execution Validation

`vm_validate()` checks before execution:
- Jump targets are within program bounds
- LOOP/ENDP are properly nested and balanced
- Register indices are valid
- At least one HALT instruction exists (warning if missing)

## Command-Line Usage

```bash
nml <program.nml> [data.nml.data] [--trace] [--max-cycles N]
```

- `--trace` — Print each instruction as it executes (to stderr)
- `--max-cycles N` — Override the default 1M cycle limit

## Example Programs

### 1. Single Dense Layer (Neural Network)
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

### 2. Backward Jump — While Loop (sum 1 to 5)
```
LEAF  RA #0.0            ; sum = 0
LEAF  RD #1.0            ; counter = 1
TACC  RA RA RD           ; sum += counter
LEAF  RC #1.0
TACC  RD RD RC           ; counter++
CMPI  RE RD #6.0         ; counter < 6?
JMPT  #-5                ; yes → back to first TACC
ST    RA @result          ; result = 15
HALT
```

### 3. Subroutine with CALL/RET
```
LEAF  R1 #7.0
CALL  #2                 ; call double_r1
ST    R1 @result          ; result = 14
JUMP  #2                 ; skip subroutine
TACC  R1 R1 R1           ; double_r1: R1 = R1 + R1
RET
HALT
```

### 4. Division
```
LEAF  R0 #100.0
SDIV  R1 R0 #3.0         ; R1 = 33.333...
LEAF  R2 #4.0
EDIV  R3 R0 R2           ; R3 = 25.0
ST    R1 @div_result
HALT
```

### 5. General Comparison
```
LEAF  R0 #42.0
CMPI  RE R0 #50.0        ; 42 < 50 → flag = 1
JMPF  #1                 ; flag is true, don't jump
ST    R0 @less_than_50
HALT
```

## Token Analysis

| Metric | NML | Python/PyTorch | Python/XGBoost |
|--------|-----|---------------|----------------|
| Vocabulary size | ~60 symbols | ~50,000+ tokens | ~50,000+ tokens |
| Avg tokens/program (NN) | 20–50 | 100–500 | — |
| Avg tokens/program (trees) | auto-transpiled | — | 100–300 + model |
| Syntactic ambiguity | Zero | High | High |
| Ways to express same op | 1 | 5–10+ | 5–10+ |
| Grammar rules | ~10 | ~100+ | ~100+ |
| Runtime size (stripped) | 50 KB | ~30+ MB | ~50+ MB |

## Transpilation Support

| Source | Status | Notes |
|--------|--------|-------|
| XGBoost | Validated | 20/20 exact match, tree dump → NML |
| Domain Data | Validated | Deterministic rule-based transpilation |
| Domain Rules | Validated | Deterministic rule-based transpilation |
| LightGBM | Planned | Similar tree structure to XGBoost |
| scikit-learn RF | Planned | Decision tree export available |
| PyTorch (NN) | Planned | Layer → NML instruction mapping |
| ONNX | Planned | Graph → NML instruction sequence |

## Symbolic Syntax (Dual-Syntax Mode)

NML supports a dual-syntax system: every opcode and register has both a classic ASCII mnemonic and a Unicode symbolic alias. Both forms assemble to identical bytecode and can be mixed freely within a single program.

### Symbolic Opcode Reference

| Classic | Symbol | Unicode | Category |
|---------|--------|---------|----------|
| `MMUL` | `×` | U+00D7 | Arithmetic |
| `MADD` | `⊕` | U+2295 | Arithmetic |
| `MSUB` | `⊖` | U+2296 | Arithmetic |
| `EMUL` | `⊗` | U+2297 | Arithmetic |
| `EDIV` | `⊘` | U+2298 | Arithmetic |
| `SDOT` | `·` | U+00B7 | Arithmetic |
| `SCLR` | `∗` | U+2217 | Arithmetic |
| `SDIV` | `÷` | U+00F7 | Arithmetic |
| `SADD` | `∔` | U+2214 | Arithmetic |
| `SSUB` | `∸` | U+2238 | Arithmetic |
| `RELU` | `⌐` | U+2310 | Activation |
| `SIGM` | `σ` | U+03C3 | Activation |
| `TANH` | `τ` | U+03C4 | Activation |
| `SOFT` | `Σ` | U+03A3 | Activation |
| `LD` | `↓` | U+2193 | Memory |
| `ST` | `↑` | U+2191 | Memory |
| `MOV` | `←` | U+2190 | Memory |
| `ALLC` | `□` | U+25A1 | Memory |
| `RSHP` | `⊟` | U+229F | Data Flow |
| `TRNS` | `⊤` | U+22A4 | Data Flow |
| `SPLT` | `⊢` | U+22A2 | Data Flow |
| `MERG` | `⊣` | U+22A3 | Data Flow |
| `CMPF` | `⋈` | U+22C8 | Comparison |
| `CMP` | `≶` | U+2276 | Comparison |
| `CMPI` | `≺` | U+227A | Comparison |
| `JMPT` | `↗` | U+2197 | Control |
| `JMPF` | `↘` | U+2198 | Control |
| `JUMP` | `→` | U+2192 | Control |
| `LOOP` | `↻` | U+21BB | Control |
| `ENDP` | `↺` | U+21BA | Control |
| `LEAF` | `∎` | U+220E | Tree |
| `TACC` | `∑` | U+2211 | Tree |
| `CALL` | `⇒` | U+21D2 | Subroutine |
| `RET` | `⇐` | U+21D0 | Subroutine |
| `SYNC` | `⏸` | U+23F8 | System |
| `HALT` | `◼` | U+25FC | System |
| `TRAP` | `⚠` | U+26A0 | System |
| `CONV` | `⊛` | U+229B | NML-V |
| `POOL` | `⊓` | U+2293 | NML-V |
| `UPSC` | `⊔` | U+2294 | NML-V |
| `PADZ` | `⊡` | U+22A1 | NML-V |
| `ATTN` | `⊙` | U+2299 | NML-T |
| `NORM` | `‖` | U+2016 | NML-T |
| `EMBD` | `⊏` | U+228F | NML-T |
| `GELU` | `ℊ` | U+210A | NML-T |
| `RDUC` | `⊥` | U+22A5 | NML-R |
| `WHER` | `⊻` | U+22BB | NML-R |
| `CLMP` | `⊧` | U+22A7 | NML-R |
| `CMPR` | `⊜` | U+229C | NML-R |
| `FFT` | `∿` | U+223F | NML-S |
| `FILT` | `⋐` | U+22D0 | NML-S |
| `BKWD` | `∇` | U+2207 | NML-TR |
| `WUPD` | `⟳` | U+27F3 | NML-TR |
| `LOSS` | `△` | U+25B3 | NML-TR |
| `TNET` | `⥁` | U+2941 | NML-TR |

### Greek Register Aliases

All 32 registers have Greek letter aliases:

| Index | Classic | Greek | Letter | Purpose |
|-------|---------|-------|--------|---------|
| 0 | `R0` | `ι` | iota | General |
| 1 | `R1` | `κ` | kappa | General |
| 2 | `R2` | `λ` | lambda | General |
| 3 | `R3` | `μ` | mu | General |
| 4 | `R4` | `ν` | nu | General |
| 5 | `R5` | `ξ` | xi | General |
| 6 | `R6` | `ο` | omicron | General |
| 7 | `R7` | `π` | pi | General |
| 8 | `R8` | `ρ` | rho | General |
| 9 | `R9` | `ς` | final sigma | General |
| 10 | `RA` | `α` | alpha | Accumulator |
| 11 | `RB` | `β` | beta | General |
| 12 | `RC` | `γ` | gamma | Scratch |
| 13 | `RD` | `δ` | delta | Counter |
| 14 | `RE` | `φ` | phi | Condition flag |
| 15 | `RF` | `ψ` | psi | Stack pointer |
| 16 | `RG` | `η` | eta | Gradient1 |
| 17 | `RH` | `θ` | theta | Gradient2 |
| 18 | `RI` | `ζ` | zeta | Gradient3 |
| 19 | `RJ` | `ω` | omega | LearningRate |
| 20 | `RK` | `χ` | chi | Training/Hive |
| 21 | `RL` | `υ` | upsilon | Training/Hive |
| 22 | `RM` | `ε` | epsilon | Training/Hive |
| 23–31 | `RN`–`RV` | — | — | Training/Hive |

### Example: Anomaly Detector in Symbolic Syntax

```
↓  ι @sensor_data
↓  κ @w1
↓  λ @b1
×  μ ι κ
⊕  μ μ λ
⌐  μ μ
↓  ν @w2
↓  ξ @b2
×  ο μ ν
⊕  ο ο ξ
⌐  ο ο
↓  π @w3
↓  ρ @b3
×  ς ο π
⊕  ς ς ρ
σ  ς ς
↑  ς @anomaly_score
◼
```

18 instructions. Every token carries semantic weight. Produces identical output to the classic-syntax version.

### Verbose Syntax (Human-Readable)

A third syntax tier uses full English words, optimized for auditability and self-documentation:

| Classic | Verbose | Category |
|---------|---------|----------|
| `MMUL` | `MATRIX_MULTIPLY` | Arithmetic |
| `MADD` | `ADD` | Arithmetic |
| `MSUB` | `SUBTRACT` | Arithmetic |
| `EMUL` | `ELEMENT_MULTIPLY` | Arithmetic |
| `EDIV` | `ELEMENT_DIVIDE` | Arithmetic |
| `SDOT` | `DOT_PRODUCT` | Arithmetic |
| `SCLR` | `SCALE` | Arithmetic |
| `SDIV` | `DIVIDE` | Arithmetic |
| `SADD` | `SCALAR_ADD` | Arithmetic |
| `SSUB` | `SCALAR_SUB` | Arithmetic |
| `SIGM` | `SIGMOID` | Activation |
| `SOFT` | `SOFTMAX` | Activation |
| `LD` | `LOAD` | Memory |
| `ST` | `STORE` | Memory |
| `MOV` | `COPY` | Memory |
| `ALLC` | `ALLOCATE` | Memory |
| `RSHP` | `RESHAPE` | Data Flow |
| `TRNS` | `TRANSPOSE` | Data Flow |
| `SPLT` | `SPLIT` | Data Flow |
| `MERG` | `MERGE` | Data Flow |
| `LOOP` | `REPEAT` | Control |
| `ENDP` | `END_REPEAT` | Control |
| `CMPF` | `COMPARE_FEATURE` | Comparison |
| `CMP` | `COMPARE` | Comparison |
| `CMPI` | `COMPARE_VALUE` | Comparison |
| `LEAF` | `SET_VALUE` | Tree |
| `TACC` | `ACCUMULATE` | Tree |
| `JMPT` | `BRANCH_TRUE` | Control |
| `JMPF` | `BRANCH_FALSE` | Control |
| `HALT` | `STOP` | System |
| `RET` | `RETURN` | System |
| `TRAP` | `FAULT` | System |
| `SYNC` | `BARRIER` | System |
| `BKWD` | `BACKPROPAGATE` | Training |
| `WUPD` | `WEIGHT_UPDATE` | Training |
| `LOSS` | `COMPUTE_LOSS` | Training |
| `TNET` | `TRAIN_NETWORK` | Training |

Verbose register aliases for named registers:

| Classic | Verbose | Aliases |
|---------|---------|---------|
| `RA` | `ACCUMULATOR` | `ACC` |
| `RB` | `GENERAL` | `GEN` |
| `RC` | `SCRATCH` | `TMP` |
| `RD` | `COUNTER` | `CTR` |
| `RE` | `FLAG` | `FLG` |
| `RF` | `STACK` | `STK` |
| `RG` | `GRADIENT1` | `GRD1` |
| `RH` | `GRADIENT2` | `GRD2` |
| `RI` | `GRADIENT3` | `GRD3` |
| `RJ` | `LEARNING_RATE` | `LR` |
| `RK`–`RV` | `TRAINING_K`–`TRAINING_V` | `TRK`–`TRV` |

### Tri-Syntax Comparison

The same FICA calculation in all three syntaxes (all produce identical bytecode):

**Classic:**
```
LEAF  RC #176100.000000
SCLR  RC RC #0.062000
TACC  RA RA RC
ST    RA @tax_amount
HALT
```

**Symbolic:**
```
∎  γ #176100.000000
∗  γ γ #0.062000
∑  α α γ
↑  α @tax_amount
◼
```

**Verbose:**
```
SET_VALUE          SCRATCH #176100.000000
SCALE              SCRATCH SCRATCH #0.062000
ACCUMULATE         ACCUMULATOR ACCUMULATOR SCRATCH
STORE              ACCUMULATOR @tax_amount
STOP
```

### Mixed Syntax

All three syntaxes can be combined freely in the same program:

```
LOAD  R0 @input        ; verbose opcode + classic register
×     R3 R0 R1         ; symbolic opcode + classic registers
⊕     μ μ λ           ; symbolic opcode + Greek registers
STORE ς @output        ; verbose opcode + Greek register
◼                      ; symbolic HALT
```

### Transpiler Syntax Flag

The transpiler and library builder support a `--syntax` flag:

```bash
python3 nml_transpiler.py transpile file.json --syntax classic
python3 nml_transpiler.py transpile file.json --syntax symbolic
python3 nml_transpiler.py transpile file.json --syntax verbose
```

### Compact Form (Single-Line Representation)

NML programs can be represented as a single line of text using the pilcrow `¶` (U+00B6) as the instruction delimiter. The runtime natively parses both newline-delimited and `¶`-delimited programs — no preprocessing required.

**Multi-line (standard):**
```
↓  ι  @gross_pay
∗  γ  ι  #0.062
∑  α  α  γ
↑  α  @tax_amount
◼
```

**Compact (single-line):**
```
↓ ι @gross_pay¶∗ γ ι #0.062¶∑ α α γ¶↑ α @tax_amount¶◼
```

Both forms produce identical bytecode. The compact form strips comments and normalizes whitespace (multiple spaces collapse to one).

**Design:**
- `¶` derives from the Greek *paragraphos* (παράγραφος), the mark scribes used to indicate a new section
- `¶` (instruction boundary) pairs with `◼` (program termination) — the two structural delimiters of an NML program
- `¶` is U+00B6 (2 bytes UTF-8), not used by any NML opcode, register, or operand
- Overhead vs newlines: +1 byte per instruction boundary, offset by whitespace normalization

**When to use compact form:**
- Agent-to-agent message payloads (M2M)
- JSON API fields (no escaping needed — no newlines)
- Token-efficient LLM context (fewer tokens than multi-line)
- Embedded in data structures, databases, or configuration

**Tooling:**
```bash
python3 nml_format.py compact  program.nml       # multi-line → compact
python3 nml_format.py format   "∎ α #42¶↑ α @r¶◼"  # compact → multi-line
```

**Runtime:**
```bash
echo '∎ α #42.0¶↑ α @result¶◼' > program.nml
./nml program.nml data.nml.data    # parses ¶ natively
```

## Data Types (v0.5)

NML v0.5 supports per-tensor data types. Each tensor carries its own type; operations auto-promote when mixing types.

### Supported Types

| Type | Size | Use Case | Default |
|------|------|----------|---------|
| `f32` | 4 bytes | ML inference, neural networks | Yes |
| `f64` | 8 bytes | Financial calculations, high-precision matching | No |
| `i32` | 4 bytes | Counters, indices, integer math | No |

### Type Promotion Rules

When two tensors of different types interact (MADD, MSUB, EMUL, etc.):
- f32 + f64 → f64
- f32 + i32 → f32
- f64 + i32 → f64
- Same type → same type

LEAF creates f64 tensors (immediate values are stored in double precision).
ALLC creates f32 tensors by default, or a specified type: `ALLC R0 #[1] f64`.

### Data File Format

The `.nml.data` format supports an optional `dtype=` field:

```
@income shape=1 dtype=f64 data=185000.00
@weights shape=4,3 data=0.2,-0.1,0.4,...          # defaults to f32
@periods shape=1 dtype=i32 data=26
```

When `dtype=` is omitted, the tensor defaults to `f32` for backward compatibility.

### Backward Compatibility

- All existing `.nml` programs work unchanged (everything defaults to f32)
- All existing `.nml.data` files work unchanged (no dtype field = f32)
- `ALLC R0 #[1]` without dtype = f32
- LEAF always creates f64 (immediates parsed as double)
- Compile with `-DNML_F32_ONLY` to strip type support and preserve original binary size

## Extension: NML-M2M — Machine-to-Machine (11 instructions)

See [NML_M2M_Spec.md](NML_M2M_Spec.md) for the full specification.

### Structure (4 instructions)
```
META  @key value             — Program metadata (no-op at runtime)
FRAG  name                   — Open named fragment scope
ENDF                         — Close fragment scope
LINK  @name                  — Import named fragment inline
```

### Communication (3 instructions)
```
PTCH  @directive args        — Differential program patch
SIGN  agent=N key=K sig=S    — Cryptographic signature (no-op at runtime)
VRFY  @hash @signer          — Verify signature (TRAP on mismatch)
```

### Collective (3 instructions)
```
VOTE  Rd Rs #strategy [#thr] — Multi-agent consensus (median/mean/quorum/min/max)
PROJ  Rd Rs Rmatrix          — Embedding projection with L2 normalization
DIST  Rd Rs1 Rs2 [#metric]   — Embedding distance (cosine/euclidean/dot)
GATH  Rd Rs Ridx          — Gather: Rd = Rs[Ridx[0]] (tensor index lookup)
SCAT  Rs Rd Ridx          — Scatter: Rd[Ridx[0]] = Rs[0] (tensor index write)
```

### Semantic Type Annotations
Register operands support optional `:type` suffix: `ι:currency`, `κ:category`, etc.
Types: `float` (default), `currency`, `ratio`, `category`, `count`, `bool`, `embedding`, `probability`.

## Extension: NML-G — General Purpose (5 instructions)

NML-G extends the instruction set for general-purpose computing: console I/O, integer math, and type conversion. See [NML_G_Spec.md](NML_G_Spec.md) for the full specification.

### System Call (1 instruction)
```
SYS   Rd #code            — System call: code selects operation
```

| Code | Name | Behavior |
|------|------|----------|
| 0 | PRINT_NUM | Print Rd[0] as number + newline |
| 1 | PRINT_CHAR | Print Rd[0] as ASCII character |
| 2 | READ_NUM | Read number from stdin into Rd |
| 3 | READ_CHAR | Read character from stdin into Rd |
| 4 | TIME | Wall-clock time (seconds since epoch) into Rd |
| 5 | RAND | Random float in [0, 1) into Rd |
| 6 | EXIT | Terminate with exit code Rd[0] |

### Integer Math (1 instruction)
```
MOD   Rd Rs1 Rs2          — Integer modulo: Rd = (int)Rs1[0] % (int)Rs2[0]
```

### Type Conversion (2 instructions)
```
ITOF  Rd Rs               — Integer to float: Rd = (float)Rs[0]
FTOI  Rd Rs               — Float to integer: Rd = (int)Rs[0] (truncates toward zero)
```

### Bitwise (1 instruction)
```
BNOT  Rd Rs               — Bitwise NOT: Rd[i] = ~(int)Rs[i]
```

### NML-G Tri-Syntax Aliases

| Classic | Symbolic | Verbose |
|---------|----------|---------|
| `SYS` | `⚙` (U+2699) | `SYSTEM` |
| `MOD` | `%` | `MODULO` |
| `ITOF` | `⊶` (U+22B6) | `INT_TO_FLOAT` |
| `FTOI` | `⊷` (U+22B7) | `FLOAT_TO_INT` |
| `BNOT` | `¬` (U+00AC) | `BITWISE_NOT` |

### NML-G Examples

**Fibonacci (13 instructions):**
```
LEAF  R0 #0.0       ; a = 0
LEAF  R1 #1.0       ; b = 1
LEAF  RD #0.0       ; counter
LEAF  R5 #20.0      ; limit
SYS   R0 #0         ; print a
TACC  R2 R0 R1      ; next = a + b
MOV   R0 R1         ; a = b
MOV   R1 R2         ; b = next
LEAF  RC #1.0
TACC  RD RD RC       ; counter++
CMP   RD R5          ; counter < limit?
JMPT  #-8           ; loop back
HALT
```

**FizzBuzz with MOD (29 instructions):**
```
LEAF  RD #1.0
LEAF  R5 #31.0
LEAF  R3 #15.0
; ...
MOD   R0 RD R3      ; R0 = n % 15
CMPI  RE R0 #0.5    ; R0 == 0?
; ... print FizzBuzz / Fizz / Buzz / number ...
HALT
```

Build with NML-G disabled: `gcc -DNML_NO_GENERAL -o nml nml.c -lm`

## Version History

| Version | Core | Extensions | Total | Key Changes |
|---------|------|-----------|-------|-------------|
| v0.2 | 28 | 0 | 28 | Initial spec (NN + tree models) |
| v0.3 | 28 | 14 | 42 | NML-V, NML-T, NML-R, NML-S extensions |
| v0.4 | 35 | 14 | 49 | SDIV, EDIV, CMP, CMPI, SPLT, MERG, CALL, RET, TRAP; backward jumps; error codes; --trace; vm_validate() |
| v0.4.1 | 35 | 14 | 49 | Symbolic dual-syntax: Unicode opcode aliases + Greek register aliases for all 16 registers |
| v0.4.2 | 35 | 14 | 49 | Verbose human-readable aliases; tri-syntax transpiler (`--syntax classic\|symbolic\|verbose`); NML library builder |
| v0.5 | 35 | 14 | 49 | Per-tensor data types (f32, f64, i32); auto type promotion; dtype= in .nml.data; ALLC dtype; NML_F32_ONLY flag |
| v0.6 | 35 | 14 + 11 M2M | 60 | META (self-describing programs); typed register annotations; FRAG/ENDF/LINK (compositional fragments); PTCH (differential programs); SIGN/VRFY (cryptographic signing); VOTE (consensus); PROJ/DIST (latent space) |
| v0.6.1 | 35 | 14 + 13 M2M | 62 | GATH (tensor index lookup), SCAT (tensor index write); register aliasing fix in tensor_add/sub/emul/ediv/mmul; f64 MMUL via tensor_getd/setd; RELU/SIGM/TANH/SOFT f64 support |
| v0.6.2 | 35 | 14 + 13 M2M + 5 GP | 67 | NML-G general-purpose extension: SYS (multiplexed I/O — print, read, time, rand, exit), MOD (integer modulo), ITOF/FTOI (type conversion), BNOT (bitwise NOT); configurable resource limits via compile flags; CMP operand count fix |
| v0.6.3 | 35 | 14 + 13 M2M + 5 GP | 67 | Compact form: `¶` (U+00B6, pilcrow) as native instruction delimiter for single-line NML; runtime parses both `\n` and `¶`; `nml_format.py` CLI for compact/format conversion; MCP toolchain server with 9 tools (spec_lookup, transpile, validate, execute, library_lookup, scan, intent, compact, format) |
| v0.6.4 | 35 | 14 + 13 M2M + 5 GP | 67 | Alternative aliases for LLM trainability: `ϟ` (Greek koppa) for CMPI, `ϛ` (Greek stigma) for RDUC, `DOT` for SDOT, `SCTR` for SCAT (Rd-first order); bare number tolerance on JUMP/JMPT/JMPF/CALL |
| v0.7.0 | 35 | 14 + 13 M2M + 5 GP + 4 TR + 11 BK | 82 | 32 registers (R0–RV); NML-TR training extension: BKWD, WUPD, LOSS, TNET; 11 backward opcodes (RELUBK, SIGMBK, TANHBK, GELUBK, SOFTBK, MMULBK, CONVBK, POOLBK, NORMBK, ATTNBK, TNDEEP); optional BLAS acceleration |

## Alternative Aliases (v0.6.4)

NML v0.6.4 adds alternative opcode aliases designed to improve LLM code generation accuracy. These are additive — all original opcodes remain valid. Programs can mix original and alternative aliases freely.

### Why Alternative Aliases

Some symbolic opcodes collide with common mathematical notation in LLM pre-training data:
- `≺` (CMPI) conflicts with the "precedes" relation in set theory
- `⊥` (RDUC) conflicts with "perpendicular/bottom" in logic and geometry

Some classic opcodes trigger hallucinations from other ISAs:
- `SDOT` — models generate `DOT` (from NumPy `dot()`)
- `SCAT` — unique Rs-first operand order causes errors

### New Aliases

| Original | New Alias | Unicode | Type | Notes |
|----------|-----------|---------|------|-------|
| `≺` (CMPI) | `ϟ` | U+03DF (Greek koppa) | Symbolic | Ancient Greek counting letter, zero pre-training contamination |
| `⊥` (RDUC) | `ϛ` | U+03DB (Greek stigma) | Symbolic | Ancient Greek numeral (=6), never appears in modern code |
| `SDOT` | `DOT` | — | Classic | Matches NumPy convention; `DOT Rd Rs1 Rs2` = `SDOT Rd Rs1 Rs2` |
| `SCAT` | `SCTR` | — | Classic | Rd-first operand order: `SCTR Rd Rs Ridx` (runtime swaps to SCAT order internally) |

### Bare Number Tolerance

Jump and call instructions now accept bare numbers without the `#` prefix:

```
JUMP  #2    ; original (still valid)
JUMP  2     ; now also valid (parsed identically)
JMPT  3     ; bare number accepted
CALL  2     ; bare number accepted
```

### SCTR Operand Order

`SCTR` uses destination-first order, matching every other NML instruction:

```
SCAT  R0 R2 R1    ; original: source R0, destination R2, index R1
SCTR  R2 R0 R1    ; alias: destination R2, source R0, index R1
```

Both produce identical bytecode and execution.
