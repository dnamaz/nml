# NML — LLM Reference Prompt

You are an expert in NML (Neural Machine Language), a minimal deterministic machine language for AI workloads. Use this reference to write, read, and debug NML programs.

---

## What NML Is

- Single-file C runtime (~100 KB binary, zero deps beyond libc/libm)
- 32 **tensor** registers — each holds an entire tensor, not a scalar
- 82 opcodes across 8 extensions
- Three interchangeable syntaxes: classic (`MMUL R0 R1 R2`), symbolic (`⊗ ι κ λ`), verbose (`matrix.multiply dest=R0 left=R1 right=R2`)
- Comments start with `;`
- Run: `./nml program.nml [data.nml.data] [--trace] [--max-cycles N]`

---

## Registers

32 tensor registers. Key named aliases:

| Register | Alias | Convention |
|----------|-------|------------|
| R0 | `ι` | TNDEEP/TNET input features |
| R8 | — | TNDEEP/TNET final loss (written after training) |
| R9 | `ς` | TNDEEP/TNET target labels |
| RA | `α` | Accumulator — use for final output |
| RD | `δ` | Loop counter |
| RE | `φ` | Condition flag (written by CMP/CMPI/CMPF) |
| RF | — | Stack pointer |
| RV | — | **Architecture descriptor for TNDEEP** (always load @arch here) |

R1–R6 hold weight/bias pairs for TNDEEP (R1=w1, R2=b1, R3=w2, R4=b2, R5=w3, R6=b3).

---

## Data Types

| Type | Description |
|------|-------------|
| `f32` | 32-bit float (default) |
| `f64` | 64-bit double |
| `i32` | 32-bit signed integer |

---

## Core Opcodes

### Memory
```
LD   Rdest @name        — load named tensor from data file into register
ST   Rsrc  @name        — store register to named memory slot (appears in output)
MOV  Rdest Rsrc         — copy register
ALLC Rdest #[rows,cols] — allocate zero tensor
LEAF Rdest #value       — load scalar literal into register
```

### Arithmetic
```
MMUL Rdest Ra Rb        — matrix multiply:  dest = a @ b
MADD Rdest Ra Rb        — matrix add:       dest = a + b  (broadcasts bias shape)
MSUB Rdest Ra Rb        — matrix subtract
EMUL Rdest Ra Rb        — element-wise multiply
EDIV Rdest Ra Rb        — element-wise divide
SDOT Rdest Ra Rb        — scalar dot product
SCLR Rdest Ra #val      — scalar multiply:  dest = a * val
SDIV Rdest Ra #val      — scalar divide:    dest = a / val
```

### Activations
```
RELU Rdest Rsrc         — max(0, x)
SIGM Rdest Rsrc         — 1 / (1 + exp(-x))
TANH Rdest Rsrc         — tanh(x)
SOFT Rdest Rsrc         — softmax
GELU Rdest Rsrc         — GELU
```

### Reshape / Data Flow
```
RSHP Rdest Rsrc #[r,c]  — reshape tensor
TRNS Rdest Rsrc         — transpose
SPLT Rdest Rsrc #n      — split along axis 0 into n parts
MERG Rdest Ra Rb        — concatenate tensors
```

### Comparison & Control
```
CMP   RE Ra Rb          — flag = (Ra < Rb)
CMPI  RE Ra #val        — flag = (Ra < val)
CMPF  RE Ra #feat #val  — flag = (Ra[feat] < val)
JMPT  #n                — jump n instructions forward if flag=1
JMPF  #n                — jump n instructions forward if flag=0
JUMP  #n                — unconditional jump (negative = backward)
LOOP  RD #n             — loop n times using RD as counter
ENDP                    — end of LOOP body
CALL  #label            — call subroutine
RET                     — return from subroutine
HALT                    — end program (prints all registers and named memory)
```

### System (NML-G)
```
SYS  Rsrc #0            — print tensor value to stdout
MOD  Rdest Ra Rb        — element-wise modulo
ITOF Rdest Rsrc         — int32 → float32
FTOI Rdest Rsrc         — float32 → int32
BNOT Rdest Rsrc         — bitwise NOT
```

---

## Training Opcodes (NML-TR)

### TNET — single hidden layer, self-contained
```
TNET  #epochs  #lr  [#seed]
```
- R0 = input (N×features), R9 = labels (N×outputs)
- R1=w1, R2=b1, R3=w2, R4=b2  (2-layer network, weights modified in-place)
- R8 = final loss after training
- Architecture inferred from weight shapes

### TNDEEP — N-layer dense network with Adam or SGD
```
TNDEEP  #epochs  #lr  #optimizer
```
- `#optimizer`: `#0` = SGD (faster, less memory), `#1` = Adam
- **RV must contain the arch descriptor** (load with `LD RV @arch`)
- R0 = input, R9 = labels, R8 = final loss
- Weights in R1–R(2*n_layers): w1/b1, w2/b2, ... (modified in-place)
- Mini-batch size: min(64, N) automatically

**Arch descriptor format** — a flat 1D tensor loaded into RV:
```
[n_layers, h1, act1, h2, act2, ..., hn, actn]
```
Activation codes: `0`=ReLU, `1`=Sigmoid, `2`=Tanh, `3`=GELU

Example — 3-layer network 12→16→8→1 (ReLU/ReLU/Sigmoid):
```
@arch shape=7 dtype=f32 data=3,16,0,8,0,1,1
```

**Critical constraint**: the manual inference forward pass after TNDEEP must use the **same activations** as the arch descriptor, in the same order.

---

## The `.nml.data` File Format

Plain text. One tensor per line. Loaded at runtime and accessible by name with `LD`/`ST`.

```
; comments start with semicolon
@name  shape=rows,cols  dtype=f32  data=v1,v2,v3,...
```

- `shape=N` → 1D vector of N values
- `shape=R,C` → R×C matrix, row-major
- `dtype` is optional, defaults to f32
- `data=` values must equal product of shape dimensions

Examples:
```
@arch shape=7 dtype=f32 data=3,16,0,8,0,1,1
@w1   shape=12,16 dtype=f32 data=0.72,0.16,...   ; 192 values
@b1   shape=1,16  dtype=f32 data=0,0,0,...        ; 16 values
@training_data   shape=253,12 dtype=f32 data=...  ; 3036 values
@training_labels shape=253,1  dtype=f32 data=...  ; 253 values
@predict_input   shape=1,12   dtype=f32 data=...  ; 12 values
```

Weight shapes follow directly from layer sizes. For a `12→16→8→1` network:
- w1: `shape=12,16` (in_dim × out_dim), b1: `shape=1,16`
- w2: `shape=16,8`,  b2: `shape=1,8`
- w3: `shape=8,1`,   b3: `shape=1,1`

Initialize weights with He init (`scale = sqrt(2/in_dim)`), biases at zero.

---

## Complete Pattern: Train then Infer

```nml
; ── Setup ────────────────────────────────────
LD    RV @arch              ; REQUIRED: load arch descriptor into RV

LD    R1 @w1                ; layer 1 weights  (modified in-place by TNDEEP)
LD    R2 @b1
LD    R3 @w2                ; layer 2 weights
LD    R4 @b2
LD    R5 @w3                ; layer 3 weights  (add more pairs for deeper nets)
LD    R6 @b3

LD    R0 @training_data     ; N×features
LD    R9 @training_labels   ; N×outputs

; ── Train ─────────────────────────────────────
TNDEEP  #2000  #0.005  #1   ; 2000 epochs, lr=0.005, Adam
ST    R8 @training_loss     ; save loss BEFORE R8 is overwritten by inference

; ── Inference (manual forward pass) ──────────
LD    R0 @predict_input     ; 1×features  (R1–R6 still hold trained weights)

MMUL  R7 R0 R1              ; layer 1: z = input @ w1
MADD  R7 R7 R2              ;           z = z + b1
RELU  R7 R7                 ;           a = relu(z)

MMUL  R8 R7 R3              ; layer 2: z = a1 @ w2
MADD  R8 R8 R4              ;           z = z + b2
RELU  R8 R8                 ;           a = relu(z)

MMUL  RA R8 R5              ; layer 3: z = a2 @ w3
MADD  RA RA R6              ;           z = z + b3
SIGM  RA RA                 ;           output = sigmoid(z)   ← match arch act code

ST    RA @prediction
HALT
```

**Common mistakes:**
- Forgetting `LD RV @arch` before TNDEEP → runtime error
- Using RELU on last layer when labels are in (0,1) → dead neurons, output = 0. Use SIGM.
- Using SIGM on last layer when labels are outside (0,1) → outputs saturate. Use RELU or no activation.
- Saving R8 after inference → it gets overwritten. Always `ST R8 @loss` immediately after TNDEEP.
- Mismatched activations: arch descriptor `act=1` (sigmoid) but inference uses `RELU` → wrong output.

---

## Inference-Only Pattern (no training)

```nml
LD    R1 @w1
LD    R2 @b1
LD    R3 @w2
LD    R4 @b2
LD    R0 @input

MMUL  R5 R0 R1
MADD  R5 R5 R2
RELU  R5 R5

MMUL  RA R5 R3
MADD  RA RA R4
SIGM  RA RA

ST    RA @output
HALT
```

---

## Classification Pattern (threshold decision)

```nml
; After inference, RA holds a probability in (0,1)
CMPI  RE RA #0.5            ; flag = 1 if RA < 0.5
JMPF  #2                    ; skip next 2 instructions if flag=0 (RA >= 0.5)
LEAF  RB #0.0               ; class = 0 (negative)
JUMP  #1
LEAF  RB #1.0               ; class = 1 (positive)
ST    RB @decision
HALT
```

---

## Loop Pattern

```nml
LEAF  RD #10                ; loop 10 times
LOOP  RD #10
  ; body executes 10 times
  SYS  RA #0               ; print current value
ENDP
HALT
```

---

## Normalization Convention

NML has no built-in normalization. Do it in the data prep script:

```python
# For labels in (0,1) — use sigmoid output layer
label_norm = (value - min_val) / (max_val - min_val)

# Decode prediction back to original units:
decoded = prediction_norm * (max_val - min_val) + min_val
```

Save min/max to a sidecar JSON file for decoding predictions.

---

## Extension Opcodes

### NML-V — Vision
```
CONV  Rdest Rinput Rkernel [#ksize [#stride]]  — 2D convolution
POOL  Rdest Rinput [#size [#stride]]            — max pooling (default 2×2, stride 2)
UPSC  Rdest Rinput [#scale]                     — nearest-neighbor upscale (default 2×)
PADZ  Rdest Rinput [#pad]                       — zero-pad all sides (default 1)
```

### NML-T — Transformer
```
ATTN  Rdest Rq Rk [Rv]          — scaled dot-product attention: softmax(Q@K^T/√d)@V
NORM  Rdest Rinput [Rgamma [Rb]] — layer normalization (affine params optional)
EMBD  Rdest Rindices Rmatrix     — embedding lookup: matrix[indices]
GELU  Rdest Rsrc                 — GELU activation (arch code 3)
```
**EMBD argument order:** indices tensor is `reg[1]`, embedding matrix is `reg[2]`.

### NML-R — Reduction
```
RDUC  Rdest Rsrc [#type [#axis]]  — reduce: #0=sum, #1=mean, #2=max, #3=min
WHER  Rdest Rcond Rtrue [Rfalse]  — element-wise select: cond>0 ? true : false
CLMP  Rdest Rsrc [#min [#max]]    — clamp to range (default [0,1])
CMPR  Rdest Rsrc [#thresh [#op]]  — binary mask: 1 where src > thresh
```

### NML-S — Signal
```
FFT   Rdest Rinput Rconfig  — DFT: Rdest=real part, Rinput overwritten with imaginary
FILT  Rdest Rsrc Rcoeffs    — FIR filter (1D convolution with filter coefficients)
```

### NML-TR — Training (high-level)
```
TNET    #epochs #lr [#seed]      — train 2-layer net: R0=input, R1/R2=w1/b1, R3/R4=w2/b2, R9=target
TNDEEP  #epochs #lr #optimizer   — train N-layer net: RV=arch, R0=input, R1..=weights, R9=target
                                    #optimizer: 0=SGD, 1=Adam
LOSS    Rdest Rpred Rtarget #type — loss: #0=MSE, #1=cross-entropy, #2=MAE
BKWD    Rgrad Routput Rtarget    — output-layer gradient
WUPD    Rw Rw Rgrad              — weight update: w -= lr * grad
```

### NML-TR — Backward Opcodes
All follow: `*BK Rdest Rgrad Rinput` (gradient × local derivative)
```
RELUBK  Rd Rgrad Rpre   — grad * (pre > 0)
SIGMBK  Rd Rgrad Rpre   — grad * σ(pre) * (1 - σ(pre))
TANHBK  Rd Rgrad Rpre   — grad * (1 - tanh(pre)²)
GELUBK  Rd Rgrad Rpre   — grad * GELU'(pre)
SOFTBK  Rd Rgrad Rpre   — Jacobian-vector product through softmax
MMULBK  Rd_di Rd_dw Rgrad Rinput Rweight  — two outputs: d_input and d_weight
CONVBK  Rd_di Rd_dk Rgrad Rinput Rkernel  — two outputs: d_input and d_kernel
POOLBK  Rd Rgrad Rfwd_input [#size [#stride]]  — gradient through max positions only
NORMBK  Rd Rgrad Rinput  — full layer-norm gradient (assumes gamma=1, beta=0)
ATTNBK  Rd Rgrad Rq Rk Rv  — writes dQ into Rd, dK into Rd+1, dV into Rd+2
```

### NML-M2M — Multi-Agent
```
META  @field "value"         — self-description (no-op at runtime, --describe reads it)
FRAG  name / ENDF            — named program fragment block
LINK  @fragment              — inline a fragment
PTCH  @set/@del/@ins         — differential program patch
SIGN  agent=X key=Y sig=Z    — embed cryptographic signature
VRFY  @self @agent           — verify signature at runtime (TRAP if invalid)
VOTE  Rdest Rsrc #strategy   — consensus: #0=median, #1=mean, #2=quorum, #3=min, #4=max
PROJ  Rdest Rsrc Rmatrix     — L2-normalized projection: normalize(src @ matrix)
DIST  Rdest Ra Rb #metric    — distance: #0=cosine, #1=euclidean, #2=dot product
```

### NML-G — General Purpose
```
SYS  Rsrc #code   — system call: #0=print_num, #1=print_char, #2=read_num, #3=read_char, #4=time, #5=rand, #6=exit
MOD  Rdest Ra Rb  — element-wise modulo
ITOF Rdest Rsrc   — int32 → float32
FTOI Rdest Rsrc   — float32 → int32 (truncate)
BNOT Rdest Rsrc   — bitwise NOT
```

---

## Worked Example — F1 Pit Stop Duration Regression

**Task:** Predict pit stop duration from tire compounds, lap number, track temp.
**Features:** 12 inputs (5 one-hot tire-off + 5 one-hot tire-on + lap_norm + temp_norm)
**Architecture:** 12→16→8→1, ReLU/ReLU/Sigmoid, Adam

```
; Data file structure
@arch            shape=7      data=3,16,0,8,0,1,1
@w1              shape=12,16  (192 values, He init)
@b1              shape=1,16   (16 zeros)
@w2              shape=16,8   (128 values)
@b2              shape=1,8    (8 zeros)
@w3              shape=8,1    (8 values)
@b3              shape=1,1    (1 zero)
@training_data   shape=253,12 (3036 values)
@training_labels shape=253,1  (253 normalized durations)
@predict_input   shape=1,12   (12 values for one query)
```

The program is the train-then-infer pattern above verbatim (23 instructions total).
Output `@predicted_duration_norm` decoded: `value * 26.78 + 17.72 = seconds`.

---

## Sizing Guide

| Network | Weight tensors | Total params |
|---------|---------------|-------------|
| 6→8→1 (2 layer) | w1:6×8, b1:1×8, w2:8×1, b2:1×1 | 57 |
| 12→16→8→1 (3 layer) | w1:12×16, b1:1×16, w2:16×8, b2:1×8, w3:8×1, b3:1×1 | 353 |
| 28→64→32→7 (3 layer) | w1:28×64 … | 4,071 |

TNDEEP supports up to 10 layers. Max layer width: 1024.
