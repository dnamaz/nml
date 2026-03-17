# NML — Getting Started

A 5-minute introduction to writing and running NML programs.

## Build

```bash
git clone <repo-url>
cd nml
make
```

This produces a single `nml` binary (~83 KB).

## Your First Program

Create `my_program.nml`:

```
LEAF  R0 #10.0
LEAF  R1 #3.0
TACC  RA R0 R1
ST    RA @result
HALT
```

Run it:

```bash
./nml my_program.nml
```

Output:

```
result: shape=[1] dtype=f64 data=[13.0000]
```

This program sets R0 to 10, R1 to 3, adds them into the accumulator (RA), stores the result, and halts.

## Key Concepts

### Registers

NML has 32 tensor registers. Each register holds an entire tensor (scalar, vector, or matrix).

| Register | Purpose |
|----------|---------|
| R0-R9 | General purpose |
| RA | Accumulator |
| RB | General |
| RC | Scratch / temporaries |
| RD | Loop counter |
| RE | Condition flag (set by CMP/CMPI/CMPF) |
| RF | Stack pointer |
| RG | Gradient1 (backpropagation) |
| RH | Gradient2 (backpropagation) |
| RI | Gradient3 (backpropagation) |
| RJ | Learning rate |
| RK-RV | Training workspace / hive collective |

### Two Ways to Get Values Into Registers

- **`LEAF R0 #42.0`** — Load an immediate constant (the number is in the instruction)
- **`LD R0 @name`** — Load from a named memory address (defined in a `.nml.data` file)

This is the most important distinction in NML. Constants use `LEAF` with `#`. Memory uses `LD` with `@`.

### Programs Must End With HALT

Every NML program must contain a `HALT` instruction. Without it, execution continues past the last instruction.

## Opcode Quick Reference

### Arithmetic
| Opcode | Symbolic | Description | Example |
|--------|----------|-------------|---------|
| `MMUL` | `×` | Matrix multiply | `MMUL R3 R0 R1` |
| `MADD` | `⊕` | Element-wise add | `MADD R2 R0 R1` |
| `MSUB` | `⊖` | Element-wise subtract | `MSUB R2 R0 R1` |
| `EMUL` | `⊗` | Element-wise multiply | `EMUL R2 R0 R1` |
| `EDIV` | `⊘` | Element-wise divide | `EDIV R2 R0 R1` |
| `SDOT`/`DOT` | `·` | Dot product | `SDOT R2 R0 R1` |
| `SCLR` | `∗` | Scalar multiply | `SCLR R1 R0 #2.0` |
| `SDIV` | `÷` | Scalar divide | `SDIV R1 R0 #4.0` |

### Activation
| Opcode | Symbolic | Description |
|--------|----------|-------------|
| `RELU` | `⌐` | ReLU: max(0, x) |
| `SIGM` | `σ` | Sigmoid: 1/(1+exp(-x)) |
| `TANH` | `τ` | Hyperbolic tangent |
| `SOFT` | `Σ` | Softmax |
| `GELU` | `ℊ` | GELU (used in transformers) |

### Memory
| Opcode | Symbolic | Description | Example |
|--------|----------|-------------|---------|
| `LD` | `↓` | Load from memory | `LD R0 @input` |
| `ST` | `↑` | Store to memory | `ST R0 @result` |
| `MOV` | `←` | Copy register | `MOV R1 R0` |
| `ALLC` | `□` | Allocate zero tensor | `ALLC R0 #[4]` |
| `LEAF` | `∎` | Load constant | `LEAF R0 #42.0` |

### Control Flow
| Opcode | Symbolic | Description | Example |
|--------|----------|-------------|---------|
| `CMPI` | `≺`/`ϟ` | Compare vs immediate | `CMPI RE R0 #50.0` |
| `CMP` | `≶` | Compare two registers | `CMP R0 R1` |
| `CMPF` | `⋈` | Tree feature compare | `CMPF RE R0 #3 #100.0` |
| `JMPT` | `↗` | Jump if flag true | `JMPT #2` |
| `JMPF` | `↘` | Jump if flag false | `JMPF #3` |
| `JUMP` | `→` | Unconditional jump | `JUMP #2` |
| `LOOP` | `↻` | Begin counted loop | `LOOP R0` |
| `ENDP` | `↺` | End loop | `ENDP` |
| `CALL` | `⇒` | Call subroutine | `CALL #2` |
| `RET` | `⇐` | Return | `RET` |

### Tree Model
| Opcode | Symbolic | Description | Example |
|--------|----------|-------------|---------|
| `LEAF` | `∎` | Set constant value | `LEAF RA #500.0` |
| `TACC` | `∑` | Scalar accumulate (add) | `TACC RA R0 R1` |

### System
| Opcode | Symbolic | Description |
|--------|----------|-------------|
| `HALT` | `◼` | Stop execution |
| `TRAP` | `⚠` | Trigger error |
| `SYNC` | `⏸` | Synchronization barrier |
| `SYS` | `⚙` | System call (print, read, time, rand) |
| `MOD` | `%` | Integer modulo |

### Data Flow
| Opcode | Symbolic | Description |
|--------|----------|-------------|
| `RSHP` | `⊞` | Reshape tensor |
| `TRNS` | `⊤` | Transpose matrix |
| `SPLT` | `⊢` | Split tensor |
| `MERG` | `⊣` | Merge/concatenate tensors |

### Extensions (Vision, Transformer, Reduction, Signal, M2M)
| Opcode | Category | Description |
|--------|----------|-------------|
| `CONV` `POOL` `UPSC` `PADZ` | Vision | Convolution, pooling, upscale, padding |
| `ATTN` `NORM` `EMBD` | Transformer | Attention, layer norm, embedding lookup |
| `RDUC` | Reduction | Reduce: sum, mean, max, min |
| `WHER` `CLMP` `CMPR` | Reduction | Conditional select, clamp, mask |
| `FFT` `FILT` | Signal | Fourier transform, FIR filter |
| `META` `FRAG` `LINK` | M2M | Metadata, fragments, composition |
| `SIGN` `VRFY` | M2M | Cryptographic signing |
| `VOTE` `PROJ` `DIST` | M2M | Consensus, projection, distance |
| `GATH` `SCAT`/`SCTR` | M2M | Tensor index operations |

### Training
| Opcode | Symbolic | Description | Example |
|--------|----------|-------------|---------|
| `BKWD` | `∇` | Backpropagation step | `BKWD RG R3 R9` |
| `WUPD` | `⟳` | Weight update (gradient descent) | `WUPD R1 R1 RG` |
| `LOSS` | `△` | Compute loss (MSE/CE/MAE) | `LOSS R5 R3 R9 #0` |
| `TNET` | `⥁` | Train 2-layer network in-place | `TNET #2000 #0.001 #0` |
| `TNDEEP` | `⥁ˈ` | Train N-layer network (arch from RV) | `TNDEEP #2000 #0.005 #1` |
| `RELUBK` | `⌐ˈ` | ReLU backward pass | `RELUBK R7 RG R2` |
| `SIGMBK` | `σˈ` | Sigmoid backward pass | `SIGMBK R7 RG R2` |
| `TANHBK` | `τˈ` | Tanh backward pass | `TANHBK R7 RG R2` |
| `GELUBK` | `ℊˈ` | GELU backward pass | `GELUBK R7 RG R2` |
| `SOFTBK` | `Σˈ` | Softmax backward pass | `SOFTBK R7 RG R2` |
| `MMULBK` | `×ˈ` | MatMul backward (d_input + d_weight) | `MMULBK R7 R8 RG R0 R1` |
| `POOLBK` | `⊓ˈ` | Max pool backward | `POOLBK R9 RG R2 #2 #2` |
| `NORMBK` | `‖ˈ` | Layer norm backward | `NORMBK R7 RG R4` |
| `ATTNBK` | `⊙ˈ` | Attention backward (writes dQ/dK/dV) | `ATTNBK R8 RG R4 R5 R6` |

## Data Files

NML programs load data from `.nml.data` files:

```
@input shape=1,4 data=0.9,0.1,0.95,0.3
@weights shape=4,3 data=0.2,-0.1,0.4,0.5,0.3,-0.2,-0.1,0.6,0.1,0.3,-0.4,0.5
@bias shape=1,3 data=0.1,-0.1,0.05
```

Each line defines a named memory address with a shape and data values. Run with:

```bash
./nml program.nml data.nml.data
```

## Common Patterns

### Neural Network Layer

```
LD    R0 @input
LD    R1 @weights
LD    R2 @bias
MMUL  R3 R0 R1       ; matrix multiply
MADD  R3 R3 R2       ; add bias
RELU  R3 R3          ; activate
ST    R3 @output
HALT
```

### Conditional (if/else)

```
LD    R0 @value
CMPI  RE R0 #100.0   ; flag = (value < 100)
JMPF  #3             ; if false, skip to else
SCLR  R1 R0 #0.1     ; then: scale by 0.1
JUMP  #2             ; skip else
SCLR  R1 R0 #0.05    ; else: scale by 0.05
ST    R1 @result
HALT
```

### Loop (sum 1 to N)

```
LEAF  R0 #10          ; N = 10
ALLC  R1 #[1]         ; sum = 0
LEAF  R2 #1           ; increment = 1
LOOP  R0              ; repeat 10 times
TACC  R1 R1 R2        ;   sum += 1
ENDP
ST    R1 @result      ; result = 10
HALT
```

### Subroutine

```
LEAF  R0 #7.0         ; input value
CALL  #2              ; call subroutine (offset +2)
ST    R0 @result      ; store doubled value
HALT
SCLR  R0 R0 #2.0     ; subroutine: double R0
RET
```

Note: `CALL #N` jumps to PC + N + 1. From line 1, `CALL #2` lands on line 4.

### Symbolic Syntax

The same neural network layer in symbolic NML:

```
↓  ι  @input
↓  κ  @weights
↓  λ  @bias
×  μ  ι  κ
⊕  μ  μ  λ
⌐  μ  μ
↑  μ  @output
◼
```

### Self-Training (TNET — 2-layer)

```
LD    R1 @w1
LD    R2 @b1
LD    R3 @w2
LD    R4 @b2
LD    R0 @input
LD    R9 @target
TNET  #2000  #0.001  #0
ST    RA @result
HALT
```

TNET trains the 2-layer network defined by R1–R4 using input R0 and target R9. After training, RA holds the final prediction.

### Deep Training (TNDEEP — N-layer)

For networks deeper than 2 layers. The architecture is declared in register RV as a flat descriptor tensor.

```
; Declare architecture: 3 layers — 12→16 (ReLU), 16→8 (ReLU), 8→1 (Sigmoid)
; @arch shape=7 data=3,16,0,8,0,1,1  ← in your .nml.data file

LD    RV @arch            ; REQUIRED before TNDEEP
LD    R1 @w1              ; 12×16
LD    R2 @b1              ; 1×16
LD    R3 @w2              ; 16×8
LD    R4 @b2              ; 1×8
LD    R5 @w3              ; 8×1
LD    R6 @b3              ; 1×1
LD    R0 @training_data   ; N×12
LD    R9 @training_labels ; N×1
TNDEEP  #2000  #0.005  #1 ; #1 = Adam optimizer (#0 = SGD)
ST    R8 @loss            ; R8 = final MSE loss
HALT
```

Activation codes in arch descriptor: `0`=ReLU, `1`=Sigmoid, `2`=Tanh, `3`=GELU.
After training, R1–R6 hold the trained weights. Use them for a manual forward pass.

## Fraud Detection Example (Train + Infer + Decide)

A complete ML pipeline in 23 instructions:

```bash
./nml programs/fraud_detection.nml programs/fraud_detection.nml.data
```

The program trains a 6→8→1 network on labeled transactions using TNET, classifies a new transaction, and flags it as fraud if the score >= 0.5. See `programs/fraud_detection.nml` for the full source.

## Signed Programs (M2M)

Build with crypto support to sign and verify programs:

```bash
make nml-crypto

# Sign a program
./nml-crypto --sign programs/fraud_detection.nml --key deadbeef01020304 --agent authority > signed.nml

# Verify and execute (tampered programs are rejected)
./nml-crypto signed.nml programs/fraud_detection.nml.data

# Run the full distributed demo
bash demos/distributed_fraud.sh
```

## Command-Line Options

```bash
./nml program.nml [data.nml.data] [--trace] [--max-cycles N] [--fragment NAME]
```

| Flag | Description |
|------|-------------|
| `--trace` | Print each instruction as it executes |
| `--max-cycles N` | Override the default 1,000,000 cycle limit |
| `--fragment NAME` | Run only the named fragment (for FRAG/LINK programs) |
| `--describe` | Print META metadata without executing |

With `nml-crypto` build:

| Flag | Description |
|------|-------------|
| `--sign` | Sign a program (requires `--key`) |
| `--key <hex>` | HMAC key in hex for signing |
| `--agent <name>` | Signer identity for SIGN header |
| `--patch <file>` | Apply differential patch (@set/@del/@ins) |

## Next Steps

- [NML Specification](NML_SPEC.md) — full instruction set reference with encoding details
- [Usage Guide](NML_Usage_Guide.md) — all 82 instructions with detailed examples
- [M2M Specification](NML_M2M_Spec.md) — machine-to-machine extensions (SIGN/VRFY, FRAG/LINK, PTCH)
