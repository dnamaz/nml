# NML Usage Guide

## A Complete Reference with Examples for All 67 Instructions

---

This document provides working examples for every NML instruction, organized by category. Each example can be run directly with the NML runtime:

```bash
make nml-v06                     # Build the v0.6 runtime
./nml-v06 program.nml data.nml.data    # Run a program
./nml-v06 program.nml --trace          # Run with instruction trace
./nml-v06 program.nml --describe       # Print metadata without executing
```

All examples are shown in classic syntax. The same programs work in symbolic or verbose syntax — see the [Tri-Syntax Reference](#tri-syntax-reference) at the end.

---

## Part 1: Core Instructions (35)

### Arithmetic (8 instructions)

#### MMUL — Matrix Multiply

Multiplies two tensors: `Rd = Rs1 @ Rs2`. The workhorse of neural network inference.

```
; Single dense layer: output = input @ weights
LD    R0 @input        ; shape [1, 4]
LD    R1 @weights      ; shape [4, 3]
MMUL  R2 R0 R1         ; R2 = [1, 3] — one row per input, one column per output neuron
ST    R2 @output
HALT
```

**When to use:** Every neural network layer, attention mechanism, and linear transformation. This is the most computationally expensive instruction and maps directly to GPU matrix-multiply units.

#### MADD — Element-wise Add

Adds two tensors element by element: `Rd = Rs1 + Rs2`. Used for bias addition in neural networks and accumulating tax amounts.

```
; Add bias to layer output
MMUL  R2 R0 R1         ; R2 = input @ weights
LD    R3 @bias         ; shape [1, 3]
MADD  R2 R2 R3         ; R2 = R2 + bias
```

#### MSUB — Element-wise Subtract

Subtracts tensors: `Rd = Rs1 - Rs2`. Used for computing marginal income in tax brackets.

```
; Marginal income = gross_pay - bracket_threshold
LD    R0 @gross_pay    ; $100,000
LEAF  R1 #54875.0      ; 22% bracket starts at $54,875
MSUB  R2 R0 R1         ; R2 = $45,125 (income in this bracket)
```

#### EMUL — Element-wise Multiply

Multiplies tensors element by element: `Rd = Rs1 * Rs2`. Useful for applying masks, attention weights, and per-element scaling.

```
; Apply attention weights to values
LD    R0 @attention_weights   ; shape [1, 4]
LD    R1 @values              ; shape [1, 4]
EMUL  R2 R0 R1               ; weighted values
```

#### EDIV — Element-wise Divide

Divides tensors: `Rd = Rs1 / Rs2`. Traps on division by zero.

```
; Normalize by dividing each element by its column sum
LD    R0 @data
LD    R1 @column_sums
EDIV  R2 R0 R1
```

#### SDOT — Scalar Dot Product

Computes dot product: `Rd = dot(Rs1, Rs2)`. Returns a scalar.

```
; Similarity between two vectors
LD    R0 @vector_a
LD    R1 @vector_b
SDOT  R2 R0 R1         ; R2 = scalar similarity score
```

#### SCLR — Scalar Multiply

Multiplies every element by a constant: `Rd = Rs1 * imm`. The primary instruction for applying tax rates.

```
; Apply 6.2% FICA rate
LD    R0 @gross_pay
SCLR  R1 R0 #0.062     ; R1 = gross_pay * 0.062
ST    R1 @fica_tax
```

**Tax usage:** Every flat-rate tax (FICA, Medicare, SDI, city taxes) uses SCLR with the rate as the immediate value. This is the most common instruction in the NML tax library.

#### SDIV — Scalar Divide

Divides every element by a constant: `Rd = Rs1 / imm`. Useful for averaging, annualizing.

```
; Convert annual salary to per-period pay
LD    R0 @annual_salary
SDIV  R1 R0 #26.0      ; R1 = biweekly pay (26 periods/year)
```

---

### Activation Functions (4 instructions)

These implement the nonlinearities used in neural networks.

#### RELU — Rectified Linear Unit

`Rd = max(0, Rs)`. The most common activation — zeroes out negative values.

```
; Hidden layer with ReLU activation
MMUL  R3 R0 R1         ; linear transform
MADD  R3 R3 R2         ; add bias
RELU  R3 R3            ; zero out negatives
```

**Why it matters:** ReLU is what makes neural networks nonlinear. Without it, stacking layers would just be one big matrix multiply.

#### SIGM — Sigmoid

`Rd = 1 / (1 + exp(-Rs))`. Squashes values to [0, 1]. Used for binary classification output.

```
; Anomaly score: 0 = normal, 1 = anomalous
MMUL  R9 R6 R7         ; final layer
MADD  R9 R9 R8         ; add bias
SIGM  R9 R9            ; squeeze to [0, 1]
ST    R9 @anomaly_score
```

#### TANH — Hyperbolic Tangent

`Rd = tanh(Rs)`. Squashes to [-1, 1]. Used in LSTMs and some attention mechanisms.

```
TANH  R3 R3            ; activation in range [-1, 1]
```

#### SOFT — Softmax

Converts a vector into a probability distribution (all positive, sums to 1).

```
; Classification: which of 10 classes?
MMUL  R5 R4 R3         ; logits for 10 classes
SOFT  R5 R5            ; probabilities summing to 1.0
ST    R5 @class_probs
```

---

### Memory (4 instructions)

#### LD — Load from Memory

Loads a tensor from a named memory slot into a register.

```
LD    R0 @sensor_data   ; load sensor readings
LD    R1 @weights       ; load pre-trained weights
```

Memory slots are defined in `.nml.data` files:
```
@sensor_data shape=1,4 data=0.9,0.1,0.95,0.3
@weights shape=4,3 data=0.2,-0.1,0.4,0.5,...
```

#### ST — Store to Memory

Stores a register value to a named memory slot. Used for outputting results.

```
ST    R9 @anomaly_score
ST    RA @tax_amount
```

After execution, all stored values are printed to stdout — this is how you get results out of an NML program.

#### MOV — Copy Register

Copies one register to another: `Rd = Rs`.

```
MOV   R7 R0            ; save original gross_pay before modifying R0
```

#### ALLC — Allocate Zero Tensor

Creates a zero-filled tensor with a given shape.

```
ALLC  RA #[1]           ; RA = scalar 0.0
ALLC  R0 #[3,4]         ; R0 = 3x4 zero matrix
ALLC  R1 #[1] f64       ; R1 = double-precision zero scalar
```

**Tax usage:** Every bracket-based tax program starts with `ALLC RA #[1]` to initialize the accumulator to zero.

---

### Data Flow (4 instructions)

#### RSHP — Reshape

Changes tensor dimensions without modifying data. Total element count must stay the same.

```
LD    R0 @data          ; shape [12]
RSHP  R1 R0 #[3,4]     ; reshape to 3x4 matrix
RSHP  R2 R0 #[2,6]     ; or 2x6 — same 12 elements
```

#### TRNS — Transpose

Swaps the last two dimensions of a 2D tensor.

```
LD    R0 @weights       ; shape [4, 3]
TRNS  R1 R0             ; R1 = shape [3, 4]
```

#### SPLT — Split

Splits a tensor into two parts along a dimension.

```
LD    R0 @data          ; shape [8]
SPLT  R1 R2 R0 #0 #4   ; R1 = first 4 elements, R2 = last 4
```

#### MERG — Merge (Concatenate)

Joins two tensors along a dimension.

```
MERG  R3 R1 R2 #0      ; R3 = concatenation of R1 and R2
```

---

### Comparison (3 instructions)

All comparison instructions set the global **condition flag** used by JMPT/JMPF.

#### CMPF — Compare Feature (Tree Models)

Sets flag = 1 if `Rs[feat] < thresh`. The core instruction for decision tree traversal.

```
; Is income < $54,875? (22% bracket threshold)
CMPF  RE R7 #0 #54875.0
JMPF  #10              ; no — skip to next bracket
; ... compute tax for this bracket ...
```

**Tax usage:** Every bracket-based tax program is a cascade of CMPF instructions checking income against thresholds. The FIT program has 21 CMPF instructions — 7 brackets x 3 filing statuses.

#### CMP — General Compare

Sets flag if `Rs1[0] < Rs2[0]`. For comparing two register values.

```
LEAF  R0 #42.0
LEAF  R1 #50.0
CMP   R0 R1            ; flag = 1 (42 < 50)
```

#### CMPI — Compare Immediate

Sets flag if `Rs[0] < imm`. Quick comparison against a constant.

```
CMPI  RE RD #6.0       ; is counter < 6?
JMPT  #-5              ; yes — loop back
```

---

### Control Flow (5 instructions)

#### JMPT — Jump if True

Jump forward or backward by offset if the condition flag is true.

```
CMPI  RE R0 #100.0     ; is R0 < 100?
JMPT  #3               ; yes — skip ahead 3 instructions
```

#### JMPF — Jump if False

Jump if the condition flag is false. Used in bracket tax processing.

```
; Check if income falls in this bracket
CMPF  RE R7 #0 #54875.0
JMPF  #10              ; income >= threshold, skip to this bracket's calculation
```

**Tax usage:** The FIT program pairs every CMPF with a JMPF to create a decision cascade. If income is above the threshold, fall through to compute the tax; otherwise jump ahead.

#### JUMP — Unconditional Jump

Always jump by offset. Used to skip over alternative branches.

```
; After computing tax for this bracket, skip remaining brackets
JUMP  #20              ; jump past all remaining bracket checks
```

#### LOOP / ENDP — Counted Loop

Execute a block a fixed number of times.

```
LOOP  #5               ; repeat 5 times
  TACC  RA RA RC       ; accumulate something
ENDP                   ; decrement counter, loop if > 0
```

---

### Subroutines (2 instructions)

#### CALL / RET — Subroutine Call and Return

CALL pushes the return address and jumps. RET pops and returns.

```
; Main program
LEAF  R1 #7.0
CALL  #2               ; call double_r1 (2 instructions ahead)
ST    R1 @result       ; result = 14.0
JUMP  #2               ; skip past subroutine
; Subroutine: double_r1
TACC  R1 R1 R1         ; R1 = R1 + R1
RET                    ; return to caller
HALT
```

Subroutines enable code reuse within a program. Max call depth: 32.

---

### Tree Model (2 instructions)

#### LEAF — Set Scalar Value

Stores an immediate value into a register. Named for decision tree leaf values but used for any constant.

```
LEAF  RA #0.0           ; initialize accumulator
LEAF  RC #29200.0       ; standard deduction for MFJ
LEAF  R0 #3.14159       ; pi
```

**Tax usage:** LEAF loads bracket thresholds, tax rates, standard deductions, and cumulative tax amounts. It's the second most common instruction after SCLR.

#### TACC — Tree Accumulate

Adds two registers: `Rd = Rs1 + Rs2`. Named for accumulating tree predictions across an ensemble.

```
; Accumulate tax from multiple brackets
LEAF  RC #1192.50       ; base tax for 12% bracket
TACC  RA RA RC          ; total_tax += base_tax_for_this_bracket
```

---

### System (3 instructions)

#### SYNC — Synchronization Barrier

No-op in single-threaded execution. Reserved for future multi-core NML processors.

#### HALT — Stop Execution

Required at the end of every program. The runtime prints all memory contents after HALT.

```
ST    RA @tax_amount
HALT                    ; done — results are printed
```

#### TRAP — Fault

Triggers a program error with a code. Used for assertion failures.

```
CMPI  RE RA #0.0        ; is tax_amount < 0?
JMPF  #1                ; no — skip trap
TRAP  #1                ; yes — this should never happen
```

---

## Part 2: Extension Instructions (14)

### NML-V: Vision (4 instructions)

For convolutional neural networks and image processing.

#### CONV — 2D Convolution

```
LD    R0 @image          ; shape [28, 28]
LD    R1 @kernel         ; shape [3, 3]
CONV  R2 R0 R1           ; 2D convolution
```

#### POOL — Max Pooling

```
POOL  R3 R2 #2 #2        ; 2x2 max pooling, stride 2
```

#### UPSC — Upscale

```
UPSC  R4 R3 #2           ; nearest-neighbor 2x upscale
```

#### PADZ — Zero Padding

```
PADZ  R5 R0 #1           ; pad 1 pixel of zeros on all sides
```

### NML-T: Transformer (4 instructions)

For transformer-based models (GPT, BERT, etc.).

#### ATTN — Scaled Dot-Product Attention

```
LD    R0 @queries        ; shape [seq_len, d_model]
LD    R1 @keys           ; shape [seq_len, d_model]
LD    R2 @values         ; shape [seq_len, d_model]
ATTN  R3 R0 R1 R2        ; R3 = softmax(Q @ K^T / sqrt(d)) @ V
```

This is the core of every transformer. One instruction replaces dozens of lines of Python.

#### NORM — Layer Normalization

```
LD    R4 @gamma          ; scale parameter
LD    R5 @beta           ; shift parameter
NORM  R3 R3 R4 R5        ; normalize activations
```

#### EMBD — Embedding Lookup

```
LD    R0 @vocab_table    ; shape [vocab_size, embed_dim]
LD    R1 @token_ids      ; shape [seq_len]
EMBD  R2 R0 R1           ; R2 = token embeddings
```

#### GELU — Gaussian Error Linear Unit

```
GELU  R3 R3              ; smoother alternative to ReLU, used in GPT
```

### NML-R: Reduction (4 instructions)

For aggregation and conditional operations.

#### RDUC — Reduce

```
LD    R0 @data           ; shape [1, 10]
RDUC  R1 R0 #0           ; R1 = sum of all elements
RDUC  R2 R0 #1           ; R2 = mean
RDUC  R3 R0 #2           ; R3 = max
RDUC  R4 R0 #3           ; R4 = min
```

#### WHER — Conditional Select

```
; R0 = condition (0/1 mask), R1 = value if true, R2 = value if false
WHER  R3 R0 R1 R2        ; R3[i] = R0[i] ? R1[i] : R2[i]
```

#### CLMP — Clamp

```
CLMP  R1 R0 #0.0 #1.0   ; clamp all values to [0, 1]
```

Useful for ensuring tax amounts are non-negative, probabilities stay in range.

#### CMPR — Element-wise Comparison Mask

```
CMPR  R1 R0 #0.5 #0     ; R1[i] = 1 if R0[i] > 0.5, else 0
```

### NML-S: Signal (2 instructions)

For signal processing and time-series analysis.

#### FFT — Discrete Fourier Transform

```
LD    R0 @time_signal
FFT   R1 R2 R0           ; R1 = real part, R2 = imaginary part
```

#### FILT — FIR Filter

```
LD    R0 @signal
LD    R1 @filter_coeffs
FILT  R2 R0 R1           ; 1D convolution (FIR filter)
```

---

## Part 3: M2M Extensions (11)

These instructions enable machine-to-machine communication — programs that describe themselves, compose with other programs, update differentially, sign their output, reach consensus, and communicate in embedding space.

### META — Self-Describing Programs

META declares what a program does, what it expects, and where it came from. It's a no-op at runtime — the runtime parses it during assembly and stores it in a descriptor.

```
META  @name         "00-000-0000-FICA-000"
META  @version      "2025.1"
META  @domain       "tax"
META  @input        gross_pay       currency    "Annual gross pay"
META  @input        is_exempt       bool        "Exempt from withholding"
META  @output       tax_amount      currency    "FICA tax amount"
META  @invariant    "tax_amount >= 0"
META  @provenance   "SSA wage base schedule, 2025"
META  @author       "ste_transpiler v2.3"

LD    R0 @gross_pay
SCLR  R1 R0 #0.062
ST    R1 @tax_amount
HALT
```

**Why it matters:** Without META, an NML program is opaque — you have to execute it or read the instructions to know what it does. With META, any agent can inspect the program's interface without running it. The `--describe` flag prints the descriptor:

```bash
./nml-v06 program.nml --describe
```

### Typed Register Annotations

Type annotations tell validators what kind of data each register holds. They don't change execution — they enable static analysis.

```
LD    R0:currency   @gross_pay        ; R0 holds a dollar amount
LD    R1:category   @filing_status    ; R1 holds an enum
SCLR  R2:currency   R0 #0.062        ; currency * ratio = currency (OK)
ST    R2:currency   @tax_amount

; This would trigger a semantic warning:
; EMUL  R3:currency  R0 R2            ; currency * currency = ???
```

**Available types:** `float` (default), `currency`, `ratio`, `category`, `count`, `bool`, `embedding`, `probability`

**Why it matters:** In a multi-agent system, Agent A produces an NML program and Agent B validates it. Types let Agent B catch errors like "you're multiplying a dollar amount by another dollar amount" without executing the program.

### FRAG / ENDF — Compositional Fragments

Fragments are named sub-programs that can be independently produced, validated, and composed.

```
; Fragment 1: FICA calculation (could be produced by Agent A)
FRAG  fica_tax
META  @input   gross_pay    currency
META  @output  fica_amount  currency
LD    R0 @gross_pay
ALLC  RA #[1]
SCLR  RA R0 #0.062
ST    RA @fica_amount
ENDF

; Fragment 2: Medicare calculation (could be produced by Agent B)
FRAG  medicare_tax
META  @input   gross_pay        currency
META  @output  medicare_amount  currency
LD    R0 @gross_pay
SCLR  RB R0 #0.0145
ST    RB @medicare_amount
ENDF

; Fragment 3: Composed payroll tax (Agent C composes the above)
FRAG  payroll_total
LINK  @fica_tax
LINK  @medicare_tax
TACC  RC RA RB
ST    RC @total_payroll_tax
ENDF
```

**Why it matters:** Different agents can specialize in different tax domains. One agent handles federal, another handles state, a third composes them. Each fragment is independently testable and validates against its declared inputs/outputs.

### PTCH — Differential Programs

When a tax law changes, you don't need to regenerate the entire program — just send the diff.

```
; This patch updates the 2025 FIT standard deductions for 2026
PTCH  @base   sha256:7fe8412b7c3e...
PTCH  @set    9   "LEAF  RC #8800.00"      ; was #8600.00
PTCH  @set    13  "LEAF  RC #13200.00"     ; was #12900.00
PTCH  @end
```

Apply with:
```bash
./nml-v06 --patch fit_2025.nml fit_2026_patch.nml
```

**Why it matters:** The full FIT program is 201 lines. A tax year update typically changes 10-20 threshold values. Sending a 10-line patch instead of a 201-line program is 20x more efficient — critical for swarm agent communication where thousands of jurisdictions update simultaneously.

### SIGN / VRFY — Cryptographic Signing

SIGN embeds a cryptographic signature. VRFY checks it at runtime.

```
; Signed by the transpiler agent
SIGN  agent=transpiler_v2.3  key=hmac-sha256:a1b2c3d4  sig=e5f6a7b8...

META  @name  "00-000-0000-FICA-000"
LD    R0 @gross_pay
SCLR  R1 R0 #0.062
ST    R1 @tax_amount

; Verify before using the result in a critical pipeline
VRFY  @self @transpiler_v2.3
HALT
```

Sign from Python:
```python
from nml_signing import generate_keypair, sign_program
pub, priv = generate_keypair()
sign_line = sign_program(nml_text, priv, "transpiler_v2.3")
signed_program = sign_line + "\n" + nml_text
```

**Why it matters:** In a trustless multi-agent system, how does the Validator Agent know the Transpiler Agent actually produced a program? Signatures provide cryptographic proof. If an NML program is tampered with after signing, VRFY will TRAP.

### VOTE — Multi-Agent Consensus

When multiple agents independently compute the same result, VOTE reconciles them.

```
; 5 agents each computed FICA tax — collect their answers
LD    R0 @agent_results     ; [6200.00, 6200.01, 6199.98, 6200.00, 6150.00]

; Median: robust to outliers
VOTE  RA R0 #0              ; RA = 6200.00 (the middle value)

; Mean: sensitive to outliers
VOTE  RB R0 #1              ; RB = 6189.998

; Quorum: do at least 3 agree?
VOTE  RE R0 #2 #3           ; RE = 1.0 (yes, 4 of 5 are within 0.01)

ST    RA @consensus_tax
ST    RE @quorum_passed
HALT
```

**Strategies:**
| Code | Name | Use Case |
|------|------|----------|
| 0 | median | Default — robust to single outlier |
| 1 | mean | When all agents are equally trusted |
| 2 | quorum | Verify agreement threshold before proceeding |
| 3 | min | Conservative estimate |
| 4 | max | Upper bound estimate |

**Why it matters:** No single LLM is 100% reliable. Running the same tax computation through 3-5 independent agents and taking the median is more trustworthy than trusting any single agent. VOTE makes this a first-class operation.

### PROJ — Embedding Projection

Projects a tensor into an embedding space with L2 normalization.

```
LD    R0 @feature_vector      ; shape [1, 128] — raw features
LD    R1 @projection_matrix   ; shape [128, 64] — learned projection
PROJ  R2 R0 R1                ; R2 = normalize(R0 @ R1), shape [1, 64]
ST    R2 @embedding
```

PROJ is `Rd = normalize(Rs @ Rmatrix)` — a matrix multiply followed by L2 normalization so the output has unit norm. This is how machines convert raw data into a shared semantic space.

### DIST — Embedding Distance

Measures how similar two embeddings are.

```
LD    R0 @embedding_a         ; from Agent A
LD    R1 @embedding_b         ; from Agent B
DIST  R2 R0 R1 #0             ; cosine distance
DIST  R3 R0 R1 #1             ; euclidean distance
DIST  R4 R0 R1 #2             ; dot product (similarity, not distance)
ST    R2 @similarity
```

**Metrics:**
| Code | Name | Range | Interpretation |
|------|------|-------|----------------|
| 0 | cosine | [0, 2] | 0 = identical, 1 = orthogonal, 2 = opposite |
| 1 | euclidean | [0, inf] | 0 = identical |
| 2 | dot product | [-1, 1] | 1 = identical (for unit vectors) |

**Why it matters:** PROJ and DIST together enable latent-space communication. Instead of agents exchanging text (which is ambiguous) or NML instructions (which are rigid), they can exchange embeddings — dense vector representations that capture the *meaning* of a computation. Two agents that independently derive the same tax rule will produce similar embeddings, detectable by DIST without comparing instructions line by line.

---

## Part 4: Complete Examples

### Example 1: A Complete Tax Calculation

This program computes FICA Social Security tax with the 2025 wage base cap.

```
META  @name       "fica_2025"
META  @version    "2025.1"
META  @domain     "tax"
META  @input      gross_pay   currency
META  @input      is_exempt   bool
META  @output     tax_amount  currency
META  @invariant  "tax_amount >= 0"
META  @provenance "SSA 2025 wage base: $176,100"

; Load inputs
LD    R0 @gross_pay
LD    R3 @is_exempt

; Check exemption
CMPF  RE R3 #0 #0.5
JMPF  #10               ; not exempt — continue

; Initialize accumulator
ALLC  RA #[1]

; Check if income exceeds wage base
CMPF  RE R0 #0 #176100.0
JMPF  #4                ; income >= cap

; Income below cap: tax = gross * 6.2%
SCLR  RC R0 #0.062
TACC  RA RA RC
JUMP  #2

; Income above cap: tax = cap * 6.2%
LEAF  RC #176100.0
SCLR  RC RC #0.062
TACC  RA RA RC

; Store result
ST    RA @tax_amount
HALT
```

Run with:
```bash
echo "@gross_pay shape=1 data=100000.0
@is_exempt shape=1 data=0.0" > /tmp/fica.data

./nml-v06 fica.nml /tmp/fica.data
# Output: tax_amount = 6200.00
```

### Example 2: Neural Network Anomaly Detector

A complete 3-layer neural network for sensor anomaly detection.

```
META  @name    "anomaly_detector"
META  @domain  "iot"
META  @input   sensor_data   float
META  @output  anomaly_score probability

; Layer 1: input → hidden (4 → 8)
LD    R0 @sensor_data
LD    R1 @w1
LD    R2 @b1
MMUL  R3 R0 R1
MADD  R3 R3 R2
RELU  R3 R3

; Layer 2: hidden → hidden (8 → 4)
LD    R4 @w2
LD    R5 @b2
MMUL  R6 R3 R4
MADD  R6 R6 R5
RELU  R6 R6

; Layer 3: hidden → output (4 → 1)
LD    R7 @w3
LD    R8 @b3
MMUL  R9 R6 R7
MADD  R9 R9 R8
SIGM  R9 R9

ST    R9 @anomaly_score
HALT
```

18 instructions. A deployable neural network.

### Example 3: Multi-Agent Tax Consensus

Five independent agents compute FICA tax. This program reconciles their results.

```
META  @name    "fica_consensus"
META  @domain  "tax"
META  @input   agent_results  float
META  @output  final_tax      currency
META  @output  agreement      bool
META  @author  "orchestrator"

; Load results from 5 agents
LD    R0 @agent_results

; Take the median (robust to one outlier)
VOTE  RA R0 #0

; Check if at least 4 of 5 agree
VOTE  RE R0 #2 #4

; Store results
ST    RA @final_tax
ST    RE @agreement
HALT
```

### Example 4: Embedding Similarity Between Tax Programs

Two agents interpreted the same regulation. How similar are their interpretations?

```
META  @name    "regulation_comparison"
META  @domain  "tax"
META  @input   features_agent_a  float
META  @input   features_agent_b  float
META  @input   projection        float
META  @output  similarity        float

; Load feature vectors from two agents
LD    R0 @features_agent_a
LD    R1 @features_agent_b
LD    R2 @projection

; Project both into shared embedding space
PROJ  R3 R0 R2
PROJ  R4 R1 R2

; How similar are they?
DIST  R5 R3 R4 #0       ; cosine distance

ST    R5 @similarity
HALT
```

If similarity < 0.05, the agents agree. If > 0.3, they disagree and should escalate to a human reviewer.

---

## Part 5: NML-G General Purpose Extensions (5)

NML-G adds console I/O, integer math, and type conversion — everything needed for general-purpose programs beyond ML inference and tax calculations. See [NML_G_Spec.md](NML_G_Spec.md) for the full specification.

### SYS — System Call

The single most impactful addition. SYS multiplexes all host I/O through one opcode with a numeric code.

```
; Print a number
LEAF  R0 #42.0
SYS   R0 #0             ; prints "42.0" to stdout

; Print a character
LEAF  R0 #72             ; ASCII 'H'
SYS   R0 #1             ; prints "H"

; Read a number from stdin
SYS   R0 #2             ; waits for input, stores in R0

; Get current time
SYS   R0 #4             ; R0 = seconds since epoch

; Generate random number
SYS   R0 #5             ; R0 = random float in [0, 1)
```

| Code | Name | Behavior |
|------|------|----------|
| 0 | PRINT_NUM | Print Rd[0] as number + newline |
| 1 | PRINT_CHAR | Print Rd[0] as ASCII character |
| 2 | READ_NUM | Read number from stdin into Rd |
| 3 | READ_CHAR | Read character from stdin into Rd |
| 4 | TIME | Wall-clock time into Rd |
| 5 | RAND | Random [0,1) into Rd |
| 6 | EXIT | Terminate with code Rd[0] |

**Why it matters:** Without SYS, NML programs could only communicate through memory slots printed after HALT. SYS enables interactive programs, streaming output, and runtime I/O — the minimum needed for general-purpose computing.

### MOD — Integer Modulo

Essential for algorithms: FizzBuzz, primality testing, hash functions, cycle detection.

```
; Is N divisible by 3?
LEAF  R0 #15.0
LEAF  R1 #3.0
MOD   R2 R0 R1           ; R2 = 0 (15 % 3 = 0)
CMPI  RE R2 #0.5         ; R2 < 0.5? (i.e. R2 == 0)
; flag = 1 → yes, divisible
```

### ITOF / FTOI — Type Conversion

Convert between integer and floating-point tensors.

```
; Convert loop counter (i32) to float for math
ALLC  R0 #[1] i32
LEAF  R0 #65             ; ASCII 'A' as integer
ITOF  R1 R0              ; R1 = 65.0 (float)

; Truncate float to integer for indexing
LEAF  R0 #3.7
FTOI  R1 R0              ; R1 = 3 (truncated)
```

### BNOT — Bitwise NOT

Bitwise complement of integer values.

```
LEAF  R0 #0
BNOT  R1 R0              ; R1 = ~0 = -1 (all bits set)
```

### Complete Example: Fibonacci Sequence (13 instructions)

```
LEAF  R0 #0.0            ; a = 0
LEAF  R1 #1.0            ; b = 1
LEAF  RD #0.0            ; counter = 0
LEAF  R5 #20.0           ; limit = 20
SYS   R0 #0              ; print current fib number
TACC  R2 R0 R1           ; next = a + b
MOV   R0 R1              ; a = b
MOV   R1 R2              ; b = next
LEAF  RC #1.0
TACC  RD RD RC            ; counter++
CMP   RD R5              ; counter < limit?
JMPT  #-8                ; yes → loop back
HALT
```

Output: 0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597, 2584, 4181

### Complete Example: Prime Numbers 2-50 (23 instructions)

```
LEAF  RD  #2.0            ; n = 2
LEAF  R9  #51.0           ; limit
LEAF  R1  #2.0            ; d = 2
LEAF  RA  #1.0            ; is_prime = 1
EMUL  R3 R1 R1            ; d²
CMP   RD R3               ; n < d²?
JMPT  #8                  ; → check is_prime
MOD   R0 RD R1            ; n % d
CMPI  RE R0 #0.5          ; == 0?
JMPF  #2                  ; → next d
LEAF  RA #0.0             ; not prime
JUMP  #3                  ; → check is_prime
LEAF  RC #1.0
TACC  R1 R1 RC            ; d++
JUMP  #-11                ; → inner loop
CMPI  RE RA #0.5          ; is_prime?
JMPT  #1                  ; no → skip
SYS   RD #0               ; print prime
LEAF  RC #1.0
TACC  RD RD RC            ; n++
CMP   RD R9               ; n < 51?
JMPT  #-20                ; → outer loop
HALT
```

Output: 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47

### Build Options

```bash
# Full runtime with NML-G (default)
make nml-gp

# Without NML-G
gcc -DNML_NO_GENERAL -o nml nml.c -lm

# With expanded limits for larger programs
gcc -O2 -o nml-gp nml.c -lm \
    -DNML_MAX_INSTRUCTIONS=65536 \
    -DNML_MAX_MEMORY_SLOTS=256 \
    -DNML_MAX_CALL_DEPTH=128
```

---

## Part 6: Tri-Syntax Reference

Every program above works in all three syntaxes. Here's the FICA calculation in each:

**Classic:**
```
SCLR  R1 R0 #0.062
ST    R1 @tax_amount
HALT
```

**Symbolic:**
```
∗  κ  ι  #0.062
↑  κ  @tax_amount
◼
```

**Verbose:**
```
SCALE              R1 R0 #0.062
STORE              R1 @tax_amount
STOP
```

All three produce identical bytecode. Use classic for documentation, symbolic for compact machine consumption, verbose for auditors.

### Compact Form

Any NML program can be collapsed to a single line using `¶` (U+00B6, pilcrow) as the instruction delimiter. The runtime parses `¶` natively — no expansion step needed.

**Standard (multi-line):**
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

Compact form strips comments and normalizes whitespace. The structural pair is `¶` (instruction boundary) and `◼` (program end).

**Convert between forms:**
```bash
python3 nml_format.py compact program.nml              # → single line
python3 nml_format.py format  "∎ α #42¶↑ α @r¶◼"      # → multi-line
```

Use compact form for JSON payloads, agent messages, API fields, and any context where newlines are inconvenient. Use multi-line for human reading, debugging, and documentation.

### Full Opcode Cross-Reference

| # | Classic | Symbolic | Verbose | Category |
|---|---------|----------|---------|----------|
| 1 | MMUL | × | MATRIX_MULTIPLY | Arithmetic |
| 2 | MADD | ⊕ | ADD | Arithmetic |
| 3 | MSUB | ⊖ | SUBTRACT | Arithmetic |
| 4 | EMUL | ⊗ | ELEMENT_MULTIPLY | Arithmetic |
| 5 | EDIV | ⊘ | ELEMENT_DIVIDE | Arithmetic |
| 6 | SDOT | · | DOT_PRODUCT | Arithmetic |
| 7 | SCLR | ∗ | SCALE | Arithmetic |
| 8 | SDIV | ÷ | DIVIDE | Arithmetic |
| 9 | RELU | ⌐ | RELU | Activation |
| 10 | SIGM | σ | SIGMOID | Activation |
| 11 | TANH | τ | TANH | Activation |
| 12 | SOFT | Σ | SOFTMAX | Activation |
| 13 | LD | ↓ | LOAD | Memory |
| 14 | ST | ↑ | STORE | Memory |
| 15 | MOV | ← | COPY | Memory |
| 16 | ALLC | □ | ALLOCATE | Memory |
| 17 | RSHP | ⊞ | RESHAPE | Data Flow |
| 18 | TRNS | ⊤ | TRANSPOSE | Data Flow |
| 19 | SPLT | ⊢ | SPLIT | Data Flow |
| 20 | MERG | ⊣ | MERGE | Data Flow |
| 21 | CMPF | ⋈ | COMPARE_FEATURE | Comparison |
| 22 | CMP | ≶ | COMPARE | Comparison |
| 23 | CMPI | ≺ | COMPARE_VALUE | Comparison |
| 24 | JMPT | ↗ | BRANCH_TRUE | Control |
| 25 | JMPF | ↘ | BRANCH_FALSE | Control |
| 26 | JUMP | → | JUMP | Control |
| 27 | LOOP | ↻ | REPEAT | Control |
| 28 | ENDP | ↺ | END_REPEAT | Control |
| 29 | CALL | ⇒ | CALL | Subroutine |
| 30 | RET | ⇐ | RETURN | Subroutine |
| 31 | LEAF | ∎ | SET_VALUE | Tree |
| 32 | TACC | ∑ | ACCUMULATE | Tree |
| 33 | SYNC | ⏸ | BARRIER | System |
| 34 | HALT | ◼ | STOP | System |
| 35 | TRAP | ⚠ | FAULT | System |
| 36 | CONV | ⊛ | CONVOLVE | NML-V |
| 37 | POOL | ⊓ | MAX_POOL | NML-V |
| 38 | UPSC | ⊔ | UPSCALE | NML-V |
| 39 | PADZ | ⊡ | ZERO_PAD | NML-V |
| 40 | ATTN | ⊙ | ATTENTION | NML-T |
| 41 | NORM | ‖ | LAYER_NORM | NML-T |
| 42 | EMBD | ⊏ | EMBED | NML-T |
| 43 | GELU | ℊ | GELU | NML-T |
| 44 | RDUC | ⊥ | REDUCE | NML-R |
| 45 | WHER | ⊻ | WHERE | NML-R |
| 46 | CLMP | ⊧ | CLAMP | NML-R |
| 47 | CMPR | ⊜ | MASK_COMPARE | NML-R |
| 48 | FFT | ∿ | FOURIER | NML-S |
| 49 | FILT | ⋐ | FILTER | NML-S |
| 50 | META | § | METADATA | NML-M2M |
| 51 | FRAG | ◆ | FRAGMENT | NML-M2M |
| 52 | ENDF | ◇ | END_FRAGMENT | NML-M2M |
| 53 | LINK | — | IMPORT | NML-M2M |
| 54 | PTCH | ⊿ | PATCH | NML-M2M |
| 55 | SIGN | ✦ | SIGN_PROGRAM | NML-M2M |
| 56 | VRFY | ✓ | VERIFY_SIGNATURE | NML-M2M |
| 57 | VOTE | ⚖ | CONSENSUS | NML-M2M |
| 58 | PROJ | ⟐ | PROJECT | NML-M2M |
| 59 | DIST | ⟂ | DISTANCE | NML-M2M |
| 60 | GATH | ⊃ | GATHER | NML-M2M |
| 61 | SCAT | ⊂ | SCATTER | NML-M2M |
| 62 | SYS | ⚙ | SYSTEM | NML-G |
| 63 | MOD | % | MODULO | NML-G |
| 64 | ITOF | ⊶ | INT_TO_FLOAT | NML-G |
| 65 | FTOI | ⊷ | FLOAT_TO_INT | NML-G |
| 66 | BNOT | ¬ | BITWISE_NOT | NML-G |

### Register Cross-Reference

| Index | Classic | Greek | Verbose | Purpose |
|-------|---------|-------|---------|---------|
| 0 | R0 | ι | R0 | General |
| 1 | R1 | κ | R1 | General |
| 2 | R2 | λ | R2 | General |
| 3 | R3 | μ | R3 | General |
| 4 | R4 | ν | R4 | General |
| 5 | R5 | ξ | R5 | General |
| 6 | R6 | ο | R6 | General |
| 7 | R7 | π | R7 | General |
| 8 | R8 | ρ | R8 | General |
| 9 | R9 | ς | R9 | General |
| 10 | RA | α | ACCUMULATOR | Accumulator |
| 11 | RB | β | GENERAL | General |
| 12 | RC | γ | SCRATCH | Scratch |
| 13 | RD | δ | COUNTER | Counter |
| 14 | RE | φ | FLAG | Condition flag |
| 15 | RF | ψ | STACK | Stack pointer |
