# NML Usage Guide

## A Complete Reference with Examples for All 82 Instructions

---

This document provides working examples for every NML instruction, organized by category. Each example can be run directly with the NML runtime:

```bash
make nml                              # Build the standard runtime
make nml-fast                         # Build with BLAS acceleration
./nml program.nml data.nml.data       # Run a program
./nml program.nml --trace             # Run with instruction trace
./nml program.nml --describe          # Print metadata without executing
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

#### GELU — Gaussian Error Linear Unit

`Rd = 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))`. Smoother than ReLU near zero. Standard in GPT, BERT, and most modern transformers.

```
; Transformer feed-forward block uses GELU between linear layers
MMUL  R5 R4 R3         ; linear projection
MADD  R5 R5 R2         ; add bias
GELU  R5 R5            ; smooth nonlinearity (not ReLU)
```

**When to use over RELU:** GELU is preferred in transformer architectures. RELU is faster and sufficient for simple dense networks. Use GELU when replicating GPT/BERT-style models. The arch descriptor code for TNDEEP is `3`.

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

`Rd = conv2d(input, kernel)`. Slides the kernel over the input, computing dot products at each position.

```
CONV  Rdest  Rinput  Rkernel  [#kernel_size  [#stride]]
```

- `#kernel_size` — filter width/height (default 1)
- `#stride` — step between filter positions (default 0 = no stride offset)

```
; Extract edges from a 28×28 image using a 3×3 Sobel filter
LD    R0 @image          ; shape [28, 28]
LD    R1 @sobel_kernel   ; shape [3, 3]
CONV  R2 R0 R1 #3 #1     ; 3×3 kernel, stride 1 → shape [26, 26]
ST    R2 @edge_map
HALT
```

#### POOL — Max Pooling

`Rd = max_pool(input, size, stride)`. Keeps only the maximum value in each window. Reduces spatial dimensions while preserving dominant features.

```
POOL  Rdest  Rinput  [#pool_size  [#stride]]
```

- Defaults: pool_size=2, stride=2

```
; Standard CNN pipeline: conv → pool → conv → pool
LD    R0 @image          ; [28, 28]
LD    R1 @kernel1        ; [3, 3]
CONV  R2 R0 R1 #3 #1     ; [26, 26]
POOL  R3 R2 #2 #2        ; [13, 13] — 2×2 max pool
LD    R4 @kernel2        ; [3, 3]
CONV  R5 R3 R4 #3 #1     ; [11, 11]
POOL  R6 R5 #2 #2        ; [5, 5]
ST    R6 @features
HALT
```

#### UPSC — Upscale

`Rd = nearest_neighbor_upsample(input, scale)`. Each value is replicated to fill a scale×scale block. The inverse of POOL for decoder/generative networks.

```
UPSC  Rdest  Rinput  [#scale_factor]
```

- Default scale: 2

```
; Decoder path: upsample then convolve (like U-Net decoder)
LD    R0 @encoded        ; [5, 5] — bottleneck features
UPSC  R1 R0 #2           ; [10, 10]
LD    R2 @decode_kernel  ; [3, 3]
CONV  R3 R1 R2 #3 #1     ; [8, 8]
ST    R3 @decoded
HALT
```

#### PADZ — Zero Padding

`Rd = zero_pad(input, padding)`. Adds a border of zeros around the input. Use before CONV to preserve spatial dimensions (same-padding).

```
PADZ  Rdest  Rinput  [#padding]
```

- Default padding: 1

```
; Same-padding convolution: output is same size as input
LD    R0 @feature_map    ; [14, 14]
PADZ  R1 R0 #1           ; [16, 16] — pad 1 pixel on all sides
LD    R2 @kernel         ; [3, 3]
CONV  R3 R1 R2 #3 #1     ; [14, 14] — same spatial size as input
ST    R3 @output
HALT
```

---

### NML-T: Transformer (4 instructions)

For transformer-based models (GPT, BERT, etc.).

#### ATTN — Scaled Dot-Product Attention

`Rd = softmax(Q @ K^T / sqrt(d_k)) @ V`. The core operation of every transformer. Computes how much each position should attend to every other position.

```
ATTN  Rdest  Rquery  Rkey  [Rvalue]
```

- If `Rvalue` is omitted, keys are used as values (self-attention with K=V)
- Scaling by `sqrt(d_k)` is applied automatically

```
; Single self-attention head
LD    R0 @Q              ; shape [seq_len, d_k]
LD    R1 @K              ; shape [seq_len, d_k]
LD    R2 @V              ; shape [seq_len, d_v]
ATTN  R3 R0 R1 R2        ; R3 = attention output [seq_len, d_v]
ST    R3 @attended
HALT
```

```
; Transformer block: attention + residual + norm
LD    R0 @input          ; [seq_len, d_model]
LD    R1 @Wq             ; [d_model, d_k]
LD    R2 @Wk
LD    R3 @Wv
MMUL  R4 R0 R1           ; Q = input @ Wq
MMUL  R5 R0 R2           ; K = input @ Wk
MMUL  R6 R0 R3           ; V = input @ Wv
ATTN  R7 R4 R5 R6        ; attention output
MADD  R7 R7 R0           ; residual connection: output + input
NORM  R7 R7              ; layer norm
ST    R7 @block_output
HALT
```

#### NORM — Layer Normalization

`Rd = (x - mean(x)) / sqrt(var(x) + ε)` optionally scaled by gamma and shifted by beta. Applied after attention and feed-forward layers in transformers.

```
NORM  Rdest  Rinput  [Rgamma  [Rbeta]]
```

- Without gamma/beta: pure normalization (gamma=1, beta=0)
- With gamma/beta: learnable affine transform (standard layer norm)

```
; Pre-norm style (norm before sublayer)
LD    R0 @x              ; activations
NORM  R1 R0              ; normalize
MMUL  R2 R1 R3           ; feed-forward projection
GELU  R2 R2              ; activation
MADD  R2 R2 R0           ; residual

; Post-norm with learnable parameters
LD    R4 @gamma
LD    R5 @beta
NORM  R3 R2 R4 R5        ; normalize with scale + shift
HALT
```

#### EMBD — Embedding Lookup

`Rd = embed_matrix[indices]`. Selects rows from an embedding matrix by integer index. Converts token IDs to dense vectors.

```
EMBD  Rdest  Rinput_indices  Rembed_matrix
```

- `Rinput_indices` — integer token IDs, shape `[seq_len]`
- `Rembed_matrix` — full vocabulary embedding table, shape `[vocab_size, d_embed]`
- Output shape: `[seq_len, d_embed]`

```
; Token embedding for a sequence
LD    R0 @token_ids      ; shape [seq_len]         — e.g. [42, 17, 9, 3]
LD    R1 @vocab_table    ; shape [vocab_size, 64]  — embedding matrix
EMBD  R2 R0 R1           ; R2 = shape [seq_len, 64]
ST    R2 @embeddings
HALT
```

**Important:** `Rinput_indices` is `reg[1]` and `Rembed_matrix` is `reg[2]`. The index tensor comes first.

#### GELU — Gaussian Error Linear Unit

See [Activation Functions](#activation-functions-4-instructions) above for the full description. In transformers, GELU replaces RELU in the feed-forward sublayer.

```
; GPT-style feed-forward block
LD    R0 @x              ; [seq_len, d_model]
LD    R1 @W1             ; [d_model, 4*d_model]  — expand
LD    R2 @b1
MMUL  R3 R0 R1
MADD  R3 R3 R2
GELU  R3 R3              ; ← GELU, not RELU
LD    R4 @W2             ; [4*d_model, d_model]  — contract
LD    R5 @b2
MMUL  R6 R3 R4
MADD  R6 R6 R5
ST    R6 @ff_output
HALT
```

---

### NML-R: Reduction (4 instructions)

For aggregation, thresholding, and conditional operations.

#### RDUC — Reduce

Collapses a tensor along an axis (or entirely) using a reduction function.

```
RDUC  Rdest  Rinput  [#reduce_type  [#axis]]
```

| Code | Operation | Example result on `[1, 3, 2, 4]` |
|------|-----------|----------------------------------|
| `#0` | sum | 10 |
| `#1` | mean | 2.5 |
| `#2` | max | 4 |
| `#3` | min | 1 |

```
; Compute mean and variance of predictions
LD    R0 @predictions    ; shape [1, 8]
RDUC  R1 R0 #1           ; R1 = mean
MSUB  R2 R0 R1           ; R2 = predictions - mean
EMUL  R3 R2 R2           ; R3 = squared deviations
RDUC  R4 R3 #1           ; R4 = variance
ST    R1 @mean
ST    R4 @variance
HALT
```

#### WHER — Conditional Select

`Rd[i] = cond[i] > 0 ? true[i] : false[i]`. Element-wise conditional, like NumPy's `np.where`.

```
WHER  Rdest  Rcond  Rtrue  [Rfalse]
```

```
; Replace negative predictions with zero (soft ReLU alternative)
LD    R0 @predictions    ; some values may be negative
ALLC  R1 #[8]            ; zero tensor
CMPR  R2 R0 #0.0         ; R2[i] = 1 if predictions[i] > 0, else 0
WHER  R3 R2 R0 R1        ; keep positive values, replace negatives with 0
ST    R3 @clipped
HALT
```

#### CLMP — Clamp

`Rd[i] = max(min_val, min(max_val, input[i]))`. Hard-clips every element to a range.

```
CLMP  Rdest  Rinput  [#min  [#max]]
```

- Defaults: min=0.0, max=1.0

```
; Ensure probability predictions stay in valid range
LD    R0 @raw_probs
CLMP  R1 R0 #0.0 #1.0    ; clip to [0, 1] — predictions can't be negative or > 1
ST    R1 @probs
HALT

; Clip reward signals in reinforcement learning
LD    R2 @rewards
CLMP  R3 R2 #-1.0 #1.0   ; clip to [-1, 1]
ST    R3 @clipped_rewards
HALT
```

#### CMPR — Element-wise Comparison Mask

`Rd[i] = input[i] > threshold ? 1 : 0`. Produces a binary mask. Used to binarize soft predictions.

```
CMPR  Rdest  Rinput  [#threshold  [#op]]
```

```
; Convert sigmoid output to binary decisions
LD    R0 @probabilities  ; values in [0, 1]
CMPR  R1 R0 #0.5         ; R1[i] = 1 if prob[i] > 0.5, else 0
ST    R1 @decisions
HALT

; Combined: predict, clamp, threshold
LD    R0 @logits
SIGM  R0 R0              ; → probabilities
CLMP  R0 R0 #0.0 #1.0    ; ensure valid range
CMPR  R1 R0 #0.5         ; binary decisions
ST    R1 @labels
HALT
```

---

### NML-S: Signal (2 instructions)

For signal processing and time-series analysis.

#### FFT — Discrete Fourier Transform

Converts a time-domain signal to frequency domain. Output has real and imaginary components.

```
FFT  Rdest  Rinput  Rconfig
```

Both `Rdest` and `Rinput` are written after execution (real and imaginary parts respectively).

```
; Analyze frequency content of a sensor reading
LD    R0 @time_series    ; shape [256] — 256 time-domain samples
LD    R2 @fft_config     ; configuration tensor
FFT   R1 R0 R2           ; R1 = real part, R0 = imaginary part (overwritten)
ST    R1 @freq_real
ST    R0 @freq_imag
HALT
```

#### FILT — FIR Filter

`Rd = convolve(signal, filter_coeffs)`. Applies a finite impulse response filter to a signal. Equivalent to 1D convolution.

```
FILT  Rdest  Rinput  Rfilter
```

```
; Low-pass filter to remove high-frequency noise
LD    R0 @noisy_signal   ; shape [512]
LD    R1 @lpf_coeffs     ; low-pass FIR coefficients, e.g. shape [31]
FILT  R2 R0 R1           ; smooth signal
ST    R2 @clean_signal
HALT

; Moving average (box filter)
LD    R0 @temperature_readings
LD    R1 @box_filter     ; e.g. [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
FILT  R2 R0 R1           ; 10-point moving average
ST    R2 @smoothed
HALT
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
META  @author       "domain_transpiler v2.3"

LD    R0 @gross_pay
SCLR  R1 R0 #0.062
ST    R1 @tax_amount
HALT
```

**Why it matters:** Without META, an NML program is opaque — you have to execute it or read the instructions to know what it does. With META, any agent can inspect the program's interface without running it. The `--describe` flag prints the descriptor:

```bash
./nml program.nml --describe
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
./nml --patch fit_2025.nml fit_2026_patch.nml
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

./nml fica.nml /tmp/fica.data
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
# Standard build (all extensions enabled)
make nml

# BLAS-accelerated build
make nml-fast

# With expanded limits for larger programs
gcc -O2 -o nml runtime/nml.c -lm \
    -DNML_MAX_INSTRUCTIONS=65536 \
    -DNML_MAX_MEMORY_SLOTS=256 \
    -DNML_MAX_CALL_DEPTH=128
```

---

## Part 6: NML-TR Training Extensions (15)

These instructions implement gradient-based training directly in the runtime. TNET and TNDEEP are high-level self-contained trainers. BKWD/WUPD/LOSS and the backward opcodes give you manual control over each step.

---

### High-Level Trainers

#### TNET — Single Hidden Layer Trainer

Trains a 2-layer network (one hidden layer) end-to-end. All state lives in registers.

```
TNET  #epochs  #lr  [#seed]
```

**Register convention:**
- R0 = input features (N × input_size)
- R1 = w1 weights, R2 = b1 bias  ← modified in-place
- R3 = w2 weights, R4 = b2 bias  ← modified in-place
- R9 = target labels (N × output_size)
- RA = final prediction after training

```
; Train a 4→8→1 network on sensor anomaly data
LD    R1 @w1             ; shape [4, 8] — He init
LD    R2 @b1             ; shape [1, 8] — zeros
LD    R3 @w2             ; shape [8, 1]
LD    R4 @b2             ; shape [1, 1]
LD    R0 @training_data  ; shape [N, 4]
LD    R9 @labels         ; shape [N, 1]
TNET  #1000  #0.01  #0   ; 1000 epochs, lr=0.01, seed=0
ST    RA @prediction
HALT
```

#### TNDEEP — N-Layer Dense Trainer

Trains networks with 1–10 layers, configurable activations, and a choice of optimizer. The architecture is declared in register RV.

```
TNDEEP  #epochs  #lr  #optimizer
```

**`#optimizer`:** `#0` = SGD (faster, less memory), `#1` = Adam (β1=0.9, β2=0.999)

**Register convention:**
- RV = architecture descriptor (must be loaded first)
- R0 = input features (N × input_size)
- R1/R2 = w1/b1 for layer 1  ← modified in-place
- R3/R4 = w2/b2 for layer 2  ← modified in-place
- R5/R6 = w3/b3 for layer 3  ← modified in-place (and so on for deeper nets)
- R9 = target labels
- R8 = final MSE loss (written after training)

**Architecture descriptor in RV:** A flat 1D tensor loaded with `LD RV @arch`.

```
[n_layers, h1, act1, h2, act2, ..., hn, actn]
```

Activation codes: `0`=ReLU, `1`=Sigmoid, `2`=Tanh, `3`=GELU

```
; Train 12→16→8→1 regression network
; arch: 3 layers, 16 neurons/ReLU, 8 neurons/ReLU, 1 neuron/Sigmoid
@arch shape=7 dtype=f32 data=3,16,0,8,0,1,1
```

Full train-then-infer program:

```
LD    RV @arch            ; REQUIRED: load before TNDEEP

LD    R1 @w1              ; 12×16
LD    R2 @b1              ; 1×16
LD    R3 @w2              ; 16×8
LD    R4 @b2              ; 1×8
LD    R5 @w3              ; 8×1
LD    R6 @b3              ; 1×1

LD    R0 @training_data   ; N×12
LD    R9 @training_labels ; N×1

TNDEEP  #2000  #0.005  #1  ; Adam, 2000 epochs
ST    R8 @training_loss    ; save loss BEFORE R8 is reused below

; Manual forward pass for inference (R1–R6 now hold trained weights)
LD    R0 @predict_input    ; 1×12

MMUL  R7 R0 R1             ; layer 1
MADD  R7 R7 R2
RELU  R7 R7

MMUL  R8 R7 R3             ; layer 2
MADD  R8 R8 R4
RELU  R8 R8

MMUL  RA R8 R5             ; layer 3
MADD  RA RA R6
SIGM  RA RA                ; matches arch act[6]=1

ST    RA @prediction
HALT
```

**Critical rules:**
- Always `LD RV @arch` before `TNDEEP` — missing it causes a runtime error
- Save R8 immediately after `TNDEEP` — the inference forward pass overwrites it
- The manual inference activations must match the arch descriptor activation codes
- Using RELU on the output layer when labels are in (0,1) causes dead neurons → prediction = 0. Use SIGM.

---

### Step-Level Training (Manual Backprop)

These four instructions let you write the training loop yourself, giving full control over loss functions, optimizers, and gradient flow.

#### LOSS — Compute Loss

```
LOSS  Rdest  Rpredicted  Rtarget  #loss_type
```

| Code | Loss | Use when |
|------|------|----------|
| `#0` | MSE `mean((pred - target)²)` | Regression |
| `#1` | Cross-entropy `-sum(target * log(pred))` | Classification |
| `#2` | MAE `mean(|pred - target|)` | Robust regression |

```
; MSE loss for regression
MMUL  R5 R4 R3            ; forward pass output
MADD  R5 R5 R2
RELU  R5 R5
LOSS  R6 R5 R9 #0         ; R6 = MSE loss scalar
ST    R6 @loss
```

#### BKWD — Backpropagation

```
BKWD  Rgrad  Routput  Rtarget
```

Computes the gradient of the loss with respect to the output layer. Stores the gradient in `Rgrad` for subsequent WUPD calls.

```
BKWD  RG R5 R9            ; RG = d_loss/d_output
```

#### WUPD — Weight Update

```
WUPD  Rweights  Rweights  Rgrad  [#lr]
```

Applies gradient descent: `weights -= lr * grad`. The learning rate defaults to the last value set.

```
WUPD  R3 R3 RG            ; update output weights
WUPD  R1 R1 RH            ; update hidden weights
```

---

### Backward Pass Opcodes

These compute gradients through individual operations. Use them when building a custom training loop with BKWD/WUPD, or when TNDEEP's architecture doesn't fit your network.

All backward ops follow the pattern: `*BK Rdest Rgrad Rforward_input`

#### RELUBK — ReLU Gradient

`Rd[i] = grad[i] * (input[i] > 0 ? 1 : 0)`

Passes the gradient through where the forward-pass input was positive; zeros it where it was negative (those neurons were off).

```
; Forward
RELU  R3 R3               ; R3 = relu(pre_activation)
; Backward — need to save pre-activation before RELU
RELUBK R7 RG R2           ; R7 = gradient through ReLU
;                           RG = incoming gradient, R2 = pre-activation values
```

#### SIGMBK — Sigmoid Gradient

`Rd[i] = grad[i] * σ(input[i]) * (1 - σ(input[i]))`

The sigmoid derivative is `σ(x) * (1 - σ(x))`. It is recomputed from the pre-activation input.

```
SIGMBK  R8 RG R5          ; RG = incoming grad, R5 = pre-sigmoid input
```

#### TANHBK — Tanh Gradient

`Rd[i] = grad[i] * (1 - tanh(input[i])²)`

```
TANHBK  R8 RG R5          ; RG = incoming grad, R5 = pre-tanh input
```

#### GELUBK — GELU Gradient

Applies the chain rule through the GELU approximation `0.5x(1 + tanh(√(2/π)(x + 0.044715x³)))`.

```
GELUBK  R8 RG R5          ; R8 = gradient through GELU
```

#### SOFTBK — Softmax Gradient

`Rd[i] = s[i] * (grad[i] - Σⱼ grad[j] * s[j])` where s = softmax(input).

Uses the full Jacobian-vector product. Numerically stable (internally applies max normalization).

```
SOFTBK  R8 RG R5          ; R8 = gradient through softmax
```

#### MMULBK — Matrix Multiply Gradient

Computes two gradients simultaneously: one for the input tensor and one for the weight matrix.

```
MMULBK  Rd_dinput  Rd_dweight  Rgrad  Rinput  Rweight
```

- `Rd_dinput = grad @ weight^T`
- `Rd_dweight = input^T @ grad`

```
; Forward: R3 = R0 @ R1
MMUL  R3 R0 R1
; Backward
MMULBK R7 R8 RG R0 R1    ; R7 = d_input, R8 = d_weights
;                           RG = incoming gradient
```

#### CONVBK — Convolution Gradient

Computes gradients for both the input and the kernel.

```
CONVBK  Rd_dinput  Rd_dkernel  Rgrad  Rinput  Rkernel
```

- `Rd_dinput` = full convolution of grad with flipped kernel
- `Rd_dkernel` = correlation of input with grad

```
CONVBK R5 R6 RG R0 R1    ; R5 = d_input, R6 = d_kernel
```

#### POOLBK — Max Pool Gradient

Routes gradients back only through the positions that were selected by max pooling (winner-take-all).

```
POOLBK  Rdest  Rgrad  Rfwd_input  [#pool_size  [#stride]]
```

`Rfwd_input` is the original pre-pool input — needed to find which positions were the maxima.

```
; Must save the pre-pool tensor before POOL for use in backward
MOV   R2 R0              ; save original for POOLBK
POOL  R3 R0 #2 #2        ; forward pool
; ... rest of forward pass, then backward ...
POOLBK R9 RG R2 #2 #2   ; R9 = gradient through pool
```

#### NORMBK — Layer Norm Gradient

Computes the full layer normalization gradient (complex multivariate chain rule through mean and variance).

```
NORMBK  Rdest  Rgrad  Rinput
```

Assumes gamma=1, beta=0. For learnable parameters, scale the gradient separately.

```
NORMBK R7 RG R4          ; R7 = gradient through layer norm
```

#### ATTNBK — Attention Gradient

Backpropagates through scaled dot-product attention. Writes three gradients at once into consecutive registers.

```
ATTNBK  Rd_dq  Rgrad  Rq  Rk  Rv
```

- `Rd_dq` (reg[0]) = gradient w.r.t. Query
- `Rd_dq + 1` (next register) = gradient w.r.t. Key
- `Rd_dq + 2` = gradient w.r.t. Value

Recomputes attention weights from Q/K/V internally (no need to save them from the forward pass).

```
; Forward
ATTN  R7 R4 R5 R6        ; R7 = attention output
; Backward (R8, R9, RA receive dQ, dK, dV respectively)
ATTNBK R8 RG R4 R5 R6   ; R8=dQ, R9=dK, RA=dV  (consecutive registers)
```

---

## Part 7: Tri-Syntax Reference

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
| 17 | RSHP | ⊟ | RESHAPE | Data Flow |
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
| 67 | BKWD | ∇ | BACKPROP | NML-TR |
| 68 | WUPD | ⟳ | WEIGHT_UPDATE | NML-TR |
| 69 | LOSS | △ | COMPUTE_LOSS | NML-TR |
| 70 | TNET | ⥁ | TRAIN_NETWORK | NML-TR |
| 71 | RELUBK | ⌐ˈ | RELU_BACKWARD | NML-TR |
| 72 | SIGMBK | σˈ | SIGMOID_BACKWARD | NML-TR |
| 73 | TANHBK | τˈ | TANH_BACKWARD | NML-TR |
| 74 | GELUBK | ℊˈ | GELU_BACKWARD | NML-TR |
| 75 | SOFTBK | Σˈ | SOFTMAX_BACKWARD | NML-TR |
| 76 | MMULBK | ×ˈ | MATMUL_BACKWARD | NML-TR |
| 77 | CONVBK | ⊛ˈ | CONV_BACKWARD | NML-TR |
| 78 | POOLBK | ⊓ˈ | POOL_BACKWARD | NML-TR |
| 79 | NORMBK | ‖ˈ | NORM_BACKWARD | NML-TR |
| 80 | ATTNBK | ⊙ˈ | ATTN_BACKWARD | NML-TR |
| 81 | TNDEEP | ⥁ˈ | TRAIN_DEEP | NML-TR |

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
| 16 | RG | — | GRAD1 | Gradient buffer 1 (backprop) |
| 17 | RH | — | GRAD2 | Gradient buffer 2 (backprop) |
| 18 | RI | — | GRAD3 | Gradient buffer 3 (backprop) |
| 19 | RJ | — | LR | Learning rate |
| 20–30 | RK–RU | — | TRAINING | Training workspace / hive |
| 31 | RV | — | ARCH | Architecture descriptor (TNDEEP reads this) |
