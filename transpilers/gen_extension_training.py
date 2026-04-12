#!/usr/bin/env python3
"""
Generate training pairs for NML extension and M2M instructions.

Covers all instructions missing from the current training data:
  - NML-V (Vision): CONV, POOL, UPSC, PADZ
  - NML-T (Transformer): ATTN, NORM, EMBD, GELU
  - NML-R (Reduction): RDUC, WHER, CLMP, CMPR
  - NML-S (Signal): FFT, FILT
  - NML-M2M: META, FRAG, ENDF, LINK, PTCH, SIGN, VRFY, VOTE, PROJ, DIST

Output: JSONL with instruction/input/output pairs for fine-tuning.

Usage:
    python3 gen_extension_training.py
    python3 gen_extension_training.py --output ../output/training/extension_pairs.jsonl
"""

import json
import random
import argparse
from pathlib import Path

OUTPUT_PATH = Path(__file__).parent.parent / "output" / "training" / "extension_pairs.jsonl"


def gen_vision_pairs() -> list[dict]:
    """NML-V: CONV, POOL, UPSC, PADZ"""
    pairs = []

    # --- CONV ---
    pairs.extend([
        {
            "instruction": "What does CONV do in NML?",
            "input": "",
            "output": "CONV Rd Rs1 Rs2 performs 2D convolution: Rd = Rs1 * Rs2 (convolution, not multiply). Rs1 is the input tensor, Rs2 is the kernel/filter. Optional stride and padding parameters.",
        },
        {
            "instruction": "Write NML to apply a 3x3 convolution filter to an image.",
            "input": "",
            "output": "LD    R0 @image\nLD    R1 @filter_3x3\nCONV  R2 R0 R1\nST    R2 @conv_output\nHALT",
        },
        {
            "instruction": "Write NML for a convolutional layer with ReLU activation.",
            "input": "",
            "output": "LD    R0 @input\nLD    R1 @conv_weights\nLD    R2 @conv_bias\nCONV  R3 R0 R1\nMADD  R3 R3 R2\nRELU  R3 R3\nST    R3 @output\nHALT",
        },
        {
            "instruction": "What is the symbolic opcode for CONV?",
            "input": "",
            "output": "The symbolic opcode for CONV (convolution) is ⊛. In symbolic syntax: ⊛ λ ι κ applies convolution of ι with kernel κ, storing result in λ.",
        },
        {
            "instruction": "Write a 2-layer CNN in NML.",
            "input": "",
            "output": (
                "LD    R0 @input\n"
                "LD    R1 @conv1_weights\nLD    R2 @conv1_bias\n"
                "CONV  R3 R0 R1\nMADD  R3 R3 R2\nRELU  R3 R3\n"
                "POOL  R3 R3\n"
                "LD    R4 @conv2_weights\nLD    R5 @conv2_bias\n"
                "CONV  R6 R3 R4\nMADD  R6 R6 R5\nRELU  R6 R6\n"
                "POOL  R6 R6\n"
                "ST    R6 @features\nHALT"
            ),
        },
    ])

    # --- POOL ---
    pairs.extend([
        {
            "instruction": "What does POOL do in NML?",
            "input": "",
            "output": "POOL Rd Rs performs max pooling on tensor Rs, storing the result in Rd. Optional parameters: #size (pool window, default 2) and #stride (default = size). Reduces spatial dimensions by the pool factor.",
        },
        {
            "instruction": "Write NML to downsample a feature map with max pooling.",
            "input": "",
            "output": "LD    R0 @feature_map\nPOOL  R1 R0\nST    R1 @pooled\nHALT",
        },
        {
            "instruction": "What is the symbolic opcode for POOL?",
            "input": "",
            "output": "The symbolic opcode for POOL (max pooling) is ⊓. Example: ⊓ κ ι pools the tensor in ι into κ.",
        },
    ])

    # --- UPSC ---
    pairs.extend([
        {
            "instruction": "What does UPSC do in NML?",
            "input": "",
            "output": "UPSC Rd Rs performs nearest-neighbor upscaling on tensor Rs, storing in Rd. Optional #scale parameter (default 2). Doubles spatial dimensions by repeating values.",
        },
        {
            "instruction": "Write NML to upscale an image by 2x.",
            "input": "",
            "output": "LD    R0 @low_res\nUPSC  R1 R0\nST    R1 @high_res\nHALT",
        },
        {
            "instruction": "What is the symbolic opcode for UPSC?",
            "input": "",
            "output": "The symbolic opcode for UPSC (upscale) is ⊔. Example: ⊔ κ ι upscales tensor ι into κ.",
        },
    ])

    # --- PADZ ---
    pairs.extend([
        {
            "instruction": "What does PADZ do in NML?",
            "input": "",
            "output": "PADZ Rd Rs adds zero-padding around tensor Rs, storing in Rd. Optional #pad parameter specifies padding amount on each side (default 1).",
        },
        {
            "instruction": "Write NML to zero-pad an input before convolution.",
            "input": "",
            "output": "LD    R0 @input\nPADZ  R1 R0\nLD    R2 @kernel\nCONV  R3 R1 R2\nST    R3 @output\nHALT",
        },
        {
            "instruction": "What is the symbolic opcode for PADZ?",
            "input": "",
            "output": "The symbolic opcode for PADZ (zero-pad) is ⊡. Example: ⊡ κ ι pads tensor ι with zeros, storing in κ.",
        },
    ])

    return pairs


def gen_transformer_pairs() -> list[dict]:
    """NML-T: ATTN, NORM, EMBD, GELU"""
    pairs = []

    # --- ATTN ---
    pairs.extend([
        {
            "instruction": "What does ATTN do in NML?",
            "input": "",
            "output": "ATTN Rd Rq Rk [Rv] performs scaled dot-product attention: Rd = softmax(Rq @ Rk^T / sqrt(d)) @ Rv. If Rv is omitted, Rv = Rk. This is the core attention mechanism used in transformers.",
        },
        {
            "instruction": "Write NML for self-attention on an input sequence.",
            "input": "",
            "output": (
                "LD    R0 @input\n"
                "LD    R1 @wq\nLD    R2 @wk\nLD    R3 @wv\n"
                "MMUL  R4 R0 R1\n"
                "MMUL  R5 R0 R2\n"
                "MMUL  R6 R0 R3\n"
                "ATTN  R7 R4 R5 R6\n"
                "ST    R7 @attn_output\nHALT"
            ),
        },
        {
            "instruction": "What is the symbolic opcode for ATTN?",
            "input": "",
            "output": "The symbolic opcode for ATTN (attention) is ⊙. Example: ⊙ π ν ξ ο computes attention with query ν, key ξ, value ο, storing in π.",
        },
        {
            "instruction": "Write NML for a single transformer block (attention + feedforward).",
            "input": "",
            "output": (
                "LD    R0 @input\n"
                "LD    R1 @wq\nLD    R2 @wk\nLD    R3 @wv\n"
                "MMUL  R4 R0 R1\nMMUL  R5 R0 R2\nMMUL  R6 R0 R3\n"
                "ATTN  R7 R4 R5 R6\n"
                "MADD  R7 R7 R0\n"
                "NORM  R7 R7\n"
                "LD    R8 @ff_w1\nLD    R9 @ff_w2\n"
                "MMUL  RA R7 R8\n"
                "GELU  RA RA\n"
                "MMUL  RA RA R9\n"
                "MADD  RA RA R7\n"
                "NORM  RA RA\n"
                "ST    RA @output\nHALT"
            ),
        },
    ])

    # --- NORM ---
    pairs.extend([
        {
            "instruction": "What does NORM do in NML?",
            "input": "",
            "output": "NORM Rd Rs performs layer normalization: normalizes Rs to zero mean and unit variance, then optionally applies learned scale (Rgamma) and shift (Rbeta). Used between transformer layers.",
        },
        {
            "instruction": "Write NML to normalize a hidden state.",
            "input": "",
            "output": "LD    R0 @hidden\nNORM  R1 R0\nST    R1 @normalized\nHALT",
        },
        {
            "instruction": "What is the symbolic opcode for NORM?",
            "input": "",
            "output": "The symbolic opcode for NORM (layer normalization) is ‖. Example: ‖ κ ι normalizes tensor ι into κ.",
        },
    ])

    # --- EMBD ---
    pairs.extend([
        {
            "instruction": "What does EMBD do in NML?",
            "input": "",
            "output": "EMBD Rd Rtable Rindices performs embedding table lookup: for each index in Rindices, looks up the corresponding row in Rtable and stores the result in Rd. Used to convert token IDs to dense vectors.",
        },
        {
            "instruction": "Write NML to embed a sequence of token IDs.",
            "input": "",
            "output": "LD    R0 @token_ids\nLD    R1 @embedding_table\nEMBD  R2 R1 R0\nST    R2 @embeddings\nHALT",
        },
        {
            "instruction": "What is the symbolic opcode for EMBD?",
            "input": "",
            "output": "The symbolic opcode for EMBD (embedding lookup) is ⊏. Example: ⊏ λ κ ι looks up indices ι in table κ, storing embeddings in λ.",
        },
    ])

    # --- GELU ---
    pairs.extend([
        {
            "instruction": "What does GELU do in NML?",
            "input": "",
            "output": "GELU Rd Rs applies the Gaussian Error Linear Unit activation: Rd = x * 0.5 * (1 + erf(x/sqrt(2))). Smoother than ReLU, used in modern transformers (GPT, BERT).",
        },
        {
            "instruction": "Write NML for a feedforward layer with GELU activation.",
            "input": "",
            "output": "LD    R0 @input\nLD    R1 @ff_weights\nLD    R2 @ff_bias\nMMUL  R3 R0 R1\nMADD  R3 R3 R2\nGELU  R3 R3\nST    R3 @output\nHALT",
        },
        {
            "instruction": "What is the difference between RELU and GELU in NML?",
            "input": "",
            "output": "RELU clips negative values to zero (max(0,x)). GELU is smoother — it gradually gates values based on their magnitude using the Gaussian CDF. GELU is preferred in transformers; RELU is simpler and used in CNNs and older networks. Both take the form: OPCODE Rd Rs.",
        },
        {
            "instruction": "What is the symbolic opcode for GELU?",
            "input": "",
            "output": "The symbolic opcode for GELU is ℊ. Example: ℊ κ ι applies GELU activation to ι, storing in κ.",
        },
    ])

    return pairs


def gen_reduction_pairs() -> list[dict]:
    """NML-R: RDUC, WHER, CLMP, CMPR"""
    pairs = []

    # --- RDUC ---
    pairs.extend([
        {
            "instruction": "What does RDUC do in NML?",
            "input": "",
            "output": "RDUC Rd Rs #op reduces tensor Rs using the specified operation: #0=sum, #1=mean, #2=max, #3=min. Optional #dim parameter specifies the dimension to reduce along. Result stored in Rd.",
        },
        {
            "instruction": "Write NML to compute the sum of all elements in a tensor.",
            "input": "",
            "output": "LD    R0 @data\nRDUC  R1 R0 #0\nST    R1 @total\nHALT",
        },
        {
            "instruction": "Write NML to compute the mean, max, and min of a dataset.",
            "input": "",
            "output": "LD    R0 @data\nRDUC  R1 R0 #1\nRDUC  R2 R0 #2\nRDUC  R3 R0 #3\nST    R1 @mean\nST    R2 @max_val\nST    R3 @min_val\nHALT",
        },
        {
            "instruction": "What is the symbolic opcode for RDUC?",
            "input": "",
            "output": "The symbolic opcode for RDUC (reduce) is ⊥. Example: ⊥ κ ι #0 computes sum of ι into κ. Operations: #0=sum, #1=mean, #2=max, #3=min.",
        },
    ])

    # --- WHER ---
    pairs.extend([
        {
            "instruction": "What does WHER do in NML?",
            "input": "",
            "output": "WHER Rd Rcond Ra [Rb] performs conditional element-wise select: where Rcond is true (nonzero), Rd takes the value from Ra; otherwise from Rb. If Rb is omitted, false elements are set to 0.",
        },
        {
            "instruction": "Write NML to zero out all negative values using WHER.",
            "input": "",
            "output": "LD    R0 @data\nALLC  R1 #[1]\nCMPR  R2 R0 #0 #4\nWHER  R3 R2 R0 R1\nST    R3 @positive_only\nHALT",
        },
        {
            "instruction": "Write NML to select between two values based on a condition.",
            "input": "",
            "output": "LD    R0 @condition\nLD    R1 @value_if_true\nLD    R2 @value_if_false\nWHER  R3 R0 R1 R2\nST    R3 @result\nHALT",
        },
        {
            "instruction": "What is the symbolic opcode for WHER?",
            "input": "",
            "output": "The symbolic opcode for WHER (conditional select) is ⊻. Example: ⊻ μ λ κ ι selects from κ where λ is true, else ι, storing in μ.",
        },
    ])

    # --- CLMP ---
    pairs.extend([
        {
            "instruction": "What does CLMP do in NML?",
            "input": "",
            "output": "CLMP Rd Rs #lo #hi clamps all elements of Rs to the range [lo, hi]: values below lo become lo, values above hi become hi. Useful for bounding outputs.",
        },
        {
            "instruction": "Write NML to clamp values between 0 and 1.",
            "input": "",
            "output": "LD    R0 @data\nCLMP  R1 R0 #0.0 #1.0\nST    R1 @clamped\nHALT",
        },
        {
            "instruction": "Write NML to clamp a result to a maximum of 10000.",
            "input": "",
            "output": "LD    R0 @input\nSCLR  R1 R0 #0.150000\nCLMP  R1 R1 #0.0 #10000.0\nST    R1 @result\nHALT",
        },
        {
            "instruction": "What is the symbolic opcode for CLMP?",
            "input": "",
            "output": "The symbolic opcode for CLMP (clamp) is ⊧. Example: ⊧ κ ι #0.0 #1.0 clamps ι to [0,1], storing in κ.",
        },
    ])

    # --- CMPR ---
    pairs.extend([
        {
            "instruction": "What does CMPR do in NML?",
            "input": "",
            "output": "CMPR Rd Rs #threshold #op produces a 0/1 mask by comparing each element of Rs against threshold. Operations: #0=less, #1=less_equal, #2=equal, #3=not_equal, #4=greater_equal, #5=greater.",
        },
        {
            "instruction": "Write NML to create a mask of all elements greater than 100.",
            "input": "",
            "output": "LD    R0 @data\nCMPR  R1 R0 #100.0 #5\nST    R1 @mask\nHALT",
        },
        {
            "instruction": "What is the symbolic opcode for CMPR?",
            "input": "",
            "output": "The symbolic opcode for CMPR (element-wise mask compare) is ⊜. Example: ⊜ κ ι #0.0 #4 creates a mask where ι >= 0, storing in κ.",
        },
    ])

    return pairs


def gen_signal_pairs() -> list[dict]:
    """NML-S: FFT, FILT"""
    pairs = []

    pairs.extend([
        {
            "instruction": "What does FFT do in NML?",
            "input": "",
            "output": "FFT Rd_real Rd_imag Rs computes the Discrete Fourier Transform of Rs. The real part goes to Rd_real and the imaginary part to Rd_imag. Converts a time-domain signal to frequency domain.",
        },
        {
            "instruction": "Write NML to compute the frequency spectrum of a signal.",
            "input": "",
            "output": "LD    R0 @signal\nFFT   R1 R2 R0\nST    R1 @spectrum_real\nST    R2 @spectrum_imag\nHALT",
        },
        {
            "instruction": "What is the symbolic opcode for FFT?",
            "input": "",
            "output": "The symbolic opcode for FFT (Fourier transform) is ∿. Example: ∿ κ λ ι transforms signal ι, real part in κ, imaginary in λ.",
        },
        {
            "instruction": "What does FILT do in NML?",
            "input": "",
            "output": "FILT Rd Rs Rs_coeffs applies a FIR (Finite Impulse Response) filter: Rd = Rs convolved with Rs_coeffs (1D convolution). Used for smoothing, low-pass, high-pass, and bandpass filtering of signals.",
        },
        {
            "instruction": "Write NML to apply a low-pass filter to a signal.",
            "input": "",
            "output": "LD    R0 @signal\nLD    R1 @filter_coeffs\nFILT  R2 R0 R1\nST    R2 @filtered_signal\nHALT",
        },
        {
            "instruction": "Write NML for signal processing: FFT, filter in frequency domain, then analyze.",
            "input": "",
            "output": (
                "LD    R0 @raw_signal\n"
                "FFT   R1 R2 R0\n"
                "LD    R3 @bandpass_coeffs\n"
                "FILT  R4 R0 R3\n"
                "RDUC  R5 R4 #2\n"
                "ST    R1 @spectrum_real\n"
                "ST    R4 @filtered\n"
                "ST    R5 @peak_value\n"
                "HALT"
            ),
        },
        {
            "instruction": "What is the symbolic opcode for FILT?",
            "input": "",
            "output": "The symbolic opcode for FILT (FIR filter) is ⋐. Example: ⋐ λ ι κ filters signal ι with coefficients κ, storing in λ.",
        },
    ])

    return pairs


def gen_m2m_pairs() -> list[dict]:
    """NML-M2M: META, FRAG, ENDF, LINK, PTCH, SIGN, VRFY, VOTE, PROJ, DIST"""
    pairs = []

    # --- META ---
    pairs.extend([
        {
            "instruction": "What does META do in NML?",
            "input": "",
            "output": "META @key value declares program metadata. It is a no-op at runtime — the runtime parses META lines during assembly and stores them in a descriptor. Valid keys: @name, @version, @input, @output, @invariant, @provenance, @author, @domain, @license, @created.",
        },
        {
            "instruction": "Write NML META headers for a self-describing interest calculator.",
            "input": "",
            "output": (
                'META  @name       "simple_interest"\n'
                'META  @version    "1.0"\n'
                'META  @domain     "finance"\n'
                'META  @input      principal   currency  "Loan principal"\n'
                'META  @input      rate        ratio     "Annual interest rate"\n'
                'META  @output     interest    currency  "Annual interest"\n'
                'META  @invariant  "interest >= 0"\n'
                'META  @author     "domain_transpiler"\n'
                "LD    R0 @principal\nLD    R1 @rate\n"
                "EMUL  RA R0 R1\nST    RA @interest\nHALT"
            ),
        },
        {
            "instruction": "What is the symbolic opcode for META?",
            "input": "",
            "output": 'The symbolic opcode for META is §. Example: § @name "my_program" declares the program name.',
        },
        {
            "instruction": "What are the valid META keys in NML?",
            "input": "",
            "output": "Valid META keys: @name (program ID), @version, @input (repeatable, declares inputs with type), @output (repeatable), @invariant (runtime assertion), @provenance (source reference), @author (producing agent), @domain (e.g. tax, finance), @license, @created (ISO-8601 timestamp).",
        },
    ])

    # --- FRAG/ENDF/LINK ---
    pairs.extend([
        {
            "instruction": "What do FRAG, ENDF, and LINK do in NML?",
            "input": "",
            "output": "FRAG name opens a named fragment scope. ENDF closes it. LINK @name imports a fragment's instructions inline at assembly time. Fragments are independently validatable and composable — multiple fragments can be linked into a single program.",
        },
        {
            "instruction": "Write NML with two composable fragments linked together.",
            "input": "",
            "output": (
                "FRAG  calc_rate\n"
                "LD    R0 @input\nSCLR  RA R0 #0.050000\nST    RA @rate_result\n"
                "ENDF\n\n"
                "FRAG  apply_cap\n"
                "LD    R0 @rate_result\nCMPF  RE R0 #0 #500.000000\n"
                "JMPT  #2\nLEAF  R0 #500.000000\nST    R0 @rate_result\n"
                "ENDF\n\n"
                "FRAG  main\n"
                "LINK  @calc_rate\n"
                "LINK  @apply_cap\n"
                "LD    RA @rate_result\nST    RA @final_result\nHALT\n"
                "ENDF"
            ),
        },
        {
            "instruction": "What are the symbolic opcodes for FRAG, ENDF, and LINK?",
            "input": "",
            "output": "FRAG = ◆ (open fragment), ENDF = ◇ (close fragment), LINK = ⊚ (import fragment). Example: ◆ my_calc opens a fragment named my_calc, ◇ closes it, ⊚ @my_calc imports it.",
        },
        {
            "instruction": "Write NML fragments for a multi-step calculation using symbolic syntax.",
            "input": "",
            "output": (
                "◆  step_one\n"
                "§  @input  value  currency\n"
                "§  @output result currency\n"
                "↓  ι  @value\n∗  α  ι  #0.100000\n↑  α  @result\n"
                "◇\n\n"
                "◆  step_two\n"
                "↓  ι  @result\n⊖  α  ι  γ\n↑  α  @final\n"
                "◇\n\n"
                "◆  pipeline\n"
                "⊕  @step_one\n⊕  @step_two\n◼\n"
                "◇"
            ),
        },
    ])

    # --- PTCH ---
    pairs.extend([
        {
            "instruction": "What does PTCH do in NML?",
            "input": "",
            "output": "PTCH applies a differential patch to an existing NML program. PTCH @base declares the SHA-256 hash of the program being patched. PTCH @set N \"instr\" replaces line N. PTCH @del N deletes line N. PTCH @ins N \"instr\" inserts at line N. PTCH @end closes the patch block.",
        },
        {
            "instruction": "Write an NML patch that updates two rate values in an existing program.",
            "input": "",
            "output": (
                "PTCH  @base  sha256:a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0\n"
                'PTCH  @set   7   "SCLR  RC R0 #0.065000"\n'
                'PTCH  @set   15  "LEAF  RC #175000.000000"\n'
                "PTCH  @end"
            ),
        },
        {
            "instruction": "What is the symbolic opcode for PTCH?",
            "input": "",
            "output": "The symbolic opcode for PTCH (differential patch) is ⊿. Example: ⊿ @base sha256:... declares the target program hash.",
        },
        {
            "instruction": "When would you use PTCH instead of sending a full NML program?",
            "input": "",
            "output": "Use PTCH when updating a few values in a large program — like annual rate changes in a 200-line tax program. A patch might be 5 lines vs retransmitting 200. The base hash ensures the patch applies to the correct version. This is critical for M2M communication where bandwidth and verification matter.",
        },
    ])

    # --- SIGN/VRFY ---
    pairs.extend([
        {
            "instruction": "What do SIGN and VRFY do in NML?",
            "input": "",
            "output": "SIGN records a cryptographic signature over the program (no-op at runtime, metadata only). VRFY is executable — it computes the program's SHA-256 hash, looks up the SIGN metadata, and verifies the signature. If verification fails, VRFY triggers a TRAP. Supported algorithms: ed25519, hmac-sha256.",
        },
        {
            "instruction": "Write NML with a signed program that verifies itself before executing.",
            "input": "",
            "output": (
                "SIGN  agent=transpiler_v2  key=ed25519:a1b2c3d4  sig=e5f6a7b8\n"
                'META  @name  "verified_calc"\n'
                "VRFY  @self  @transpiler_v2\n"
                "LD    R0 @input\n"
                "SCLR  RA R0 #0.050000\n"
                "ST    RA @output\n"
                "HALT"
            ),
        },
        {
            "instruction": "What are the symbolic opcodes for SIGN and VRFY?",
            "input": "",
            "output": "SIGN = ✦ (sign program), VRFY = ✓ (verify signature). Example: ✦ agent=my_agent key=ed25519:... sig=... signs the program. ✓ @self @my_agent verifies it.",
        },
        {
            "instruction": "Why is SIGN/VRFY important for machine-to-machine NML communication?",
            "input": "",
            "output": "When LLMs exchange NML programs, SIGN/VRFY provides a trust chain. The producing model signs its output; the consuming model verifies before executing. This prevents tampering, ensures provenance, and enables audit trails. Without it, a receiving model has no way to know if the program was modified in transit.",
        },
    ])

    # --- VOTE ---
    pairs.extend([
        {
            "instruction": "What does VOTE do in NML?",
            "input": "",
            "output": "VOTE Rd Rs #strategy [#threshold] performs multi-agent consensus on a vector Rs of results from N agents. Strategies: #0=median, #1=mean, #2=quorum, #3=min, #4=max. For quorum (strategy 2), threshold specifies the minimum number of agents that must agree.",
        },
        {
            "instruction": "Write NML to take the median of 5 agent results.",
            "input": "",
            "output": "LD    R0 @agent_results\nVOTE  RA R0 #0\nST    RA @consensus_value\nHALT",
        },
        {
            "instruction": "Write NML for quorum voting: check if at least 3 of 5 agents agree.",
            "input": "",
            "output": "LD    R0 @agent_results\nVOTE  RE R0 #2 #3\nJMPF  #2\nTRAP  #1\nJUMP  #1\nLD    RA @agent_results\nVOTE  RA RA #0\nST    RA @verified_result\nHALT",
        },
        {
            "instruction": "What is the symbolic opcode for VOTE?",
            "input": "",
            "output": "The symbolic opcode for VOTE (consensus) is ⚖. Example: ⚖ α ι #0 computes the median of values in ι. ⚖ φ ι #2 #4 checks if at least 4 agents agree (quorum).",
        },
        {
            "instruction": "How would you use VOTE in a multi-model NML architecture?",
            "input": "",
            "output": "Multiple specialist LLMs each generate an NML program for the same task. Each program runs through the C runtime producing a result. The results are collected into a tensor, and VOTE #0 (median) or VOTE #2 (quorum) determines the consensus answer. This eliminates outlier hallucinations — if 4 of 5 models agree, the answer is trustworthy.",
        },
    ])

    # --- PROJ ---
    pairs.extend([
        {
            "instruction": "What does PROJ do in NML?",
            "input": "",
            "output": "PROJ Rd Rs Rmatrix performs embedding projection with L2 normalization: Rd = normalize(Rs @ Rmatrix). The output has unit norm. Used to project features into a shared embedding space for comparison.",
        },
        {
            "instruction": "Write NML to project a feature vector into a 64-dim embedding.",
            "input": "",
            "output": "LD    R0 @features\nLD    R1 @projection_matrix\nPROJ  R2 R0 R1\nST    R2 @embedding\nHALT",
        },
        {
            "instruction": "What is the symbolic opcode for PROJ?",
            "input": "",
            "output": "The symbolic opcode for PROJ (embedding projection) is ⟐. Example: ⟐ λ ι κ projects ι through matrix κ with L2 normalization, storing in λ.",
        },
    ])

    # --- DIST ---
    pairs.extend([
        {
            "instruction": "What does DIST do in NML?",
            "input": "",
            "output": "DIST Rd Rs1 Rs2 #metric computes the distance between two embeddings. Metrics: #0=cosine distance, #1=euclidean distance, #2=dot product. Both inputs should have the same shape.",
        },
        {
            "instruction": "Write NML to compute cosine similarity between two embeddings.",
            "input": "",
            "output": "LD    R0 @embedding_a\nLD    R1 @embedding_b\nDIST  R2 R0 R1 #0\nST    R2 @cosine_distance\nHALT",
        },
        {
            "instruction": "Write NML to find the nearest neighbor in an embedding space.",
            "input": "",
            "output": (
                "LD    R0 @query_embedding\n"
                "LD    R1 @candidate_1\nLD    R2 @candidate_2\nLD    R3 @candidate_3\n"
                "DIST  R4 R0 R1 #0\nDIST  R5 R0 R2 #0\nDIST  R6 R0 R3 #0\n"
                "CMP   R4 R5\nJMPT  #2\nMOV   R4 R5\nMOV   R7 R2\n"
                "CMP   R4 R6\nJMPT  #2\nMOV   R4 R6\nMOV   R7 R3\n"
                "ST    R7 @nearest\nST    R4 @min_distance\nHALT"
            ),
        },
        {
            "instruction": "What is the symbolic opcode for DIST?",
            "input": "",
            "output": "The symbolic opcode for DIST (embedding distance) is ⟂. Example: ⟂ λ ι κ #0 computes cosine distance between ι and κ, storing in λ. Metrics: #0=cosine, #1=euclidean, #2=dot.",
        },
    ])

    # --- Semantic type annotations ---
    pairs.extend([
        {
            "instruction": "What are NML semantic type annotations?",
            "input": "",
            "output": "NML v0.6 supports optional :type suffixes on register operands. Types: float (default), currency, ratio, category, count, bool, embedding, probability. Example: ↓ ι:currency @gross_pay declares that R0 holds a monetary value. Types are advisory by default; compile with -DNML_TYPE_CHECK for runtime enforcement.",
        },
        {
            "instruction": "Write NML with typed register annotations for a rate calculation.",
            "input": "",
            "output": (
                "LD    R0:currency @principal\n"
                "LD    R1:ratio @interest_rate\n"
                "EMUL  RA:currency R0 R1\n"
                "ST    RA:currency @interest_amount\n"
                "HALT"
            ),
        },
        {
            "instruction": "What type compatibility rules does NML enforce?",
            "input": "",
            "output": "currency * ratio = currency. currency + currency = currency. currency - currency = currency. currency / count = currency. CMPF on any type = bool. SOFT output = probability. PROJ output = embedding. DIST of embeddings = float. Incompatible ops (e.g. currency * currency) produce semantic warnings.",
        },
    ])

    # --- Combined M2M examples ---
    pairs.extend([
        {
            "instruction": "Write a complete self-describing NML program with META, fragments, and signing using symbolic syntax.",
            "input": "",
            "output": (
                "✦  agent=nml_core_v1  key=ed25519:abc123  sig=def456\n"
                '§  @name       "multi_rate_calc"\n'
                '§  @version    "1.0"\n'
                '§  @domain     "finance"\n'
                '§  @input      amount  currency  "Input amount"\n'
                '§  @output     result  currency  "Final amount"\n\n'
                "◆  apply_rate\n"
                "↓  ι:currency  @amount\n"
                "∗  α:currency  ι  #0.050000\n"
                "↑  α:currency  @rate_result\n"
                "◇\n\n"
                "◆  main\n"
                "✓  @self  @nml_core_v1\n"
                "⊕  @apply_rate\n"
                "↓  α  @rate_result\n"
                "↑  α  @result\n"
                "◼\n"
                "◇"
            ),
        },
        {
            "instruction": "Explain the NML M2M extension and why it matters for multi-agent systems.",
            "input": "",
            "output": "NML M2M (v0.6) adds 11 instructions enabling programs to be self-describing (META), composable (FRAG/ENDF/LINK), differentially updatable (PTCH), cryptographically signed (SIGN/VRFY), consensus-driven (VOTE), and embedding-aware (PROJ/DIST). This matters because when multiple LLMs communicate via NML, they need: trust (SIGN/VRFY), composition (FRAG/LINK), efficient updates (PTCH), and agreement mechanisms (VOTE). Together these turn NML from a single-model execution language into a multi-agent communication protocol.",
        },
    ])

    return pairs


def main():
    parser = argparse.ArgumentParser(description="Generate NML extension training pairs")
    parser.add_argument("--output", default=str(OUTPUT_PATH))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("Generating NML extension + M2M training pairs...\n")

    vision = gen_vision_pairs()
    print(f"  NML-V (Vision):      {len(vision)} pairs")

    transformer = gen_transformer_pairs()
    print(f"  NML-T (Transformer): {len(transformer)} pairs")

    reduction = gen_reduction_pairs()
    print(f"  NML-R (Reduction):   {len(reduction)} pairs")

    signal = gen_signal_pairs()
    print(f"  NML-S (Signal):      {len(signal)} pairs")

    m2m = gen_m2m_pairs()
    print(f"  NML-M2M:             {len(m2m)} pairs")

    all_pairs = vision + transformer + reduction + signal + m2m
    random.shuffle(all_pairs)

    with open(output_path, "w") as f:
        for pair in all_pairs:
            f.write(json.dumps(pair) + "\n")

    print(f"\n  Total: {len(all_pairs)} pairs → {output_path}")

    by_type = {}
    for p in all_pairs:
        text = p.get("output", "")
        for op in ["CONV","POOL","UPSC","PADZ","ATTN","NORM","EMBD","GELU",
                    "RDUC","WHER","CLMP","CMPR","FFT","FILT",
                    "META","FRAG","ENDF","LINK","PTCH","SIGN","VRFY","VOTE","PROJ","DIST",
                    "⊛","⊓","⊔","⊡","⊙","‖","⊏","ℊ",
                    "⊥","⊻","⊧","⊜","∿","⋐",
                    "§","◆","◇","⊿","✦","✓","⚖","⟐","⟂"]:
            if op in text or op in p.get("instruction", ""):
                by_type[op] = by_type.get(op, 0) + 1

    print(f"\n  Opcode coverage:")
    for op, count in sorted(by_type.items()):
        if len(op) <= 4:
            print(f"    {op:6s} {count} pairs")


if __name__ == "__main__":
    main()
