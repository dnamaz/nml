#!/usr/bin/env python3
"""
Generate ~15K training pairs for backward opcodes (NML-TR v0.9).

Covers backward opcodes + fused training:
  RELUBK, SIGMBK, TANHBK, GELUBK, SOFTBK (activation backward)
  MMULBK (matmul backward)
  CONVBK, POOLBK (vision backward)
  NORMBK (layer norm backward)
  ATTNBK (attention backward)
  TRAIN + INFER (fused N-layer dense training and inference)

Patterns:
  - Individual backward opcode usage (~3K)
  - Dense forward-backward-update loops, 2-4 layers (~4K)
  - CNN forward-backward-update loops (~3K)
  - Attention + norm backward loops (~2K)
  - TRAIN+INFER convenience examples (~1K)
  - Mixed architecture training loops (~2K)

Tri-syntax: ~60% classic, 25% symbolic, 15% verbose.

Output: domain/output/training/nml_backward_pairs.jsonl
"""

import json
import random
import argparse
from pathlib import Path

OUTPUT_PATH = Path(__file__).parent.parent / "domain" / "output" / "training" / "nml_backward_pairs.jsonl"

random.seed(2028)

from nml_core_training_gen import (
    REGS_CLASSIC, REGS_GREEK, SYM, VERBOSE,
    INPUT_NAMES, OUTPUT_NAMES, MEM_NAMES,
    _pick, _fval, _ival, _fmt, _pair,
    apply_syntax, pick_syntax, syntax_tag, _inp, _out, _mem,
)

SYM.update({
    "RELUBK": "⌐ˈ", "SIGMBK": "σˈ", "TANHBK": "τˈ",
    "GELUBK": "ℊˈ", "SOFTBK": "Σˈ",
    "MMULBK": "×ˈ", "CONVBK": "⊛ˈ", "POOLBK": "⊓ˈ",
    "NORMBK": "‖ˈ", "ATTNBK": "⊙ˈ", "TRAIN": "⥁", "INFER": "⊸",
})
VERBOSE.update({
    "RELUBK": "RELU_BACKWARD", "SIGMBK": "SIGMOID_BACKWARD",
    "TANHBK": "TANH_BACKWARD", "GELUBK": "GELU_BACKWARD",
    "SOFTBK": "SOFTMAX_BACKWARD", "MMULBK": "MATMUL_BACKWARD",
    "CONVBK": "CONV_BACKWARD", "POOLBK": "POOL_BACKWARD",
    "NORMBK": "NORM_BACKWARD", "ATTNBK": "ATTN_BACKWARD",
    "TRAIN": "TRAIN", "INFER": "INFER",
})

ACTIVATIONS = ["RELU", "SIGM", "TANH", "GELU"]
ACT_BK          = {"RELU": "RELUBK", "SIGM": "SIGMBK", "TANH": "TANHBK", "GELU": "GELUBK"}
ACT_BK_ALIAS    = {"RELU": "RELU_BK", "SIGM": "SIGM_BK", "TANH": "TANH_BK", "GELU": "GELU_BK"}
ACT_NAMES = {"RELU": "ReLU", "SIGM": "sigmoid", "TANH": "tanh", "GELU": "GELU"}

# Pick canonical or _BK alias form (~30% alias) so training data covers both
def _bk(act):
    return ACT_BK_ALIAS[act] if random.random() < 0.3 else ACT_BK[act]

_OTHER_BK = {
    "SIGMBK": "SIGM_BK", "TANHBK": "TANH_BK", "GELUBK": "GELU_BK", "SOFTBK": "SOFT_BK",
    "MMULBK": "MMUL_BK", "CONVBK": "CONV_BK", "POOLBK": "POOL_BK",
    "NORMBK": "NORM_BK", "ATTNBK": "ATTN_BK",
}

def _obk(canonical):
    return _OTHER_BK[canonical] if random.random() < 0.3 else canonical
GRAD_REGS = ["RG", "RH", "RI", "RJ", "RK", "RL"]
WEIGHT_NAMES = ["weights", "w1", "w2", "w3", "kernel", "params", "matrix", "projection"]
BIAS_NAMES = ["bias", "b1", "b2", "b3", "offset", "shift"]
LR_VALUES = ["0.001", "0.01", "0.005", "0.0001", "0.1", "0.05", "0.0005"]
EPOCH_VALUES = ["100", "500", "1000", "2000", "5000", "200", "300"]


def _lr():
    return random.choice(LR_VALUES)

def _epochs():
    return random.choice(EPOCH_VALUES)

def _r(exclude=()):
    pool = [r for r in REGS_CLASSIC[:16] if r not in exclude]
    return random.choice(pool)

def _gr():
    return random.choice(GRAD_REGS)


# ═══════════════════════════════════════════════════════════════════════════════
# Category 1: Individual backward opcode usage
# ═══════════════════════════════════════════════════════════════════════════════

def gen_individual_activation_backward(count):
    """Individual RELUBK/SIGMBK/TANHBK/GELUBK/SOFTBK usage."""
    pairs = []
    prompts_bk = [
        "Write NML to compute the {act} backward pass on {inp}",
        "Use {op} to propagate gradient through {act} activation",
        "Compute gradient of {act} applied to {inp}",
        "Write NML for {act} backward: multiply upstream gradient by {act} derivative",
        "Backpropagate through {act} activation on {inp}",
    ]
    for _ in range(count):
        syntax = pick_syntax()
        act = random.choice(ACTIVATIONS)
        bk_op = _bk(act)
        act_name = ACT_NAMES[act]
        inp = _inp(); grad_name = random.choice(["gradient", "upstream_grad", "d_loss", "grad"])
        out_name = random.choice(["d_input", "grad_out", "backward_result", "d_activation"])

        q = random.choice(prompts_bk).format(act=act_name, op=bk_op, inp=inp) + syntax_tag(syntax)
        lines = [
            _fmt("LD", "R0", f"@{grad_name}"),
            _fmt("LD", "R1", f"@{inp}"),
            _fmt(bk_op, "R2", "R0", "R1"),
            _fmt("ST", "R2", f"@{out_name}"),
            "HALT",
        ]
        pairs.append(_pair(q, apply_syntax(lines, syntax)))
    return pairs


def gen_individual_softmax_backward(count):
    """SOFTBK usage with context about Jacobian-vector product."""
    pairs = []
    prompts = [
        "Write NML for softmax backward pass on {inp}",
        "Compute softmax gradient using SOFTBK on {inp}",
        "Backpropagate through softmax layer applied to {inp}",
    ]
    for _ in range(count):
        syntax = pick_syntax()
        inp = _inp()
        q = random.choice(prompts).format(inp=inp) + syntax_tag(syntax)
        lines = [
            _fmt("LD", "R0", f"@upstream_grad"),
            _fmt("LD", "R1", f"@{inp}"),
            _fmt("SOFTBK", "R2", "R0", "R1"),
            _fmt("ST", "R2", f"@d_{inp}"),
            "HALT",
        ]
        pairs.append(_pair(q, apply_syntax(lines, syntax)))
    return pairs


def gen_individual_mmulbk(count):
    """MMULBK usage producing both d_input and d_weight."""
    pairs = []
    prompts = [
        "Write NML to compute matmul backward: get gradients for both input and weights",
        "Use MMULBK to backpropagate through a linear layer",
        "Compute d_input and d_weight from upstream gradient through matrix multiply",
        "Write NML for matmul backward pass on {w}",
        "Backpropagate through MMUL using MMULBK",
    ]
    for _ in range(count):
        syntax = pick_syntax()
        w = random.choice(WEIGHT_NAMES)
        inp = _inp()
        q = random.choice(prompts).format(w=w, inp=inp) + syntax_tag(syntax)
        lines = [
            _fmt("LD", "R0", f"@upstream_grad"),
            _fmt("LD", "R1", f"@{inp}"),
            _fmt("LD", "R2", f"@{w}"),
            _fmt("MMULBK", "R3", "R4", "R0", "R1", "R2"),
            _fmt("ST", "R3", f"@d_{inp}"),
            _fmt("ST", "R4", f"@d_{w}"),
            "HALT",
        ]
        pairs.append(_pair(q, apply_syntax(lines, syntax)))
    return pairs


def gen_individual_convbk(count):
    """CONVBK usage for conv2d backward."""
    pairs = []
    prompts = [
        "Write NML for convolution backward pass",
        "Use CONVBK to get input gradient and kernel gradient",
        "Backpropagate through a conv layer using CONVBK",
        "Compute d_input and d_kernel for a convolution layer",
    ]
    for _ in range(count):
        syntax = pick_syntax()
        q = random.choice(prompts) + syntax_tag(syntax)
        lines = [
            _fmt("LD", "R0", f"@conv_grad"),
            _fmt("LD", "R1", f"@image"),
            _fmt("LD", "R2", f"@kernel"),
            _fmt("CONVBK", "R3", "R4", "R0", "R1", "R2"),
            _fmt("ST", "R3", f"@d_image"),
            _fmt("ST", "R4", f"@d_kernel"),
            "HALT",
        ]
        pairs.append(_pair(q, apply_syntax(lines, syntax)))
    return pairs


def gen_individual_poolbk(count):
    """POOLBK usage for max pool backward."""
    pairs = []
    prompts = [
        "Write NML for max pool backward pass",
        "Route gradient through max pool using POOLBK",
        "Backpropagate through a 2x2 max pool layer",
        "Use POOLBK to compute gradient through pooling",
    ]
    pool_sizes = [2, 3]
    for _ in range(count):
        syntax = pick_syntax()
        ps = random.choice(pool_sizes)
        q = random.choice(prompts) + syntax_tag(syntax)
        lines = [
            _fmt("LD", "R0", f"@pool_grad"),
            _fmt("LD", "R1", f"@pool_input"),
            _fmt("POOLBK", "R2", "R0", "R1", f"#{ps}", f"#{ps}"),
            _fmt("ST", "R2", f"@d_input"),
            "HALT",
        ]
        pairs.append(_pair(q, apply_syntax(lines, syntax)))
    return pairs


def gen_individual_normbk(count):
    """NORMBK usage for layer norm backward."""
    pairs = []
    prompts = [
        "Write NML for layer normalization backward pass",
        "Backpropagate through layer norm using NORMBK",
        "Compute gradient through LayerNorm on {inp}",
        "Use NORMBK for norm backward",
    ]
    for _ in range(count):
        syntax = pick_syntax()
        inp = _inp()
        q = random.choice(prompts).format(inp=inp) + syntax_tag(syntax)
        lines = [
            _fmt("LD", "R0", f"@upstream_grad"),
            _fmt("LD", "R1", f"@{inp}"),
            _fmt("NORMBK", "R2", "R0", "R1"),
            _fmt("ST", "R2", f"@d_{inp}"),
            "HALT",
        ]
        pairs.append(_pair(q, apply_syntax(lines, syntax)))
    return pairs


def gen_individual_attnbk(count):
    """ATTNBK usage for attention backward."""
    pairs = []
    prompts = [
        "Write NML for attention backward pass producing dQ, dK, dV",
        "Backpropagate through self-attention using ATTNBK",
        "Compute gradients for Q, K, V using ATTNBK",
        "Use ATTNBK to get attention backward gradients",
    ]
    for _ in range(count):
        syntax = pick_syntax()
        q = random.choice(prompts) + syntax_tag(syntax)
        lines = [
            _fmt("LD", "R0", f"@attn_grad"),
            _fmt("LD", "R1", f"@query"),
            _fmt("LD", "R2", f"@key"),
            _fmt("LD", "R3", f"@value"),
            _fmt("ATTNBK", "R5", "R0", "R1", "R2", "R3"),
            _fmt("ST", "R5", f"@d_query"),
            _fmt("ST", "R6", f"@d_key"),
            _fmt("ST", "R7", f"@d_value"),
            "HALT",
        ]
        pairs.append(_pair(q, apply_syntax(lines, syntax)))
    return pairs


# ═══════════════════════════════════════════════════════════════════════════════
# Category 2: Dense forward-backward-update training loops
# ═══════════════════════════════════════════════════════════════════════════════

def gen_dense_2layer_training(count):
    """2-layer dense network training: forward -> loss -> backward -> update."""
    pairs = []
    prompts = [
        "Write NML to train a 2-layer dense network with {act} activation",
        "Train a {act} network: forward pass, loss, backward, weight update",
        "Write a complete NML training loop for a 2-layer {act} network",
        "NML training: dense({act}) -> dense(sigmoid) with MSE loss and weight update",
    ]
    for _ in range(count):
        syntax = pick_syntax()
        act = random.choice(ACTIVATIONS)
        act_name = ACT_NAMES[act]
        bk = _bk(act)
        lr = _lr()
        epochs = _epochs()
        q = random.choice(prompts).format(act=act_name) + syntax_tag(syntax)
        lines = [
            _fmt("LD", "R0", "@input"),
            _fmt("LD", "R1", "@w1"),
            _fmt("LD", "R2", "@b1"),
            _fmt("LD", "R3", "@w2"),
            _fmt("LD", "R4", "@b2"),
            _fmt("LD", "R9", "@target"),
            "",
            _fmt("LOOP", f"#{epochs}"),
            f"  ; Forward",
            f"  {_fmt('MMUL', 'R5', 'R0', 'R1')}",
            f"  {_fmt('MADD', 'R5', 'R5', 'R2')}",
            f"  {_fmt(act, 'R6', 'R5')}",
            f"  {_fmt('MMUL', 'R7', 'R6', 'R3')}",
            f"  {_fmt('MADD', 'R7', 'R7', 'R4')}",
            f"  {_fmt('SIGM', 'R8', 'R7')}",
            f"  {_fmt('LOSS', 'RA', 'R8', 'R9', '#0')}",
            f"  ; Backward",
            f"  {_fmt(_obk('SIGMBK'), 'RG', 'RA', 'R7')}",
            f"  {_fmt(_obk('MMULBK'), 'RH', 'RI', 'RG', 'R6', 'R3')}",
            f"  {_fmt(bk, 'RJ', 'RH', 'R5')}",
            f"  {_fmt(_obk('MMULBK'), 'RK', 'RL', 'RJ', 'R0', 'R1')}",
            f"  ; Update",
            f"  {_fmt('WUPD', 'R3', 'RI', f'#{lr}')}",
            f"  {_fmt('WUPD', 'R4', 'RG', f'#{lr}')}",
            f"  {_fmt('WUPD', 'R1', 'RL', f'#{lr}')}",
            f"  {_fmt('WUPD', 'R2', 'RJ', f'#{lr}')}",
            "ENDP",
            "",
            _fmt("ST", "RA", "@final_loss"),
            "HALT",
        ]
        pairs.append(_pair(q, apply_syntax([l for l in lines if l is not None], syntax)))
    return pairs


def gen_dense_3layer_training(count):
    """3-layer dense network training loop."""
    pairs = []
    prompts = [
        "Write NML for a 3-layer dense training loop with {a1} and {a2}",
        "Train a 3-layer network: dense({a1}) -> dense({a2}) -> sigmoid",
        "Complete 3-layer NML training with backward opcodes",
    ]
    for _ in range(count):
        syntax = pick_syntax()
        a1, a2 = random.sample(ACTIVATIONS, 2)
        lr = _lr()
        epochs = _epochs()
        q = random.choice(prompts).format(a1=ACT_NAMES[a1], a2=ACT_NAMES[a2]) + syntax_tag(syntax)
        lines = [
            _fmt("LD", "R0", "@input"),
            _fmt("LD", "R1", "@w1"), _fmt("LD", "R2", "@b1"),
            _fmt("LD", "R3", "@w2"), _fmt("LD", "R4", "@b2"),
            _fmt("LD", "R5", "@w3"), _fmt("LD", "R6", "@b3"),
            _fmt("LD", "R9", "@target"),
            _fmt("LOOP", f"#{epochs}"),
            f"  {_fmt('MMUL', 'R7', 'R0', 'R1')}",
            f"  {_fmt('MADD', 'R7', 'R7', 'R2')}",
            f"  {_fmt(a1, 'R8', 'R7')}",
            f"  {_fmt('MMUL', 'RA', 'R8', 'R3')}",
            f"  {_fmt('MADD', 'RA', 'RA', 'R4')}",
            f"  {_fmt(a2, 'RB', 'RA')}",
            f"  {_fmt('MMUL', 'RC', 'RB', 'R5')}",
            f"  {_fmt('MADD', 'RC', 'RC', 'R6')}",
            f"  {_fmt('SIGM', 'RD', 'RC')}",
            f"  {_fmt('LOSS', 'RE', 'RD', 'R9', '#0')}",
            f"  {_fmt(_obk('SIGMBK'), 'RG', 'RE', 'RC')}",
            f"  {_fmt(_obk('MMULBK'), 'RH', 'RI', 'RG', 'RB', 'R5')}",
            f"  {_fmt(_bk(a2), 'RJ', 'RH', 'RA')}",
            f"  {_fmt(_obk('MMULBK'), 'RK', 'RL', 'RJ', 'R8', 'R3')}",
            f"  {_fmt(_bk(a1), 'RM', 'RK', 'R7')}",
            f"  {_fmt(_obk('MMULBK'), 'RN', 'RO', 'RM', 'R0', 'R1')}",
            f"  {_fmt('WUPD', 'R5', 'RI', f'#{lr}')}",
            f"  {_fmt('WUPD', 'R6', 'RG', f'#{lr}')}",
            f"  {_fmt('WUPD', 'R3', 'RL', f'#{lr}')}",
            f"  {_fmt('WUPD', 'R4', 'RJ', f'#{lr}')}",
            f"  {_fmt('WUPD', 'R1', 'RO', f'#{lr}')}",
            f"  {_fmt('WUPD', 'R2', 'RM', f'#{lr}')}",
            "ENDP",
            _fmt("ST", "RE", "@final_loss"),
            "HALT",
        ]
        pairs.append(_pair(q, apply_syntax(lines, syntax)))
    return pairs


# ═══════════════════════════════════════════════════════════════════════════════
# Category 3: CNN forward-backward-update loops
# ═══════════════════════════════════════════════════════════════════════════════

def gen_cnn_training(count):
    """CNN training: conv -> relu -> pool -> dense -> loss -> backward."""
    pairs = []
    prompts = [
        "Write NML to train a CNN: conv -> {act} -> pool -> dense",
        "Train a convolutional neural network with backward opcodes",
        "Complete CNN training loop with CONVBK, {actbk}, POOLBK, MMULBK",
        "NML CNN training: convolution forward and backward with weight updates",
    ]
    for _ in range(count):
        syntax = pick_syntax()
        act = random.choice(["RELU", "GELU"])
        act_name = ACT_NAMES[act]
        bk = _bk(act)
        lr = _lr()
        epochs = _epochs()
        q = random.choice(prompts).format(act=act_name, actbk=bk) + syntax_tag(syntax)
        lines = [
            _fmt("LD", "R0", "@image"),
            _fmt("LD", "R1", "@conv_kernel"),
            _fmt("LD", "R2", "@dense_weights"),
            _fmt("LD", "R3", "@dense_bias"),
            _fmt("LD", "R9", "@target"),
            _fmt("LOOP", f"#{epochs}"),
            f"  ; Forward: conv -> {act_name} -> pool -> dense -> sigmoid",
            f"  {_fmt('CONV', 'R4', 'R0', 'R1')}",
            f"  {_fmt(act, 'R5', 'R4')}",
            f"  {_fmt('POOL', 'R6', 'R5')}",
            f"  {_fmt('MMUL', 'R7', 'R6', 'R2')}",
            f"  {_fmt('MADD', 'R7', 'R7', 'R3')}",
            f"  {_fmt('SIGM', 'R8', 'R7')}",
            f"  {_fmt('LOSS', 'RA', 'R8', 'R9', '#0')}",
            f"  ; Backward",
            f"  {_fmt(_obk('SIGMBK'), 'RG', 'RA', 'R7')}",
            f"  {_fmt(_obk('MMULBK'), 'RH', 'RI', 'RG', 'R6', 'R2')}",
            f"  {_fmt(_obk('POOLBK'), 'RJ', 'RH', 'R5')}",
            f"  {_fmt(bk, 'RK', 'RJ', 'R4')}",
            f"  {_fmt(_obk('CONVBK'), 'RL', 'RM', 'RK', 'R0', 'R1')}",
            f"  ; Update",
            f"  {_fmt('WUPD', 'R1', 'RM', f'#{lr}')}",
            f"  {_fmt('WUPD', 'R2', 'RI', f'#{lr}')}",
            f"  {_fmt('WUPD', 'R3', 'RG', f'#{lr}')}",
            "ENDP",
            _fmt("ST", "RA", "@final_loss"),
            "HALT",
        ]
        pairs.append(_pair(q, apply_syntax(lines, syntax)))
    return pairs


# ═══════════════════════════════════════════════════════════════════════════════
# Category 4: Attention + norm backward loops
# ═══════════════════════════════════════════════════════════════════════════════

def gen_attention_training(count):
    """Attention + LayerNorm + dense training loop."""
    pairs = []
    prompts = [
        "Write NML to train an attention layer with ATTNBK and NORMBK",
        "Train a transformer block: attention -> norm -> dense with backward ops",
        "NML attention training loop with ATTNBK, NORMBK, MMULBK",
        "Complete backward pass through attention + layer norm",
    ]
    for _ in range(count):
        syntax = pick_syntax()
        lr = _lr()
        epochs = _epochs()
        q = random.choice(prompts) + syntax_tag(syntax)
        lines = [
            _fmt("LD", "R0", "@query"),
            _fmt("LD", "R1", "@key"),
            _fmt("LD", "R2", "@value"),
            _fmt("LD", "R3", "@dense_w"),
            _fmt("LD", "R9", "@target"),
            _fmt("LOOP", f"#{epochs}"),
            f"  ; Forward: attention -> norm -> dense",
            f"  {_fmt('ATTN', 'R4', 'R0', 'R1', 'R2')}",
            f"  {_fmt('NORM', 'R5', 'R4')}",
            f"  {_fmt('MMUL', 'R6', 'R5', 'R3')}",
            f"  {_fmt('LOSS', 'R7', 'R6', 'R9', '#0')}",
            f"  ; Backward",
            f"  {_fmt(_obk('MMULBK'), 'R8', 'RA', 'R7', 'R5', 'R3')}",
            f"  {_fmt(_obk('NORMBK'), 'RB', 'R8', 'R4')}",
            f"  {_fmt(_obk('ATTNBK'), 'RC', 'RB', 'R0', 'R1', 'R2')}",
            f"  ; Update",
            f"  {_fmt('WUPD', 'R3', 'RA', f'#{lr}')}",
            "ENDP",
            _fmt("ST", "R7", "@final_loss"),
            "HALT",
        ]
        pairs.append(_pair(q, apply_syntax(lines, syntax)))
    return pairs


# ═══════════════════════════════════════════════════════════════════════════════
# Category 5: TRAIN+INFER convenience examples
# ═══════════════════════════════════════════════════════════════════════════════

def gen_train_infer(count):
    """TRAIN+INFER fused N-layer dense training with architecture descriptor."""
    pairs = []
    prompts = [
        "Write NML to train a {n}-layer dense network using TRAIN+INFER",
        "Use TRAIN and INFER for fused {n}-layer training with {opt}",
        "Train a {n}-layer network with TRAIN+INFER (epochs={ep}, lr={lr})",
        "NML fused deep training: TRAIN with {n} layers then INFER",
        "Write NML using TRAIN to train a {desc} network",
        "Use TRAIN with {opt} optimizer to train on @training_data for {ep} epochs",
        "Write NML that loads architecture into RV, config into RU, and calls TRAIN",
        "Train a neural network using the TRAIN opcode with architecture in RV and config in RU",
        "Write NML using TRAIN with named data references @training_data and @training_labels",
        "Use the TRAIN opcode to train a {n}-layer {desc} network with {opt}, then INFER",
        "Write NML: load RV=arch, ALLC RU=config, TRAIN on data, INFER for prediction",
        "Train a {desc} network with TRAIN — pass data as @training_data @training_labels, INFER result into R8",
    ]
    configs = [
        (2, [4, 0, 1, 0], "2-layer 4-hidden ReLU"),
        (2, [8, 0, 1, 1], "2-layer 8-hidden ReLU+sigmoid"),
        (3, [8, 0, 4, 0, 1, 0], "3-layer 8-4-1"),
        (2, [16, 0, 1, 0], "2-layer 16-hidden ReLU"),
        (2, [4, 2, 1, 1], "2-layer tanh+sigmoid"),
        (2, [4, 0, 1, 0], "2-layer 4-hidden ReLU"),
        (2, [8, 1, 1, 0], "2-layer sigmoid+ReLU"),
        (2, [32, 0, 1, 0], "2-layer 32-hidden ReLU"),
    ]
    data_names = [
        ("training_data", "training_labels"),
        ("features", "targets"),
        ("input_data", "labels"),
        ("train_x", "train_y"),
        ("samples", "ground_truth"),
    ]
    for _ in range(count):
        syntax = pick_syntax()
        n_layers, arch, desc = random.choice(configs)
        opt = random.choice(["SGD", "Adam"])
        opt_val = "0" if opt == "SGD" else "1"
        lr = _lr()
        ep = _epochs()
        data_ref, label_ref = random.choice(data_names)
        q = random.choice(prompts).format(n=n_layers, opt=opt, ep=ep, lr=lr, desc=desc) + syntax_tag(syntax)

        arch_data = [float(n_layers)] + [float(x) for x in arch]

        # TRAIN+INFER pattern:
        #   ALLC RV [size] arch_descriptor
        #   ALLC RU [6] epochs,lr,optimizer,print_every,patience,min_delta
        #   TRAIN RU @data @labels
        #   INFER R8 R0
        lines = [
            f"; TRAIN+INFER: {desc} with {opt}",
            f"; RV = architecture descriptor, RU = training config",
        ]
        arch_size = len(arch_data)
        arch_str = ",".join(str(int(v)) if v == int(v) else str(v) for v in arch_data)
        lines.append(_fmt("ALLC", "RV", f"[{arch_size}]", arch_str))
        lines.append(_fmt("ALLC", "RU", "[6]", f"{ep},{lr},{opt_val},0,0,0"))
        lines.append(_fmt("TRAIN", "RU", f"@{data_ref}", f"@{label_ref}"))
        lines.append(_fmt("INFER", "R8", "R0"))
        lines.append(_fmt("ST", "R8", "@final_loss"))
        lines.append("HALT")
        pairs.append(_pair(q, apply_syntax([l for l in lines], syntax)))
    return pairs


# ═══════════════════════════════════════════════════════════════════════════════
# Category 6: Mixed architecture training
# ═══════════════════════════════════════════════════════════════════════════════

def gen_mixed_training(count):
    """Mixed: dense + activation backward in varied patterns."""
    pairs = []
    prompts = [
        "Write NML for a single training step: forward, loss, backward, update on {inp}",
        "One training iteration for a {act} dense layer on {inp}",
        "Manual gradient descent step with MMULBK and {bk}",
        "Forward + backward pass for a dense + {act} layer",
    ]
    for _ in range(count):
        syntax = pick_syntax()
        act = random.choice(ACTIVATIONS)
        bk = _bk(act)
        inp = _inp()
        w = random.choice(WEIGHT_NAMES)
        lr = _lr()
        q = random.choice(prompts).format(act=ACT_NAMES[act], bk=bk, inp=inp) + syntax_tag(syntax)
        lines = [
            _fmt("LD", "R0", f"@{inp}"),
            _fmt("LD", "R1", f"@{w}"),
            _fmt("LD", "R9", "@target"),
            _fmt("MMUL", "R2", "R0", "R1"),
            _fmt(act, "R3", "R2"),
            _fmt("LOSS", "R4", "R3", "R9", "#0"),
            _fmt(bk, "RG", "R4", "R2"),
            _fmt("MMULBK", "RH", "RI", "RG", "R0", "R1"),
            _fmt("WUPD", "R1", "RI", f"#{lr}"),
            _fmt("ST", "R4", "@loss"),
            "HALT",
        ]
        pairs.append(_pair(q, apply_syntax(lines, syntax)))
    return pairs


def gen_inference_after_training(count):
    """Train then immediately run inference — the self-training use case."""
    pairs = []
    prompts = [
        "Write NML that trains on local data, then runs inference on new input",
        "Self-training: TRAIN to learn, then INFER for prediction",
        "Train a network and immediately use it for inference",
        "Edge self-training: train on calibration data, predict on live data",
    ]
    for _ in range(count):
        syntax = pick_syntax()
        lr = _lr()
        ep = _epochs()
        q = random.choice(prompts) + syntax_tag(syntax)
        lines = [
            "; Train on local data using TRAIN+INFER",
            _fmt("ALLC", "RV", "[5]", "2,8,0,1,0"),
            _fmt("ALLC", "RU", "[6]", f"{ep},{lr},1,0,0,0"),
            _fmt("TRAIN", "RU", "@training_input", "@training_target"),
            "",
            "; Inference on new data",
            _fmt("LD", "R0", "@live_input"),
            _fmt("INFER", "R8", "R0"),
            _fmt("ST", "R8", "@prediction"),
            "HALT",
        ]
        pairs.append(_pair(q, apply_syntax([l for l in lines if l is not None], syntax)))
    return pairs


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Generate NML backward opcode training pairs")
    parser.add_argument("--output", type=str, default=str(OUTPUT_PATH))
    args = parser.parse_args()

    print("=" * 60)
    print("  NML Backward Opcode Training Generator")
    print("=" * 60)

    categories = [
        ("Activation backward",       gen_individual_activation_backward, 1500),
        ("Softmax backward",           gen_individual_softmax_backward,   300),
        ("MMULBK individual",          gen_individual_mmulbk,             500),
        ("CONVBK individual",          gen_individual_convbk,             400),
        ("POOLBK individual",          gen_individual_poolbk,             400),
        ("NORMBK individual",          gen_individual_normbk,             400),
        ("ATTNBK individual",          gen_individual_attnbk,             400),
        ("Dense 2-layer training",     gen_dense_2layer_training,        2500),
        ("Dense 3-layer training",     gen_dense_3layer_training,        1500),
        ("CNN training loop",          gen_cnn_training,                 2500),
        ("Attention training loop",    gen_attention_training,           1500),
        ("TRAIN+INFER examples",        gen_train_infer,                  1000),
        ("Mixed single-step training", gen_mixed_training,              1500),
        ("Self-training + inference",  gen_inference_after_training,      600),
    ]

    all_pairs = []
    for name, gen_fn, count in categories:
        pairs = gen_fn(count)
        all_pairs.extend(pairs)
        print(f"  {name:<30} {len(pairs):>6}")

    random.shuffle(all_pairs)

    print(f"{'─' * 60}")
    print(f"  TOTAL:                        {len(all_pairs):>6}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for pair in all_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    print(f"\n  Written to: {out_path}")


if __name__ == "__main__":
    main()
