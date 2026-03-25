#!/usr/bin/env python3
"""
NML Coverage Fix Generator — targeted training pairs for the 8 opcodes
identified as under-represented after the first corpus build:

  BN, DROP, TLOG, TRAIN, INFER, WDECAY, SYS, SOFTBK

Each category targets ~400 pairs (well above the 200-example threshold).

Output: domain/output/training/nml_coverage_fix_pairs.jsonl
"""

import json
import random
from pathlib import Path

random.seed(42)
OUTPUT = Path(__file__).parent.parent / "domain" / "output" / "training" / "nml_coverage_fix_pairs.jsonl"


def pair(prompt, code):
    return {"messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": code}]}


def r():
    return random.choice(["R0","R1","R2","R3","R4","R5","R6","R7","R8","R9"])

def rand_reg(exclude=None):
    regs = ["R0","R1","R2","R3","R4","R5","R6","R7","R8","R9","RA","RB"]
    if exclude:
        regs = [r for r in regs if r not in exclude]
    return random.choice(regs)

def rand_lr():
    return random.choice([0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05])

def rand_epochs():
    return random.choice([100, 200, 500, 1000, 2000, 5000])

def rand_shape():
    return random.choice(["4,8", "8,16", "16,32", "32,64", "64,128", "128,256", "4,4", "8,8"])

INPUT_NAMES = [
    "x", "input", "features", "data", "batch", "samples", "signal",
    "activations", "hidden", "embeddings", "layer_out", "encoder_out",
    "repr", "latent", "context", "query", "key", "value", "logits",
    "raw_input", "preprocessed", "normalized", "pooled", "projected",
]

WEIGHT_NAMES = [
    ("w1", "b1"), ("w2", "b2"), ("w3", "b3"),
    ("fc1_w", "fc1_b"), ("fc2_w", "fc2_b"),
    ("layer1_w", "layer1_b"), ("layer2_w", "layer2_b"),
    ("enc_w", "enc_b"), ("dec_w", "dec_b"),
    ("proj_w", "proj_b"), ("out_w", "out_b"),
    ("weights", "bias"), ("kernel", "offset"),
]

OUTPUT_NAMES = [
    "predictions", "output", "scores", "logits", "probs",
    "result", "out", "y_hat", "y_pred", "activations_out",
    "final", "response", "decoded",
]

ARCH_NAMES = [
    "arch_desc", "model_arch", "network_config", "arch",
    "model_config", "layer_config", "net_arch",
]

LABEL_NAMES = [
    "labels", "targets", "y", "ground_truth", "expected",
    "train_y", "class_labels", "true_labels",
]

GAMMA_NAMES = ["gamma", "scale", "bn_gamma", "norm_scale"]
BETA_NAMES  = ["beta",  "shift", "bn_beta",  "norm_shift"]

def rand_name():
    return random.choice(INPUT_NAMES)

def rand_input():
    return random.choice(INPUT_NAMES)

def rand_output():
    return random.choice(OUTPUT_NAMES)

def rand_weights():
    return random.choice(WEIGHT_NAMES)

def rand_arch():
    return random.choice(ARCH_NAMES)

def rand_labels():
    return random.choice(LABEL_NAMES)

def rand_data_name():
    return random.choice(["train_x", "train_labels", "input_data", "features", "samples"])

def rand_decay():
    return random.choice([0.0001, 0.0005, 0.001, 0.01, 0.1])

def rand_rate():
    return random.choice([0.1, 0.2, 0.3, 0.4, 0.5])


# ═══════════════════════════════════════════════════════════════
# 1. BN — batch normalization
# ═══════════════════════════════════════════════════════════════

def gen_bn():
    pairs = []
    prompts_2op = [
        "Normalize a batch of {inp} tensors.",
        "Apply batch normalization to {inp}.",
        "Batch-normalize the {inp} activations before passing to the next layer.",
        "Normalize {inp} using NML BN opcode.",
        "Write a batch norm step on the {inp} tensor.",
        "Stabilize training: apply BN to {inp}.",
        "Apply BN before the activation function on {inp}.",
        "Add batch normalization to {inp}.",
    ]
    prompts_sym = [
        "Normalize {inp} using NML symbolic syntax.",
        "Batch normalization on {inp} — symbolic NML.",
        "Apply BN to {inp} (symbolic syntax).",
    ]
    prompts_verb = [
        "Normalize {inp} using verbose NML syntax.",
        "Batch normalization on {inp} — verbose NML.",
        "Apply BATCH_NORM to {inp} (verbose syntax).",
    ]
    prompts_4op = [
        "Apply batch normalization with learnable gamma and beta to {inp}.",
        "BN with scale ({gamma}) and shift ({beta}) parameters on {inp}.",
        "Normalize {inp} and apply affine transform: gamma={gamma}, beta={beta}.",
        "Batch-norm {inp} with trainable scale and bias.",
    ]

    # 2-operand
    for _ in range(200):
        src = rand_reg()
        dst = rand_reg(exclude=[src])
        inp = rand_input()
        out = rand_output()
        prompt = random.choice(prompts_2op).format(inp=inp)
        pairs.append(pair(prompt,
            f"LD {src} @{inp}\nBN {dst} {src}\nST {dst} @{out}\nHALT"))
        # symbolic
        inp2 = rand_input()
        out2 = rand_output()
        pairs.append(pair(random.choice(prompts_sym).format(inp=inp2),
            f"LD {src} @{inp2}\n\u229e {dst} {src}\nST {dst} @{out2}\nHALT"))
        # verbose
        inp3 = rand_input()
        out3 = rand_output()
        pairs.append(pair(random.choice(prompts_verb).format(inp=inp3),
            f"LD {src} @{inp3}\nBATCH_NORM {dst} {src}\nST {dst} @{out3}\nHALT"))

    # 4-operand (with gamma/beta)
    for _ in range(150):
        inp    = rand_input()
        out    = rand_output()
        g_name = random.choice(GAMMA_NAMES)
        b_name = random.choice(BETA_NAMES)
        prompt = random.choice(prompts_4op).format(inp=inp, gamma=g_name, beta=b_name)
        pairs.append(pair(prompt,
            f"LD R0 @{inp}\nLD R2 @{g_name}\nLD R3 @{b_name}\n"
            f"BN R1 R0 R2 R3\nST R1 @{out}\nHALT"))
        # in training context
        lbl = rand_labels()
        w, b = rand_weights()
        pairs.append(pair(
            f"Train with BN: load {inp}, matmul with {w}, batch-norm, ReLU, compute loss against {lbl}.",
            f"LD R0 @{inp}\nLD R1 @{w}\nLD R2 @{g_name}\nLD R3 @{b_name}\n"
            f"MMUL R4 R0 R1\nBN R5 R4 R2 R3\nRELU R6 R5\n"
            f"LD R9 @{lbl}\nLOSS R8 R6 R9\nST R8 @loss\nHALT"))

    return pairs


# ═══════════════════════════════════════════════════════════════
# 2. DROP — inverted dropout
# ═══════════════════════════════════════════════════════════════

def gen_drop():
    pairs = []
    prompts_classic = [
        "Apply dropout (rate={rate}) to {inp} to prevent overfitting.",
        "Add {rate} dropout regularization to {inp}.",
        "Write an inverted dropout layer on {inp} with p={rate}.",
        "Regularize {inp} with dropout probability {rate}.",
        "Apply DROP to {inp} at rate {rate}.",
        "Use inverted dropout (p={rate}) on the {inp} tensor.",
    ]
    prompts_sym = [
        "Apply {rate} dropout to {inp} in symbolic NML.",
        "Dropout on {inp} (rate={rate}) — symbolic syntax.",
        "Symbolic DROP on {inp}, p={rate}.",
    ]
    prompts_verb = [
        "Apply {rate} dropout to {inp} using verbose NML.",
        "DROPOUT on {inp} (rate={rate}) — verbose syntax.",
    ]
    prompts_infer = [
        "Disable dropout at inference time for {inp} (pass-through, p=0).",
        "Inference mode: bypass dropout on {inp}.",
        "Set dropout rate to 0 for {inp} during evaluation.",
    ]

    for _ in range(200):
        src  = rand_reg()
        dst  = rand_reg(exclude=[src])
        rate = rand_rate()
        inp  = rand_input()
        out  = rand_output()
        # classic
        pairs.append(pair(
            random.choice(prompts_classic).format(rate=rate, inp=inp),
            f"LD {src} @{inp}\nDROP {dst} {src} #{rate}\nST {dst} @{out}\nHALT"))
        # symbolic
        inp2 = rand_input()
        out2 = rand_output()
        pairs.append(pair(
            random.choice(prompts_sym).format(rate=rate, inp=inp2),
            f"LD {src} @{inp2}\n\u224b {dst} {src} #{rate}\nST {dst} @{out2}\nHALT"))
        # verbose
        inp3 = rand_input()
        out3 = rand_output()
        pairs.append(pair(
            random.choice(prompts_verb).format(rate=rate, inp=inp3),
            f"LD {src} @{inp3}\nDROPOUT {dst} {src} #{rate}\nST {dst} @{out3}\nHALT"))

    # p=0 inference passthrough
    for _ in range(80):
        src  = rand_reg()
        dst  = rand_reg(exclude=[src])
        inp  = rand_input()
        out  = rand_output()
        pairs.append(pair(
            random.choice(prompts_infer).format(inp=inp),
            f"LD {src} @{inp}\nDROP {dst} {src} #0.0\nST {dst} @{out}\nHALT"))

    # DROP + BN together
    for _ in range(60):
        inp  = rand_input()
        out  = rand_output()
        rate = rand_rate()
        pairs.append(pair(
            f"Apply batch normalization then {rate} dropout to {inp}.",
            f"LD R0 @{inp}\nBN R1 R0\nDROP R2 R1 #{rate}\nST R2 @{out}\nHALT"))

    return pairs


# ═══════════════════════════════════════════════════════════════
# 3. TLOG — set training log interval
# ═══════════════════════════════════════════════════════════════

def gen_tlog():
    pairs = []
    intervals = [10, 25, 50, 100, 200, 500, 1000]
    contexts = [
        "Log training loss every {n} epochs",
        "Print progress every {n} epochs during training",
        "Set the logging interval to {n} for the training run",
        "Enable training diagnostics every {n} steps",
        "Configure the model to report loss every {n} epochs",
    ]

    for _ in range(200):
        n   = random.choice(intervals)
        ctx = random.choice(contexts).format(n=n)
        # TLOG then TNET
        epochs = rand_epochs()
        lr     = rand_lr()
        loss   = random.choice([0, 1, 2])
        loss_name = {0: "MSE", 1: "cross-entropy", 2: "MAE"}[loss]
        pairs.append(pair(
            f"{ctx}. Train a 2-layer network with {loss_name} loss.",
            f"LD R0 @input\nLD R1 @w1\nLD R2 @b1\nLD R3 @w2\nLD R4 @b2\nLD R9 @labels\n"
            f"TLOG #{n}\nTNET #{epochs} #{lr} #{loss}\nST RA @result\nHALT"
        ))
        # TLOG then TRAIN (config-driven)
        pairs.append(pair(
            f"{ctx}. Use config-driven training (TRAIN opcode).",
            f"; arch: 1 layer, input=16, hidden=8\n"
            f"LD RV @arch_desc\nLD R0 @input\nLD R9 @labels\n"
            f"LD R1 @w1\nLD R2 @b1\n"
            f"LD R5 @train_config\n"
            f"TLOG #{n}\nTRAIN R5\nST R8 @loss\nHALT"
        ))

    return pairs


# ═══════════════════════════════════════════════════════════════
# 4. TRAIN — config-driven training
# ═══════════════════════════════════════════════════════════════
# Config tensor (RV holds arch, RS holds [epochs,lr,opt,print_every,patience,min_delta])

def gen_train():
    pairs = []
    opt_names = {0: "SGD", 1: "Adam"}

    descriptions = [
        "Train a 2-layer MLP using config-driven TRAIN",
        "Use the TRAIN opcode to train a neural network from a config tensor",
        "Config-driven training loop with early stopping",
        "Train a network with Adam optimizer using TRAIN",
        "Fit a model using the NML TRAIN opcode",
        "Use TRAIN to run a multi-epoch training loop",
    ]

    for _ in range(200):
        epochs    = rand_epochs()
        lr        = rand_lr()
        opt       = random.choice([0, 1])
        patience  = random.choice([0, 10, 20, 50])
        print_ev  = random.choice([0, 50, 100, 200])
        desc      = random.choice(descriptions)

        cfg_vals = f"{epochs},{lr},{opt},{print_ev},{patience},0.00001"
        pairs.append(pair(
            f"{desc}. Epochs={epochs}, lr={lr}, optimizer={opt_names[opt]}.",
            f"; RV = architecture descriptor [n_layers, h1, act1, h2, act2, ...]\n"
            f"; config: epochs={epochs} lr={lr} opt={opt} print_every={print_ev} patience={patience}\n"
            f"LD RV @arch_desc\n"
            f"LD R0 @train_x\nLD R9 @train_y\n"
            f"LD R1 @w1\nLD R2 @b1\nLD R3 @w2\nLD R4 @b2\n"
            f"LD R5 @train_config\n"
            f"TRAIN R5\n"
            f"ST R8 @final_loss\n"
            f"ST R1 @w1\nST R2 @b1\nST R3 @w2\nST R4 @b2\nHALT"
        ))

    # TRAIN + INFER pair (train then run inference)
    for _ in range(100):
        epochs = rand_epochs()
        lr     = rand_lr()
        pairs.append(pair(
            f"Train a network for {epochs} epochs then run inference on test data.",
            f"LD RV @arch_desc\n"
            f"LD R0 @train_x\nLD R9 @train_y\n"
            f"LD R1 @w1\nLD R2 @b1\nLD R3 @w2\nLD R4 @b2\n"
            f"LD R5 @train_config\n"
            f"TRAIN R5\n"
            f"ST R1 @w1\nST R2 @b1\nST R3 @w2\nST R4 @b2\n"
            f"; inference on test set\n"
            f"LD R0 @test_x\n"
            f"INFER RA R0\n"
            f"ST RA @predictions\nHALT"
        ))

    return pairs


# ═══════════════════════════════════════════════════════════════
# 5. INFER — forward pass only (no weight update)
# ═══════════════════════════════════════════════════════════════

def gen_infer():
    pairs = []
    descriptions = [
        "Run inference on new data using a trained model",
        "Forward pass only — no weight update",
        "Use INFER to score test samples",
        "Apply a trained network to make predictions",
        "Run the model in inference mode on a batch",
        "Predict outputs for input data using INFER",
        "Generate predictions without updating weights",
    ]

    for _ in range(300):
        desc  = random.choice(descriptions)
        dst   = random.choice(["RA", "RB", "R8"])
        arch  = rand_arch()
        inp   = rand_input()
        out   = rand_output()
        lbl   = rand_labels()
        w, b  = rand_weights()
        n_layers = random.randint(1, 3)
        ld_weights = "".join(
            f"LD R{i*2+1} @layer{i+1}_w\nLD R{i*2+2} @layer{i+1}_b\n"
            for i in range(n_layers)
        )
        pairs.append(pair(
            f"{desc} ({n_layers}-layer network, input={inp}).",
            f"LD RV @{arch}\nLD R0 @{inp}\n{ld_weights}"
            f"INFER {dst} R0\nST {dst} @{out}\nHALT"))

    # symbolic
    for _ in range(120):
        arch = rand_arch()
        inp  = rand_input()
        out  = rand_output()
        w, b = rand_weights()
        pairs.append(pair(
            f"Run inference on {inp} in symbolic NML (arch in @{arch}).",
            f"LD RV @{arch}\nLD R0 @{inp}\nLD R1 @{w}\nLD R2 @{b}\n"
            f"\u27f6 RA R0\nST RA @{out}\nHALT"))

    # batch inference
    for _ in range(80):
        batch = random.choice([1, 4, 8, 16, 32, 64])
        arch  = rand_arch()
        inp   = rand_input()
        out   = rand_output()
        pairs.append(pair(
            f"Batch inference (batch_size={batch}) on {inp} using arch @{arch}.",
            f"; batch_size={batch}\nLD RV @{arch}\nLD R0 @{inp}\n"
            f"LD R1 @w1\nLD R2 @b1\nLD R3 @w2\nLD R4 @b2\n"
            f"INFER RA R0\nST RA @{out}\nHALT"))

    return pairs


# ═══════════════════════════════════════════════════════════════
# 6. WDECAY — L2 weight decay
# ═══════════════════════════════════════════════════════════════

def gen_wdecay():
    pairs = []
    descriptions = [
        "Apply L2 weight decay to prevent overfitting",
        "Regularize weights with weight decay",
        "Add AdamW-style weight decay",
        "Apply weight decay to all weight matrices",
        "L2 regularization via WDECAY",
        "Decay weights by factor lambda before the update step",
        "Use WDECAY for L2 regularization in a training loop",
    ]

    for _ in range(200):
        decay = rand_decay()
        desc  = random.choice(descriptions)
        w1, b1 = rand_weights()
        w2, b2 = rand_weights()
        pairs.append(pair(
            f"{desc} (lambda={decay}) applied to {w1} and {w2}.",
            f"LD R1 @{w1}\nLD R2 @{w2}\n"
            f"WDECAY R1 #{decay}\nWDECAY R2 #{decay}\n"
            f"ST R1 @{w1}\nST R2 @{w2}\nHALT"))
        # symbolic
        w3, _ = rand_weights()
        pairs.append(pair(
            f"{desc} (lambda={decay}) on {w3} using symbolic NML.",
            f"LD R1 @{w3}\n\u03c9 R1 #{decay}\nST R1 @{w3}\nHALT"))

    # WDECAY inside a manual training loop
    for _ in range(150):
        epochs = rand_epochs()
        lr     = rand_lr()
        decay  = rand_decay()
        inp    = rand_input()
        lbl    = rand_labels()
        w1, b1 = rand_weights()
        w2, b2 = rand_weights()
        pairs.append(pair(
            f"Manual training loop on {inp}/{lbl}: {epochs} epochs, lr={lr}, weight decay {decay}.",
            f"LD R0 @{inp}\nLD R1 @{w1}\nLD R2 @{b1}\nLD R3 @{w2}\nLD R4 @{b2}\nLD R9 @{lbl}\n"
            f"LOOP #{epochs}\n"
            f"  MMUL R5 R0 R1\nMADD R5 R5 R2\nRELU R5 R5\n"
            f"  MMUL R6 R5 R3\nMADD R6 R6 R4\n"
            f"  LOSS R8 R6 R9\n"
            f"  BKWD R7 R6 R9\n"
            f"  WUPD R3 R7 #{lr}\nWUPD R4 R7 #{lr}\n"
            f"  WDECAY R1 #{decay}\nWDECAY R3 #{decay}\n"
            f"ENDP\nST R1 @{w1}\nST R3 @{w2}\nHALT"))

    # verbose
    for _ in range(50):
        decay = rand_decay()
        w, _  = rand_weights()
        pairs.append(pair(
            f"Apply weight decay (lambda={decay}) to {w} using verbose NML syntax.",
            f"LD R1 @{w}\nWEIGHT_DECAY R1 #{decay}\nST R1 @{w}\nHALT"))

    return pairs


# ═══════════════════════════════════════════════════════════════
# 7. SYS — system calls (print, shape, size, clock)
# ═══════════════════════════════════════════════════════════════

def gen_sys():
    pairs = []

    # SYS #0 — PRINT_NUM
    for _ in range(100):
        reg   = rand_reg()
        name  = rand_name()
        pairs.append(pair(
            f"Load a value and print it using the SYS print opcode.",
            f"LD {reg} @{name}\nSYS {reg} #0\nHALT"
        ))
        pairs.append(pair(
            f"Compute the dot product of two vectors and print the scalar result.",
            f"LD R0 @vec_a\nLD R1 @vec_b\nSDOT R2 R0 R1\nSYS R2 #0\nHALT"
        ))

    # SYS #2 — PRINT_SHAPE
    for _ in range(80):
        reg  = rand_reg()
        name = rand_name()
        pairs.append(pair(
            f"Print the shape of a tensor loaded from @{name}.",
            f"LD {reg} @{name}\nSYS {reg} #2\nHALT"
        ))

    # SYS #3 — PRINT_SIZE
    for _ in range(50):
        reg  = rand_reg()
        name = rand_name()
        pairs.append(pair(
            f"Print the number of elements in @{name}.",
            f"LD {reg} @{name}\nSYS {reg} #3\nHALT"
        ))

    # SYS in debugging context
    for _ in range(80):
        name = rand_name()
        pairs.append(pair(
            f"Compute a matrix multiply then print the output shape for debugging.",
            f"LD R0 @{name}\nLD R1 @weights\nMMUL R2 R0 R1\nSYS R2 #2\nST R2 @out\nHALT"
        ))

    # SYS #1 — PRINT_TENSOR (a few)
    for _ in range(40):
        reg  = rand_reg()
        name = rand_name()
        pairs.append(pair(
            f"Print the full contents of tensor @{name}.",
            f"LD {reg} @{name}\nSYS {reg} #1\nHALT"
        ))

    return pairs


# ═══════════════════════════════════════════════════════════════
# 8. SOFTBK — softmax backward pass
# ═══════════════════════════════════════════════════════════════

def gen_softbk():
    pairs = []
    descriptions = [
        "Backpropagate through a softmax layer",
        "Compute the softmax backward pass",
        "Manual gradient through softmax",
        "Backward pass for a softmax output layer",
        "Apply SOFTBK to propagate gradients through softmax",
        "Gradient of loss through the softmax activation",
    ]

    # MMULBK signature: MMULBK dW dX grad X W  (5 operands)
    for _ in range(250):
        desc = random.choice(descriptions)
        inp  = rand_input()
        out  = rand_output()
        lbl  = rand_labels()
        w, b = rand_weights()
        lr   = rand_lr()
        pairs.append(pair(
            f"{desc} on {inp}/{lbl}.",
            f"LD R0 @{inp}\nLD R1 @{w}\nLD R9 @{lbl}\n"
            f"MMUL R2 R0 R1\nSOFT R3 R2\n"
            f"LOSS R8 R3 R9\n"
            f"SOFTBK R4 R3 R9\n"
            f"MMULBK R5 R6 R4 R0 R1\n"
            f"WUPD R1 R5 #{lr}\nST R1 @{w}\nHALT"))

    # SOFTBK in a full training loop (LOOP ends with ENDP)
    for _ in range(150):
        epochs = rand_epochs()
        lr     = rand_lr()
        inp    = rand_input()
        lbl    = rand_labels()
        w, b   = rand_weights()
        out    = rand_output()
        pairs.append(pair(
            f"Train softmax classifier on {inp}/{lbl}: {epochs} epochs, lr={lr}.",
            f"LD R0 @{inp}\nLD R1 @{w}\nLD R2 @{b}\nLD R9 @{lbl}\n"
            f"LOOP #{epochs}\n"
            f"  MMUL R3 R0 R1\nMADD R3 R3 R2\n"
            f"  SOFT R4 R3\n"
            f"  LOSS R8 R4 R9\n"
            f"  SOFTBK R5 R4 R9\n"
            f"  MMULBK R6 R7 R5 R0 R1\n"
            f"  WUPD R1 R6 #{lr}\nWUPD R2 R5 #{lr}\n"
            f"ENDP\nST R1 @{out}\nHALT"))

    # symbolic (Σˈ = SOFTBK, × = MMUL, Σ = SOFT)
    for _ in range(60):
        inp  = rand_input()
        lbl  = rand_labels()
        w, _ = rand_weights()
        pairs.append(pair(
            f"Softmax backward pass on {inp}/{lbl} in symbolic NML.",
            f"LD R0 @{inp}\nLD R1 @{w}\nLD R9 @{lbl}\n"
            "\u00d7 R2 R0 R1\n"          # × MMUL
            "\u03a3 R3 R2\n"              # Σ SOFT
            "\u03a3\u02c8 R4 R3 R9\n"    # Σˈ SOFTBK
            "ST R4 @grad\nHALT"))

    return pairs


# ═══════════════════════════════════════════════════════════════
# 9. JMP / JMPZ / JMPNZ — unconditional and conditional jumps
# ═══════════════════════════════════════════════════════════════

def gen_jumps():
    pairs = []
    for _ in range(150):
        inp = rand_input()
        out = rand_output()
        pairs.append(pair(
            f"Write NML that skips a block when {inp} is zero using JMPZ.",
            f"LD R0 @{inp}\nJMPZ R0 @done\nRELU R0 R0\n@done\nST R0 @{out}\nHALT"))
    for _ in range(150):
        inp = rand_input()
        pairs.append(pair(
            f"Write NML that loops while {inp} counter is non-zero using JMPNZ.",
            f"LD R0 @{inp}\nLD R1 @weights\n@loop\nMMADD R2 R2 R0\nSSUB R0 R0 #1.0\nJMPNZ R0 @loop\nST R2 @result\nHALT"))
    for _ in range(100):
        pairs.append(pair(
            "Write NML that unconditionally jumps past a block using JMP.",
            "LD R0 @input\nJMP @compute\nRELU R0 R0\n@compute\nMMUL R1 R0 R0\nST R1 @result\nHALT"))
    return pairs


# ═══════════════════════════════════════════════════════════════
# 10. Scalar ops — SADD, SSUB, SCLR, SDIV, SOFT
# ═══════════════════════════════════════════════════════════════

def gen_scalar_ops():
    pairs = []
    for _ in range(100):
        inp = rand_input(); out = rand_output()
        v = round(random.uniform(0.1, 5.0), 2)
        pairs.append(pair(f"Add scalar {v} to every element of {inp}.",
            f"LD R0 @{inp}\nSADD R1 R0 #{v}\nST R1 @{out}\nHALT"))
    for _ in range(100):
        inp = rand_input(); out = rand_output()
        v = round(random.uniform(0.1, 2.0), 2)
        pairs.append(pair(f"Multiply every element of {inp} by scalar {v} using SCLR.",
            f"LD R0 @{inp}\nSCLR R1 R0 #{v}\nST R1 @{out}\nHALT"))
    for _ in range(100):
        inp = rand_input(); out = rand_output()
        v = round(random.uniform(0.1, 2.0), 2)
        pairs.append(pair(f"Subtract scalar {v} from every element of {inp} using SSUB.",
            f"LD R0 @{inp}\nSSUB R1 R0 #{v}\nST R1 @{out}\nHALT"))
    for _ in range(100):
        inp = rand_input(); out = rand_output()
        v = round(random.uniform(0.5, 4.0), 2)
        pairs.append(pair(f"Scale {inp} by {v}.",
            f"LD R0 @{inp}\nSCLR R1 R0 #{v}\nST R1 @{out}\nHALT"))
    for _ in range(100):
        inp = rand_input(); out = rand_output()
        pairs.append(pair(f"Compute softmax of {inp}.",
            f"LD R0 @{inp}\nSOFT R1 R0\nST R1 @{out}\nHALT"))
    return pairs


# ═══════════════════════════════════════════════════════════════
# 11. Vision — UPSC, PADZ
# ═══════════════════════════════════════════════════════════════

def gen_vision_ops():
    pairs = []
    for _ in range(150):
        inp = rand_input(); out = rand_output()
        scale = random.choice([2, 4])
        pairs.append(pair(f"Upsample feature map {inp} by factor {scale} using UPSC.",
            f"LD R0 @{inp}\nUPSC R1 R0 #{scale}\nST R1 @{out}\nHALT"))
    for _ in range(150):
        inp = rand_input(); out = rand_output()
        pad = random.choice([1, 2, 3])
        pairs.append(pair(f"Zero-pad tensor {inp} with padding {pad} using PADZ.",
            f"LD R0 @{inp}\nPADZ R1 R0 #{pad}\nST R1 @{out}\nHALT"))
    for _ in range(100):
        inp = rand_input(); out = rand_output()
        pairs.append(pair(f"Zero-pad {inp} then apply CONV.",
            f"LD R0 @{inp}\nLD R1 @weights\nPADZ R2 R0 #1\nCONV R3 R2 R1\nST R3 @{out}\nHALT"))
    return pairs


# ═══════════════════════════════════════════════════════════════
# 12. Reduction — WHER, CMPR
# ═══════════════════════════════════════════════════════════════

def gen_reduction_ops():
    pairs = []
    for _ in range(150):
        inp = rand_input(); out = rand_output()
        thresh = round(random.uniform(0.0, 1.0), 2)
        pairs.append(pair(f"Select elements of {inp} where value > {thresh} using WHER.",
            f"LD R0 @{inp}\nLD R1 @mask\nWHER R2 R0 R1\nST R2 @{out}\nHALT"))
    for _ in range(150):
        inp = rand_input(); out = rand_output()
        pairs.append(pair(f"Compare {inp} against threshold and replace using CMPR.",
            f"LD R0 @{inp}\nLD R1 @threshold\nCMPR R2 R0 R1\nST R2 @{out}\nHALT"))
    return pairs


# ═══════════════════════════════════════════════════════════════
# 13. Signal — FFT
# ═══════════════════════════════════════════════════════════════

def gen_signal_ops():
    pairs = []
    for _ in range(200):
        inp = random.choice(["signal", "audio", "waveform", "samples", "time_series"])
        out = rand_output()
        pairs.append(pair(f"Compute the FFT of {inp}.",
            f"LD R0 @{inp}\nFFT R1 R0\nST R1 @{out}\nHALT"))
    for _ in range(100):
        inp = random.choice(["signal", "audio", "samples"])
        out = rand_output()
        pairs.append(pair(f"Compute FFT of {inp} then apply FILT.",
            f"LD R0 @{inp}\nLD R1 @filter_coeffs\nFFT R2 R0\nFILT R3 R2 R1\nST R3 @{out}\nHALT"))
    return pairs


# ═══════════════════════════════════════════════════════════════
# 14. Backward passes — RELU_BK, SIGM_BK, TANH_BK, GELU_BK,
#                       NORM_BK, LOSS_BK
# ═══════════════════════════════════════════════════════════════

def gen_backward_ops():
    pairs = []
    bk_ops = [
        ("RELU_BK",  "ReLU backward pass"),
        ("SIGM_BK",  "sigmoid backward pass"),
        ("TANH_BK",  "tanh backward pass"),
        ("GELU_BK",  "GELU backward pass"),
        ("NORM_BK",  "layer-norm backward pass"),
        ("LOSS_BK",  "loss backward pass"),
    ]
    for op, desc in bk_ops:
        for _ in range(80):
            inp = rand_input()
            pairs.append(pair(
                f"Compute the {desc} for {inp}.",
                f"LD R0 @{inp}\nLD R1 @grad_out\n{op} R2 R0 R1\nST R2 @grad_in\nHALT"))
        for _ in range(40):
            inp = rand_input(); w, b = rand_weights(); lr = rand_lr()
            pairs.append(pair(
                f"Full backprop with {op} and weight update for {inp}.",
                f"LD R0 @{inp}\nLD R1 @{w}\nLD R9 @targets\n"
                f"MMUL R2 R0 R1\n{op} R3 R2 R9\n"
                f"MMUL_BK R4 R5 R3 R0 R1\n"
                f"WUPD R1 R4 #{lr}\nST R1 @{w}\nHALT"))
    return pairs


# ═══════════════════════════════════════════════════════════════
# 15. TNDEEP — deep N-layer training
# ═══════════════════════════════════════════════════════════════

def gen_tndeep():
    pairs = []
    descs = [
        "Train a deep MLP using TNDEEP.",
        "Deep network training with TNDEEP opcode.",
        "Multi-layer deep training using NML TNDEEP.",
        "Train a deep neural network with TNDEEP and weight decay.",
    ]
    for _ in range(200):
        inp  = rand_input()
        arch = rand_arch()
        lbl  = rand_labels()
        lr   = rand_lr()
        ep   = rand_epochs()
        desc = random.choice(descs)
        pairs.append(pair(desc,
            f"LD R0 @{inp}\nLD R1 @{arch}\nLD R9 @{lbl}\n"
            f"TNDEEP R1 #{ep}\nST R1 @trained_model\nHALT"))
    return pairs


# ═══════════════════════════════════════════════════════════════
# 16. M2M — VRFY, VOTE
# ═══════════════════════════════════════════════════════════════

def gen_m2m_ops():
    pairs = []
    for _ in range(150):
        pairs.append(pair(
            "Verify the signature of a received NML program using VRFY.",
            "LD R0 @program_hash\nLD R1 @signature\nLD R2 @public_key\n"
            "VRFY R3 R0 R1 R2\nST R3 @verified\nHALT"))
    for _ in range(150):
        pairs.append(pair(
            "Cast a consensus vote across agent outputs using VOTE.",
            "LD R0 @agent_outputs\nLD R1 @weights\n"
            "VOTE R2 R0 R1\nST R2 @consensus\nHALT"))
    return pairs


# ═══════════════════════════════════════════════════════════════
# 17. NML-G — ITOF, FTOI, BNOT
# ═══════════════════════════════════════════════════════════════

def gen_nmlg_ops():
    pairs = []
    for _ in range(120):
        inp = rand_input(); out = rand_output()
        pairs.append(pair(f"Convert float tensor {inp} to int using FTOI.",
            f"LD R0 @{inp}\nFTOI R1 R0\nST R1 @{out}\nHALT"))
    for _ in range(120):
        inp = rand_input(); out = rand_output()
        pairs.append(pair(f"Convert integer tensor {inp} to float using ITOF.",
            f"LD R0 @{inp}\nITOF R1 R0\nST R1 @{out}\nHALT"))
    for _ in range(80):
        inp = rand_input(); out = rand_output()
        pairs.append(pair(f"Bitwise NOT of {inp} using BNOT.",
            f"LD R0 @{inp}\nBNOT R1 R0\nST R1 @{out}\nHALT"))
    return pairs


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    categories = [
        ("BN",          gen_bn),
        ("DROP",        gen_drop),
        ("TLOG",        gen_tlog),
        ("TRAIN",       gen_train),
        ("INFER",       gen_infer),
        ("WDECAY",      gen_wdecay),
        ("SYS",         gen_sys),
        ("SOFTBK",      gen_softbk),
        ("JUMPS",       gen_jumps),
        ("SCALAR_OPS",  gen_scalar_ops),
        ("VISION_OPS",  gen_vision_ops),
        ("REDUCTION",   gen_reduction_ops),
        ("SIGNAL",      gen_signal_ops),
        ("BACKWARD",    gen_backward_ops),
        ("TNDEEP",      gen_tndeep),
        ("M2M",         gen_m2m_ops),
        ("NML_G",       gen_nmlg_ops),
    ]

    all_pairs = []
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    print("NML Coverage Fix Generator")
    print("-" * 50)
    for name, fn in categories:
        p = fn()
        random.shuffle(p)
        all_pairs.extend(p)
        print(f"  {name:<10} {len(p):>5} pairs")

    random.shuffle(all_pairs)

    with open(OUTPUT, "w", encoding="utf-8") as f:
        for p in all_pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    print("-" * 50)
    print(f"  TOTAL      {len(all_pairs):>5} pairs")
    print(f"\nWritten to: {OUTPUT}")


if __name__ == "__main__":
    main()
