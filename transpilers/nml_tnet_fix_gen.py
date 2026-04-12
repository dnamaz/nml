#!/usr/bin/env python3
"""
TRAIN+INFER training data — drills the modern v0.9 pattern:
  LD   RV @arch   /  ALLC RV #[n]   ; architecture descriptor
  LD   RU @config /  ALLC RU #[6]   ; training config
  LD   R0 @input  /  LD R9 @target
  TRAIN RU                           ; config-driven training
  INFER R8 R0                        ; forward-only inference
  ST   R8 @result / HALT

Replaces the legacy TNET pattern. Also includes DPO-style correction
pairs showing TNET as the wrong approach and TRAIN+INFER as the right one.
"""

import json
import random
from pathlib import Path

random.seed(77)
OUTPUT = Path(__file__).parent.parent / "domain" / "output" / "training" / "nml_tnet_fix_pairs.jsonl"

def pair(prompt, code):
    return {"messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": code}]}

EPOCHS = [50, 100, 200, 500, 1000, 2000, 5000, 10000]
LRS = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1]
OPTIMIZERS = [0, 1]
OPT_NAMES = {0: "SGD", 1: "Adam"}
TASKS = [
    ("learn y=2x+1", "linear"),
    ("classify XOR", "xor"),
    ("approximate sine", "sine"),
    ("learn AND gate", "and_gate"),
    ("learn OR gate", "or_gate"),
    ("predict housing prices", "housing"),
    ("classify digits", "digits"),
    ("learn square root", "sqrt"),
    ("approximate cosine", "cosine"),
    ("fit a polynomial", "polynomial"),
]

INPUT_NAMES = ["input", "x", "training_data", "features", "samples", "train_x", "data"]
TARGET_NAMES = ["target", "y", "labels", "expected", "train_y", "ground_truth"]
RESULT_NAMES = ["result", "prediction", "output", "inference", "answer"]

# Architecture descriptors: (description, n_layers, RV values list)
# RV = [n_layers, h1, act1, h2, act2, ...]  where act: 0=ReLU, 1=Sigmoid, 2=Tanh
ARCHITECTURES = [
    ("2-layer hidden=8 ReLU",        2, [2, 8, 0, 1, 0]),
    ("2-layer hidden=16 ReLU",       2, [2, 16, 0, 1, 0]),
    ("2-layer hidden=32 ReLU",       2, [2, 32, 0, 1, 0]),
    ("2-layer hidden=4 Sigmoid",     2, [2, 4, 1, 1, 1]),
    ("2-layer hidden=8 Sigmoid",     2, [2, 8, 1, 1, 1]),
    ("2-layer hidden=16 Tanh",       2, [2, 16, 2, 1, 2]),
    ("3-layer 16,8 ReLU",            3, [3, 16, 0, 8, 0, 1, 0]),
    ("3-layer 32,16 ReLU",           3, [3, 32, 0, 16, 0, 1, 0]),
    ("3-layer 8,4 Sigmoid",          3, [3, 8, 1, 4, 1, 1, 1]),
]

ARCH_NAMES = ["arch", "arch_desc", "architecture", "network_arch", "layer_desc"]
CONFIG_NAMES = ["config", "train_config", "hyperparams", "train_params", "training_cfg"]


def _rv_comment(arch_vals):
    return ",".join(str(v) for v in arch_vals)


def _ru_comment(epochs, lr, opt=0, print_every=0, patience=0, min_delta=0):
    return f"{epochs},{lr},{opt},{print_every},{patience},{min_delta}"


def gen_classic():
    """Classic TRAIN+INFER pattern using LD from memory (grammar-valid)."""
    pairs = []
    for _ in range(300):
        ep = random.choice(EPOCHS)
        lr = random.choice(LRS)
        opt = random.choice(OPTIMIZERS)
        opt_name = OPT_NAMES[opt]
        task, _ = random.choice(TASKS)
        inp = random.choice(INPUT_NAMES)
        tgt = random.choice(TARGET_NAMES)
        res = random.choice(RESULT_NAMES)
        arch_desc, n_layers, arch_vals = random.choice(ARCHITECTURES)
        arch_name = random.choice(ARCH_NAMES)
        cfg_name = random.choice(CONFIG_NAMES)

        code = f"""; {arch_desc}, {opt_name}, {ep} epochs, lr={lr}
LD    RV @{arch_name}
LD    RU @{cfg_name}
LD    R0 @{inp}
LD    R9 @{tgt}
TRAIN RU
INFER R8 R0
ST    R8 @{res}
HALT"""
        prompts = [
            f"Write NML to train a neural network to {task} using TRAIN with {ep} epochs and lr={lr}.",
            f"NML: train a {n_layers}-layer network with TRAIN. Use {ep} epochs, learning rate {lr}, {opt_name} optimizer.",
            f"Write NML using TRAIN+INFER to {task}. {ep} epochs, lr={lr}, {arch_desc}.",
            f"Write NML for config-driven training with TRAIN opcode. {ep} epochs, lr={lr}.",
            f"NML program that uses TRAIN to train on data. Parameters: epochs={ep}, lr={lr}, optimizer={opt_name}.",
            f"Write NML: load architecture into RV, config into RU, then TRAIN+INFER.",
        ]
        pairs.append(pair(random.choice(prompts), code))

    return pairs


def gen_allc_inline():
    """TRAIN+INFER with inline ALLC descriptors (teaches the ALLC+values pattern)."""
    pairs = []
    for _ in range(200):
        ep = random.choice(EPOCHS)
        lr = random.choice(LRS)
        opt = random.choice(OPTIMIZERS)
        opt_name = OPT_NAMES[opt]
        task, _ = random.choice(TASKS)
        inp = random.choice(INPUT_NAMES)
        tgt = random.choice(TARGET_NAMES)
        res = random.choice(RESULT_NAMES)
        arch_desc, n_layers, arch_vals = random.choice(ARCHITECTURES)
        rv_size = len(arch_vals)

        # Use ALLC #[N] (grammar-valid shape) + comment showing values
        # The runtime will allocate zeros; the values are documentation
        code = f"""; {arch_desc}, {opt_name}, {ep} epochs
; RV = [{_rv_comment(arch_vals)}]
ALLC  RV #[{rv_size}]
; RU = [{_ru_comment(ep, lr, opt)}]
ALLC  RU #[6]
LD    R0 @{inp}
LD    R9 @{tgt}
TRAIN RU
INFER R8 R0
ST    R8 @{res}
HALT"""
        prompts = [
            f"Write NML to {task} using ALLC to set up architecture and config tensors, then TRAIN+INFER.",
            f"NML with inline tensor allocation: {n_layers}-layer network, {ep} epochs, lr={lr}, {opt_name}.",
            f"Write NML using ALLC for RV (architecture) and RU (config), then TRAIN RU and INFER R8 R0.",
        ]
        pairs.append(pair(random.choice(prompts), code))

    return pairs


def gen_with_inference():
    """TRAIN followed by inference on new data."""
    pairs = []
    for _ in range(200):
        ep = random.choice(EPOCHS)
        lr = random.choice(LRS)
        opt = random.choice(OPTIMIZERS)
        task, _ = random.choice(TASKS)
        arch_desc, n_layers, arch_vals = random.choice(ARCHITECTURES)
        arch_name = random.choice(ARCH_NAMES)
        cfg_name = random.choice(CONFIG_NAMES)

        code = f"""; Train then predict on test data
LD    RV @{arch_name}
LD    RU @{cfg_name}
LD    R0 @train_input
LD    R9 @train_target
TRAIN RU
; inference on test set
LD    R0 @test_input
INFER R8 R0
ST    R8 @prediction
HALT"""
        prompt = random.choice([
            f"Write NML to train with TRAIN ({ep} epochs, lr={lr}), then run inference on test input.",
            f"NML: TRAIN-based training loop followed by INFER on new data.",
            f"Write NML that trains a network with TRAIN then predicts on test_input using INFER.",
        ])
        pairs.append(pair(prompt, code))

    return pairs


def gen_symbolic():
    pairs = []
    for _ in range(200):
        ep = random.choice(EPOCHS)
        lr = random.choice(LRS)
        opt = random.choice(OPTIMIZERS)
        arch_desc, n_layers, arch_vals = random.choice(ARCHITECTURES)

        code = f"""; {arch_desc}
↓     RV @arch
↓     RU @config
↓     ι  @input
↓     ς  @target
⟴     RU
⟶     R8 ι
↑     R8 @result
◼"""
        prompt = random.choice([
            f"Write symbolic NML to train with ⟴ (TRAIN) for {ep} epochs, lr={lr}.",
            f"Symbolic NML: load arch into RV, config into RU, input ι, target ς, then ⟴ RU and ⟶ R8.",
            f"Write NML in symbolic syntax using ⟴ for config-driven training.",
        ])
        pairs.append(pair(prompt, code))

    return pairs


def gen_verbose():
    pairs = []
    for _ in range(100):
        ep = random.choice(EPOCHS)
        lr = random.choice(LRS)
        opt = random.choice(OPTIMIZERS)
        arch_desc, n_layers, arch_vals = random.choice(ARCHITECTURES)

        code = f"""; {arch_desc}
LOAD          RV  @arch
LOAD          RU  @config
LOAD          R0  @input
LOAD          R9  @target
TRAIN_CONFIG  RU
FORWARD_PASS  R8  R0
STORE         R8  @result
STOP"""
        prompt = f"Write verbose NML to train with TRAIN for {ep} epochs, {OPT_NAMES[opt]} optimizer."
        pairs.append(pair(prompt, code))

    return pairs


def gen_wrong_to_right():
    """Correction pairs: show the wrong TNET form and the modern TRAIN+INFER fix."""
    pairs = []
    for _ in range(200):
        ep = random.choice(EPOCHS)
        lr = random.choice(LRS)
        opt = random.choice(OPTIMIZERS)
        arch_desc, n_layers, arch_vals = random.choice(ARCHITECTURES)
        arch_name = random.choice(ARCH_NAMES)
        cfg_name = random.choice(CONFIG_NAMES)

        wrong = f"""LD    R1 @w1
LD    R2 @b1
LD    R3 @w2
LD    R4 @b2
LD    R0 @input
LD    R9 @target
TNET  #{ep} #{lr} #0
ST    RA @result
HALT"""

        right = f"""; Use TRAIN+INFER instead of legacy TNET
LD    RV @{arch_name}
LD    RU @{cfg_name}
LD    R0 @input
LD    R9 @target
TRAIN RU
INFER R8 R0
ST    R8 @result
HALT"""

        prompts = [
            f"Fix this NML — it uses the legacy TNET opcode. Convert to modern TRAIN+INFER:\n```\n{wrong}\n```",
            f"This program uses TNET which is deprecated. Rewrite using TRAIN+INFER with config tensors.",
            f"Convert this legacy TNET training code to the modern TRAIN+INFER pattern:\n```\n{wrong}\n```",
        ]
        pairs.append(pair(random.choice(prompts), right))

    return pairs


def gen_dpo_pairs():
    """DPO-style (rejected, chosen) pairs — TNET is rejected, TRAIN+INFER is chosen."""
    pairs = []
    for _ in range(200):
        ep = random.choice(EPOCHS)
        lr = random.choice(LRS)
        opt = random.choice(OPTIMIZERS)
        task, _ = random.choice(TASKS)
        arch_desc, n_layers, arch_vals = random.choice(ARCHITECTURES)
        arch_name = random.choice(ARCH_NAMES)
        cfg_name = random.choice(CONFIG_NAMES)

        rejected = f"""LD    R1 @w1
LD    R2 @b1
LD    R3 @w2
LD    R4 @b2
LD    R0 @input
LD    R9 @target
TNET  #{ep} #{lr} #0
ST    RA @result
HALT"""

        chosen = f"""; {arch_desc}
LD    RV @{arch_name}
LD    RU @{cfg_name}
LD    R0 @input
LD    R9 @target
TRAIN RU
INFER R8 R0
ST    R8 @result
HALT"""

        prompt = random.choice([
            f"Write NML to {task} with {ep} epochs and lr={lr}.",
            f"NML: train a network to {task}. {ep} epochs, learning rate {lr}.",
            f"Write NML training code to {task}.",
        ])

        # DPO pair: show both with preference labels
        pairs.append({"messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": chosen},
        ], "rejected": rejected, "chosen": chosen})

    return pairs


def gen_train_explanation():
    """Q&A pairs explaining TRAIN+INFER syntax."""
    pairs = []
    explanations = [
        ("What is the correct syntax for NML TRAIN+INFER?",
         "TRAIN uses a config-driven approach with two tensor descriptors:\n\n"
         "1. **RV** — architecture descriptor: `[n_layers, h1, act1, h2, act2, ...]`\n"
         "   - Activations: 0=ReLU, 1=Sigmoid, 2=Tanh\n"
         "2. **RU** — training config: `[epochs, lr, optimizer, print_every, patience, min_delta]`\n"
         "   - Optimizers: 0=SGD, 1=Adam\n\n"
         "Load both from memory with LD, then `TRAIN RU` runs the training loop "
         "and `INFER R8 R0` runs forward-only inference.\n\n"
         "Example:\n```\n"
         "; 2-layer: hidden=8 ReLU, output=1 ReLU\n"
         "LD    RV @arch\n"
         "; config: 1000 epochs, lr=0.01, Adam\n"
         "LD    RU @config\n"
         "LD    R0 @input\n"
         "LD    R9 @target\n"
         "TRAIN RU\n"
         "INFER R8 R0\n"
         "ST    R8 @result\n"
         "HALT\n```"),
        ("How does TRAIN work in NML?",
         "TRAIN is the v0.9 config-driven training opcode. It replaces the legacy TNET opcode.\n\n"
         "Setup:\n"
         "- Load RV with the architecture descriptor (layer sizes and activations)\n"
         "- Load RU with [epochs, lr, optimizer, print_every, patience, min_delta]\n"
         "- Load input into R0 and target into R9\n"
         "- Call `TRAIN RU` to run the training loop\n"
         "- Call `INFER R8 R0` for forward-only inference after training\n\n"
         "Correct usage:\n```\n"
         "; 3-layer: 32 ReLU, 16 ReLU, 1 ReLU\n"
         "LD    RV @arch\n"
         "LD    RU @config\n"
         "LD    R0 @input\n"
         "LD    R9 @target\n"
         "TRAIN RU\n"
         "INFER R8 R0\n"
         "ST    R8 @result\n"
         "HALT\n```"),
        ("What is the difference between TNET and TRAIN in NML?",
         "TNET is the legacy v0.7 training opcode. TRAIN is the modern v0.9 replacement.\n\n"
         "**TNET (legacy, deprecated):**\n"
         "- Fixed 2-layer architecture\n"
         "- Implicit register convention (R0-R4, R9)\n"
         "- Syntax: `TNET #epochs #lr #loss_type`\n\n"
         "**TRAIN (modern, preferred):**\n"
         "- Flexible N-layer architecture via RV descriptor\n"
         "- Config tensor in RU: [epochs, lr, optimizer, print_every, patience, min_delta]\n"
         "- Separate INFER opcode for forward-only inference\n"
         "- Syntax: `LD RV @arch` / `LD RU @config` / `TRAIN RU` / `INFER R8 R0`\n\n"
         "Always prefer TRAIN+INFER over TNET for new code."),
        ("Why should I use TRAIN instead of TNET?",
         "TRAIN is the modern v0.9 opcode that replaces TNET:\n\n"
         "1. **Flexible architecture** — TNET is locked to 2 layers; TRAIN supports N layers via the RV descriptor\n"
         "2. **Optimizer choice** — TRAIN supports SGD (0) and Adam (1); TNET only has SGD\n"
         "3. **Early stopping** — RU config supports patience and min_delta for early stopping\n"
         "4. **Separate inference** — INFER R8 R0 gives a clean forward-only pass without retraining\n"
         "5. **Config-driven** — architecture and hyperparameters are explicit tensor values, not implicit\n\n"
         "TNET still works (it redirects to TRAIN internally) but new code should always use TRAIN+INFER."),
    ]
    for prompt, response in explanations:
        for _ in range(30):
            pairs.append(pair(prompt, response))
    return pairs


def main():
    all_pairs = []
    generators = [
        ("Classic TRAIN",      gen_classic),
        ("ALLC inline",        gen_allc_inline),
        ("TRAIN + inference",  gen_with_inference),
        ("Symbolic TRAIN",     gen_symbolic),
        ("Verbose TRAIN",      gen_verbose),
        ("TNET→TRAIN fixes",   gen_wrong_to_right),
        ("DPO correction",     gen_dpo_pairs),
        ("Explanations",       gen_train_explanation),
    ]

    for name, fn in generators:
        p = fn()
        all_pairs.extend(p)
        print(f"  {name:<20} {len(p):>5} pairs")

    random.shuffle(all_pairs)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w") as f:
        for p in all_pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    print(f"\n  Total: {len(all_pairs)} pairs")
    print(f"  Written to: {OUTPUT}")

    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    import nml_grammar
    valid = 0
    code_pairs = [p for p in all_pairs if not p["messages"][0]["content"].startswith(("What", "How", "Why"))]
    for p in code_pairs:
        code = p["messages"][1]["content"]
        if "```" in code:
            continue
        try:
            r = nml_grammar.validate_grammar(code)
            if r.valid:
                valid += 1
        except:
            pass
    print(f"  Grammar validation (code pairs): {valid}/{len(code_pairs)} ({valid/len(code_pairs)*100:.0f}%)")


if __name__ == "__main__":
    main()
