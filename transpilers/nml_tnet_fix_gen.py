#!/usr/bin/env python3
"""
TNET-focused training data — drills the correct pattern:
  LD R1 @w1 / LD R2 @b1 / LD R3 @w2 / LD R4 @b2
  LD R0 @input / LD R9 @target
  TNET #epochs #lr #loss_type
  ST RA @result / HALT

Every pair shows the complete pattern: loads then TNET with exactly 3 immediates.
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
LOSS_TYPES = [0, 1, 2]
LOSS_NAMES = {0: "MSE", 1: "cross-entropy", 2: "MAE"}
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
W_PATTERNS = [
    ("w1", "b1", "w2", "b2"),
    ("weights1", "bias1", "weights2", "bias2"),
    ("layer1_w", "layer1_b", "layer2_w", "layer2_b"),
    ("hidden_w", "hidden_b", "output_w", "output_b"),
]


def gen_classic():
    pairs = []
    for _ in range(500):
        ep = random.choice(EPOCHS)
        lr = random.choice(LRS)
        lt = random.choice(LOSS_TYPES)
        ln = LOSS_NAMES[lt]
        task, _ = random.choice(TASKS)
        inp = random.choice(INPUT_NAMES)
        tgt = random.choice(TARGET_NAMES)
        res = random.choice(RESULT_NAMES)
        w1, b1, w2, b2 = random.choice(W_PATTERNS)

        code = f"""LD    R1 @{w1}
LD    R2 @{b1}
LD    R3 @{w2}
LD    R4 @{b2}
LD    R0 @{inp}
LD    R9 @{tgt}
TNET  #{ep} #{lr} #{lt}
ST    RA @{res}
HALT"""
        prompts = [
            f"Write NML to train a neural network to {task} using TNET with {ep} epochs and lr={lr}.",
            f"NML: train a 2-layer network with TNET. Use {ep} epochs, learning rate {lr}, {ln} loss.",
            f"Write NML using TNET to {task}. Load weights, input, target, then TNET #{ep} #{lr} #{lt}.",
            f"Write NML for self-training with TNET opcode. {ep} epochs, lr={lr}.",
            f"NML program that uses TNET to train on data. Parameters: epochs={ep}, lr={lr}, loss={ln}.",
            f"Write NML: load w1, b1, w2, b2 into R1-R4, input into R0, target into R9, then TNET #{ep} #{lr} #{lt}.",
        ]
        pairs.append(pair(random.choice(prompts), code))

    return pairs


def gen_with_inference():
    """TNET followed by inference on new data."""
    pairs = []
    for _ in range(200):
        ep = random.choice(EPOCHS)
        lr = random.choice(LRS)
        lt = random.choice(LOSS_TYPES)
        task, _ = random.choice(TASKS)

        code = f"""LD    R1 @w1
LD    R2 @b1
LD    R3 @w2
LD    R4 @b2
LD    R0 @train_input
LD    R9 @train_target
TNET  #{ep} #{lr} #{lt}
LD    R0 @test_input
MMUL  R5 R0 R1
MADD  R5 R5 R2
RELU  R5 R5
MMUL  R6 R5 R3
MADD  RA R6 R4
ST    RA @prediction
HALT"""
        prompt = random.choice([
            f"Write NML to train with TNET ({ep} epochs, lr={lr}), then run inference on test input.",
            f"NML: TNET training loop followed by a forward pass on new data.",
            f"Write NML that trains a network with TNET then predicts on test_input.",
        ])
        pairs.append(pair(prompt, code))

    return pairs


def gen_symbolic():
    pairs = []
    for _ in range(200):
        ep = random.choice(EPOCHS)
        lr = random.choice(LRS)
        lt = random.choice(LOSS_TYPES)

        code = f"""↓  κ  @w1
↓  λ  @b1
↓  μ  @w2
↓  ν  @b2
↓  ι  @input
↓  ς  @target
⥁  #{ep} #{lr} #{lt}
↑  α  @result
◼"""
        prompt = random.choice([
            f"Write symbolic NML to train with ⥁ (TNET) for {ep} epochs, lr={lr}.",
            f"Symbolic NML: load weights into κ-ν, input ι, target ς, then ⥁ #{ep} #{lr} #{lt}.",
            f"Write NML in symbolic syntax using ⥁ for self-training.",
        ])
        pairs.append(pair(prompt, code))

    return pairs


def gen_verbose():
    pairs = []
    for _ in range(100):
        ep = random.choice(EPOCHS)
        lr = random.choice(LRS)
        lt = random.choice(LOSS_TYPES)

        code = f"""LOAD  R1  @w1
LOAD  R2  @b1
LOAD  R3  @w2
LOAD  R4  @b2
LOAD  R0  @input
LOAD  R9  @target
TNET  #{ep}  #{lr}  #{lt}
STORE  ACCUMULATOR  @result
STOP"""
        prompt = f"Write verbose NML to self-train with TNET for {ep} epochs."
        pairs.append(pair(prompt, code))

    return pairs


def gen_wrong_to_right():
    """Correction pairs: show the wrong form and the fix."""
    pairs = []
    for _ in range(200):
        ep = random.choice(EPOCHS)
        lr = random.choice(LRS)
        lt = random.choice(LOSS_TYPES)

        wrong = f"TNET  R0 R1 R2 R3 R4 R5 #{ep} #{lr}"
        right = f"""LD    R1 @w1
LD    R2 @b1
LD    R3 @w2
LD    R4 @b2
LD    R0 @input
LD    R9 @target
TNET  #{ep} #{lr} #{lt}
ST    RA @result
HALT"""

        prompts = [
            f"Fix this NML: `{wrong}`. TNET takes 3 immediates, not registers.",
            f"The program `{wrong}` is wrong. TNET uses implicit registers R0-R4/R9. Fix it.",
            f"Correct this TNET usage: `{wrong}`. Show the proper pattern with LD then TNET #epochs #lr #loss.",
        ]
        pairs.append(pair(random.choice(prompts), right))

    return pairs


def gen_tnet_explanation():
    """Q&A pairs explaining TNET syntax."""
    pairs = []
    explanations = [
        ("What is the correct syntax for NML TNET?",
         "TNET takes exactly 3 immediates: `TNET #epochs #lr #loss_type` where loss_type is 0=MSE, 1=cross-entropy, 2=MAE. Registers R0 (input), R1 (w1), R2 (b1), R3 (w2), R4 (b2), and R9 (target) must be loaded beforehand. After training, RA holds the final prediction.\n\nExample:\n```\nLD    R1 @w1\nLD    R2 @b1\nLD    R3 @w2\nLD    R4 @b2\nLD    R0 @input\nLD    R9 @target\nTNET  #1000 #0.01 #0\nST    RA @result\nHALT\n```"),
        ("How does TNET work in NML?",
         "TNET is a single-instruction training loop. It reads weights from R1-R4 (w1, b1, w2, b2), input from R0, and target from R9. The 3 parameters are: #epochs (iteration count), #lr (learning rate), #loss_type (0=MSE, 1=CE, 2=MAE).\n\nCorrect usage:\n```\nLD    R1 @w1\nLD    R2 @b1\nLD    R3 @w2\nLD    R4 @b2\nLD    R0 @input\nLD    R9 @target\nTNET  #2000 #0.001 #0\nST    RA @result\nHALT\n```"),
        ("Why does TNET only take 3 operands instead of registers?",
         "TNET uses an implicit register convention: R0=input, R1=w1, R2=b1, R3=w2, R4=b2, R9=target. You load these registers with LD before calling TNET. The 3 operands are all immediates: `TNET #epochs #learning_rate #loss_type`. Do NOT pass registers to TNET — load them first, then call TNET with the 3 parameters."),
    ]
    for prompt, response in explanations:
        for _ in range(30):
            pairs.append(pair(prompt, response))
    return pairs


def main():
    all_pairs = []
    generators = [
        ("Classic TNET", gen_classic),
        ("TNET + inference", gen_with_inference),
        ("Symbolic TNET", gen_symbolic),
        ("Verbose TNET", gen_verbose),
        ("Wrong→Right", gen_wrong_to_right),
        ("Explanations", gen_tnet_explanation),
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
