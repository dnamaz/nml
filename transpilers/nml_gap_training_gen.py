#!/usr/bin/env python3
"""
NML Gap Training Data Generator — targeted pairs for opcodes the LLM
gets wrong or never generates. Run after llm_opcode_test.py identifies gaps.

Output: domain/output/training/nml_gap_fix_pairs.jsonl

Categories:
  1. TRAIN+INFER (training config via ALLC RU, then TRAIN RU / INFER R8 R0)
  2. BKWD/WUPD/LOSS (manual training loops)
  3. CMPF (exactly 4 operands)
  4. Under-represented: BNOT, DOT, FRAG/LINK, SCAT/SCTR, SIGN/VRFY, TRAP
"""

import json
import random
from pathlib import Path

random.seed(42)
OUTPUT = Path(__file__).parent.parent / "domain" / "output" / "training" / "nml_gap_fix_pairs.jsonl"

def pair(prompt, code):
    return {"messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": code}]}

def rand_lr():
    return random.choice([0.001, 0.005, 0.01, 0.02, 0.05, 0.1])

def rand_epochs():
    return random.choice([100, 200, 500, 1000, 2000, 5000])

def rand_loss():
    return random.choice([0, 1, 2])

def rand_float():
    return round(random.uniform(0.1, 100.0), 1)

def rand_thresh():
    return round(random.uniform(0.01, 500.0), 2)

def rand_feat():
    return random.randint(0, 7)

def rand_name():
    return random.choice(["data", "input", "x", "features", "values", "signal", "tensor", "weights", "params", "samples"])

def rand_name2():
    return random.choice(["result", "output", "y", "prediction", "score", "answer", "out", "target"])

LOSS_NAMES = {0: "MSE", 1: "cross-entropy", 2: "MAE"}
SYNTAXES = ["classic", "symbolic", "verbose"]

# ═══════════════════════════════════════════════════════════════
# 1. TRAIN+INFER — self-training loop
# ═══════════════════════════════════════════════════════════════

def gen_tnet():
    pairs = []
    for _ in range(300):
        epochs = rand_epochs()
        lr = rand_lr()
        loss = rand_loss()
        loss_name = LOSS_NAMES[loss]

        prompts = [
            f"Write NML to train a neural network using TRAIN+INFER for {epochs} epochs with learning rate {lr}.",
            f"Write NML using TRAIN+INFER to self-train on input/target data. Use {loss_name} loss, {epochs} epochs, lr={lr}.",
            f"NML program: load weights w1,b1,w2,b2, input, target, then train with ALLC RU / TRAIN RU / INFER.",
            f"Write NML for self-training: load 2-layer weights, input in R0, target in R9, use TRAIN+INFER.",
        ]
        prompt = random.choice(prompts)

        code = f"""LD    R1 @w1
LD    R2 @b1
LD    R3 @w2
LD    R4 @b2
LD    R0 @input
LD    R9 @target
ALLC  RU #[6] {epochs},{lr},{loss},0,0,0
TRAIN RU
INFER R8 R0
ST    R8 @result
HALT"""
        pairs.append(pair(prompt, code))

    for _ in range(100):
        epochs = rand_epochs()
        lr = rand_lr()
        loss = rand_loss()
        code = f"""↓  κ  @w1
↓  λ  @b1
↓  μ  @w2
↓  ν  @b2
↓  ι  @input
↓  ς  @target
□  RU  #[6]  {epochs},{lr},{loss},0,0,0
⥁  RU
⥂  R8  ι
↑  R8  @result
◼"""
        prompt = random.choice([
            f"Write symbolic NML to train with TRAIN+INFER for {epochs} epochs.",
            f"Symbolic NML: self-train a network using ⥁ (TRAIN) + ⥂ (INFER) with lr={lr}.",
        ])
        pairs.append(pair(prompt, code))

    for _ in range(100):
        epochs = rand_epochs()
        lr = rand_lr()
        loss = rand_loss()
        code = f"""LOAD  R1  @w1
LOAD  R2  @b1
LOAD  R3  @w2
LOAD  R4  @b2
LOAD  R0  @input
LOAD  R9  @target
ALLOCATE  RU  #[6]  {epochs},{lr},{loss},0,0,0
TRAIN_NETWORK  RU
INFER_FORWARD  R8  R0
STORE  R8  @result
STOP"""
        prompt = f"Write verbose NML to train a network with TRAIN+INFER for {epochs} epochs."
        pairs.append(pair(prompt, code))

    return pairs


# ═══════════════════════════════════════════════════════════════
# 2. BKWD / WUPD / LOSS — manual training loops
# ═══════════════════════════════════════════════════════════════

def gen_training_manual():
    pairs = []
    for _ in range(200):
        lr = rand_lr()
        loss_type = rand_loss()
        loop_count = random.choice([50, 100, 200, 500, 1000])

        code = f"""LD    R0 @input
LD    R1 @weights
LD    R2 @bias
LD    R9 @target
LOOP  #{loop_count}
MMUL  R3 R0 R1
MADD  R3 R3 R2
RELU  R3 R3
LOSS  R5 R3 R9 #{loss_type}
BKWD  RG R3 R9
WUPD  R1 RG #{lr}
ENDP
ST    R3 @output
ST    R5 @final_loss
HALT"""
        prompts = [
            f"Write NML for a manual training loop: forward pass, LOSS, BKWD, WUPD. Use {loop_count} iterations and lr={lr}.",
            f"Write NML to train weights with BKWD and WUPD in a LOOP. Compute LOSS with type {loss_type}.",
            f"NML training loop: MMUL forward pass, RELU, compute LOSS, backpropagate with BKWD, update with WUPD.",
        ]
        pairs.append(pair(random.choice(prompts), code))

    for _ in range(100):
        lr = rand_lr()
        code = f"""LD    R0 @input
LD    R1 @weights
LD    R9 @target
MMUL  R3 R0 R1
LOSS  R5 R3 R9 #0
BKWD  RG R3 R9
WUPD  R1 RG #{lr}
MMUL  R3 R0 R1
ST    R3 @output
ST    R5 @loss
HALT"""
        prompt = random.choice([
            f"Write NML for one training step: forward, LOSS, BKWD, WUPD with lr={lr}.",
            "Write NML to compute MSE LOSS, backpropagate with BKWD, then update weights with WUPD.",
        ])
        pairs.append(pair(prompt, code))

    for _ in range(100):
        lr = rand_lr()
        loss_type = rand_loss()
        code = f"""↓  ι  @input
↓  κ  @weights
↓  λ  @bias
↓  ς  @target
×  μ  ι  κ
⊕  μ  μ  λ
⌐  μ  μ
△  ξ  μ  ς  #{loss_type}
∇  η  μ  ς
⟳  κ  η  #{lr}
↑  μ  @output
↑  ξ  @loss
◼"""
        prompt = random.choice([
            f"Symbolic NML: forward pass, compute loss with △, backprop with ∇, update with ⟳.",
            f"Write symbolic NML for a training step using △ (LOSS), ∇ (BKWD), ⟳ (WUPD).",
        ])
        pairs.append(pair(prompt, code))

    return pairs


# ═══════════════════════════════════════════════════════════════
# 3. CMPF — feature comparison (exactly 4 operands)
# ═══════════════════════════════════════════════════════════════

def gen_cmpf():
    pairs = []
    for _ in range(300):
        feat = rand_feat()
        thresh = rand_thresh()
        val_true = rand_float()
        val_false = rand_float()
        name = rand_name()

        code = f"""LD    R0 @{name}
CMPF  RE R0 #{feat} #{thresh}
JMPT  #3
LEAF  RA #{val_false}
JUMP  #2
LEAF  RA #{val_true}
ST    RA @result
HALT"""
        prompts = [
            f"Write NML to compare feature {feat} of a tensor against {thresh} using CMPF.",
            f"NML decision tree: if feature[{feat}] < {thresh} then {val_true} else {val_false}. Use CMPF.",
            f"Write NML using CMPF to check if feature index {feat} is below {thresh}.",
        ]
        pairs.append(pair(random.choice(prompts), code))

    for _ in range(100):
        feat = rand_feat()
        thresh = rand_thresh()
        code = f"""↓  ι  @features
⋈  φ  ι  #{feat}  #{thresh}
↗  #3
∎  α  #0.0
→  #2
∎  α  #1.0
↑  α  @result
◼"""
        prompt = f"Symbolic NML: use ⋈ (CMPF) to compare feature {feat} against {thresh}."
        pairs.append(pair(prompt, code))

    for _ in range(100):
        feat = rand_feat()
        thresh = rand_thresh()
        feat2 = rand_feat()
        thresh2 = rand_thresh()
        code = f"""LD    R0 @input
CMPF  RE R0 #{feat} #{thresh}
JMPF  #4
CMPF  RE R0 #{feat2} #{thresh2}
JMPF  #3
LEAF  RA #100.0
JUMP  #4
LEAF  RA #200.0
JUMP  #2
LEAF  RA #300.0
ST    RA @result
HALT"""
        prompt = f"Write NML for a 2-level decision tree: first compare feature {feat} < {thresh}, then feature {feat2} < {thresh2}."
        pairs.append(pair(prompt, code))

    return pairs


# ═══════════════════════════════════════════════════════════════
# 4. Under-represented opcodes
# ═══════════════════════════════════════════════════════════════

def gen_bnot():
    pairs = []
    for _ in range(200):
        val = random.randint(0, 255)
        code = f"""LEAF  R0 #{val}.0
BNOT  R1 R0
ST    R1 @result
HALT"""
        prompts = [
            f"Write NML to compute bitwise NOT of {val} using BNOT.",
            f"NML: load {val}, apply BNOT, store result.",
        ]
        pairs.append(pair(random.choice(prompts), code))

    for _ in range(50):
        code = f"""∎  ι  #{random.randint(0,255)}.0
¬  κ  ι
↑  κ  @result
◼"""
        pairs.append(pair("Symbolic NML: apply ¬ (BNOT) to a value.", code))
    return pairs


def gen_dot():
    pairs = []
    for _ in range(200):
        code = f"""LD    R0 @a
LD    R1 @b
DOT   R2 R0 R1
ST    R2 @result
HALT"""
        prompts = [
            "Write NML to compute the dot product using DOT.",
            "NML: dot product of two vectors with DOT (alias for SDOT).",
            "Write NML using DOT to compute inner product of a and b.",
        ]
        pairs.append(pair(random.choice(prompts), code))
    return pairs


def gen_frag_link():
    pairs = []
    for _ in range(200):
        name1 = random.choice(["compute_tax", "scale_value", "normalize", "layer1", "activation"])
        name2 = random.choice(["apply_bias", "output_layer", "postprocess", "layer2", "clamp_output"])

        code = f"""META  @name "composed_program"
META  @version "0.7.0"

FRAG  {name1}
LD    R0 @input
SCLR  R1 R0 #{round(random.uniform(0.01, 0.5), 3)}
ST    R1 @intermediate
ENDF

FRAG  {name2}
LD    R2 @intermediate
LEAF  R3 #{round(random.uniform(1.0, 100.0), 1)}
TACC  RA R2 R3
ST    RA @result
ENDF

LINK  @{name1}
LINK  @{name2}
HALT"""
        prompts = [
            f"Write NML with two composable fragments '{name1}' and '{name2}' using FRAG, ENDF, and LINK.",
            f"NML with FRAG/ENDF/LINK: define fragment {name1} and {name2}, then link them.",
            "Write NML using fragments (FRAG/ENDF) and composition (LINK).",
        ]
        pairs.append(pair(random.choice(prompts), code))

    for _ in range(50):
        code = f"""§  @name "symbolic_fragments"
◆  compute
↓  ι  @input
∗  κ  ι  #0.1
↑  κ  @result
◇
◆  ι
HALT"""
        pairs.append(pair("Symbolic NML with fragments using ◆ (FRAG) and ◇ (ENDF).", code))

    return pairs


def gen_scat():
    pairs = []
    for _ in range(150):
        idx = random.randint(0, 4)
        val = rand_float()
        size = random.randint(5, 10)
        code = f"""ALLC  R0 #[{size}]
LEAF  R1 #{val}
LEAF  R2 #{idx}.0
SCTR  R0 R1 R2
ST    R0 @result
HALT"""
        prompts = [
            f"Write NML to scatter value {val} at index {idx} in a tensor using SCTR.",
            f"NML: allocate a tensor, write {val} at position {idx} with SCTR.",
            f"Write NML using SCTR to set index {idx} of a zero tensor to {val}.",
        ]
        pairs.append(pair(random.choice(prompts), code))

    for _ in range(50):
        code = f"""LD    R0 @tensor
LEAF  R1 #{rand_float()}
LEAF  R2 #{random.randint(0,4)}.0
SCAT  R1 R0 R2
ST    R0 @result
HALT"""
        pairs.append(pair("Write NML using SCAT to write a value into a tensor by index.", code))

    for _ in range(50):
        code = f"""□  ι  #[5]
∎  κ  #{rand_float()}
∎  λ  #{random.randint(0,4)}.0
⊂  κ  ι  λ
↑  ι  @result
◼"""
        pairs.append(pair("Symbolic NML: scatter a value into a tensor using ⊂ (SCAT).", code))

    return pairs


def gen_sign_vrfy():
    pairs = []
    for _ in range(100):
        agent = random.choice(["transpiler_v1", "agent_001", "validator", "builder", "signer"])
        code = f"""SIGN  agent={agent}  key=hmac-sha256:abc123  sig=def456
META  @name "signed_program"
META  @author "{agent}"
LD    R0 @input
SCLR  R1 R0 #2.0
ST    R1 @result
VRFY  @self  @{agent}
HALT"""
        prompts = [
            f"Write NML for a signed program using SIGN and VRFY with agent '{agent}'.",
            "Write NML with cryptographic signing using SIGN and verification with VRFY.",
        ]
        pairs.append(pair(random.choice(prompts), code))

    for _ in range(50):
        code = f"""✦  agent=signer  key=hmac-sha256:abc123  sig=def456
§  @name "signed"
↓  ι  @input
∗  κ  ι  #2.0
↑  κ  @result
✓  @self  @signer
◼"""
        pairs.append(pair("Symbolic NML with ✦ (SIGN) and ✓ (VRFY).", code))
    return pairs


def gen_trap():
    pairs = []
    for _ in range(150):
        code_val = random.randint(1, 10)
        thresh = rand_thresh()
        code = f"""LD    R0 @input
CMPI  RE R0 #{thresh}
JMPF  #1
TRAP  #{code_val}
SCLR  R1 R0 #2.0
ST    R1 @result
HALT"""
        prompts = [
            f"Write NML that uses TRAP #{code_val} to abort if input < {thresh}.",
            f"NML with error handling: TRAP if a condition is met.",
            f"Write NML using TRAP to signal an error with code {code_val}.",
        ]
        pairs.append(pair(random.choice(prompts), code))

    for _ in range(50):
        code = f"""↓  ι  @input
≺  φ  ι  #0.0
↘  #1
⚠  #1
∗  κ  ι  #2.0
↑  κ  @result
◼"""
        pairs.append(pair("Symbolic NML: use ⚠ (TRAP) to abort on negative input.", code))
    return pairs


# ═══════════════════════════════════════════════════════════════
# 5. CMPI — immediate comparison with threshold branching
# ═══════════════════════════════════════════════════════════════

def gen_cmpi():
    pairs = []
    for _ in range(100):
        thresh = round(random.choice([0.3, 0.4, 0.5, 0.5, 0.6, 0.7, 0.8]), 1)
        inp = rand_name()
        out = rand_name2()

        code = f"""LD    R0 @{inp}
CMPI  RE R0 #{thresh}
JMPF  #2
LEAF  RB #0.0
JUMP  #1
LEAF  RB #1.0
ST    RB @{out}
HALT"""
        prompts = [
            f"Write NML using CMPI to flag {inp} as 1.0 if value >= {thresh}, else 0.0.",
            f"NML: if {inp} >= {thresh} then {out}=1 else {out}=0. Use CMPI and JMPF.",
            f"Write NML for a threshold check at {thresh} using CMPI. Output 1.0 above, 0.0 below.",
        ]
        pairs.append(pair(random.choice(prompts), code))

    for _ in range(50):
        thresh = round(random.uniform(0.1, 0.9), 1)
        code = f"""↓  ι  @input
≺  φ  ι  #{thresh}
↘  #2
∎  β  #0.0
→  #1
∎  β  #1.0
↑  β  @result
◼"""
        pairs.append(pair(f"Symbolic NML: flag input as 1.0 if >= {thresh} using ≺ (CMPI) and ↘ (JMPF).", code))

    for _ in range(50):
        thresh = round(random.uniform(0.1, 0.9), 1)
        code = f"""LOAD  R0  @input
COMPARE_VALUE  FLAG  R0  #{thresh}
BRANCH_FALSE  #2
SET_VALUE  GENERAL  #0.0
JUMP  #1
SET_VALUE  GENERAL  #1.0
STORE  GENERAL  @result
STOP"""
        pairs.append(pair(f"Verbose NML: threshold check at {thresh}, output 1.0 if above.", code))

    return pairs


# ═══════════════════════════════════════════════════════════════
# 6. CMP — register-to-register comparison
# ═══════════════════════════════════════════════════════════════

def gen_cmp():
    pairs = []
    for _ in range(60):
        code = f"""LD    R0 @a
LD    R1 @b
CMP   R0 R1
JMPF  #2
MOV   RA R0
JUMP  #1
MOV   RA R1
ST    RA @result
HALT"""
        prompts = [
            "Write NML to compare two values with CMP and return the smaller one.",
            "NML: load a and b, use CMP to compare, output the minimum.",
            "Write NML using CMP and JMPF to select the smaller of two values.",
        ]
        pairs.append(pair(random.choice(prompts), code))

    for _ in range(40):
        code = f"""LD    R0 @a
LD    R1 @b
CMP   R0 R1
JMPT  #2
MOV   RA R1
JUMP  #1
MOV   RA R0
ST    RA @result
HALT"""
        prompts = [
            "Write NML to compare two values with CMP and return the larger one.",
            "NML: use CMP to find the maximum of a and b.",
        ]
        pairs.append(pair(random.choice(prompts), code))

    return pairs


# ═══════════════════════════════════════════════════════════════
# 7. SYNC — pipeline barrier
# ═══════════════════════════════════════════════════════════════

def gen_sync():
    pairs = []
    for _ in range(30):
        code = f"""LD    R0 @stage1_input
SCLR  R1 R0 #2.0
ST    R1 @stage1_output
SYNC
LD    R2 @stage1_output
SCLR  RA R2 #3.0
ST    RA @result
HALT"""
        prompts = [
            "Write NML with SYNC as a barrier between two pipeline stages.",
            "NML: two-stage pipeline with SYNC barrier between stages.",
            "Write NML using SYNC to ensure stage 1 completes before stage 2 reads its output.",
        ]
        pairs.append(pair(random.choice(prompts), code))

    for _ in range(20):
        code = f"""↓  ι  @input
∗  κ  ι  #2.0
↑  κ  @intermediate
⏸
↓  λ  @intermediate
∗  α  λ  #3.0
↑  α  @result
◼"""
        pairs.append(pair("Symbolic NML: two-stage pipeline with ⏸ (SYNC) barrier.", code))

    return pairs


# ═══════════════════════════════════════════════════════════════
# 8. MOV — register copy
# ═══════════════════════════════════════════════════════════════

def gen_mov():
    pairs = []
    for _ in range(30):
        val = rand_float()
        code = f"""LEAF  R0 #{val}
MOV   R1 R0
SCLR  R1 R1 #2.0
ST    R0 @original
ST    R1 @doubled
HALT"""
        prompts = [
            f"Write NML to copy R0 to R1 using MOV, then scale R1 while keeping R0 unchanged.",
            "NML: use MOV to duplicate a register, then modify the copy.",
            "Write NML using MOV to copy a value to another register.",
        ]
        pairs.append(pair(random.choice(prompts), code))

    for _ in range(20):
        code = f"""∎  ι  #{rand_float()}
←  κ  ι
∗  κ  κ  #2.0
↑  ι  @original
↑  κ  @doubled
◼"""
        pairs.append(pair("Symbolic NML: copy a register with ← (MOV) and scale the copy.", code))

    return pairs


# ═══════════════════════════════════════════════════════════════
# 9. ALLC — allocate zero tensor
# ═══════════════════════════════════════════════════════════════

def gen_allc():
    pairs = []
    for _ in range(30):
        size = random.randint(1, 20)
        code = f"""ALLC  R0 #[{size}]
ST    R0 @zeros
HALT"""
        prompts = [
            f"Write NML to allocate a zero tensor of size {size} using ALLC.",
            f"NML: create a {size}-element zero tensor with ALLC.",
            f"Write NML using ALLC to initialize R0 as a {size}-element zero vector.",
        ]
        pairs.append(pair(random.choice(prompts), code))

    for _ in range(20):
        size = random.randint(1, 10)
        code = f"""ALLC  R0 #[{size}]
LEAF  R1 #1.0
TACC  R0 R0 R1
ST    R0 @ones
HALT"""
        prompts = [
            f"Write NML to create a {size}-element tensor of ones: ALLC then TACC with 1.0.",
            f"NML: allocate {size} zeros with ALLC, then add 1.0 to make an all-ones vector.",
        ]
        pairs.append(pair(random.choice(prompts), code))

    return pairs


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    all_pairs = []

    generators = [
        ("TRAIN+INFER", gen_tnet),
        ("BKWD/WUPD/LOSS", gen_training_manual),
        ("CMPF", gen_cmpf),
        ("BNOT", gen_bnot),
        ("DOT", gen_dot),
        ("FRAG/LINK", gen_frag_link),
        ("SCAT/SCTR", gen_scat),
        ("SIGN/VRFY", gen_sign_vrfy),
        ("TRAP", gen_trap),
        ("CMPI", gen_cmpi),
        ("CMP", gen_cmp),
        ("SYNC", gen_sync),
        ("MOV", gen_mov),
        ("ALLC", gen_allc),
    ]

    for name, gen_fn in generators:
        pairs = gen_fn()
        all_pairs.extend(pairs)
        print(f"  {name:<16} {len(pairs):>5} pairs")

    random.shuffle(all_pairs)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w") as f:
        for p in all_pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    print(f"\n  Total: {len(all_pairs)} pairs")
    print(f"  Written to: {OUTPUT}")

    # Validate a sample
    print("\n  Validating sample...")
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    import nml_grammar

    valid = 0
    checked = 0
    for p in random.sample(all_pairs, min(200, len(all_pairs))):
        code = p["messages"][1]["content"]
        try:
            report = nml_grammar.validate_grammar(code)
            if report.valid:
                valid += 1
            checked += 1
        except:
            checked += 1

    print(f"  Sample validation: {valid}/{checked} ({valid/checked*100:.0f}%) grammar-valid")


if __name__ == "__main__":
    main()
