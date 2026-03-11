#!/usr/bin/env python3
"""
NML Syntax Training Data Generator

Generates training pairs focused on teaching NML v0.4 as a language:
- Instruction reference (what each opcode does)
- Small program examples with descriptions
- Line-by-line explanations
- Intent-to-NML translation
- Register and pattern knowledge

Usage:
    python3 nml_syntax_gen.py --output output/training/nml_syntax.jsonl
"""

import json
import os
import random

# ═══════════════════════════════════════════
# NML v0.4 Instruction Definitions
# ═══════════════════════════════════════════

INSTRUCTIONS = {
    "MMUL": {"syntax": "MMUL Rd Rs1 Rs2", "desc": "Matrix multiply: Rd = Rs1 @ Rs2", "category": "arithmetic", "example": "MMUL  R3 R0 R1"},
    "MADD": {"syntax": "MADD Rd Rs1 Rs2", "desc": "Element-wise add: Rd = Rs1 + Rs2", "category": "arithmetic", "example": "MADD  R3 R3 R2"},
    "MSUB": {"syntax": "MSUB Rd Rs1 Rs2", "desc": "Element-wise subtract: Rd = Rs1 - Rs2", "category": "arithmetic", "example": "MSUB  R7 R0 R3"},
    "EMUL": {"syntax": "EMUL Rd Rs1 Rs2", "desc": "Element-wise multiply: Rd = Rs1 * Rs2", "category": "arithmetic", "example": "EMUL  R4 R0 R1"},
    "EDIV": {"syntax": "EDIV Rd Rs1 Rs2", "desc": "Element-wise divide: Rd = Rs1 / Rs2 (v0.4)", "category": "arithmetic", "example": "EDIV  R3 R0 R2"},
    "SDOT": {"syntax": "SDOT Rd Rs1 Rs2", "desc": "Scalar dot product: Rd = dot(Rs1, Rs2)", "category": "arithmetic", "example": "SDOT  R5 R0 R1"},
    "SCLR": {"syntax": "SCLR Rd Rs1 #imm", "desc": "Scalar multiply: Rd = Rs1 * imm", "category": "arithmetic", "example": "SCLR  R3 R0 #0.062000"},
    "SDIV": {"syntax": "SDIV Rd Rs1 #imm", "desc": "Scalar divide: Rd = Rs1 / imm (v0.4)", "category": "arithmetic", "example": "SDIV  R1 R0 #3.0"},
    "RELU": {"syntax": "RELU Rd Rs", "desc": "ReLU activation: Rd = max(0, Rs)", "category": "activation", "example": "RELU  R3 R3"},
    "SIGM": {"syntax": "SIGM Rd Rs", "desc": "Sigmoid: Rd = 1/(1+exp(-Rs))", "category": "activation", "example": "SIGM  R9 R9"},
    "TANH": {"syntax": "TANH Rd Rs", "desc": "Tanh activation", "category": "activation", "example": "TANH  R3 R3"},
    "SOFT": {"syntax": "SOFT Rd Rs", "desc": "Softmax across last dimension", "category": "activation", "example": "SOFT  R5 R5"},
    "LD":   {"syntax": "LD Rd @addr", "desc": "Load tensor from memory address", "category": "memory", "example": "LD    R0 @input"},
    "ST":   {"syntax": "ST Rs @addr", "desc": "Store tensor to memory address", "category": "memory", "example": "ST    R3 @output"},
    "MOV":  {"syntax": "MOV Rd Rs", "desc": "Copy register: Rd = Rs", "category": "memory", "example": "MOV   R7 R0"},
    "ALLC": {"syntax": "ALLC Rd #shape", "desc": "Allocate zero tensor with given shape", "category": "memory", "example": "ALLC  RA #[1]"},
    "RSHP": {"syntax": "RSHP Rd Rs #shape", "desc": "Reshape tensor preserving data", "category": "dataflow", "example": "RSHP  R3 R0 #[2,4]"},
    "TRNS": {"syntax": "TRNS Rd Rs", "desc": "Transpose (swap last two dims)", "category": "dataflow", "example": "TRNS  R2 R1"},
    "SPLT": {"syntax": "SPLT Rd Re Rs #dim", "desc": "Split tensor along dimension into two parts", "category": "dataflow", "example": "SPLT  R3 R4 R0 #0"},
    "MERG": {"syntax": "MERG Rd Rs1 Rs2 #dim", "desc": "Concatenate two tensors along dimension", "category": "dataflow", "example": "MERG  R5 R3 R4 #0"},
    "CMPF": {"syntax": "CMPF Rd Rs #feat #thresh", "desc": "Tree feature compare: flag = (Rs[feat] < thresh)", "category": "comparison", "example": "CMPF  RE R0 #0 #50000.0"},
    "CMP":  {"syntax": "CMP Rs1 Rs2", "desc": "General compare: flag = (Rs1[0] < Rs2[0]) (v0.4)", "category": "comparison", "example": "CMP   R0 R1"},
    "CMPI": {"syntax": "CMPI Rd Rs #imm", "desc": "Immediate compare: flag = (Rs[0] < imm) (v0.4)", "category": "comparison", "example": "CMPI  RE R0 #100.0"},
    "JMPT": {"syntax": "JMPT #offset", "desc": "Jump by offset if condition flag is true", "category": "control", "example": "JMPT  #3"},
    "JMPF": {"syntax": "JMPF #offset", "desc": "Jump by offset if condition flag is false", "category": "control", "example": "JMPF  #5"},
    "JUMP": {"syntax": "JUMP #offset", "desc": "Unconditional jump by offset", "category": "control", "example": "JUMP  #2"},
    "LOOP": {"syntax": "LOOP #count", "desc": "Begin counted loop for N iterations", "category": "control", "example": "LOOP  #10"},
    "ENDP": {"syntax": "ENDP", "desc": "End loop block", "category": "control", "example": "ENDP"},
    "CALL": {"syntax": "CALL #offset", "desc": "Push return address, jump by offset (v0.4)", "category": "subroutine", "example": "CALL  #4"},
    "RET":  {"syntax": "RET", "desc": "Pop return address, resume after CALL (v0.4)", "category": "subroutine", "example": "RET"},
    "LEAF": {"syntax": "LEAF Rd #value", "desc": "Store scalar immediate value in register", "category": "tree", "example": "LEAF  RC #1234.56"},
    "TACC": {"syntax": "TACC Rd Rs1 Rs2", "desc": "Tree accumulate: scalar Rd = Rs1 + Rs2", "category": "tree", "example": "TACC  RA RA RC"},
    "SYNC": {"syntax": "SYNC", "desc": "Synchronization barrier (no-op in single-threaded)", "category": "system", "example": "SYNC"},
    "HALT": {"syntax": "HALT", "desc": "Stop execution", "category": "system", "example": "HALT"},
    "TRAP": {"syntax": "TRAP #code", "desc": "Trigger program fault with error code (v0.4)", "category": "system", "example": "TRAP  #1"},
}

REGISTERS = {
    "R0": "General-purpose tensor register",
    "R1": "General-purpose tensor register",
    "R2": "General-purpose tensor register",
    "R3": "General-purpose tensor register",
    "R4": "General-purpose tensor register",
    "R5": "General-purpose tensor register",
    "R6": "General-purpose tensor register",
    "R7": "General-purpose tensor register",
    "R8": "General-purpose tensor register",
    "R9": "General-purpose tensor register",
    "RA": "Accumulator — tree prediction accumulation",
    "RB": "General-purpose",
    "RC": "Scratch — leaf values, intermediate results",
    "RD": "Counter — loop counter",
    "RE": "Flag — condition flag set by CMPF, CMP, CMPI",
    "RF": "Stack pointer",
}

# ═══════════════════════════════════════════
# Small NML Program Templates
# ═══════════════════════════════════════════

PROGRAM_TEMPLATES = [
    {
        "desc": "Load a value and store it",
        "code": "LD    R0 @input\nST    R0 @output\nHALT",
        "explanation": "Loads tensor from @input into R0, stores it to @output, then halts.",
    },
    {
        "desc": "Single dense layer with ReLU activation",
        "code": "LD    R0 @input\nLD    R1 @weights\nLD    R2 @bias\nMMUL  R3 R0 R1\nMADD  R3 R3 R2\nRELU  R3 R3\nST    R3 @output\nHALT",
        "explanation": "Neural network dense layer: output = ReLU(input @ weights + bias).",
    },
    {
        "desc": "Two-layer neural network with sigmoid output",
        "code": "LD    R0 @input\nLD    R1 @w1\nLD    R2 @b1\nMMUL  R3 R0 R1\nMADD  R3 R3 R2\nRELU  R3 R3\nLD    R4 @w2\nLD    R5 @b2\nMMUL  R6 R3 R4\nMADD  R6 R6 R5\nSIGM  R6 R6\nST    R6 @output\nHALT",
        "explanation": "Two-layer MLP: hidden = ReLU(input @ w1 + b1), output = sigmoid(hidden @ w2 + b2).",
    },
    {
        "desc": "Flat rate tax: rate times gross pay",
        "code": "LD    R0 @gross_pay\nSCLR  R1 R0 #0.050000\nST    R1 @tax\nHALT",
        "explanation": "Calculates 5% flat tax: tax = gross_pay * 0.05.",
    },
    {
        "desc": "Flat rate tax with wage base cap",
        "code": "LD    R0 @gross_pay\nALLC  RA #[1]\nCMPF  RE R0 #0 #176100.000000\nJMPT  #4\nLEAF  RC #176100.000000\nSCLR  RC RC #0.062000\nTACC  RA RA RC\nJUMP  #2\nSCLR  RC R0 #0.062000\nTACC  RA RA RC\nST    RA @tax\nHALT",
        "explanation": "FICA tax: if gross > $176,100 wage base, tax = 176100 * 6.2%; otherwise tax = gross * 6.2%.",
    },
    {
        "desc": "Progressive bracket tax lookup using domain calc() formula",
        "code": "LD    R0 @income\nALLC  RA #[1]\nCMPF  RE R0 #0 #47150.000000\nJMPF  #6\nCMPF  RE R0 #0 #11600.000000\nJMPF  #3\nSCLR  RA R0 #0.100000\nJUMP  #7\nLEAF  RA #1160.000000\nLEAF  RC #11600.000000\nMSUB  R8 R0 RC\nSCLR  R8 R8 #0.120000\nTACC  RA RA R8\nJUMP  #3\nLEAF  RA #5266.000000\nLEAF  RC #47150.000000\nMSUB  R8 R0 RC\nSCLR  R8 R8 #0.220000\nTACC  RA RA R8\nST    RA @tax\nHALT",
        "explanation": "Three-bracket progressive tax: 10% up to $11,600, 12% up to $47,150, 22% above. Uses domain formula: tax = addition + (income - threshold) * rate.",
    },
    {
        "desc": "While loop summing 1 to 5 using backward jumps (v0.4)",
        "code": "LEAF  RA #0.0\nLEAF  RD #1.0\nTACC  RA RA RD\nLEAF  RC #1.0\nTACC  RD RD RC\nCMPI  RE RD #6.0\nJMPT  #-5\nST    RA @result\nHALT",
        "explanation": "v0.4 backward jump loop: sum = 0, counter = 1. Loop: sum += counter, counter++. While counter < 6, jump back 5 instructions. Result = 15.",
    },
    {
        "desc": "Subroutine call that doubles a value (v0.4)",
        "code": "LEAF  R1 #7.0\nCALL  #2\nST    R1 @result\nJUMP  #2\nTACC  R1 R1 R1\nRET\nHALT",
        "explanation": "v0.4 subroutine: CALL pushes return address and jumps to the TACC instruction which doubles R1. RET returns to the ST instruction. Result = 14.",
    },
    {
        "desc": "Scalar and element-wise division (v0.4)",
        "code": "LEAF  R0 #100.0\nSDIV  R1 R0 #3.0\nLEAF  R2 #4.0\nEDIV  R3 R0 R2\nST    R1 @scalar_div\nST    R3 @elem_div\nHALT",
        "explanation": "v0.4 division: SDIV divides R0 by immediate 3.0 giving 33.33. EDIV divides R0 element-wise by R2 giving 25.0.",
    },
    {
        "desc": "Conditional branch with CMPI (v0.4)",
        "code": "LD    R0 @value\nCMPI  RE R0 #50.0\nJMPT  #2\nLEAF  R1 #0.030000\nJUMP  #1\nLEAF  R1 #0.050000\nSCLR  R2 R0 R1\nST    R2 @result\nHALT",
        "explanation": "v0.4 CMPI: if value < 50, use rate 5%; else use rate 3%. CMPI sets flag directly without needing a feature index like CMPF.",
    },
    {
        "desc": "Exempt check: skip tax if exempt flag is set",
        "code": "LD    R0 @gross_pay\nLD    R3 @is_exempt\nCMPF  RE R3 #0 #0.5\nJMPF  #3\nSCLR  R1 R0 #0.050000\nST    R1 @tax\nHALT",
        "explanation": "If is_exempt >= 0.5 (true), JMPF jumps past the tax calculation to HALT. Otherwise computes 5% tax.",
    },
    {
        "desc": "Resident vs nonresident rate branch",
        "code": "LD    R0 @gross_pay\nLD    R4 @is_resident\nCMPF  RE R4 #0 #0.5\nJMPT  #2\nSCLR  R1 R0 #0.020000\nJUMP  #1\nSCLR  R1 R0 #0.015000\nST    R1 @tax\nHALT",
        "explanation": "If is_resident < 0.5 (nonresident), applies 1.5% rate. Otherwise applies 2.0% resident rate.",
    },
    {
        "desc": "Accumulate multiple tax components",
        "code": "LD    R0 @gross_pay\nSCLR  RA R0 #0.100000\nSCLR  RB R0 #0.062000\nSCLR  R3 R0 #0.014500\nTACC  R8 RA RB\nTACC  R8 R8 R3\nMSUB  R9 R0 R8\nST    R8 @total_tax\nST    R9 @net_pay\nHALT",
        "explanation": "Computes three taxes (10% FIT, 6.2% FICA, 1.45% Medicare), accumulates with TACC, then net_pay = gross - total_tax.",
    },
]

# ═══════════════════════════════════════════
# Generators
# ═══════════════════════════════════════════

def gen_instruction_reference() -> list[dict]:
    """Generate Q&A pairs for each instruction."""
    pairs = []
    for opcode, info in INSTRUCTIONS.items():
        pairs.append({
            "instruction": f"What does the NML instruction {opcode} do?",
            "input": "",
            "output": f"{info['syntax']} — {info['desc']}",
            "metadata": {"type": "nml_instruction_ref", "opcode": opcode},
        })
        pairs.append({
            "instruction": f"Show me the syntax for {opcode} in NML v0.4.",
            "input": "",
            "output": f"Syntax: {info['syntax']}\nDescription: {info['desc']}\nExample: {info['example']}",
            "metadata": {"type": "nml_instruction_ref", "opcode": opcode},
        })
        pairs.append({
            "instruction": f"Write a single NML instruction that {info['desc'].lower().rstrip('.')}.",
            "input": "",
            "output": info["example"],
            "metadata": {"type": "nml_single_instruction", "opcode": opcode},
        })
    return pairs


def gen_category_knowledge() -> list[dict]:
    """Generate knowledge about instruction categories."""
    by_cat = {}
    for op, info in INSTRUCTIONS.items():
        cat = info["category"]
        if cat not in by_cat:
            by_cat[cat] = []
        by_cat[cat].append(op)

    pairs = []
    for cat, ops in by_cat.items():
        pairs.append({
            "instruction": f"What are all the NML v0.4 {cat} instructions?",
            "input": "",
            "output": f"NML v0.4 {cat} instructions: {', '.join(ops)}",
            "metadata": {"type": "nml_category", "category": cat},
        })

    pairs.append({
        "instruction": "How many instructions does NML v0.4 have?",
        "input": "",
        "output": "NML v0.4 has 49 total instructions: 35 core + 14 extensions (NML-V: 4, NML-T: 4, NML-R: 4, NML-S: 2).",
        "metadata": {"type": "nml_knowledge"},
    })
    pairs.append({
        "instruction": "What instructions were added in NML v0.4?",
        "input": "",
        "output": "NML v0.4 added 9 new core instructions: SDIV (scalar divide), EDIV (element-wise divide), CMP (general register compare), CMPI (immediate compare), CALL (subroutine call), RET (subroutine return), TRAP (program fault). It also added backward jumps (negative offsets in JMPT/JMPF/JUMP) and the --trace execution mode.",
        "metadata": {"type": "nml_knowledge"},
    })
    return pairs


def gen_register_knowledge() -> list[dict]:
    """Generate pairs about registers."""
    pairs = []
    for reg, desc in REGISTERS.items():
        pairs.append({
            "instruction": f"What is NML register {reg} used for?",
            "input": "",
            "output": f"{reg}: {desc}",
            "metadata": {"type": "nml_register", "register": reg},
        })

    pairs.append({
        "instruction": "How many registers does NML have?",
        "input": "",
        "output": "NML has 16 tensor registers: R0-R9 (10 general purpose), RA (accumulator), RB (general), RC (scratch), RD (counter), RE (condition flag), RF (stack pointer).",
        "metadata": {"type": "nml_register"},
    })
    return pairs


def gen_program_examples() -> list[dict]:
    """Generate training pairs from program templates."""
    pairs = []
    for tmpl in PROGRAM_TEMPLATES:
        pairs.append({
            "instruction": f"Write an NML v0.4 program to: {tmpl['desc']}",
            "input": "",
            "output": tmpl["code"],
            "metadata": {"type": "nml_program"},
        })
        pairs.append({
            "instruction": f"Explain this NML program:\n{tmpl['code']}",
            "input": "",
            "output": tmpl["explanation"],
            "metadata": {"type": "nml_explanation"},
        })

    return pairs


def gen_line_by_line() -> list[dict]:
    """Generate line-by-line explanations of programs."""
    pairs = []
    for tmpl in PROGRAM_TEMPLATES:
        lines = [l for l in tmpl["code"].split("\n") if l.strip()]
        explanation_lines = []
        for i, line in enumerate(lines):
            parts = line.split()
            opcode = parts[0]
            info = INSTRUCTIONS.get(opcode, {})
            desc = info.get("desc", "Unknown instruction")
            explanation_lines.append(f"Line {i+1}: {line.strip()}")
            explanation_lines.append(f"  → {desc}")

        pairs.append({
            "instruction": f"Explain this NML program line by line:\n{tmpl['code']}",
            "input": "",
            "output": "\n".join(explanation_lines),
            "metadata": {"type": "nml_line_by_line"},
        })
    return pairs


def gen_pattern_variations(rng: random.Random) -> list[dict]:
    """Generate variations of common NML patterns with different values."""
    pairs = []

    rates = [0.01, 0.02, 0.03, 0.05, 0.062, 0.0145, 0.08, 0.10, 0.12, 0.22, 0.24, 0.32, 0.35]
    thresholds = [10000, 25000, 50000, 75000, 100000, 150000, 176100, 200000, 250000, 500000]
    labels = ["gross_pay", "income", "wages", "salary", "earnings"]

    for _ in range(500):
        rate = rng.choice(rates)
        label = rng.choice(labels)
        pairs.append({
            "instruction": f"Write NML to compute {rate*100:.1f}% of @{label} and store in RA.",
            "input": "",
            "output": f"LD    R0 @{label}\nALLC  RA #[1]\nSCLR  RC R0 #{rate:.6f}\nTACC  RA RA RC\nHALT",
            "metadata": {"type": "nml_pattern", "pattern": "flat_rate"},
        })

    for _ in range(500):
        rate = rng.choice(rates)
        cap = rng.choice(thresholds)
        label = rng.choice(labels)
        pairs.append({
            "instruction": f"Write NML to compute {rate*100:.2f}% tax on @{label} with a ${cap:,} wage base cap.",
            "input": "",
            "output": f"LD    R0 @{label}\nALLC  RA #[1]\nCMPF  RE R0 #0 #{cap:.6f}\nJMPT  #4\nLEAF  RC #{cap:.6f}\nSCLR  RC RC #{rate:.6f}\nTACC  RA RA RC\nJUMP  #2\nSCLR  RC R0 #{rate:.6f}\nTACC  RA RA RC\nST    RA @tax\nHALT",
            "metadata": {"type": "nml_pattern", "pattern": "rate_with_cap"},
        })

    for _ in range(300):
        threshold = rng.choice(thresholds)
        rate_low = rng.choice([0.03, 0.05, 0.08, 0.10])
        rate_high = rng.choice([0.12, 0.15, 0.20, 0.22])
        pairs.append({
            "instruction": f"Write NML for a two-bracket tax: {rate_low*100:.0f}% below ${threshold:,}, {rate_high*100:.0f}% above.",
            "input": "",
            "output": f"LD    R0 @income\nALLC  RA #[1]\nCMPF  RE R0 #0 #{threshold:.6f}\nJMPF  #3\nSCLR  RA R0 #{rate_low:.6f}\nJUMP  #5\nLEAF  RA #{threshold * rate_low:.6f}\nLEAF  RC #{threshold:.6f}\nMSUB  R8 R0 RC\nSCLR  R8 R8 #{rate_high:.6f}\nTACC  RA RA R8\nST    RA @tax\nHALT",
            "metadata": {"type": "nml_pattern", "pattern": "two_bracket"},
        })

    nn_configs = [
        (4, 8, "RELU"), (8, 4, "SIGM"), (3, 6, "TANH"),
        (10, 16, "RELU"), (16, 8, "RELU"), (4, 4, "SOFT"),
    ]
    for inp, hid, act in nn_configs:
        pairs.append({
            "instruction": f"Write NML for a single dense layer: {inp} inputs, {hid} outputs, {act} activation.",
            "input": "",
            "output": f"LD    R0 @input\nLD    R1 @weights\nLD    R2 @bias\nMMUL  R3 R0 R1\nMADD  R3 R3 R2\n{act}  R3 R3\nST    R3 @output\nHALT",
            "metadata": {"type": "nml_pattern", "pattern": "dense_layer"},
        })

    for n in range(2, 8):
        pairs.append({
            "instruction": f"Write NML to sum integers 1 to {n} using a backward jump loop.",
            "input": "",
            "output": f"LEAF  RA #0.0\nLEAF  RD #1.0\nTACC  RA RA RD\nLEAF  RC #1.0\nTACC  RD RD RC\nCMPI  RE RD #{n+1}.0\nJMPT  #-5\nST    RA @result\nHALT",
            "metadata": {"type": "nml_pattern", "pattern": "loop"},
        })

    for val, mult in [(3, 6), (5, 10), (7, 14), (10, 20), (100, 200)]:
        pairs.append({
            "instruction": f"Write NML to double the value {val} using a CALL/RET subroutine.",
            "input": "",
            "output": f"LEAF  R1 #{val}.0\nCALL  #2\nST    R1 @result\nJUMP  #2\nTACC  R1 R1 R1\nRET\nHALT",
            "metadata": {"type": "nml_pattern", "pattern": "subroutine"},
        })

    return pairs


def gen_error_handling() -> list[dict]:
    """Generate error handling knowledge."""
    errors = [
        ("NML_ERR_SHAPE", -1, "Tensor shape mismatch in an operation like MMUL or MADD"),
        ("NML_ERR_OOB", -2, "Out-of-bounds memory or register access"),
        ("NML_ERR_DIVZERO", -5, "Division by zero in SDIV or EDIV"),
        ("NML_ERR_OPCODE", -6, "Unknown opcode encountered"),
        ("NML_ERR_CYCLE_LIMIT", -9, "Program exceeded the maximum cycle count"),
        ("NML_ERR_TRAP", -10, "TRAP instruction was executed"),
        ("NML_ERR_CALL_DEPTH", -11, "Call stack overflow from too many nested CALLs"),
        ("NML_ERR_RET_EMPTY", -12, "RET executed with empty call stack"),
    ]
    pairs = []
    for name, code, desc in errors:
        pairs.append({
            "instruction": f"What does NML error code {code} ({name}) mean?",
            "input": "",
            "output": f"Error {code} ({name}): {desc}.",
            "metadata": {"type": "nml_error", "error_code": code},
        })
    return pairs


def gen_data_format() -> list[dict]:
    """Generate knowledge about .nml.data files."""
    return [
        {
            "instruction": "What is the format of NML data files (.nml.data)?",
            "input": "",
            "output": "NML data files use the format:\n@label shape=dim1,dim2 data=val1,val2,val3,...\n\nExample:\n@weights shape=4,3 data=0.2,-0.1,0.4,0.5,0.3,-0.2\n@bias shape=1,3 data=0.1,-0.05,0.0\n\nComments start with #.",
            "metadata": {"type": "nml_data_format"},
        },
        {
            "instruction": "Write an NML data file for an employee with $85,000 gross pay, single filing status.",
            "input": "",
            "output": "@gross_pay shape=1 data=85000.00\n@filing_status shape=1 data=1.0\n@is_exempt shape=1 data=0.0\n@is_resident shape=1 data=1.0\n@pay_periods_inv shape=1 data=0.03846154",
            "metadata": {"type": "nml_data_format"},
        },
    ]


# ═══════════════════════════════════════════
# Main
# ═══════════════════════════════════════════

def main():
    import argparse

    parser = argparse.ArgumentParser(description="NML Syntax Training Data Generator")
    parser.add_argument("--output", "-o", default="../output/training/nml_syntax.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    all_pairs = []

    print("Generating NML syntax training data...")

    ref = gen_instruction_reference()
    all_pairs.extend(ref)
    print(f"  Instruction reference: {len(ref)}")

    cat = gen_category_knowledge()
    all_pairs.extend(cat)
    print(f"  Category knowledge: {len(cat)}")

    reg = gen_register_knowledge()
    all_pairs.extend(reg)
    print(f"  Register knowledge: {len(reg)}")

    prog = gen_program_examples()
    all_pairs.extend(prog)
    print(f"  Program examples: {len(prog)}")

    lbl = gen_line_by_line()
    all_pairs.extend(lbl)
    print(f"  Line-by-line explanations: {len(lbl)}")

    pat = gen_pattern_variations(rng)
    all_pairs.extend(pat)
    print(f"  Pattern variations: {len(pat)}")

    err = gen_error_handling()
    all_pairs.extend(err)
    print(f"  Error handling: {len(err)}")

    df = gen_data_format()
    all_pairs.extend(df)
    print(f"  Data format: {len(df)}")

    rng.shuffle(all_pairs)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        for pair in all_pairs:
            f.write(json.dumps(pair) + "\n")

    print(f"\nTotal: {len(all_pairs)} NML syntax pairs -> {args.output}")


if __name__ == "__main__":
    main()
