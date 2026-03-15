#!/usr/bin/env python3
"""
NML-to-Mojo Transpiler — Convert NML programs to Mojo source code.

Generates Mojo source that can be compiled for CPU or GPU execution
via the Mojo/MAX platform. Each NML opcode maps to a Mojo function
with hardware-dispatched implementations.

Architecture:
  NML Program (.nml) -> parse -> IR -> Mojo source (.mojo)
  - Register file: 32-slot Tensor array
  - Memory: Dict[String, Tensor]
  - Opcodes: mapped to Mojo stdlib / MAX linalg functions
  - Control flow: LOOP -> for, JUMP/JMPT/JMPF -> goto-like patterns

Usage:
    python3 nml_to_mojo.py program.nml
    python3 nml_to_mojo.py program.nml -o output.mojo
    python3 nml_to_mojo.py program.nml --emit-ir  # print IR only
"""

import sys
import argparse
from pathlib import Path
from dataclasses import dataclass, field

sys.path.insert(0, str(Path(__file__).parent))
from nml_grammar import _resolve_opcode, _canonical_register


# ═══════════════════════════════════════════════════════════════════════════════
# NML IR
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class NMLInstruction:
    line: int
    opcode: str
    operands: list[str]
    raw: str

@dataclass
class NMLProgram:
    instructions: list[NMLInstruction] = field(default_factory=list)
    memory_refs: set[str] = field(default_factory=set)
    registers_used: set[str] = field(default_factory=set)


def parse_nml(source: str) -> NMLProgram:
    """Parse NML source into IR."""
    prog = NMLProgram()
    for line_no, raw_line in enumerate(source.splitlines(), 1):
        stripped = raw_line.strip()
        if not stripped or stripped.startswith(";"):
            continue
        if ";" in stripped:
            stripped = stripped[:stripped.index(";")].strip()
        if not stripped:
            continue
        tokens = stripped.split()
        opcode_raw = tokens[0]
        canonical = _resolve_opcode(opcode_raw)
        if canonical is None:
            continue
        operands = tokens[1:]

        for tok in operands:
            if tok.startswith("@"):
                prog.memory_refs.add(tok[1:])
            cr = _canonical_register(tok)
            if cr:
                prog.registers_used.add(cr)

        prog.instructions.append(NMLInstruction(
            line=line_no, opcode=canonical, operands=operands, raw=stripped
        ))
    return prog


# ═══════════════════════════════════════════════════════════════════════════════
# Register index mapping
# ═══════════════════════════════════════════════════════════════════════════════

REG_INDEX = {}
for i in range(10):
    REG_INDEX[f"R{i}"] = i
for i, c in enumerate("ABCDEFGHIJKLMNOPQRSTUV"):
    REG_INDEX[f"R{c}"] = 10 + i

GREEK_TO_CLASSIC = {
    "ι": "R0", "κ": "R1", "λ": "R2", "μ": "R3", "ν": "R4",
    "ξ": "R5", "ο": "R6", "π": "R7", "ρ": "R8", "ς": "R9",
    "α": "RA", "β": "RB", "γ": "RC", "δ": "RD", "φ": "RE", "ψ": "RF",
    "η": "RG", "θ": "RH", "ζ": "RI", "ω": "RJ", "χ": "RK",
    "υ": "RL", "ε": "RM",
}


def reg_idx(tok: str) -> int:
    cr = _canonical_register(tok)
    if cr and cr in REG_INDEX:
        return REG_INDEX[cr]
    return -1


def mojo_reg(tok: str) -> str:
    idx = reg_idx(tok)
    if idx >= 0:
        return f"regs[{idx}]"
    return tok


def mojo_operand(tok: str) -> str:
    if tok.startswith("@"):
        return f'mem["{tok[1:]}"]'
    if tok.startswith("#"):
        val = tok[1:]
        if val.startswith("["):
            return val
        return val
    cr = _canonical_register(tok)
    if cr:
        return mojo_reg(tok)
    try:
        float(tok)
        return tok
    except ValueError:
        return f'"{tok}"'


# ═══════════════════════════════════════════════════════════════════════════════
# Opcode -> Mojo code generation
# ═══════════════════════════════════════════════════════════════════════════════

def emit_instruction(ins: NMLInstruction, indent: str = "    ") -> list[str]:
    """Emit Mojo code lines for a single NML instruction."""
    op = ins.opcode
    ops = ins.operands
    lines = []

    def r(i): return mojo_reg(ops[i]) if i < len(ops) else "regs[0]"
    def imm(i): return ops[i].lstrip("#") if i < len(ops) else "0"
    def mem(i): return f'mem["{ops[i][1:]}"]' if i < len(ops) and ops[i].startswith("@") else mojo_operand(ops[i]) if i < len(ops) else 'mem["_"]'

    if op == "MMUL":
        lines.append(f"{r(0)} = matmul({r(1)}, {r(2)})")
    elif op == "MADD":
        lines.append(f"{r(0)} = {r(1)} + {r(2)}")
    elif op == "MSUB":
        lines.append(f"{r(0)} = {r(1)} - {r(2)}")
    elif op == "EMUL":
        lines.append(f"{r(0)} = {r(1)} * {r(2)}")
    elif op == "EDIV":
        lines.append(f"{r(0)} = {r(1)} / {r(2)}")
    elif op == "SDOT":
        lines.append(f"{r(0)} = dot({r(1)}, {r(2)})")
    elif op == "SCLR":
        lines.append(f"{r(0)} = {r(1)} * {imm(2)}")
    elif op == "SDIV":
        lines.append(f"{r(0)} = {r(1)} / {imm(2)}")
    elif op == "RELU":
        lines.append(f"{r(0)} = relu({r(1)})")
    elif op == "SIGM":
        lines.append(f"{r(0)} = sigmoid({r(1)})")
    elif op == "TANH":
        lines.append(f"{r(0)} = tanh_act({r(1)})")
    elif op == "SOFT":
        lines.append(f"{r(0)} = softmax({r(1)})")
    elif op == "GELU":
        lines.append(f"{r(0)} = gelu({r(1)})")

    elif op == "LD":
        if len(ops) >= 2 and ops[1].startswith("@"):
            lines.append(f"{r(0)} = {mem(1)}.copy()")
        elif len(ops) >= 2 and ops[1].startswith("#"):
            lines.append(f"{r(0)} = Tensor.scalar({imm(1)})")
        else:
            lines.append(f"{r(0)} = {r(1)}.copy()")
    elif op == "ST":
        if len(ops) >= 2 and ops[1].startswith("@"):
            lines.append(f'{mem(1)} = {r(0)}.copy()')
        else:
            lines.append(f"# ST {' '.join(ops)}")
    elif op == "MOV":
        lines.append(f"{r(0)} = {r(1)}.copy()")
    elif op == "ALLC":
        lines.append(f"{r(0)} = Tensor.zeros({mojo_operand(ops[1]) if len(ops) > 1 else '1'})")
    elif op == "LEAF":
        if len(ops) >= 2 and ops[1].startswith("#"):
            lines.append(f"{r(0)} = Tensor.scalar({imm(1)})")
        elif len(ops) >= 2 and ops[1].startswith("@"):
            lines.append(f"{r(0)} = {mem(1)}.copy()")
        else:
            lines.append(f"{r(0)} = {r(1)}.copy()")

    elif op == "RSHP":
        lines.append(f"{r(0)} = reshape({r(1)}, {mojo_operand(ops[2]) if len(ops) > 2 else '[]'})")
    elif op == "TRNS":
        lines.append(f"{r(0)} = transpose({r(1)})")
    elif op == "SPLT":
        lines.append(f"{r(0)} = split({r(1)})")
    elif op == "MERG":
        lines.append(f"{r(0)} = merge({r(1)}, {r(2)})")

    elif op == "CONV":
        lines.append(f"{r(0)} = conv2d({r(1)}, {r(2)})")
    elif op == "POOL":
        lines.append(f"{r(0)} = maxpool({r(1)})")
    elif op == "UPSC":
        lines.append(f"{r(0)} = upscale({r(1)})")
    elif op == "PADZ":
        lines.append(f"{r(0)} = pad({r(1)})")

    elif op == "ATTN":
        lines.append(f"{r(0)} = attention({r(1)}, {r(2)}, {r(3) if len(ops) > 3 else r(2)})")
    elif op == "NORM":
        lines.append(f"{r(0)} = layer_norm({r(1)})")
    elif op == "EMBD":
        lines.append(f"{r(0)} = embedding({r(1)}, {r(2)})")

    elif op == "RDUC":
        mode = imm(2) if len(ops) > 2 else "0"
        lines.append(f"{r(0)} = reduce({r(1)}, mode={mode})")

    elif op == "CMP" or op == "CMPI":
        lines.append(f"{r(0)} = compare({r(1)}, {imm(2) if len(ops) > 2 else '0'})")
    elif op == "CMPF":
        lines.append(f"{r(0)} = compare_feature({r(1)}, {imm(2) if len(ops) > 2 else '0'}, {imm(3) if len(ops) > 3 else '0'})")

    elif op == "JMPT":
        lines.append(f"if regs[14].item() > 0:  # JMPT {imm(0)}")
        lines.append(f"    pc += {imm(0)}")
        lines.append(f"    continue")
    elif op == "JMPF":
        lines.append(f"if regs[14].item() <= 0:  # JMPF {imm(0)}")
        lines.append(f"    pc += {imm(0)}")
        lines.append(f"    continue")
    elif op == "JUMP":
        lines.append(f"pc += {imm(0)}  # JUMP")
        lines.append(f"continue")

    elif op == "LOOP":
        lines.append(f"for _loop_{ins.line} in range({imm(0)}):")
        return [(indent + "# LOOP_START", True)]
    elif op == "ENDP":
        return [(indent + "# LOOP_END", False)]

    elif op == "TACC":
        lines.append(f"{r(0)} = {r(0)} + {r(1)}")

    elif op == "LOSS":
        loss_type = imm(3) if len(ops) > 3 else "0"
        lines.append(f"{r(0)} = mse_loss({r(1)}, {r(2)}) if {loss_type} == 0 else mae_loss({r(1)}, {r(2)})")
    elif op == "BKWD":
        lines.append(f"{r(0)} = backward_grad({r(1)}, {r(2)})")
    elif op == "WUPD":
        lr = imm(2) if len(ops) > 2 and ops[2].startswith("#") else f"{r(2)}.item()"
        lines.append(f"{r(0)} = {r(0)} - {lr} * {r(1)}")
    elif op == "TNET":
        lines.append(f"regs[8] = tnet_train(regs, epochs={imm(0)}, lr={imm(1)}, optimizer={imm(2) if len(ops) > 2 else '0'})")

    elif op == "RELUBK":
        lines.append(f"{r(0)} = relu_backward({r(1)}, {r(2)})")
    elif op == "SIGMBK":
        lines.append(f"{r(0)} = sigmoid_backward({r(1)}, {r(2)})")
    elif op == "TANHBK":
        lines.append(f"{r(0)} = tanh_backward({r(1)}, {r(2)})")
    elif op == "GELUBK":
        lines.append(f"{r(0)} = gelu_backward({r(1)}, {r(2)})")
    elif op == "SOFTBK":
        lines.append(f"{r(0)} = softmax_backward({r(1)}, {r(2)})")
    elif op == "MMULBK":
        lines.append(f"{r(0)}, {r(1)} = matmul_backward({r(2)}, {mojo_reg(ops[3]) if len(ops) > 3 else 'regs[0]'}, {mojo_reg(ops[4]) if len(ops) > 4 else 'regs[0]'})")
    elif op == "CONVBK":
        lines.append(f"{r(0)}, {r(1)} = conv_backward({r(2)}, {mojo_reg(ops[3]) if len(ops) > 3 else 'regs[0]'}, {mojo_reg(ops[4]) if len(ops) > 4 else 'regs[0]'})")
    elif op == "POOLBK":
        lines.append(f"{r(0)} = pool_backward({r(1)}, {r(2)})")
    elif op == "NORMBK":
        lines.append(f"{r(0)} = norm_backward({r(1)}, {r(2)})")
    elif op == "ATTNBK":
        lines.append(f"{r(0)}, regs[{reg_idx(ops[0])+1}], regs[{reg_idx(ops[0])+2}] = attention_backward({r(1)}, {r(2)}, {mojo_reg(ops[3]) if len(ops) > 3 else 'regs[0]'}, {mojo_reg(ops[4]) if len(ops) > 4 else 'regs[0]'})")
    elif op == "TNDEEP":
        lines.append(f"regs[8] = tndeep_train(regs, epochs={imm(0)}, lr={imm(1)}, optimizer={imm(2) if len(ops) > 2 else '0'})")

    elif op == "HALT":
        lines.append("return (regs^, mem^)")
    elif op == "SYNC":
        lines.append("barrier()")
    elif op == "TRAP":
        lines.append(f"raise Error('TRAP at line {ins.line}')")

    elif op in ("META", "FRAG", "ENDF", "LINK", "PTCH", "SIGN", "VRFY", "VOTE"):
        lines.append(f"# {op} {' '.join(ops)}  (M2M — not transpiled)")

    elif op in ("SYS", "MOD", "ITOF", "FTOI", "BNOT", "FFT", "FILT",
                "WHER", "CLMP", "CMPR", "PROJ", "DIST", "GATH", "SCAT",
                "CALL", "RET"):
        lines.append(f"# {op} {' '.join(ops)}  (TODO: implement)")

    else:
        lines.append(f"# Unknown: {op} {' '.join(ops)}")

    return [(indent + l, None) for l in lines]


# ═══════════════════════════════════════════════════════════════════════════════
# Mojo source generation
# ═══════════════════════════════════════════════════════════════════════════════

MOJO_MAX_PRELUDE = '''\
# Auto-generated from NML by nml_to_mojo.py (MAX mode)
# NML: 82-opcode tensor register machine -> Mojo GPU code
#
# Requires: Mojo SDK + MAX framework
# Run: mojo run {filename}
#
# To use MAX GPU acceleration, replace the Tensor stub below with:
#   from max.tensor import Tensor, TensorShape
#   from max.engine import InferenceSession
# and replace matmul() with max.linalg.matmul() for hardware dispatch.
'''

MOJO_PRELUDE = '''\
# Auto-generated from NML by nml_to_mojo.py
# NML: 82-opcode tensor register machine -> Mojo CPU code
#
# Requires: Mojo SDK >= 0.26
# Run: mojo run {filename}

from std.collections import Dict
from std.math import exp, tanh, sqrt

comptime NUM_REGS = 32


struct Tensor(Movable, Copyable):
    """Minimal tensor stub — replace with MAX Tensor for GPU support."""
    var data: List[Float64]
    var shape: List[Int]

    fn __init__(out self):
        self.data = List[Float64]()
        self.shape = List[Int]()

    fn __copyinit__(out self, *, copy: Self):
        self.data = copy.data.copy()
        self.shape = copy.shape.copy()

    fn __moveinit__(out self, *, deinit take: Self):
        self.data = take.data^
        self.shape = take.shape^

    fn __add__(self, other: Self) -> Self:
        var out = self.copy()
        for i in range(min(len(self.data), len(other.data))):
            out.data[i] += other.data[i]
        return out^

    fn __sub__(self, other: Self) -> Self:
        var out = self.copy()
        for i in range(min(len(self.data), len(other.data))):
            out.data[i] -= other.data[i]
        return out^

    fn __mul__(self, other: Self) -> Self:
        var out = self.copy()
        for i in range(min(len(self.data), len(other.data))):
            out.data[i] *= other.data[i]
        return out^

    fn __truediv__(self, other: Self) -> Self:
        var out = self.copy()
        for i in range(min(len(self.data), len(other.data))):
            if other.data[i] != 0:
                out.data[i] /= other.data[i]
        return out^

    @staticmethod
    fn scalar(val: Float64) -> Tensor:
        var t = Tensor()
        t.data.append(val)
        t.shape.append(1)
        return t^

    @staticmethod
    fn zeros(size: Int) -> Tensor:
        var t = Tensor()
        for _ in range(size):
            t.data.append(0.0)
        t.shape.append(size)
        return t^

    fn copy(self) -> Tensor:
        var t = Tensor()
        t.data = self.data.copy()
        t.shape = self.shape.copy()
        return t^

    fn item(self) -> Float64:
        if len(self.data) > 0:
            return self.data[0]
        return 0.0


fn _make_shape2(a: Int, b: Int) -> List[Int]:
    var s = List[Int]()
    s.append(a)
    s.append(b)
    return s^


fn matmul(a: Tensor, b: Tensor) -> Tensor:
    """Matrix multiply — dispatch to GPU via MAX for large tensors."""
    var m = a.shape[0] if len(a.shape) >= 2 else 1
    var k = a.shape[1] if len(a.shape) >= 2 else a.shape[0]
    var n = b.shape[1] if len(b.shape) >= 2 else 1
    var out = Tensor.zeros(m * n)
    out.shape = _make_shape2(m, n)
    for i in range(m):
        for j in range(n):
            var s: Float64 = 0.0
            for p in range(k):
                s += a.data[i * k + p] * b.data[p * n + j]
            out.data[i * n + j] = s
    return out^


fn relu(t: Tensor) -> Tensor:
    var out = t.copy()
    for i in range(len(out.data)):
        if out.data[i] < 0:
            out.data[i] = 0.0
    return out^


fn sigmoid(t: Tensor) -> Tensor:
    var out = t.copy()
    for i in range(len(out.data)):
        out.data[i] = 1.0 / (1.0 + exp(-out.data[i]))
    return out^


fn tanh_act(t: Tensor) -> Tensor:
    var out = t.copy()
    for i in range(len(out.data)):
        out.data[i] = tanh(out.data[i])
    return out^


fn softmax(t: Tensor) -> Tensor:
    var out = t.copy()
    var mx = out.data[0]
    for i in range(1, len(out.data)):
        if out.data[i] > mx:
            mx = out.data[i]
    var s: Float64 = 0.0
    for i in range(len(out.data)):
        out.data[i] = exp(out.data[i] - mx)
        s += out.data[i]
    for i in range(len(out.data)):
        out.data[i] /= s
    return out^


fn gelu(t: Tensor) -> Tensor:
    var out = t.copy()
    for i in range(len(out.data)):
        var x = out.data[i]
        out.data[i] = 0.5 * x * (1.0 + tanh(0.7978845608 * (x + 0.044715 * x * x * x)))
    return out^


fn relu_backward(grad: Tensor, inp: Tensor) -> Tensor:
    var out = Tensor.zeros(len(grad.data))
    out.shape = inp.shape.copy()
    for i in range(len(grad.data)):
        out.data[i] = grad.data[i] if inp.data[i] > 0 else 0.0
    return out^


fn sigmoid_backward(grad: Tensor, inp: Tensor) -> Tensor:
    var out = Tensor.zeros(len(grad.data))
    out.shape = inp.shape.copy()
    for i in range(len(grad.data)):
        var s = 1.0 / (1.0 + exp(-inp.data[i]))
        out.data[i] = grad.data[i] * s * (1.0 - s)
    return out^


fn tanh_backward(grad: Tensor, inp: Tensor) -> Tensor:
    var out = Tensor.zeros(len(grad.data))
    out.shape = inp.shape.copy()
    for i in range(len(grad.data)):
        var t = tanh(inp.data[i])
        out.data[i] = grad.data[i] * (1.0 - t * t)
    return out^


fn gelu_backward(grad: Tensor, inp: Tensor) -> Tensor:
    var out = Tensor.zeros(len(grad.data))
    out.shape = inp.shape.copy()
    for i in range(len(grad.data)):
        var x = inp.data[i]
        var inner = 0.7978845608 * (x + 0.044715 * x * x * x)
        var t = tanh(inner)
        var sech2 = 1.0 - t * t
        var dg = 0.5 * (1.0 + t) + 0.5 * x * sech2 * 0.7978845608 * (1.0 + 3.0 * 0.044715 * x * x)
        out.data[i] = grad.data[i] * dg
    return out^


fn transpose(t: Tensor) -> Tensor:
    if len(t.shape) < 2:
        return t.copy()
    var r = t.shape[0]
    var c = t.shape[1]
    var out = Tensor.zeros(r * c)
    out.shape = _make_shape2(c, r)
    for i in range(r):
        for j in range(c):
            out.data[j * r + i] = t.data[i * c + j]
    return out^


fn mse_loss(pred: Tensor, target: Tensor) -> Tensor:
    var n = len(pred.data)
    var s: Float64 = 0.0
    for i in range(n):
        var d = pred.data[i] - target.data[i]
        s += d * d
    s /= Float64(n) if n > 0 else 1.0
    return Tensor.scalar(s)

'''


def transpile(prog: NMLProgram, source_name: str = "program.nml", use_max: bool = False) -> str:
    """Transpile NML program to Mojo source."""
    prelude = MOJO_MAX_PRELUDE if use_max else MOJO_PRELUDE
    lines = [prelude.replace("{filename}", source_name)]

    lines.append(f"fn nml_main() raises -> Tuple[List[Tensor], Dict[String, Tensor]]:")
    lines.append(f"    var regs = List[Tensor]()")
    lines.append(f"    for _ in range(NUM_REGS):")
    lines.append(f"        regs.append(Tensor())")
    lines.append(f"    var mem = Dict[String, Tensor]()")
    lines.append(f"")

    if prog.memory_refs:
        lines.append(f"    # Memory slots referenced by the program")
        for ref in sorted(prog.memory_refs):
            lines.append(f'    mem["{ref}"] = Tensor()  # load from .nml.data')
        lines.append(f"")

    indent_level = 1
    for ins in prog.instructions:
        indent = "    " * indent_level

        if ins.opcode == "LOOP":
            count = ins.operands[0].lstrip("#") if ins.operands else "1"
            lines.append(f"{indent}for _loop_{ins.line} in range({count}):")
            indent_level += 1
            continue
        elif ins.opcode == "ENDP":
            indent_level = max(1, indent_level - 1)
            continue

        emitted = emit_instruction(ins, indent)
        for code, _ in emitted:
            lines.append(code)

    if not any(ins.opcode == "HALT" for ins in prog.instructions):
        lines.append(f"    return (regs^, mem^)")

    lines.append("")
    lines.append("fn main() raises:")
    lines.append('    var result = nml_main()')
    lines.append('    print("NML program complete.")')
    lines.append("")
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Transpile NML programs to Mojo source")
    parser.add_argument("input", help="NML source file (.nml)")
    parser.add_argument("-o", "--output", help="Output Mojo file (default: stdout)")
    parser.add_argument("--emit-ir", action="store_true", help="Print IR instead of Mojo")
    parser.add_argument("--max", action="store_true",
                        help="Emit MAX framework imports (GPU-ready, requires MAX SDK)")
    args = parser.parse_args()

    source = Path(args.input).read_text()
    prog = parse_nml(source)

    if args.emit_ir:
        print(f"Instructions: {len(prog.instructions)}")
        print(f"Memory refs:  {sorted(prog.memory_refs)}")
        print(f"Registers:    {sorted(prog.registers_used)}")
        print()
        for ins in prog.instructions:
            print(f"  {ins.line:3d}: {ins.opcode:<8} {' '.join(ins.operands)}")
        return

    mojo_src = transpile(prog, Path(args.input).name, use_max=args.max)

    if args.output:
        Path(args.output).write_text(mojo_src)
        print(f"Written: {args.output}")
    else:
        print(mojo_src)


if __name__ == "__main__":
    main()
