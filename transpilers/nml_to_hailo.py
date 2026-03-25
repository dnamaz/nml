#!/usr/bin/env python3
"""
nml_to_hailo.py — Offline NML inference program → Hailo HEF compiler

Converts an NML inference program + weight data file into a Hailo Executable
Format (.hef) that can be loaded at runtime by nml_backend_hailo.cpp.

Requires:
    pip install hailo-sdk-client hailo-sdk-common onnx numpy
    (Hailo Dataflow Compiler — available from https://developer.hailo.ai)

Usage:
    python3 transpilers/nml_to_hailo.py <program.nml> <weights.nml.data> [OPTIONS]

    # Basic — produces program.hef in the same directory as program.nml
    python3 transpilers/nml_to_hailo.py programs/anomaly_detector.nml \\
            programs/anomaly_weights.nml.data

    # Specify output path explicitly
    python3 transpilers/nml_to_hailo.py programs/anomaly_detector.nml \\
            programs/anomaly_weights.nml.data --output programs/anomaly_detector.hef

    # Dry-run: print the network graph without compiling
    python3 transpilers/nml_to_hailo.py programs/anomaly_detector.nml \\
            programs/anomaly_weights.nml.data --dry-run

How it works
────────────
1. Parse the NML program to extract the feedforward graph:
     LD → MMUL → activation → MMUL → activation → ... → ST
2. Load weights from the .nml.data file.
3. Build an ONNX model with weights as initializers (baked in).
4. Run the Hailo Dataflow Compiler (hailomz / hailo_sdk_client) to produce .hef.

Supported NML ops
─────────────────
   LD   (memory load — marks data tensors and weight tensors)
   ST   (store — marks output tensor names for HEF output streams)
   MMUL (matrix multiply → ONNX MatMul)
   MADD (matrix add / bias → ONNX Add)
   RELU (→ ONNX Relu)
   SIGM (→ ONNX Sigmoid)
   TANH (→ ONNX Tanh)
   GELU (→ ONNX Gelu)
   SOFT (→ ONNX Softmax)
   NORM (→ ONNX LayerNormalization)
   CONV (4D NCHW convolution → ONNX Conv; kernel must be in .nml.data)

Programs using unsupported ops (ATTN, LOOP, CALL/RET, etc.) will print
a warning and abort — use the standard NML CPU/SYCL path for those.
"""

import argparse
import os
import re
import sys
from pathlib import Path

import numpy as np

# ── Optional imports: only required for actual compilation ───────────────────
try:
    import onnx
    from onnx import helper, numpy_helper, TensorProto
    HAVE_ONNX = True
except ImportError:
    HAVE_ONNX = False

try:
    from hailo_sdk_client import ClientRunner
    from hailo_sdk_common.targets.inference_targets import SdkNative
    HAVE_HAILO_SDK = True
except ImportError:
    HAVE_HAILO_SDK = False

# ── NML data-file parser ─────────────────────────────────────────────────────

def parse_nml_data(path: str) -> dict:
    """Parse a .nml.data file.  Returns {label: np.ndarray}."""
    tensors = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(";"):
                continue
            # @name shape=R,C dtype=f32 data=v0,v1,...
            m = re.match(
                r"@(\S+)\s+shape=([\d,]+)\s+dtype=(\S+)\s+data=(.*)", line
            )
            if not m:
                continue
            name   = m.group(1)
            shape  = [int(x) for x in m.group(2).split(",")]
            dtype  = m.group(3)
            values = [float(x) for x in m.group(4).split(",")]

            np_dtype = np.float32 if dtype in ("f32", "float32") else np.float64
            tensors[name] = np.array(values, dtype=np_dtype).reshape(shape)
    return tensors


# ── NML program parser ───────────────────────────────────────────────────────

SUPPORTED_OPS = {"LD", "ST", "MMUL", "MADD", "MSUB", "EMUL",
                 "RELU", "SIGM", "TANH", "GELU", "SOFT", "NORM", "CONV"}

# Symbolic aliases → canonical names
SYMBOLIC = {
    "↓": "LD", "↑": "ST",
    "×": "MMUL", "⊗": "MMUL",
    "⊕": "MADD",
    "⌐": "RELU",
    "σ": "SIGM",
}

# Greek register aliases
GREEK = {
    "ι":"R0","κ":"R1","λ":"R2","μ":"R3","ν":"R4",
    "ξ":"R5","ο":"R6","π":"R7","ρ":"R8","ς":"R9",
    "α":"RA","β":"RB","γ":"RC","δ":"RD","ε":"RE","ζ":"RF",
}


def normalise_token(t: str) -> str:
    t = SYMBOLIC.get(t, t).upper()
    return GREEK.get(t, t)


def parse_nml_program(path: str):
    """
    Returns a list of (opcode, [args]) tuples for supported instructions.
    Raises ValueError on unsupported ops.
    """
    instructions = []
    unsupported  = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.split(";")[0].strip()       # strip comments
            if not line:
                continue
            tokens = line.split()
            op  = normalise_token(tokens[0])
            args = [normalise_token(t) for t in tokens[1:]]
            if op in ("HALT", "◼", "NOP"):
                continue
            if op not in SUPPORTED_OPS:
                unsupported.append((lineno, op))
            instructions.append((op, args))

    if unsupported:
        msg = "\n".join(f"  line {ln}: {op}" for ln, op in unsupported)
        raise ValueError(
            f"Program contains ops not supported for Hailo compilation:\n{msg}\n"
            "Use the standard CPU/SYCL path for programs with these ops."
        )
    return instructions


# ── Graph builder ─────────────────────────────────────────────────────────────

def build_onnx_graph(instructions, tensors: dict, program_name: str):
    """
    Walk the instruction list and build an ONNX graph.

    Register state is tracked symbolically: reg_value[Rx] = tensor_name_str
    Weights are ONNX initializers (baked in).
    The data input (first LD that loads a tensor WITHOUT a weight counterpart)
    becomes the ONNX graph input.  The ST destination name becomes the output.
    """
    if not HAVE_ONNX:
        raise ImportError("onnx not installed: pip install onnx")

    reg   = {}           # Rx → ONNX value name
    nodes = []
    initializers = []
    graph_inputs  = []
    graph_outputs = []
    node_counter  = [0]

    weight_names = set(tensors.keys())  # everything in .nml.data starts as potential weight

    def fresh(prefix="t"):
        node_counter[0] += 1
        return f"{prefix}_{node_counter[0]}"

    def add_initializer(name: str, arr: np.ndarray):
        initializers.append(numpy_helper.from_array(arr, name=name))

    # First pass: find the "data" input (LD of a slot NOT in weight_names,
    # or the first LD whose value changes across calls)
    # Heuristic: if a register is loaded and later used as the LEFT argument
    # of MMUL (not the right), it's the data input.
    data_inputs = {}   # slot_name → placeholder ONNX name

    # Bake all weights as initializers up front
    for wname, arr in tensors.items():
        add_initializer(wname, arr)

    def onnx_type(arr: np.ndarray) -> int:
        return TensorProto.FLOAT if arr.dtype == np.float32 else TensorProto.DOUBLE

    def tensor_shape(arr: np.ndarray):
        return list(arr.shape)

    for op, args in instructions:
        if op == "LD":
            # LD Rx @slot
            rx, slot = args[0], args[1].lstrip("@")
            if slot in tensors:
                reg[rx] = slot           # register points to weight tensor name
            else:
                # Runtime data input — create ONNX graph input placeholder
                if slot not in data_inputs:
                    # Shape unknown at this point; use dynamic (-1) dimensions
                    # The Hailo compiler will resolve these from calibration data.
                    inp_name = slot
                    data_inputs[slot] = inp_name
                    # We add the graph input with unknown shape — Hailo DFC
                    # will infer it from the provided calibration tensors.
                    graph_inputs.append(
                        helper.make_tensor_value_info(inp_name, TensorProto.FLOAT, None)
                    )
                reg[rx] = data_inputs[slot]

        elif op == "ST":
            # ST Rx @slot
            rx, slot = args[0], args[1].lstrip("@")
            out_name = slot
            # Rename the last value in that register to the slot name
            if rx in reg and reg[rx] != out_name:
                rename_node = helper.make_node(
                    "Identity", inputs=[reg[rx]], outputs=[out_name]
                )
                nodes.append(rename_node)
            graph_outputs.append(
                helper.make_tensor_value_info(out_name, TensorProto.FLOAT, None)
            )

        elif op == "MMUL":
            # MMUL Rdest Ra Rb
            rdest, ra, rb = args[0], args[1], args[2]
            out = fresh("mmul")
            nodes.append(helper.make_node(
                "MatMul", inputs=[reg[ra], reg[rb]], outputs=[out]
            ))
            reg[rdest] = out

        elif op == "MADD":
            # MADD Rdest Ra Rb
            rdest, ra, rb = args[0], args[1], args[2]
            out = fresh("add")
            nodes.append(helper.make_node(
                "Add", inputs=[reg[ra], reg[rb]], outputs=[out]
            ))
            reg[rdest] = out

        elif op == "MSUB":
            rdest, ra, rb = args[0], args[1], args[2]
            out = fresh("sub")
            nodes.append(helper.make_node(
                "Sub", inputs=[reg[ra], reg[rb]], outputs=[out]
            ))
            reg[rdest] = out

        elif op == "EMUL":
            rdest, ra, rb = args[0], args[1], args[2]
            out = fresh("mul")
            nodes.append(helper.make_node(
                "Mul", inputs=[reg[ra], reg[rb]], outputs=[out]
            ))
            reg[rdest] = out

        elif op == "RELU":
            rdest, rsrc = args[0], args[1]
            out = fresh("relu")
            nodes.append(helper.make_node("Relu", inputs=[reg[rsrc]], outputs=[out]))
            reg[rdest] = out

        elif op == "SIGM":
            rdest, rsrc = args[0], args[1]
            out = fresh("sigmoid")
            nodes.append(helper.make_node("Sigmoid", inputs=[reg[rsrc]], outputs=[out]))
            reg[rdest] = out

        elif op == "TANH":
            rdest, rsrc = args[0], args[1]
            out = fresh("tanh")
            nodes.append(helper.make_node("Tanh", inputs=[reg[rsrc]], outputs=[out]))
            reg[rdest] = out

        elif op == "GELU":
            rdest, rsrc = args[0], args[1]
            out = fresh("gelu")
            nodes.append(helper.make_node("Gelu", inputs=[reg[rsrc]], outputs=[out],
                                          domain="com.microsoft"))
            reg[rdest] = out

        elif op == "SOFT":
            rdest, rsrc = args[0], args[1]
            out = fresh("softmax")
            nodes.append(helper.make_node("Softmax", inputs=[reg[rsrc]], outputs=[out]))
            reg[rdest] = out

        elif op == "NORM":
            rdest, rsrc = args[0], args[1]
            out = fresh("norm")
            # Simple mean normalisation (LayerNorm with unit scale/bias)
            nodes.append(helper.make_node(
                "LayerNormalization", inputs=[reg[rsrc]], outputs=[out]
            ))
            reg[rdest] = out

        elif op == "CONV":
            # CONV Rd R_input R_kernel #stride [#pad]
            # args: [output_reg, input_reg, kernel_reg, #stride, #pad?]
            output_reg  = args[0]
            input_reg   = args[1]
            kernel_reg  = args[2]
            stride = int(args[3].lstrip('#')) if len(args) > 3 else 1
            pad    = int(args[4].lstrip('#')) if len(args) > 4 else 0

            input_name  = reg.get(input_reg)
            kernel_name = reg.get(kernel_reg)

            if input_name is None:
                print(f"WARNING: CONV input register {input_reg} has no value — skipping")
                continue
            if kernel_name is None:
                print(f"WARNING: CONV kernel register {kernel_reg} has no value — skipping")
                continue

            # Verify kernel shape is available (needed to emit a valid Conv node)
            kernel_arr = tensors.get(kernel_name)
            if kernel_arr is None:
                print(f"WARNING: CONV kernel '{kernel_name}' not found in weights — skipping")
                continue

            # kernel_arr shape: [C_out, C_in, KH, KW]
            if kernel_arr.ndim != 4:
                print(f"WARNING: CONV kernel '{kernel_name}' has shape {kernel_arr.shape} "
                      f"(expected 4D [C_out, C_in, KH, KW]) — skipping")
                continue

            out = fresh("conv")
            nodes.append(helper.make_node(
                'Conv',
                inputs=[input_name, kernel_name],
                outputs=[out],
                strides=[stride, stride],
                pads=[pad, pad, pad, pad],
                group=1,
            ))
            reg[output_reg] = out

    if not graph_inputs:
        raise ValueError("No runtime data input found — all tensors are weights?\n"
                         "Ensure at least one LD loads a tensor NOT in the .nml.data file.")
    if not graph_outputs:
        raise ValueError("No ST instruction found — no output stream name available.")

    graph = helper.make_graph(nodes, program_name,
                               graph_inputs, graph_outputs, initializers)
    model = helper.make_model(graph, opset_imports=[
        helper.make_opsetid("", 13),           # standard ONNX opset 13
        helper.make_opsetid("com.microsoft", 1) # for GELU
    ])
    model.doc_string = f"Compiled from NML program: {program_name}"
    onnx.checker.check_model(model)
    return model


# ── Hailo compilation ─────────────────────────────────────────────────────────

def compile_to_hef(onnx_model, output_hef: str, hw_arch: str = "hailo8"):
    """
    Use the Hailo Dataflow Compiler SDK to compile an ONNX model to .hef.

    hw_arch options:
        "hailo8"   — Hailo-8  (AI HAT+, 26 TOPS)
        "hailo8l"  — Hailo-8L (AI Kit,  13 TOPS)
        "hailo10h" — Hailo-10H (AI HAT+ 2, if applicable)
    """
    if not HAVE_HAILO_SDK:
        raise ImportError(
            "hailo-sdk-client not installed.\n"
            "Install the Hailo Dataflow Compiler:\n"
            "  https://developer.hailo.ai/developer-zone/documentation/"
        )

    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        onnx_path = f.name
    try:
        onnx.save(onnx_model, onnx_path)

        runner = ClientRunner(hw_arch=hw_arch)

        # Parse ONNX model
        runner.translate_onnx_model(
            onnx_path,
            net_name=os.path.splitext(os.path.basename(output_hef))[0],
            start_node_names=None,
            end_node_names=None,
        )

        # Optimise (uses Hailo's model optimiser — applies quantisation, etc.)
        # For float32 input/output streams as used by nml_backend_hailo.cpp:
        runner.optimize_full_precision()  # skip quantisation for now

        # Compile to HEF
        hef_bytes = runner.compile()
        with open(output_hef, "wb") as fout:
            fout.write(hef_bytes)

        print(f"[nml_to_hailo] HEF written: {output_hef} ({len(hef_bytes)//1024} KB)")
    finally:
        os.unlink(onnx_path)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Compile an NML inference program to a Hailo HEF file."
    )
    ap.add_argument("program",  help="Path to .nml program")
    ap.add_argument("data",     help="Path to .nml.data weights file")
    ap.add_argument("--output", help="Output .hef path (default: <program>.<arch>.hef)")
    # HAILO_ARCH env var sets the default; --arch overrides it explicitly.
    _env_arch = os.environ.get("HAILO_ARCH", "hailo8")
    ap.add_argument("--arch",   default=_env_arch,
                    choices=["hailo8", "hailo8l", "hailo10h", "hailo15h"],
                    help=(
                        "Hailo chip variant "
                        "(default: $HAILO_ARCH or 'hailo8').\n"
                        "  hailo8   = AI HAT+  26 TOPS\n"
                        "  hailo8l  = AI Kit   13 TOPS\n"
                        "  hailo10h = AI HAT+ 2 / Hailo-10H\n"
                        "  hailo15h = Hailo-15H\n"
                        "Output file: <program>.<arch>.hef"
                    ))
    ap.add_argument("--onnx-only", action="store_true",
                    help="Write <program>.onnx and stop (skip Hailo DFC step)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Parse and print graph structure, do not compile")
    args = ap.parse_args()

    program_path = args.program
    data_path    = args.data

    arch = args.arch  # e.g. "hailo8", "hailo8l", "hailo10h"

    if args.output:
        hef_path = args.output
    else:
        # Chip-specific name: anomaly_detector.hailo8.hef
        # Matches the runtime lookup order in nml.c / nml_backend_hailo.cpp.
        stem     = re.sub(r"\.[^.]+$", "", program_path)
        hef_path = f"{stem}.{arch}.hef"

    onnx_path = re.sub(r"\.[^.]+$", "", program_path) + ".onnx"
    program_name = Path(program_path).stem

    print(f"[nml_to_hailo] Parsing {program_path}")
    try:
        instructions = parse_nml_program(program_path)
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"[nml_to_hailo] Loading weights from {data_path}")
    tensors = parse_nml_data(data_path)
    print(f"[nml_to_hailo] Loaded {len(tensors)} weight tensor(s): "
          f"{', '.join(tensors.keys())}")

    if args.dry_run:
        print("\nInstruction graph:")
        for op, a in instructions:
            print(f"  {op:6s} {' '.join(a)}")
        return

    if not HAVE_ONNX:
        print("ERROR: onnx not installed. Run: pip install onnx numpy",
              file=sys.stderr)
        sys.exit(1)

    print(f"[nml_to_hailo] Building ONNX graph …")
    try:
        model = build_onnx_graph(instructions, tensors, program_name)
    except (ValueError, Exception) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    if args.onnx_only:
        onnx.save(model, onnx_path)
        print(f"[nml_to_hailo] ONNX model written: {onnx_path}")
        return

    print(f"[nml_to_hailo] Compiling to HEF (arch={arch}) …")
    try:
        compile_to_hef(model, hef_path, hw_arch=arch)
    except ImportError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        print("\nFallback: writing ONNX only (use hailo_sdk_client separately)",
              file=sys.stderr)
        onnx.save(model, onnx_path)
        print(f"[nml_to_hailo] ONNX written: {onnx_path}")
        sys.exit(1)

    print(f"\n[nml_to_hailo] Done.")
    print(f"  Arch: {arch}")
    print(f"  HEF:  {hef_path}")
    print(f"  Run:  ./nml-rpi-hailo {program_path} {data_path}")
    print(f"  Or:   HAILO_ARCH={arch} ./nml-rpi-hailo {program_path} {data_path}")


if __name__ == "__main__":
    main()
