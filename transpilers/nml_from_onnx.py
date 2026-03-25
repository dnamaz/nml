#!/usr/bin/env python3
"""
nml_from_onnx.py — ONNX model → NML program + weight data file

Converts a pre-trained ONNX model into an NML program that can be executed
by the NML runtime (nml, nml-cuda, nml-sycl, nml-rpi-hailo).

Requires: pip install onnx numpy

Usage:
    python3 transpilers/nml_from_onnx.py model.onnx
    python3 transpilers/nml_from_onnx.py model.onnx --output-nml out.nml --output-data out.nml.data
    python3 transpilers/nml_from_onnx.py model.onnx --dry-run
"""

import argparse
import sys
import os
import struct
from pathlib import Path
from collections import OrderedDict

try:
    import onnx
    import onnx.numpy_helper as onph
    import numpy as np
except ImportError:
    print("ERROR: pip install onnx numpy", file=sys.stderr)
    sys.exit(1)


# Register allocation: we have R0-RV (32 registers)
# R0 = model input (always)
# R1-R28 = intermediate activations (allocated in order of first use)
# R29-R31 (RT, RU, RV) = reserved for scratch
# Named memory slots (@name) = weight tensors (initializers)
NML_REGS = ['R0','R1','R2','R3','R4','R5','R6','R7','R8','R9',
             'RA','RB','RC','RD','RE','RF','RG','RH','RI','RJ',
             'RK','RL','RM','RN','RO','RP','RQ','RR','RS','RT','RU','RV']

UNSUPPORTED_OPS = set()

class ONNXToNML:
    def __init__(self, model_path, verbose=False):
        self.model = onnx.load(model_path)
        onnx.checker.check_model(self.model)
        self.graph = self.model.graph
        self.verbose = verbose

        # Map: ONNX tensor name -> NML register name or @memory_slot
        self.tensor_map = {}  # name -> 'R0' or '@weight_name'
        self.reg_alloc = 1    # next available register (R0 = input)
        self.instructions = []
        self.data_tensors = OrderedDict()  # @name -> numpy array
        self.warnings = []

    def _alloc_reg(self):
        if self.reg_alloc >= 29:
            raise RuntimeError("Register overflow: model too deep (>28 intermediate tensors). "
                               "Consider splitting the model.")
        reg = NML_REGS[self.reg_alloc]
        self.reg_alloc += 1
        return reg

    def _safe_name(self, name):
        """Convert ONNX tensor name to a valid NML memory slot label."""
        return name.replace('/', '_').replace('.', '_').replace(':', '_').replace('-', '_')[:32]

    def _get_initializer_map(self):
        """Map initializer name -> numpy array."""
        return {init.name: onph.to_array(init) for init in self.graph.initializer}

    def convert(self):
        init_map = self._get_initializer_map()

        # Register initializers as named memory slots
        for name, arr in init_map.items():
            slot = '@' + self._safe_name(name)
            self.tensor_map[name] = slot
            self.data_tensors[self._safe_name(name)] = arr

        # Model input -> R0
        for inp in self.graph.input:
            if inp.name not in init_map:
                self.tensor_map[inp.name] = 'R0'
                break  # assume single primary input

        # Convert each node
        for node in self.graph.node:
            self._convert_node(node, init_map)

        # Model output -> store to @output
        for out in self.graph.output:
            if out.name in self.tensor_map:
                src = self.tensor_map[out.name]
                if src.startswith('R'):
                    self.instructions.append(f'ST    {src} @output')

        self.instructions.append('HALT')
        return self

    def _get_or_load(self, name):
        """Get register for a tensor, loading from memory if it's a weight."""
        if name not in self.tensor_map:
            raise KeyError(f"Unknown tensor: {name!r}")
        ref = self.tensor_map[name]
        if ref.startswith('@'):
            # Weight tensor: need to LD into a register
            reg = self._alloc_reg()
            self.instructions.append(f'LD    {reg} {ref}')
            # Update map so subsequent uses of this weight reuse the same register
            self.tensor_map[name] = reg
            return reg
        return ref

    def _convert_node(self, node, init_map):
        op = node.op_type
        inputs = list(node.input)
        outputs = list(node.output)
        attrs = {a.name: a for a in node.attribute}

        if self.verbose:
            print(f"  {op}: {inputs} -> {outputs}")

        handler = getattr(self, f'_op_{op.lower()}', None)
        if handler:
            handler(inputs, outputs, attrs, init_map)
        else:
            self.warnings.append(f"Unsupported op: {op} — skipped")
            UNSUPPORTED_OPS.add(op)
            # Map output to R0 as placeholder
            for o in outputs:
                if o:
                    self.tensor_map[o] = 'R0'

    def _op_matmul(self, inputs, outputs, attrs, init_map):
        a = self._get_or_load(inputs[0])
        b = self._get_or_load(inputs[1])
        out = self._alloc_reg()
        self.instructions.append(f'MMUL  {out} {a} {b}')
        self.tensor_map[outputs[0]] = out

    def _op_gemm(self, inputs, outputs, attrs, init_map):
        # Gemm: Y = alpha * A * B^T + beta * C (typically alpha=1, beta=1, transB=1)
        a = self._get_or_load(inputs[0])
        b = self._get_or_load(inputs[1])
        out = self._alloc_reg()
        trans_b = attrs.get('transB')
        if trans_b and trans_b.i == 1:
            # Need transpose: TRNS then MMUL
            tmp = self._alloc_reg()
            self.instructions.append(f'TRNS  {tmp} {b}')
            self.instructions.append(f'MMUL  {out} {a} {tmp}')
        else:
            self.instructions.append(f'MMUL  {out} {a} {b}')
        if len(inputs) > 2 and inputs[2]:
            bias = self._get_or_load(inputs[2])
            out2 = self._alloc_reg()
            self.instructions.append(f'MADD  {out2} {out} {bias}')
            self.tensor_map[outputs[0]] = out2
        else:
            self.tensor_map[outputs[0]] = out

    def _op_add(self, inputs, outputs, attrs, init_map):
        a = self._get_or_load(inputs[0])
        b = self._get_or_load(inputs[1])
        out = self._alloc_reg()
        self.instructions.append(f'MADD  {out} {a} {b}')
        self.tensor_map[outputs[0]] = out

    def _op_sub(self, inputs, outputs, attrs, init_map):
        a = self._get_or_load(inputs[0])
        b = self._get_or_load(inputs[1])
        out = self._alloc_reg()
        self.instructions.append(f'MSUB  {out} {a} {b}')
        self.tensor_map[outputs[0]] = out

    def _op_mul(self, inputs, outputs, attrs, init_map):
        a = self._get_or_load(inputs[0])
        b = self._get_or_load(inputs[1])
        out = self._alloc_reg()
        self.instructions.append(f'EMUL  {out} {a} {b}')
        self.tensor_map[outputs[0]] = out

    def _op_div(self, inputs, outputs, attrs, init_map):
        a = self._get_or_load(inputs[0])
        b = self._get_or_load(inputs[1])
        out = self._alloc_reg()
        self.instructions.append(f'EDIV  {out} {a} {b}')
        self.tensor_map[outputs[0]] = out

    def _op_relu(self, inputs, outputs, attrs, init_map):
        inp = self._get_or_load(inputs[0])
        out = self._alloc_reg()
        self.instructions.append(f'RELU  {out} {inp}')
        self.tensor_map[outputs[0]] = out

    def _op_sigmoid(self, inputs, outputs, attrs, init_map):
        inp = self._get_or_load(inputs[0])
        out = self._alloc_reg()
        self.instructions.append(f'SIGM  {out} {inp}')
        self.tensor_map[outputs[0]] = out

    def _op_tanh(self, inputs, outputs, attrs, init_map):
        inp = self._get_or_load(inputs[0])
        out = self._alloc_reg()
        self.instructions.append(f'TANH  {out} {inp}')
        self.tensor_map[outputs[0]] = out

    def _op_gelu(self, inputs, outputs, attrs, init_map):
        inp = self._get_or_load(inputs[0])
        out = self._alloc_reg()
        self.instructions.append(f'GELU  {out} {inp}')
        self.tensor_map[outputs[0]] = out

    def _op_softmax(self, inputs, outputs, attrs, init_map):
        inp = self._get_or_load(inputs[0])
        out = self._alloc_reg()
        self.instructions.append(f'SOFT  {out} {inp}')
        self.tensor_map[outputs[0]] = out

    def _op_layernormalization(self, inputs, outputs, attrs, init_map):
        inp = self._get_or_load(inputs[0])
        scale = self._get_or_load(inputs[1]) if len(inputs) > 1 and inputs[1] else None
        bias_t = self._get_or_load(inputs[2]) if len(inputs) > 2 and inputs[2] else None
        out = self._alloc_reg()
        if scale and bias_t:
            self.instructions.append(f'NORM  {out} {inp} {scale} {bias_t}')
        elif scale:
            self.instructions.append(f'NORM  {out} {inp} {scale}')
        else:
            self.instructions.append(f'NORM  {out} {inp}')
        self.tensor_map[outputs[0]] = out

    def _op_batchnormalization(self, inputs, outputs, attrs, init_map):
        inp = self._get_or_load(inputs[0])
        scale = self._get_or_load(inputs[1]) if len(inputs) > 1 and inputs[1] else None
        bias_t = self._get_or_load(inputs[2]) if len(inputs) > 2 and inputs[2] else None
        out = self._alloc_reg()
        if scale and bias_t:
            self.instructions.append(f'BN    {out} {inp} {scale} {bias_t}')
        else:
            self.instructions.append(f'BN    {out} {inp}')
        self.tensor_map[outputs[0]] = out

    def _op_conv(self, inputs, outputs, attrs, init_map):
        inp = self._get_or_load(inputs[0])
        weight = self._get_or_load(inputs[1])
        # Extract stride and pad from attrs
        strides = [1, 1]
        pads = [0, 0, 0, 0]
        if 'strides' in attrs:
            strides = list(attrs['strides'].ints)
        if 'pads' in attrs:
            pads = list(attrs['pads'].ints)
        stride = strides[0]
        pad = pads[0]  # NML CONV takes single pad value (symmetric)
        out = self._alloc_reg()
        self.instructions.append(f'CONV  {out} {inp} {weight} #{stride} #{pad}')
        # Handle bias if present
        if len(inputs) > 2 and inputs[2]:
            bias = self._get_or_load(inputs[2])
            out2 = self._alloc_reg()
            self.instructions.append(f'MADD  {out2} {out} {bias}')
            self.tensor_map[outputs[0]] = out2
        else:
            self.tensor_map[outputs[0]] = out

    def _op_maxpool(self, inputs, outputs, attrs, init_map):
        inp = self._get_or_load(inputs[0])
        kernel_shape = list(attrs['kernel_shape'].ints) if 'kernel_shape' in attrs else [2, 2]
        strides = list(attrs['strides'].ints) if 'strides' in attrs else [2, 2]
        out = self._alloc_reg()
        self.instructions.append(f'POOL  {out} {inp} #{kernel_shape[0]} #{strides[0]}')
        self.tensor_map[outputs[0]] = out

    def _op_averagepool(self, inputs, outputs, attrs, init_map):
        self.warnings.append("AveragePool mapped to POOL (max-pool) — result may differ")
        self._op_maxpool(inputs, outputs, attrs, init_map)

    def _op_globalaveragepool(self, inputs, outputs, attrs, init_map):
        inp = self._get_or_load(inputs[0])
        out = self._alloc_reg()
        self.instructions.append(f'RDUC  {out} {inp} #1')  # mode 1 = mean
        self.tensor_map[outputs[0]] = out

    def _op_reshape(self, inputs, outputs, attrs, init_map):
        inp = self._get_or_load(inputs[0])
        # Shape from second input (ONNX Reshape has shape as input, not attr)
        if len(inputs) > 1 and inputs[1] in self.tensor_map:
            shape_ref = self.tensor_map[inputs[1]]
            out = self._alloc_reg()
            self.instructions.append(f'RSHP  {out} {inp} {shape_ref}')
        else:
            # Flatten to 2D [batch, rest] — most common case
            out = self._alloc_reg()
            self.instructions.append(f'RSHP  {out} {inp}')
            self.warnings.append(f"Reshape: shape tensor unavailable, using RSHP (flatten)")
        self.tensor_map[outputs[0]] = out

    def _op_flatten(self, inputs, outputs, attrs, init_map):
        inp = self._get_or_load(inputs[0])
        out = self._alloc_reg()
        self.instructions.append(f'RSHP  {out} {inp}')
        self.tensor_map[outputs[0]] = out

    def _op_transpose(self, inputs, outputs, attrs, init_map):
        inp = self._get_or_load(inputs[0])
        out = self._alloc_reg()
        self.instructions.append(f'TRNS  {out} {inp}')
        self.tensor_map[outputs[0]] = out

    def _op_dropout(self, inputs, outputs, attrs, init_map):
        inp = self._get_or_load(inputs[0])
        out = self._alloc_reg()
        # ONNX Dropout p from inputs[1] (opset 12+) or attr
        p = 0.5
        if 'ratio' in attrs:
            p = attrs['ratio'].f
        self.instructions.append(f'DROP  {out} {inp} #{p}')
        self.tensor_map[outputs[0]] = out
        if len(outputs) > 1 and outputs[1]:
            self.tensor_map[outputs[1]] = out  # mask output — approximate

    def _op_gather(self, inputs, outputs, attrs, init_map):
        # Treat as embedding lookup (axis=0)
        table = self._get_or_load(inputs[0])
        indices = self._get_or_load(inputs[1])
        out = self._alloc_reg()
        self.instructions.append(f'EMBD  {out} {table} {indices}')
        self.tensor_map[outputs[0]] = out

    def _op_concat(self, inputs, outputs, attrs, init_map):
        axis = attrs['axis'].i if 'axis' in attrs else 0
        regs = [self._get_or_load(i) for i in inputs if i]
        if len(regs) == 2:
            out = self._alloc_reg()
            self.instructions.append(f'MERG  {out} {regs[0]} {regs[1]} #{axis}')
            self.tensor_map[outputs[0]] = out
        else:
            self.warnings.append(f"Concat with {len(regs)} inputs: only 2-input MERG supported")
            self.tensor_map[outputs[0]] = regs[0]

    def _op_split(self, inputs, outputs, attrs, init_map):
        inp = self._get_or_load(inputs[0])
        axis = attrs['axis'].i if 'axis' in attrs else 0
        # NML SPLT: SPLT R_out R_src #axis #n_splits
        n = len(outputs)
        out = self._alloc_reg()
        self.instructions.append(f'SPLT  {out} {inp} #{axis} #{n}')
        for o in outputs:
            if o:
                self.tensor_map[o] = out
        if n > 1:
            self.warnings.append(f"Split with {n} outputs: all outputs mapped to first split result")

    def _op_clip(self, inputs, outputs, attrs, init_map):
        inp = self._get_or_load(inputs[0])
        out = self._alloc_reg()
        # min/max from attrs (opset 6) or inputs (opset 11+)
        mn = attrs['min'].f if 'min' in attrs else 0.0
        mx = attrs['max'].f if 'max' in attrs else 6.0
        self.instructions.append(f'CLMP  {out} {inp} #{mn} #{mx}')
        self.tensor_map[outputs[0]] = out

    def _op_where(self, inputs, outputs, attrs, init_map):
        cond = self._get_or_load(inputs[0])
        x = self._get_or_load(inputs[1])
        y = self._get_or_load(inputs[2])
        out = self._alloc_reg()
        self.instructions.append(f'WHER  {out} {cond} {x} {y}')
        self.tensor_map[outputs[0]] = out

    def _op_unsqueeze(self, inputs, outputs, attrs, init_map):
        # Map to RSHP — pass through
        inp = self._get_or_load(inputs[0])
        out = self._alloc_reg()
        self.instructions.append(f'RSHP  {out} {inp}')
        self.tensor_map[outputs[0]] = out

    def _op_squeeze(self, inputs, outputs, attrs, init_map):
        inp = self._get_or_load(inputs[0])
        out = self._alloc_reg()
        self.instructions.append(f'RSHP  {out} {inp}')
        self.tensor_map[outputs[0]] = out

    def _op_identity(self, inputs, outputs, attrs, init_map):
        # Pass through
        src = self.tensor_map.get(inputs[0], 'R0')
        self.tensor_map[outputs[0]] = src

    def _op_cast(self, inputs, outputs, attrs, init_map):
        # NML converts types on load — approximate with identity
        src = self.tensor_map.get(inputs[0], 'R0')
        self.tensor_map[outputs[0]] = src

    def generate_nml(self):
        """Return NML program as string."""
        header = [
            f'; NML program generated by nml_from_onnx.py',
            f'; Source: ONNX model',
            f'; Inputs: load your input tensor as R0 before calling',
            f'; Output: @output memory slot after HALT',
            f'',
        ]
        return '\n'.join(header + self.instructions) + '\n'

    def generate_data(self):
        """Return .nml.data file content."""
        lines = []
        for name, arr in self.data_tensors.items():
            arr_f32 = arr.astype(np.float32).flatten()
            shape_str = ','.join(str(s) for s in arr.shape) if arr.ndim > 0 else '1'
            data_str = ','.join(f'{v:.6g}' for v in arr_f32)
            lines.append(f'@{name} shape={shape_str} dtype=f32 data={data_str}')
        return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(
        description='Convert ONNX model to NML program',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 transpilers/nml_from_onnx.py model.onnx
  python3 transpilers/nml_from_onnx.py model.onnx --output-nml out.nml --output-data out.nml.data
  python3 transpilers/nml_from_onnx.py model.onnx --dry-run --verbose
  python3 transpilers/nml_from_onnx.py model.onnx --input-name input_tensor
""")
    parser.add_argument('model', help='Input ONNX model file')
    parser.add_argument('--output-nml',  help='Output .nml file (default: model.nml)')
    parser.add_argument('--output-data', help='Output .nml.data file (default: model.nml.data)')
    parser.add_argument('--input-name',  help='Override input tensor name (default: from ONNX)')
    parser.add_argument('--dry-run',     action='store_true', help='Print graph without writing files')
    parser.add_argument('--verbose',     action='store_true', help='Show each node as it is converted')
    args = parser.parse_args()

    model_path = Path(args.model)
    out_nml  = Path(args.output_nml)  if args.output_nml  else model_path.with_suffix('.nml')
    out_data = Path(args.output_data) if args.output_data else model_path.with_suffix('.nml.data')

    print(f"Loading {model_path}...")
    converter = ONNXToNML(str(model_path), verbose=args.verbose)

    # Override input name if requested
    if args.input_name:
        # Will be resolved during convert() — we pre-seed the map
        converter.tensor_map[args.input_name] = 'R0'

    converter.convert()

    nml_text  = converter.generate_nml()
    data_text = converter.generate_data()

    if args.dry_run:
        print("\n=== NML Program ===")
        print(nml_text)
        print(f"\n=== Data ({len(converter.data_tensors)} weight tensors) ===")
        for name, arr in converter.data_tensors.items():
            print(f"  @{name}: shape={arr.shape} dtype={arr.dtype}")
    else:
        out_nml.write_text(nml_text)
        out_data.write_text(data_text)
        print(f"Written: {out_nml}")
        print(f"Written: {out_data} ({len(converter.data_tensors)} weight tensors)")

    if converter.warnings:
        print(f"\nWarnings ({len(converter.warnings)}):")
        for w in converter.warnings:
            print(f"  [WARN]  {w}")

    if UNSUPPORTED_OPS:
        print(f"\nUnsupported ops (skipped): {', '.join(sorted(UNSUPPORTED_OPS))}")
        print("  These ops are mapped to pass-throughs. Review the output NML carefully.")

    print(f"\nDone. {len(converter.instructions)} instructions generated.")


if __name__ == '__main__':
    main()
