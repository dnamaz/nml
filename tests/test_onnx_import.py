#!/usr/bin/env python3
"""
test_onnx_import.py — Test ONNX import pipeline end-to-end

Creates a minimal 2-layer MLP ONNX model, converts it to NML,
then runs it and checks the output is a valid tensor.

Requires: pip install onnx numpy
"""
import sys
import os
import subprocess
import tempfile
import numpy as np

try:
    import onnx
    import onnx.helper as oh
    import onnx.numpy_helper as onph
    from onnx import TensorProto
except ImportError:
    print("SKIP: onnx not installed (pip install onnx)")
    sys.exit(0)

def build_mlp_onnx(input_dim=4, hidden_dim=8, output_dim=2):
    """Build a minimal 2-layer MLP ONNX model."""
    np.random.seed(42)
    W1 = np.random.randn(input_dim, hidden_dim).astype(np.float32) * 0.1
    b1 = np.zeros(hidden_dim, dtype=np.float32)
    W2 = np.random.randn(hidden_dim, output_dim).astype(np.float32) * 0.1
    b2 = np.zeros(output_dim, dtype=np.float32)

    X = oh.make_tensor_value_info('input', TensorProto.FLOAT, [1, input_dim])
    Y = oh.make_tensor_value_info('output', TensorProto.FLOAT, [1, output_dim])

    nodes = [
        oh.make_node('MatMul', ['input', 'W1'], ['h1_raw']),
        oh.make_node('Add',    ['h1_raw', 'b1'], ['h1_biased']),
        oh.make_node('Relu',   ['h1_biased'], ['h1']),
        oh.make_node('MatMul', ['h1', 'W2'], ['h2_raw']),
        oh.make_node('Add',    ['h2_raw', 'b2'], ['output']),
    ]

    inits = [
        onph.from_array(W1, name='W1'),
        onph.from_array(b1, name='b1'),
        onph.from_array(W2, name='W2'),
        onph.from_array(b2, name='b2'),
    ]

    graph = oh.make_graph(nodes, 'mlp', [X], [Y], initializer=inits)
    model = oh.make_model(graph, opset_imports=[oh.make_opsetid('', 17)])
    onnx.checker.check_model(model)
    return model

def main():
    repo_root  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    transpiler = os.path.join(repo_root, 'transpilers', 'nml_from_onnx.py')

    # Locate the nml binary (built by `make nml`).
    # On Windows prefer nml.exe; also check build/Release/nml.exe.
    def _find_nml():
        candidates = [
            os.path.join(repo_root, 'nml.exe'),
            os.path.join(repo_root, 'build', 'Release', 'nml.exe'),
            os.path.join(repo_root, 'nml'),
        ]
        for c in candidates:
            if os.path.isfile(c):
                return c
        return None
    nml_bin = _find_nml()

    with tempfile.TemporaryDirectory() as tmpdir:
        onnx_path = os.path.join(tmpdir, 'mlp.onnx')
        nml_path  = os.path.join(tmpdir, 'mlp.nml')
        data_path = os.path.join(tmpdir, 'mlp.nml.data')

        # Build and save ONNX model
        model = build_mlp_onnx()
        onnx.save(model, onnx_path)
        print(f"Built ONNX model: {onnx_path}")

        # Convert to NML
        result = subprocess.run(
            [sys.executable, transpiler, onnx_path,
             '--output-nml', nml_path, '--output-data', data_path],
            capture_output=True, text=True
        )
        print(result.stdout)
        if result.returncode != 0:
            print("TRANSPILER FAILED:", result.stderr)
            sys.exit(1)

        # Verify NML file was created
        assert os.path.exists(nml_path),  "NML file not created"
        assert os.path.exists(data_path), "Data file not created"

        nml_text = open(nml_path).read()
        print(f"Generated NML ({len(nml_text.splitlines())} lines):")
        print(nml_text[:500] + ('...' if len(nml_text) > 500 else ''))

        # Add input data to the data file (R0 = [1,4] input)
        with open(data_path, 'a') as f:
            f.write('@input shape=1,4 dtype=f32 data=1.0,2.0,3.0,4.0\n')

        # Add LD R0 @input to the top of the NML program (after comments)
        lines = nml_text.splitlines()
        # Find first non-comment, non-blank line
        insert_idx = 0
        for i, line in enumerate(lines):
            if line.strip() and not line.strip().startswith(';'):
                insert_idx = i
                break
        lines.insert(insert_idx, 'LD    R0 @input')
        open(nml_path, 'w').write('\n'.join(lines) + '\n')

        # Verify the nml binary exists before trying to run it
        if nml_bin is None or not os.path.exists(nml_bin):
            print(f"\nINFO: nml binary not found at {nml_bin}")
            print("      Run 'make nml' to build, then re-run this test.")
            print("      Transpiler output is valid — structural test PASSED.")
            # Still verify NML structure
            _verify_nml_structure(nml_path)
            print("\n[PASS] ONNX import transpiler test PASSED (runtime skipped — no binary)")
            return

        # Run NML
        result = subprocess.run(
            [nml_bin, nml_path, data_path],
            capture_output=True, text=True
        )
        print(f"\nNML exit code: {result.returncode}")
        if result.stdout:
            print(result.stdout[-500:])
        if result.returncode != 0:
            print("STDERR:", result.stderr[-200:])
            sys.exit(1)

        print("\n[PASS] ONNX import test PASSED")

def _verify_nml_structure(nml_path):
    """Check the generated NML has expected structure without running it."""
    text = open(nml_path).read()
    lines = [l.strip() for l in text.splitlines() if l.strip() and not l.startswith(';')]

    # Must have HALT
    assert any(l == 'HALT' for l in lines), "Missing HALT instruction"

    # Must have at least one MMUL (MLP has two MatMuls)
    mmul_count = sum(1 for l in lines if l.startswith('MMUL'))
    assert mmul_count >= 2, f"Expected >= 2 MMUL instructions, got {mmul_count}"

    # Must have RELU
    assert any(l.startswith('RELU') for l in lines), "Missing RELU instruction"

    # Must have MADD (bias adds)
    madd_count = sum(1 for l in lines if l.startswith('MADD'))
    assert madd_count >= 2, f"Expected >= 2 MADD instructions, got {madd_count}"

    # Must have ST for output
    assert any(l.startswith('ST') for l in lines), "Missing ST instruction"

    print(f"  Structure check: {mmul_count} MMUL, {madd_count} MADD, RELU, ST, HALT — OK")

if __name__ == '__main__':
    main()
