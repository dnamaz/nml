#!/usr/bin/env python3
"""
Training Quality Investigation — compare NML vs Python training.

1. PyTorch/NumPy identical training loop vs NML
2. Convergence curves at various epoch counts
3. TNET vs manual BKWD/WUPD in NML
"""

import subprocess
import tempfile
import os
import json
import numpy as np
from pathlib import Path

NML_BINARY = str(Path(__file__).parent.parent / "nml")

# ═══════════════════════════════════════════════
# Common setup: 1→4→1 network, y=2x+1, x=3→y=7
# ═══════════════════════════════════════════════

W1_INIT = np.array([[0.5, -0.3, 0.2, -0.1]])        # 1×4
B1_INIT = np.array([[0.1, 0.1, 0.1, 0.1]])           # 1×4
W2_INIT = np.array([[0.4], [-0.2], [0.3], [0.1]])    # 4×1
B2_INIT = np.array([[0.0]])                            # 1×1
X_TRAIN = np.array([[3.0]])                            # 1×1
Y_TRAIN = np.array([[7.0]])                            # 1×1


def python_train(epochs, lr, dtype=np.float64):
    """Train with pure NumPy — identical math to NML's TNET."""
    w1 = W1_INIT.astype(dtype).copy()
    b1 = B1_INIT.astype(dtype).copy()
    w2 = W2_INIT.astype(dtype).copy()
    b2 = B2_INIT.astype(dtype).copy()
    x = X_TRAIN.astype(dtype)
    y = Y_TRAIN.astype(dtype)

    losses = []
    for epoch in range(epochs):
        # Forward
        pre_h = x @ w1 + b1          # 1×4
        hidden = np.maximum(0, pre_h)  # ReLU
        output = hidden @ w2 + b2      # 1×1

        # MSE loss
        diff = output - y
        loss = float((diff ** 2).mean())
        losses.append(loss)

        # Backward (same as NML TNET)
        d_out = 2.0 * diff / x.shape[0]
        d_w2 = hidden.T @ d_out
        d_b2 = d_out.sum(axis=0, keepdims=True)
        d_hidden = d_out @ w2.T * (pre_h > 0).astype(dtype)
        d_w1 = x.T @ d_hidden
        d_b1 = d_hidden.sum(axis=0, keepdims=True)

        # SGD update
        w2 -= lr * d_w2
        b2 -= lr * d_b2
        w1 -= lr * d_w1
        b1 -= lr * d_b1

    # Final inference
    pre_h = x @ w1 + b1
    hidden = np.maximum(0, pre_h)
    prediction = float((hidden @ w2 + b2)[0, 0])
    final_loss = float(((prediction - y[0, 0]) ** 2))

    return prediction, final_loss, losses


def nml_tnet_train(epochs, lr):
    """Train with NML TNET opcode."""
    program = f"""LD    R1 @w1
LD    R2 @b1
LD    R3 @w2
LD    R4 @b2
LD    R0 @input
LD    R9 @target
TNET  #{epochs} #{lr} #0
MMUL  R5 R0 R1
MADD  R5 R5 R2
RELU  R5 R5
MMUL  R6 R5 R3
MADD  RA R6 R4
ST    RA @prediction
HALT"""

    data = """@w1 shape=1,4 dtype=f64 data=0.5,-0.3,0.2,-0.1
@b1 shape=1,4 dtype=f64 data=0.1,0.1,0.1,0.1
@w2 shape=4,1 dtype=f64 data=0.4,-0.2,0.3,0.1
@b2 shape=1,1 dtype=f64 data=0.0
@input shape=1,1 data=3.0
@target shape=1,1 data=7.0"""

    return _run_nml(program, data)


def nml_manual_train(epochs, lr):
    """Train with manual BKWD/WUPD loop in NML."""
    program = f"""LD    R1 @w1
LD    R2 @b1
LD    R3 @w2
LD    R4 @b2
LD    R0 @input
LD    R9 @target
LOOP  #{epochs}
MMUL  R5 R0 R1
MADD  R5 R5 R2
RELU  R6 R5
MMUL  R7 R6 R3
MADD  R7 R7 R4
LOSS  R8 R7 R9 #0
BKWD  RG R7 R9
TRNS  RH R6
MMUL  RI RH RG
WUPD  R3 RI #{lr}
WUPD  R4 RG #{lr}
ENDP
MMUL  R5 R0 R1
MADD  R5 R5 R2
RELU  R5 R5
MMUL  R6 R5 R3
MADD  RA R6 R4
ST    RA @prediction
ST    R8 @final_loss
HALT"""

    data = """@w1 shape=1,4 dtype=f64 data=0.5,-0.3,0.2,-0.1
@b1 shape=1,4 dtype=f64 data=0.1,0.1,0.1,0.1
@w2 shape=4,1 dtype=f64 data=0.4,-0.2,0.3,0.1
@b2 shape=1,1 dtype=f64 data=0.0
@input shape=1,1 data=3.0
@target shape=1,1 data=7.0"""

    return _run_nml(program, data)


def _run_nml(program, data):
    """Run an NML program and extract prediction + loss."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".nml", delete=False) as pf:
        pf.write(program)
        prog_path = pf.name
    with tempfile.NamedTemporaryFile(mode="w", suffix=".nml.data", delete=False) as df:
        df.write(data)
        data_path = df.name

    try:
        r = subprocess.run([NML_BINARY, prog_path, data_path, "--max-cycles", "10000000"],
                          capture_output=True, text=True, timeout=60)
        prediction = None
        loss = None
        for line in r.stdout.split("\n"):
            if "prediction:" in line:
                nums = line.split("data=[")[1].split("]")[0] if "data=[" in line else ""
                if nums:
                    prediction = float(nums.split(",")[0])
            if "final_loss:" in line:
                nums = line.split("data=[")[1].split("]")[0] if "data=[" in line else ""
                if nums:
                    loss = float(nums.split(",")[0])
        return prediction, loss
    finally:
        os.unlink(prog_path)
        os.unlink(data_path)


def main():
    print("=" * 70)
    print("  NML Training Quality Investigation")
    print("  Architecture: 1→4→1 ReLU, y=2x+1, x=3→y=7")
    print("=" * 70)

    # ═══════════════════════════════════════════
    # Test 1: Side-by-side at 2000 epochs
    # ═══════════════════════════════════════════
    print("\n  TEST 1: Python vs NML at 2000 epochs, lr=0.001")
    print("  " + "─" * 55)

    py_f64_pred, py_f64_loss, _ = python_train(2000, 0.001, np.float64)
    py_f32_pred, py_f32_loss, _ = python_train(2000, 0.001, np.float32)
    nml_tnet_pred, _ = nml_tnet_train(2000, 0.001)
    nml_manual_pred, nml_manual_loss = nml_manual_train(2000, 0.001)

    print(f"  {'Method':<25} {'Prediction':>12} {'Loss':>12} {'Error':>8}")
    print(f"  {'─'*60}")
    print(f"  {'Python f64':<25} {py_f64_pred:>12.6f} {py_f64_loss:>12.8f} {abs(py_f64_pred - 7.0):>8.4f}")
    print(f"  {'Python f32':<25} {py_f32_pred:>12.6f} {py_f32_loss:>12.8f} {abs(py_f32_pred - 7.0):>8.4f}")
    print(f"  {'NML TNET (f32 internal)':<25} {nml_tnet_pred or 0:>12.6f} {'—':>12} {abs((nml_tnet_pred or 0) - 7.0):>8.4f}")
    print(f"  {'NML Manual BKWD/WUPD':<25} {nml_manual_pred or 0:>12.6f} {nml_manual_loss or 0:>12.8f} {abs((nml_manual_pred or 0) - 7.0):>8.4f}")

    # ═══════════════════════════════════════════
    # Test 2: Convergence curves
    # ═══════════════════════════════════════════
    print(f"\n  TEST 2: Convergence curves (lr=0.001)")
    print("  " + "─" * 70)
    epoch_counts = [100, 500, 1000, 2000, 5000, 10000]

    print(f"  {'Epochs':>8}", end="")
    print(f" {'Py f64':>12} {'Py f32':>12} {'NML TNET':>12} {'NML Manual':>12}")
    print(f"  {'─'*60}")

    for ep in epoch_counts:
        py64, py64_l, _ = python_train(ep, 0.001, np.float64)
        py32, py32_l, _ = python_train(ep, 0.001, np.float32)
        nml_t, _ = nml_tnet_train(ep, 0.001)
        nml_m, nml_m_l = nml_manual_train(ep, 0.001)
        print(f"  {ep:>8} {py64:>12.6f} {py32:>12.6f} {nml_t or 0:>12.6f} {nml_m or 0:>12.6f}")

    # ═══════════════════════════════════════════
    # Test 3: TNET vs Manual — same epochs
    # ═══════════════════════════════════════════
    print(f"\n  TEST 3: TNET vs Manual NML (trains all 4 weight tensors vs output layer only)")
    print("  " + "─" * 55)
    print("  Note: TNET trains w1,b1,w2,b2 (all layers).")
    print("  Note: Manual loop above only trains w2,b2 (output layer) via BKWD/WUPD.")
    print("  Note: self_train.nml also only trains w2,b2.")
    print()

    # Full Python training (all layers, like TNET)
    py_all, py_all_loss, _ = python_train(2000, 0.001, np.float64)

    # Python training output layer only (like manual NML)
    w1 = W1_INIT.copy(); b1 = B1_INIT.copy()
    w2 = W2_INIT.copy(); b2 = B2_INIT.copy()
    x = X_TRAIN; y = Y_TRAIN
    for _ in range(2000):
        pre_h = x @ w1 + b1
        hidden = np.maximum(0, pre_h)
        output = hidden @ w2 + b2
        diff = output - y
        d_out = 2.0 * diff
        d_w2 = hidden.T @ d_out
        d_b2 = d_out.sum(axis=0, keepdims=True)
        w2 -= 0.001 * d_w2
        b2 -= 0.001 * d_b2
    pred_out_only = float((np.maximum(0, x @ w1 + b1) @ w2 + b2)[0, 0])
    loss_out_only = float((pred_out_only - 7.0) ** 2)

    print(f"  {'Method':<35} {'Prediction':>12} {'Error':>8}")
    print(f"  {'─'*58}")
    print(f"  {'Python all layers (like TNET)':<35} {py_all:>12.6f} {abs(py_all - 7.0):>8.4f}")
    print(f"  {'Python output-only (like Manual)':<35} {pred_out_only:>12.6f} {abs(pred_out_only - 7.0):>8.4f}")
    print(f"  {'NML TNET':<35} {nml_tnet_pred or 0:>12.6f} {abs((nml_tnet_pred or 0) - 7.0):>8.4f}")
    print(f"  {'NML Manual':<35} {nml_manual_pred or 0:>12.6f} {abs((nml_manual_pred or 0) - 7.0):>8.4f}")


if __name__ == "__main__":
    main()
