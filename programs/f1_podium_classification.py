"""
F1 Podium Predictor — side-by-side comparison

Runs NML training as a subprocess, parses its register dump, then trains the
same network in NumPy. Metrics are computed from each system's own predictions
using the same labels — genuinely apples-to-apples.

Usage:
    python3 programs/f1_podium_classification.py
    python3 programs/f1_podium_classification.py --py   # Python only
    python3 programs/f1_podium_classification.py --nml  # NML only (must have nml-fast built)

Reads:  programs/f1_podium.nml.data    (features, labels, predict_input)
        programs/f1_podium_meta.json   (prediction example description)
"""

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE   = Path(__file__).parent
ROOT   = HERE.parent
LAYER_DIMS = [(4, 8), (8, 4), (4, 1)]


# ---------------------------------------------------------------------------
# Network (Python reference)
# ---------------------------------------------------------------------------

def sigmoid(x):        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
def relu(x):           return np.maximum(0.0, x)
def sigmoid_bk(z, a):  return a * (1.0 - a)
def relu_bk(z, a):     return (z > 0).astype(np.float32)

ACTIVATIONS    = [relu, relu, sigmoid]
ACTIVATION_BKS = [relu_bk, relu_bk, sigmoid_bk]


def he_init(in_d, out_d, seed=0):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((in_d, out_d)) * np.sqrt(2.0 / in_d)).astype(np.float32)


def forward(X, params):
    cache = {'X': X}
    A = X
    for i, ((W, b), act) in enumerate(zip(params, ACTIVATIONS)):
        Z = A @ W + b
        A = act(Z)
        cache[f'Z{i}'] = Z
        cache[f'A{i}'] = A
    return A, cache


def backward(y_pred, y_true, cache, params):
    N = y_true.shape[0]
    grads = []
    dA = (y_pred - y_true) / N
    for i in reversed(range(len(params))):
        Z      = cache[f'Z{i}']
        A_prev = cache[f'A{i-1}'] if i > 0 else cache['X']
        W, _   = params[i]
        dZ     = dA * ACTIVATION_BKS[i](Z, cache[f'A{i}'])
        dW     = A_prev.T @ dZ
        db     = dZ.sum(axis=0, keepdims=True)
        dA     = dZ @ W.T
        grads.insert(0, (dW, db))
    return grads


def train_numpy(features, labels, epochs=3000, lr=0.003, batch_size=64):
    params = [(he_init(i, o, seed=k), np.zeros((1, o), dtype=np.float32))
              for k, (i, o) in enumerate(LAYER_DIMS)]
    m = [(np.zeros_like(w), np.zeros_like(b)) for w, b in params]
    v = [(np.zeros_like(w), np.zeros_like(b)) for w, b in params]
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    N = len(features)
    for epoch in range(1, epochs + 1):
        idx = np.random.permutation(N)
        for start in range(0, N, batch_size):
            b_idx  = idx[start: start + batch_size]
            X_b, y_b = features[b_idx], labels[b_idx]
            y_pred, cache = forward(X_b, params)
            grads = backward(y_pred, y_b, cache, params)
            for i, ((dW, db), (W, b)) in enumerate(zip(grads, params)):
                t = epoch
                m[i] = (beta1*m[i][0]+(1-beta1)*dW,  beta1*m[i][1]+(1-beta1)*db)
                v[i] = (beta2*v[i][0]+(1-beta2)*dW**2, beta2*v[i][1]+(1-beta2)*db**2)
                W -= lr * (m[i][0]/(1-beta1**t)) / (np.sqrt(v[i][0]/(1-beta2**t)) + eps)
                b -= lr * (m[i][1]/(1-beta1**t)) / (np.sqrt(v[i][1]/(1-beta2**t)) + eps)
                params[i] = (W, b)
    y_all, _ = forward(features, params)
    return params, float(np.mean((y_all - labels) ** 2)), y_all


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def classification_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int).ravel()
    y_true = y_true.ravel().astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    acc  = (tp + tn) / len(y_true)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1   = 2*prec*rec / (prec+rec) if (prec+rec) > 0 else 0.0
    return dict(acc=acc, prec=prec, rec=rec, f1=f1, tp=tp, fp=fp, fn=fn, tn=tn)


# ---------------------------------------------------------------------------
# .nml.data reader
# ---------------------------------------------------------------------------

def read_nml_data(path):
    tensors = {}
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line.startswith('@'):
            continue
        name      = line.split()[0][1:]
        shape_str = [p for p in line.split() if p.startswith('shape=')]
        data_idx  = line.find('data=')
        if not shape_str or data_idx < 0:
            continue
        dims = [int(d) for d in shape_str[0].replace('shape=', '').split(',') if d]
        vals = np.array([float(v) for v in line[data_idx+5:].split(',') if v],
                        dtype=np.float32)
        tensors[name] = vals.reshape(dims) if dims else vals
    return tensors


# ---------------------------------------------------------------------------
# NML subprocess runner + output parser
# ---------------------------------------------------------------------------

def run_nml(nml_bin, nml_prog, nml_data, output_file):
    """Run NML with --output, return (elapsed_ms, dict of tensors from output file)."""
    t0 = time.perf_counter()
    result = subprocess.run(
        [str(nml_bin), str(nml_prog), str(nml_data), '--output', str(output_file)],
        capture_output=True, text=True
    )
    elapsed = (time.perf_counter() - t0) * 1000

    if result.returncode != 0:
        print('NML error:')
        print(result.stdout[-2000:])
        return elapsed, {}

    tensors = read_nml_data(output_file) if Path(output_file).exists() else {}

    # Parse NML's own reported execution time
    tm = re.search(r'Time:\s+([\d.]+)\s+µs', result.stdout)
    nml_reported_ms = float(tm.group(1)) / 1000.0 if tm else elapsed

    return nml_reported_ms, tensors


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--nml', action='store_true')
    parser.add_argument('--py',  action='store_true')
    args = parser.parse_args()
    show_nml = not args.py
    show_py  = not args.nml

    data_path = HERE / 'f1_podium.nml.data'
    meta_path = HERE / 'f1_podium_meta.json'
    if not data_path.exists():
        print('Run python3 programs/f1_podium_prep.py first.')
        sys.exit(1)

    meta     = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    tensors  = read_nml_data(data_path)
    features = tensors['training_data']
    labels   = tensors['training_labels']
    X_pred   = tensors.get('predict_input')

    N   = len(features)
    pos = int(labels.sum())
    neg = N - pos
    baseline_acc = neg / N

    print(f'F1 Podium Classifier — 4→8→4→1  |  Adam  |  3000 epochs  |  {N} samples')
    print(f'Class balance: {pos} podium ({100*pos/N:.1f}%)  {neg} no-podium ({100*neg/N:.1f}%)')
    print(f'Baseline (always "no podium"): {baseline_acc:.3f} accuracy\n')

    nml_m = py_m = None
    nml_loss = py_loss = None
    nml_time = py_time = None
    nml_prob = py_prob = None

    # ── NML ──────────────────────────────────────────────────────────────
    if show_nml:
        nml_bin  = ROOT / 'nml-fast'
        nml_prog = HERE / 'f1_podium_classification.nml'
        if not nml_bin.exists():
            print('nml-fast not found — run: make nml-fast')
        else:
            print('Running NML...', flush=True)
            out_file = HERE / 'f1_podium_nml_output.nml.data'
            nml_time, nml_tensors = run_nml(nml_bin, nml_prog, data_path, out_file)
            nml_preds = nml_tensors.get('training_predictions')
            nml_sloss = nml_tensors.get('training_loss')
            if nml_preds is not None:
                nml_loss = float(nml_sloss.ravel()[0]) if nml_sloss is not None else None
                nml_m    = classification_metrics(labels, nml_preds)
            nml_pred_single = nml_tensors.get('predicted_podium_prob')
            nml_prob = float(nml_pred_single.ravel()[0]) if nml_pred_single is not None else None

    # ── Python ───────────────────────────────────────────────────────────
    if show_py:
        print('Training Python+NumPy...', flush=True)
        t0 = time.perf_counter()
        py_params, py_loss, py_preds = train_numpy(features, labels)
        py_time = (time.perf_counter() - t0) * 1000
        py_m = classification_metrics(labels, py_preds)
        if X_pred is not None:
            p, _ = forward(X_pred, py_params)
            py_prob = float(p[0, 0])

    # ── Table ────────────────────────────────────────────────────────────
    def fmt(v): return v if v is not None else '—'
    W = 20

    print(f'\n{"":30}  {"nml-fast (BLAS)":>{W}}  {"Python + NumPy":>{W}}')
    print('─' * (30 + 2*W + 6))

    def row(label, nv, pv):
        print(f'  {label:<28}  {fmt(nv):>{W}}  {fmt(pv):>{W}}')

    row('MSE loss',
        f'{nml_loss:.5f}' if nml_loss is not None else None,
        f'{py_loss:.5f}'  if py_loss  is not None else None)
    row('Accuracy',
        f'{nml_m["acc"]:.3f}'  if nml_m else None,
        f'{py_m["acc"]:.3f}'   if py_m  else None)
    row('Precision',
        f'{nml_m["prec"]:.3f}' if nml_m else None,
        f'{py_m["prec"]:.3f}'  if py_m  else None)
    row('Recall',
        f'{nml_m["rec"]:.3f}'  if nml_m else None,
        f'{py_m["rec"]:.3f}'   if py_m  else None)
    row('F1-score',
        f'{nml_m["f1"]:.3f}'   if nml_m else None,
        f'{py_m["f1"]:.3f}'    if py_m  else None)
    if nml_m and py_m:
        row('TP/FP/FN/TN',
            f'{nml_m["tp"]}/{nml_m["fp"]}/{nml_m["fn"]}/{nml_m["tn"]}',
            f'{py_m["tp"]}/{py_m["fp"]}/{py_m["fn"]}/{py_m["tn"]}')
    row('Training time',
        f'{nml_time:.0f} ms'  if nml_time is not None else None,
        f'{py_time:.0f} ms'   if py_time  is not None else None)

    print('─' * (30 + 2*W + 6))

    ex = meta.get('predict_example', {})
    if ex:
        print(f'\nPrediction: {ex.get("description", "")}')
        row('Podium probability',
            f'{nml_prob:.4f}' if nml_prob is not None else None,
            f'{py_prob:.4f}'  if py_prob  is not None else None)
        if nml_prob is not None:
            verdict = 'PODIUM' if nml_prob >= 0.5 else 'no podium'
            print(f'  NML:    {nml_prob:.4f}  →  {verdict}')
        if py_prob is not None:
            verdict = 'PODIUM' if py_prob >= 0.5 else 'no podium'
            print(f'  Python: {py_prob:.4f}  →  {verdict}')


if __name__ == '__main__':
    main()
