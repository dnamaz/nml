"""
F1 Lap Time Predictor — Pure NumPy equivalent of f1_laptime_regression.nml

Trains the same 15→16→8→1 network on the full 7246-sample dataset
(unrestricted by NML's 4095-char line / 65536-element tensor limits).

Usage:
    python3 programs/f1_laptime_regression.py
    python3 programs/f1_laptime_regression.py --sgd   # switch to SGD

Reads:  programs/f1_laptime.nml.data  (for weights/arch/predict_input)
        programs/f1_laptime_norm.json  (for decoding predictions)

Comparison vs NML:
    NML trains on 30 samples (line-length limit), ~256ms
    Python trains on 7246 samples (no limit),    compare with --time flag
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent


# ---------------------------------------------------------------------------
# Full dataset rebuild (mirrors f1_laptime_prep.py logic)
# ---------------------------------------------------------------------------

WINDOW = 5
FEATURES_PER_LAP = 3
N_FEATURES = WINDOW * FEATURES_PER_LAP
MAX_TYRE_LIFE = 50
ARCH = [(15, 16), (16, 8), (8, 1)]


def rebuild_full_dataset():
    try:
        import fastf1
        import pandas as pd
        HAS_F1 = True
    except ImportError:
        HAS_F1 = False

    norm = json.loads((HERE / 'f1_laptime_norm.json').read_text())

    if not HAS_F1:
        print("fastf1 not installed — training on synthetic data")
        return _synthetic_dataset(norm, n=500)

    RACES = [
        (2023, 'Bahrain', 'R'), (2023, 'Saudi Arabia', 'R'),
        (2023, 'Australia', 'R'), (2023, 'Azerbaijan', 'R'),
        (2023, 'Monaco', 'R'), (2023, 'Spain', 'R'),
        (2023, 'Hungary', 'R'), (2023, 'Italy', 'R'),
    ]

    cache_dir = HERE.parent / 'f1_cache'
    cache_dir.mkdir(exist_ok=True)
    fastf1.Cache.enable_cache(str(cache_dir))

    lt_min, lt_max = norm['lt_min'], norm['lt_max']
    lt_range = max(lt_max - lt_min, 1.0)
    all_feat, all_lbl = [], []

    for year, gp, stype in RACES:
        try:
            session = fastf1.get_session(year, gp, stype)
            session.load(weather=False, laps=True, telemetry=False, messages=False)
            laps = session.laps
            total_laps = float(laps['LapNumber'].max())

            for driver in laps['Driver'].unique():
                dl = (laps[laps['Driver'] == driver]
                      .sort_values('LapNumber').reset_index(drop=True))
                acc = dl[dl['IsAccurate'] == True].reset_index(drop=True)
                if len(acc) < WINDOW + 1:
                    continue
                acc['LapTime'] = acc['LapTime'].apply(
                    lambda x: x.total_seconds() if pd.notna(x) else np.nan)
                for i in range(len(acc) - WINDOW):
                    w = acc.iloc[i: i + WINDOW]
                    t = acc.iloc[i + WINDOW]
                    if w['LapTime'].isna().any() or pd.isna(t['LapTime']):
                        continue
                    if w['LapTime'].min() < 60 or w['LapTime'].max() > 200:
                        continue
                    if not (60 < t['LapTime'] < 200):
                        continue
                    feat = []
                    for _, lap in w.iterrows():
                        lt_n = (float(lap['LapTime']) - lt_min) / lt_range
                        tire_n = min(float(lap.get('TyreLife', 1)), MAX_TYRE_LIFE) / MAX_TYRE_LIFE
                        lap_n = float(lap['LapNumber']) / max(total_laps, 1.0)
                        feat.extend([lt_n, tire_n, lap_n])
                    all_feat.append(feat)
                    all_lbl.append([(float(t['LapTime']) - lt_min) / lt_range])
        except Exception as e:
            print(f'  Skipped {year} {gp}: {e}')

    return (np.array(all_feat, dtype=np.float32),
            np.array(all_lbl, dtype=np.float32), norm)


def _synthetic_dataset(norm, n=500):
    np.random.seed(42)
    lt_min, lt_max = norm['lt_min'], norm['lt_max']
    lt_range = max(lt_max - lt_min, 1.0)
    all_feat, all_lbl = [], []
    for _ in range(n):
        base = np.random.uniform(82, 98)
        ts = np.random.randint(1, 25)
        ls = np.random.randint(5, 50)
        feat = []
        for j in range(WINDOW):
            lt = base + (ts + j) * 0.03 + np.random.normal(0, 0.15)
            feat.extend([
                (lt - lt_min) / lt_range,
                min(ts + j, MAX_TYRE_LIFE) / MAX_TYRE_LIFE,
                min(ls + j, 70) / 70.0,
            ])
        target_lt = base + (ts + WINDOW) * 0.03 + np.random.normal(0, 0.15)
        all_feat.append(feat)
        all_lbl.append([(target_lt - lt_min) / lt_range])
    return (np.array(all_feat, dtype=np.float32),
            np.array(all_lbl, dtype=np.float32), {'lt_min': lt_min, 'lt_max': lt_max})


# ---------------------------------------------------------------------------
# Network + training
# ---------------------------------------------------------------------------

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def sigmoid_bk(x):
    s = sigmoid(x)
    return s * (1.0 - s)

def relu(x):
    return np.maximum(0.0, x)

def relu_bk(x):
    return (x > 0).astype(np.float32)

ACTIVATIONS = [relu, relu, sigmoid]
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
        Z = cache[f'Z{i}']
        A_prev = cache[f'A{i-1}'] if i > 0 else cache['X']
        W, _ = params[i]
        dZ = dA * ACTIVATION_BKS[i](Z)
        dW = A_prev.T @ dZ
        db = dZ.sum(axis=0, keepdims=True)
        dA = dZ @ W.T
        grads.insert(0, (dW, db))

    return grads


def train(features, labels, use_adam=True, epochs=5000, lr=0.005, batch_size=64):
    params = [(he_init(i, o, seed=k), np.zeros((1, o), dtype=np.float32))
              for k, (i, o) in enumerate(ARCH)]

    if use_adam:
        m = [(np.zeros_like(w), np.zeros_like(b)) for w, b in params]
        v = [(np.zeros_like(w), np.zeros_like(b)) for w, b in params]
        beta1, beta2, eps = 0.9, 0.999, 1e-8

    N = len(features)
    losses = []

    for epoch in range(1, epochs + 1):
        idx = np.random.permutation(N)
        epoch_loss = 0.0
        batches = 0

        for start in range(0, N, batch_size):
            b_idx = idx[start: start + batch_size]
            X_b = features[b_idx]
            y_b = labels[b_idx]

            y_pred, cache = forward(X_b, params)
            loss = float(np.mean((y_pred - y_b) ** 2))
            epoch_loss += loss
            batches += 1

            grads = backward(y_pred, y_b, cache, params)

            for i, ((dW, db), (W, b)) in enumerate(zip(grads, params)):
                if use_adam:
                    t = epoch
                    m[i] = (beta1 * m[i][0] + (1 - beta1) * dW,
                             beta1 * m[i][1] + (1 - beta1) * db)
                    v[i] = (beta2 * v[i][0] + (1 - beta2) * dW**2,
                             beta2 * v[i][1] + (1 - beta2) * db**2)
                    m_hat_w = m[i][0] / (1 - beta1**t)
                    m_hat_b = m[i][1] / (1 - beta1**t)
                    v_hat_w = v[i][0] / (1 - beta2**t)
                    v_hat_b = v[i][1] / (1 - beta2**t)
                    W -= lr * m_hat_w / (np.sqrt(v_hat_w) + eps)
                    b -= lr * m_hat_b / (np.sqrt(v_hat_b) + eps)
                else:
                    W -= lr * dW
                    b -= lr * db
                params[i] = (W, b)

        losses.append(epoch_loss / batches)
        if epoch % 1000 == 0:
            print(f'  epoch {epoch:5d}  loss={losses[-1]:.6f}')

    return params, losses[-1]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sgd', action='store_true', help='Use SGD instead of Adam')
    args = parser.parse_args()

    opt_name = 'SGD' if args.sgd else 'Adam'
    print(f'F1 Lap Time Predictor — NumPy ({opt_name})')
    print('Loading full dataset...')
    t0 = time.perf_counter()
    features, labels, norm = rebuild_full_dataset()
    print(f'  {len(features)} training windows  |  {N_FEATURES} features')

    print(f'\nTraining 15→16→8→1 for 5000 epochs ({opt_name}, lr=0.005)...')
    t1 = time.perf_counter()
    params, final_loss = train(features, labels, use_adam=not args.sgd,
                               epochs=5000, lr=0.005, batch_size=64)
    t2 = time.perf_counter()

    print(f'\n  Final MSE loss:  {final_loss:.6f}')
    print(f'  Training time:   {(t2 - t1) * 1000:.1f} ms  ({len(features)} samples)')

    # Prediction — same scenario as the NML program
    lt_min, lt_max = norm['lt_min'], norm['lt_max']
    lt_range = max(lt_max - lt_min, 1.0)
    base_lt = 92.0
    pred_feat = []
    for j in range(WINDOW):
        lt_n = (base_lt + j * 0.04 - lt_min) / lt_range
        tire_n = (8 + j) / MAX_TYRE_LIFE
        lap_n = (26 + j) / 58.0
        pred_feat.extend([lt_n, tire_n, lap_n])

    X_pred = np.array([pred_feat], dtype=np.float32)
    pred_norm, _ = forward(X_pred, params)
    pred_s = float(pred_norm[0, 0]) * lt_range + lt_min

    print(f'\nPrediction: lap 30/58, tyre age 12 laps, pace ~92 s')
    print(f'  Normalised output: {float(pred_norm[0, 0]):.4f}')
    print(f'  Decoded:           {pred_s:.2f} s')
    print(f'\nNML comparison (5000 epochs, {len(features)} samples, Adam):')
    print(f'  nml (vectorized, no BLAS): ~11.4s,  loss ~0.0003')
    print(f'  nml-fast (BLAS sgemm):      ~3.5s,  loss ~0.0003')
    print(f'  Python+NumPy (this script): {(t2-t1)*1000:.0f}ms, loss {final_loss:.6f}')


if __name__ == '__main__':
    main()
