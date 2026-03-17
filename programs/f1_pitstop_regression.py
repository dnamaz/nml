"""
F1 Pit Stop Duration Predictor — Pure NumPy equivalent of f1_pitstop_regression.nml
Architecture: 12 → 16 → 8 → 1  (ReLU / ReLU / Sigmoid)
Training:     Adam, 2000 epochs, lr=0.005
"""

import json
import time
import numpy as np

DATA_FILE = "programs/f1_pitstop.nml.data"
NORM_FILE = "programs/f1_pitstop_norm.json"


def parse_nml_data(path):
    """Minimal parser for .nml.data text format."""
    tensors = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(';'):
                continue
            parts = line.split()
            name = parts[0].lstrip('@')
            meta = {k: v for k, v in (p.split('=') for p in parts[1:] if '=' in p)}
            shape = tuple(int(x) for x in meta['shape'].split(','))
            data = np.array([float(x) for x in meta['data'].split(',')],
                            dtype=np.float32).reshape(shape)
            tensors[name] = data
    return tensors


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def relu(x):
    return np.maximum(0, x)


def forward(x, weights):
    w1, b1, w2, b2, w3, b3 = weights
    h1 = relu(x @ w1 + b1)
    h2 = relu(h1 @ w2 + b2)
    out = sigmoid(h2 @ w3 + b3)
    return out, h1, h2


def train(tensors, epochs=2000, lr=0.005):
    X = tensors['training_data'].astype(np.float64)
    y = tensors['training_labels'].astype(np.float64)

    w1 = tensors['w1'].astype(np.float64)
    b1 = tensors['b1'].astype(np.float64)
    w2 = tensors['w2'].astype(np.float64)
    b2 = tensors['b2'].astype(np.float64)
    w3 = tensors['w3'].astype(np.float64)
    b3 = tensors['b3'].astype(np.float64)

    # Adam state
    B1, B2, EPS = 0.9, 0.999, 1e-8
    params = [w1, b1, w2, b2, w3, b3]
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]

    N = len(X)
    batch_size = min(64, N)
    n_batches = (N + batch_size - 1) // batch_size
    loss = 0.0
    t = 0

    for epoch in range(epochs):
        loss = 0.0
        for b in range(n_batches):
            Xb = X[b * batch_size: (b + 1) * batch_size]
            yb = y[b * batch_size: (b + 1) * batch_size]
            Bn = len(Xb)
            t += 1

            # Forward
            z1 = Xb @ w1 + b1;  a1 = np.maximum(0, z1)
            z2 = a1 @ w2 + b2;  a2 = np.maximum(0, z2)
            z3 = a2 @ w3 + b3;  a3 = 1.0 / (1.0 + np.exp(-z3))

            loss += np.sum((a3 - yb) ** 2)

            # Backward
            d3 = 2.0 * (a3 - yb) / Bn * a3 * (1 - a3)   # sigmoid backward
            d2 = (d3 @ w3.T) * (z2 > 0)                   # ReLU backward
            d1 = (d2 @ w2.T) * (z1 > 0)                   # ReLU backward

            grads = [
                Xb.T @ d1, d1.sum(axis=0, keepdims=True),
                a1.T @ d2, d2.sum(axis=0, keepdims=True),
                a2.T @ d3, d3.sum(axis=0, keepdims=True),
            ]

            # Adam update
            for i, (p, g) in enumerate(zip(params, grads)):
                m[i] = B1 * m[i] + (1 - B1) * g
                v[i] = B2 * v[i] + (1 - B2) * g ** 2
                m_hat = m[i] / (1 - B1 ** t)
                v_hat = v[i] / (1 - B2 ** t)
                params[i] -= lr * m_hat / (np.sqrt(v_hat) + EPS)

        w1, b1, w2, b2, w3, b3 = params

    return params, loss


def predict(params, x):
    w1, b1, w2, b2, w3, b3 = params
    x = x.astype(np.float64)
    h1 = np.maximum(0, x @ w1 + b1)
    h2 = np.maximum(0, h1 @ w2 + b2)
    return 1.0 / (1.0 + np.exp(-(h2 @ w3 + b3)))


if __name__ == '__main__':
    t0 = time.perf_counter()
    tensors = parse_nml_data(DATA_FILE)
    params, final_loss = train(tensors)
    elapsed = time.perf_counter() - t0

    pred_norm = float(predict(params, tensors['predict_input']))

    stats = json.load(open(NORM_FILE))
    pred_s = pred_norm * (stats['dur_max'] - stats['dur_min']) + stats['dur_min']

    print(f"training_loss:            {final_loss / len(tensors['training_data']):.4f}")
    print(f"predicted_duration_norm:  {pred_norm:.4f}")
    print(f"predicted_duration_s:     {pred_s:.2f}s")
    print(f"elapsed:                  {elapsed * 1000:.1f} ms")
