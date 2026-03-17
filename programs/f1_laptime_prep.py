"""
F1 Lap Time Predictor — Data Preparation
Fetches lap-level data via FastF1 and builds a sliding-window training set.

NML Data:
    The parser reads data= values directly from the heap buffer (no line-length
    limit). NML_MAX_TENSOR_SIZE defaults to 16M elements; override at build time
    for constrained targets: gcc ... -DNML_MAX_TENSOR_SIZE=65536

Setup:
    pip install fastf1 numpy

Usage:
    python3 programs/f1_laptime_prep.py
    ./nml programs/f1_laptime_regression.nml programs/f1_laptime.nml.data

Output:
    programs/f1_laptime.nml.data   — training data + initial weights
    programs/f1_laptime_norm.json  — normalization stats for decoding predictions

Feature Engineering (15 total):
    Sliding window of 5 consecutive accurate laps.
    Per lap (×5): lt_norm, tire_norm, lap_norm
        lt_norm    — (laptime - lt_min) / (lt_max - lt_min)
        tire_norm  — tyre_life / MAX_TYRE_LIFE
        lap_norm   — lap_number / total_laps

Target:
    laptime_norm — normalised next-lap time
    Decode: predicted_s = value * (lt_max - lt_min) + lt_min
"""

import json
import sys
from pathlib import Path

import numpy as np

try:
    import fastf1
    import pandas as pd
    HAS_FASTF1 = True
except ImportError:
    HAS_FASTF1 = False
    print("fastf1 not installed — using synthetic data. Run: pip install fastf1 numpy")

RACES = [
    (2023, 'Bahrain', 'R'),
    (2023, 'Saudi Arabia', 'R'),
    (2023, 'Australia', 'R'),
    (2023, 'Azerbaijan', 'R'),
    (2023, 'Monaco', 'R'),
    (2023, 'Spain', 'R'),
    (2023, 'Hungary', 'R'),
    (2023, 'Italy', 'R'),
]

WINDOW = 5              # history laps used as input
FEATURES_PER_LAP = 3   # lt_norm, tire_norm, lap_norm
N_FEATURES = WINDOW * FEATURES_PER_LAP  # 15
MAX_TYRE_LIFE = 50

# Architecture: 15 → 16 (ReLU) → 8 (ReLU) → 1 (Sigmoid)
# act codes: 0=ReLU, 1=Sigmoid
ARCH = [3, 16, 0, 8, 0, 1, 1]
LAYER_DIMS = [(15, 16), (16, 8), (8, 1)]


# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------

def fetch_race_laps(year, gp, session_type):
    """Return raw windows (list of dicts) from one race session."""
    print(f"  Loading {year} {gp} {session_type}...", end=' ', flush=True)
    session = fastf1.get_session(year, gp, session_type)
    session.load(weather=False, laps=True, telemetry=False, messages=False)

    laps = session.laps
    total_laps = float(laps['LapNumber'].max()) if len(laps) > 0 else 70.0

    raw_rows = []
    for driver in laps['Driver'].unique():
        driver_laps = (
            laps[laps['Driver'] == driver]
            .sort_values('LapNumber')
            .reset_index(drop=True)
        )
        accurate = driver_laps[driver_laps['IsAccurate'] == True].reset_index(drop=True)
        if len(accurate) < WINDOW + 1:
            continue

        for col in ['LapTime']:
            if col in accurate.columns:
                accurate[col] = accurate[col].apply(
                    lambda x: x.total_seconds() if pd.notna(x) else np.nan
                )

        for i in range(len(accurate) - WINDOW):
            window = accurate.iloc[i: i + WINDOW]
            target = accurate.iloc[i + WINDOW]

            if window['LapTime'].isna().any() or pd.isna(target['LapTime']):
                continue
            if window['LapTime'].min() < 60 or window['LapTime'].max() > 200:
                continue
            if not (60 < target['LapTime'] < 200):
                continue

            raw_rows.append({
                'lts': window['LapTime'].tolist(),
                'tyres': [float(t) for t in window.get('TyreLife', [1]*WINDOW).tolist()],
                'laps': window['LapNumber'].tolist(),
                'total_laps': total_laps,
                'target': float(target['LapTime']),
            })

    print(f"{len(raw_rows)} windows")
    return raw_rows


def make_synthetic_data(n=200):
    """Realistic fallback when FastF1 is unavailable."""
    print(f"  Generating {n} synthetic windows")
    np.random.seed(42)
    rows = []
    for _ in range(n):
        base = np.random.uniform(82, 98)
        tire_start = np.random.randint(1, 25)
        lap_start = np.random.randint(5, 50)
        total_laps = 70.0
        lts, tyres, laps = [], [], []
        for j in range(WINDOW + 1):
            deg = (tire_start + j) * 0.03
            lt = base + deg + np.random.normal(0, 0.15)
            lts.append(lt)
            tyres.append(float(min(tire_start + j, MAX_TYRE_LIFE)))
            laps.append(float(min(lap_start + j, total_laps)))
        rows.append({
            'lts': lts[:WINDOW],
            'tyres': tyres[:WINDOW],
            'laps': laps[:WINDOW],
            'total_laps': total_laps,
            'target': lts[WINDOW],
        })
    return rows


# ---------------------------------------------------------------------------
# Feature encoding
# ---------------------------------------------------------------------------

def compute_norm_stats(rows):
    all_lts = []
    for r in rows:
        all_lts.extend(r['lts'])
        all_lts.append(r['target'])
    lt_min = float(np.min(all_lts))
    lt_max = float(np.max(all_lts))
    return {'lt_min': lt_min, 'lt_max': lt_max}


def encode_row(row, norm):
    lt_range = max(norm['lt_max'] - norm['lt_min'], 1.0)
    features = []
    for lt, tyre, lap in zip(row['lts'], row['tyres'], row['laps']):
        lt_n = (lt - norm['lt_min']) / lt_range
        tire_n = min(tyre, MAX_TYRE_LIFE) / MAX_TYRE_LIFE
        lap_n = lap / max(row['total_laps'], 1.0)
        features.extend([lt_n, tire_n, lap_n])
    label = (row['target'] - norm['lt_min']) / lt_range
    return np.array(features, dtype=np.float32), np.float32(label)


def select_samples(rows, n):
    """Pick n representative windows spread evenly across all available rows."""
    if len(rows) <= n:
        return rows
    idx = np.linspace(0, len(rows) - 1, n, dtype=int)
    return [rows[i] for i in idx]


# ---------------------------------------------------------------------------
# Weight initialisation + .nml.data writer
# ---------------------------------------------------------------------------

def he_init(in_dim, out_dim):
    scale = np.sqrt(2.0 / in_dim)
    return (np.random.randn(in_dim, out_dim) * scale).astype(np.float32)


def fmt_tensor(arr):
    return ','.join(f'{v:.6f}' for v in arr.flatten())


def write_nml_data(out_path, features, labels, norm):
    N = len(features)
    np.random.seed(0)
    weights = [(he_init(i, o), np.zeros(o, dtype=np.float32)) for i, o in LAYER_DIMS]

    lt_range = norm['lt_max'] - norm['lt_min']
    lines = [
        f'; F1 Lap Time Prediction — {N} training windows',
        f'; Generated by f1_laptime_prep.py',
        f'; Lap time range: {norm["lt_min"]:.3f}s – {norm["lt_max"]:.3f}s',
        f'; Decode: predicted_s = value * {lt_range:.4f} + {norm["lt_min"]:.4f}',
        f';',
        f'; Parser reads from heap — no line-length limit.',
        '',
        '; Architecture descriptor for TNDEEP (stored in RV):',
        '; [n_layers=3, h1=16, act1=ReLU, h2=8, act2=ReLU, h3=1, act3=Sigmoid]',
        f'@arch shape=7 dtype=f32 data={",".join(str(v) for v in ARCH)}',
        '',
    ]

    for i, ((w, b), (in_d, out_d)) in enumerate(zip(weights, LAYER_DIMS), start=1):
        lines += [
            f'@w{i} shape={in_d},{out_d} dtype=f32 data={fmt_tensor(w)}',
            f'@b{i} shape=1,{out_d} dtype=f32 data={fmt_tensor(b)}',
        ]

    feat_line = f'@training_data shape={N},{N_FEATURES} dtype=f32 data={fmt_tensor(features)}'
    lbl_line = f'@training_labels shape={N},1 dtype=f32 data={fmt_tensor(labels)}'

    lines += [
        '',
        feat_line,
        lbl_line,
        '',
    ]

    # Prediction example: lap 30/58, tyre age 12 laps, typical pace ~92s
    base_lt = 92.0
    ex_lts   = [base_lt + j * 0.04 for j in range(WINDOW)]
    ex_tyres = [float(8 + j) for j in range(WINDOW)]
    ex_laps  = [float(26 + j) for j in range(WINDOW)]
    pred_row = {'lts': ex_lts, 'tyres': ex_tyres, 'laps': ex_laps, 'total_laps': 58.0, 'target': 92.2}
    pred_feat, _ = encode_row(pred_row, norm)

    lines += [
        '; Prediction example: lap 30/58, tyre age 12 laps, pace ~92 s',
        f'@predict_input shape=1,{N_FEATURES} dtype=f32 data={fmt_tensor(pred_feat)}',
    ]

    Path(out_path).write_text('\n'.join(lines) + '\n')
    print(f'\nWrote {out_path}')
    print(f'  {N} training windows  |  {N_FEATURES} features  |  1 target')
    print(f'  Lap time range: {norm["lt_min"]:.3f}s – {norm["lt_max"]:.3f}s')
    print(f'  training_data: {N * N_FEATURES} elements  ({N * N_FEATURES * 4 // 1024} KB at f32)')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    np.random.seed(42)
    here = Path(__file__).parent

    if HAS_FASTF1:
        cache_dir = here.parent / 'f1_cache'
        cache_dir.mkdir(exist_ok=True)
        fastf1.Cache.enable_cache(str(cache_dir))

        all_rows = []
        for year, gp, stype in RACES:
            try:
                all_rows.extend(fetch_race_laps(year, gp, stype))
            except Exception as e:
                print(f'  Warning — skipped {year} {gp}: {e}')

        if len(all_rows) < 20:
            print(f'Only {len(all_rows)} windows from FastF1, falling back to synthetic')
            all_rows = make_synthetic_data(200)
            use_synthetic = True
        else:
            use_synthetic = False
    else:
        all_rows = make_synthetic_data(200)
        use_synthetic = True

    print(f'\nTotal candidate windows: {len(all_rows)}')

    norm = compute_norm_stats(all_rows)

    selected = all_rows
    print(f'Using all {len(selected)} samples')

    features = np.array([encode_row(r, norm)[0] for r in selected], dtype=np.float32)
    labels = np.array([[encode_row(r, norm)[1]] for r in selected], dtype=np.float32)

    nml_data_path = here / 'f1_laptime.nml.data'
    write_nml_data(nml_data_path, features, labels, norm)

    norm_path = here / 'f1_laptime_norm.json'
    norm_path.write_text(json.dumps(norm, indent=2))
    print(f'Wrote {norm_path}  (needed to decode predictions)')

    print('\nNext steps:')
    print(f'  ./nml programs/f1_laptime_regression.nml programs/f1_laptime.nml.data')
    print(f'  python3 programs/f1_laptime_regression.py  (full dataset, NumPy)')
