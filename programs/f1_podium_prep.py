"""
F1 Podium Predictor — Data Preparation
Binary classification: will a driver finish on the podium (P1–P3)?

Features (4 per sample):
    qual_norm        — qualifying (grid) position / grid_size  (lower = better)
    teammate_norm    — teammate's qualifying position / grid_size
    hist_ckt_rate    — driver's historical podium rate at this specific circuit
                       (computed from strictly prior races — no data leakage)
    hist_ovr_rate    — driver's overall historical podium rate across all circuits

Target: 1.0 if finished P1/P2/P3, else 0.0

Class imbalance:
    ~15% of entries are podium finishes (3 podiums / ~20 starters).
    Handled by repeating each positive row OVERSAMPLE_FACTOR times.

Usage:
    python3 programs/f1_podium_prep.py
    ./nml programs/f1_podium_classification.nml programs/f1_podium.nml.data

Output:
    programs/f1_podium.nml.data   — training data + weights
    programs/f1_podium_meta.json  — class stats + prediction example metadata
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

try:
    import fastf1
    import pandas as pd
    HAS_FASTF1 = True
except ImportError:
    HAS_FASTF1 = False
    print("fastf1 not installed — using synthetic data. Run: pip install fastf1 numpy")

# Races processed in chronological order (important for leak-free historical rates)
RACES = [
    (2021, 'Bahrain',   'R'), (2021, 'Monaco',    'R'), (2021, 'France',  'R'),
    (2021, 'Hungary',   'R'), (2021, 'Italy',      'R'), (2021, 'Russia',  'R'),
    (2021, 'Mexico',    'R'), (2021, 'Abu Dhabi',  'R'),
    (2022, 'Bahrain',   'R'), (2022, 'Monaco',     'R'), (2022, 'Spain',   'R'),
    (2022, 'Hungary',   'R'), (2022, 'Italy',      'R'), (2022, 'Singapore','R'),
    (2022, 'Mexico',    'R'), (2022, 'Abu Dhabi',  'R'),
    (2023, 'Bahrain',   'R'), (2023, 'Monaco',     'R'), (2023, 'Spain',   'R'),
    (2023, 'Hungary',   'R'), (2023, 'Italy',      'R'), (2023, 'Singapore','R'),
    (2023, 'Mexico',    'R'), (2023, 'Abu Dhabi',  'R'),
]

N_FEATURES        = 4
OVERSAMPLE_FACTOR = 4   # repeat podium rows this many times to address class imbalance

# Architecture: 4 → 8 (ReLU) → 4 (ReLU) → 1 (Sigmoid)
# act codes: 0=ReLU, 1=Sigmoid
ARCH       = [3, 8, 0, 4, 0, 1, 1]
LAYER_DIMS = [(4, 8), (8, 4), (4, 1)]


# ---------------------------------------------------------------------------
# FastF1 data fetching
# ---------------------------------------------------------------------------

def fetch_race(year, gp, session_type):
    """Return list of dicts, one per driver, from a race session."""
    print(f"  Loading {year} {gp}...", end=" ", flush=True)
    session = fastf1.get_session(year, gp, session_type)
    session.load(weather=False, laps=False, telemetry=False, messages=False)

    results = session.results
    if results is None or len(results) == 0:
        print("(empty)")
        return []

    # Normalise column names — FastF1 uses TeamName
    grid_col   = 'GridPosition'
    finish_col = 'Position'
    team_col   = 'TeamName'
    abbr_col   = 'Abbreviation'

    needed = [grid_col, finish_col, team_col, abbr_col]
    if not all(c in results.columns for c in needed):
        print(f"(missing columns: {[c for c in needed if c not in results.columns]})")
        return []

    rows = []
    grid_size = results[grid_col].dropna()
    grid_size = float(grid_size.max()) if len(grid_size) > 0 else 20.0

    # Build teammate map: driver → teammate abbreviation (first other driver on same team)
    team_to_drivers = defaultdict(list)
    for _, r in results.iterrows():
        team_to_drivers[r[team_col]].append(r[abbr_col])

    teammate_grid = {}
    for _, r in results.iterrows():
        team_mates = [d for d in team_to_drivers[r[team_col]] if d != r[abbr_col]]
        if team_mates:
            mate_row = results[results[abbr_col] == team_mates[0]]
            if len(mate_row) > 0:
                gp_val = mate_row.iloc[0][grid_col]
                teammate_grid[r[abbr_col]] = float(gp_val) if pd.notna(gp_val) else grid_size
            else:
                teammate_grid[r[abbr_col]] = grid_size
        else:
            teammate_grid[r[abbr_col]] = grid_size

    for _, r in results.iterrows():
        gp_val = r[grid_col]
        fp_val = r[finish_col]
        if pd.isna(gp_val) or float(gp_val) <= 0:
            continue
        podium = 1 if (pd.notna(fp_val) and float(fp_val) <= 3) else 0
        rows.append({
            'driver':     str(r[abbr_col]),
            'circuit':    gp,
            'grid_pos':   float(gp_val),
            'grid_size':  grid_size,
            'tm_grid':    teammate_grid.get(str(r[abbr_col]), grid_size),
            'podium':     podium,
        })

    positives = sum(r['podium'] for r in rows)
    print(f"{len(rows)} drivers, {positives} podiums")
    return rows


# ---------------------------------------------------------------------------
# Historical rate computation (strictly chronological, no data leakage)
# ---------------------------------------------------------------------------

def attach_historical_rates(all_rows):
    """
    Add hist_ckt_rate and hist_ovr_rate to each row using only *prior* races.
    all_rows must be in chronological race order.
    """
    # Counters: driver → circuit → {starts, podiums}
    ckt_starts  = defaultdict(lambda: defaultdict(int))
    ckt_podiums = defaultdict(lambda: defaultdict(int))
    ovr_starts  = defaultdict(int)
    ovr_podiums = defaultdict(int)

    enriched = []
    for row in all_rows:
        d = row['driver']
        c = row['circuit']

        # Compute rates from history so far (Laplace smoothing: +1 start in denominator)
        hist_ckt  = ckt_podiums[d][c] / (ckt_starts[d][c] + 1)
        hist_ovr  = ovr_podiums[d]    / (ovr_starts[d]    + 1)

        enriched.append({**row, 'hist_ckt_rate': hist_ckt, 'hist_ovr_rate': hist_ovr})

        # Update counters *after* recording (so this race counts for future races)
        ckt_starts[d][c]  += 1
        ckt_podiums[d][c] += row['podium']
        ovr_starts[d]     += 1
        ovr_podiums[d]    += row['podium']

    return enriched


# ---------------------------------------------------------------------------
# Synthetic fallback
# ---------------------------------------------------------------------------

def make_synthetic_data(n=400):
    """Generates plausible synthetic rows when FastF1 is unavailable."""
    print(f"  Generating {n} synthetic driver-race entries")
    np.random.seed(42)
    drivers  = [f'DRV{i:02d}' for i in range(20)]
    circuits = ['Bahrain', 'Monaco', 'Italy', 'Spain', 'Hungary']
    rows = []
    for _ in range(n):
        d = np.random.choice(drivers)
        c = np.random.choice(circuits)
        gp  = float(np.random.randint(1, 21))
        tm  = float(np.random.randint(1, 21))
        gs  = 20.0
        # Better qualifying → higher podium chance (roughly realistic)
        p_podium = max(0.0, 0.6 - gp * 0.03 + np.random.normal(0, 0.05))
        podium = 1 if np.random.random() < p_podium else 0
        rows.append({'driver': d, 'circuit': c, 'grid_pos': gp,
                     'grid_size': gs, 'tm_grid': tm, 'podium': podium,
                     'hist_ckt_rate': np.random.uniform(0, 0.4),
                     'hist_ovr_rate': np.random.uniform(0, 0.4)})
    return rows


# ---------------------------------------------------------------------------
# Feature encoding
# ---------------------------------------------------------------------------

def encode_row(row):
    gs = max(row['grid_size'], 1.0)
    feat = np.array([
        row['grid_pos']      / gs,
        row['tm_grid']       / gs,
        float(row['hist_ckt_rate']),
        float(row['hist_ovr_rate']),
    ], dtype=np.float32)
    label = np.float32(row['podium'])
    return feat, label


# ---------------------------------------------------------------------------
# Weight initialisation + .nml.data writer
# ---------------------------------------------------------------------------

def he_init(in_dim, out_dim):
    scale = np.sqrt(2.0 / in_dim)
    return (np.random.randn(in_dim, out_dim) * scale).astype(np.float32)


def fmt_tensor(arr):
    return ','.join(f'{v:.6f}' for v in arr.flatten())


def write_nml_data(out_path, features, labels, meta):
    N = len(features)
    np.random.seed(0)
    weights = [(he_init(i, o), np.zeros(o, dtype=np.float32)) for i, o in LAYER_DIMS]

    pos_count = int(labels.sum())
    neg_count = N - pos_count

    lines = [
        f'; F1 Podium Classification — {N} training samples',
        f'; {pos_count} podium / {neg_count} no-podium  '
        f'(positives oversampled {OVERSAMPLE_FACTOR}x)',
        f'; Generated by f1_podium_prep.py',
        f';',
        f'; Decode: output >= 0.5 → predict podium',
        '',
        '; Architecture: 4 → 8 (ReLU) → 4 (ReLU) → 1 (Sigmoid)',
        f'@arch shape=7 dtype=f32 data={",".join(str(v) for v in ARCH)}',
        '',
    ]

    for i, ((w, b), (in_d, out_d)) in enumerate(zip(weights, LAYER_DIMS), start=1):
        lines += [
            f'@w{i} shape={in_d},{out_d} dtype=f32 data={fmt_tensor(w)}',
            f'@b{i} shape=1,{out_d} dtype=f32 data={fmt_tensor(b)}',
        ]

    lines += [
        '',
        f'@training_data   shape={N},{N_FEATURES} dtype=f32 data={fmt_tensor(features)}',
        f'@training_labels shape={N},1 dtype=f32 data={fmt_tensor(labels)}',
        '',
    ]

    # Prediction example from meta
    ex = meta['predict_example']
    pred_feat = np.array([
        ex['qual_norm'], ex['teammate_qual_norm'],
        ex['hist_ckt_rate'], ex['hist_ovr_rate'],
    ], dtype=np.float32)
    lines += [
        f'; Prediction example: {ex["description"]}',
        f'@predict_input shape=1,{N_FEATURES} dtype=f32 data={fmt_tensor(pred_feat)}',
    ]

    Path(out_path).write_text('\n'.join(lines) + '\n')
    print(f'\nWrote {out_path}')
    print(f'  {N} training samples  |  {N_FEATURES} features  |  binary target')
    print(f'  Class balance: {pos_count} podium ({100*pos_count/N:.1f}%)  '
          f'{neg_count} no-podium ({100*neg_count/N:.1f}%)')
    print(f'  training_data: {N * N_FEATURES} elements  '
          f'({N * N_FEATURES * 4 // 1024} KB at f32)')


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
                all_rows.extend(fetch_race(year, gp, stype))
            except Exception as e:
                print(f'  Warning — skipped {year} {gp}: {e}')

        if len(all_rows) < 40:
            print(f'Only {len(all_rows)} rows from FastF1, falling back to synthetic')
            all_rows = make_synthetic_data(400)
        else:
            all_rows = attach_historical_rates(all_rows)
    else:
        all_rows = make_synthetic_data(400)

    print(f'\nTotal raw rows: {len(all_rows)}')

    features_raw = np.array([encode_row(r)[0] for r in all_rows], dtype=np.float32)
    labels_raw   = np.array([[encode_row(r)[1]] for r in all_rows], dtype=np.float32)

    # Oversample positives to counter class imbalance
    pos_idx = np.where(labels_raw[:, 0] == 1.0)[0]
    neg_idx = np.where(labels_raw[:, 0] == 0.0)[0]
    over_idx = np.tile(pos_idx, OVERSAMPLE_FACTOR)
    combined_idx = np.concatenate([neg_idx, over_idx])
    np.random.shuffle(combined_idx)

    features = features_raw[combined_idx]
    labels   = labels_raw[combined_idx]

    print(f'After {OVERSAMPLE_FACTOR}x oversampling of positives: {len(features)} rows')

    # Prediction example: strong qualifier at a familiar circuit
    predict_example = {
        'description':       'P3 qualifier at Monaco, strong circuit history',
        'qual_norm':          3.0 / 20.0,
        'teammate_qual_norm': 7.0 / 20.0,
        'hist_ckt_rate':      0.45,
        'hist_ovr_rate':      0.38,
    }

    meta = {
        'n_raw':            len(all_rows),
        'n_training':       len(features),
        'n_pos_raw':        int(labels_raw.sum()),
        'n_neg_raw':        int(len(all_rows) - labels_raw.sum()),
        'oversample_factor': OVERSAMPLE_FACTOR,
        'predict_example':  predict_example,
    }

    nml_data_path = here / 'f1_podium.nml.data'
    write_nml_data(nml_data_path, features, labels, meta)

    meta_path = here / 'f1_podium_meta.json'
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f'Wrote {meta_path}')

    print('\nNext steps:')
    print(f'  ./nml programs/f1_podium_classification.nml programs/f1_podium.nml.data')
    print(f'  python3 programs/f1_podium_classification.py  (evaluation + metrics)')
