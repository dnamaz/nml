"""
F1 Pit Stop Duration Predictor — Data Preparation
Fetches pit stop data via FastF1 and writes an NML data file for training.

Setup:
    pip install fastf1 numpy

Usage:
    python3 programs/f1_pitstop_prep.py
    ./nml programs/f1_pitstop_regression.nml programs/f1_pitstop.nml.data

Output:
    programs/f1_pitstop.nml.data   — training data + initial weights
    programs/f1_pitstop_norm.json  — normalization stats for decoding predictions

Features (12 total):
    tire_off[0:5]  — one-hot: SOFT / MEDIUM / HARD / INTERMEDIATE / WET
    tire_on[5:10]  — one-hot: same encoding
    lap_norm[10]   — lap number / total laps in session
    temp_norm[11]  — (track_temp - min) / (max - min)

Target:
    duration_norm  — (pit_duration_s - min) / (max - min)
    Decode with: predicted_s = value * (dur_max - dur_min) + dur_min
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

COMPOUNDS = ['SOFT', 'MEDIUM', 'HARD', 'INTERMEDIATE', 'WET']
COMPOUND_IDX = {c: i for i, c in enumerate(COMPOUNDS)}

# Races to pull — spread across season for compound variety
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

# Network architecture: 12 → 16 → 8 → 1
# arch descriptor for RV: [n_layers, h1, act1, h2, act2, h3, act3]
# act: 0=ReLU, 1=sigmoid
# Last layer uses sigmoid because normalized labels live in (0, 1)
ARCH = [3, 16, 0, 8, 0, 1, 1]
LAYER_DIMS = [(12, 16), (16, 8), (8, 1)]


def one_hot(compound_name):
    vec = [0.0] * 5
    idx = COMPOUND_IDX.get(str(compound_name).upper(), 1)  # default MEDIUM
    vec[idx] = 1.0
    return vec


def fetch_race_pitstops(year, gp, session_type):
    """Return list of raw pit stop dicts for one session.

    In FastF1, PitInTime and PitOutTime are on consecutive lap rows:
      - Lap N  has PitInTime set  (tire being removed = Compound on this lap)
      - Lap N+1 has PitOutTime set (new tire = Compound on this lap)
    Duration = PitOutTime(N+1) - PitInTime(N)  (~18–35 s total pit lane time)
    """
    print(f"  Loading {year} {gp} {session_type}...", end=' ', flush=True)
    session = fastf1.get_session(year, gp, session_type)
    session.load(weather=True, laps=True, telemetry=False, messages=False)

    laps = session.laps
    weather = session.weather_data

    # Index weather time in seconds for fast nearest-neighbour lookup
    if weather is not None and len(weather) > 0:
        w_times = weather['Time'].dt.total_seconds().values
        w_temps = weather['TrackTemp'].values
    else:
        w_times = None

    total_laps = float(laps['LapNumber'].max()) if len(laps) > 0 else 80.0

    rows = []
    for driver in laps['Driver'].unique():
        driver_laps = laps[laps['Driver'] == driver].sort_values('LapNumber').reset_index(drop=True)

        for idx, lap in driver_laps.iterrows():
            if pd.isna(lap['PitInTime']):
                continue

            # PitOutTime is on the next lap row for this driver
            next_rows = driver_laps[driver_laps['LapNumber'] == lap['LapNumber'] + 1]
            if len(next_rows) == 0:
                continue
            next_lap = next_rows.iloc[0]
            if pd.isna(next_lap['PitOutTime']):
                continue

            pit_in_s = lap['PitInTime'].total_seconds()
            pit_out_s = next_lap['PitOutTime'].total_seconds()
            duration = pit_out_s - pit_in_s

            # Keep only realistic pit lane times (excludes SC/VSC slow releases)
            if not (15.0 < duration < 50.0):
                continue

            compound_off = str(lap.get('Compound', 'MEDIUM')).upper()
            compound_on = str(next_lap.get('Compound', 'MEDIUM')).upper()
            if compound_off not in COMPOUND_IDX:
                compound_off = 'MEDIUM'
            if compound_on not in COMPOUND_IDX:
                compound_on = 'MEDIUM'

            # Track temp at pit-in moment (nearest weather sample)
            if w_times is not None:
                closest = int(np.argmin(np.abs(w_times - pit_in_s)))
                track_temp = float(w_temps[closest])
            else:
                track_temp = 35.0

            rows.append({
                'compound_off': compound_off,
                'compound_on': compound_on,
                'lap_number': float(lap['LapNumber']),
                'total_laps': total_laps,
                'track_temp': track_temp,
                'duration': duration,
            })

    print(f"{len(rows)} pit stops")
    return rows


def make_synthetic_data(n=120):
    """Fallback dataset when FastF1 is unavailable."""
    print(f"  Generating {n} synthetic pit stops")
    np.random.seed(42)
    rows = []
    for _ in range(n):
        comp_off = np.random.choice(COMPOUNDS[:3])  # slicks only
        comp_on = np.random.choice(COMPOUNDS[:3])
        lap = float(np.random.randint(5, 65))
        temp = float(np.random.uniform(20, 55))
        # Total pit lane time: ~22–28s depending on circuit, temp, compound
        base = 24.0 + np.random.normal(0, 1.5)
        if temp < 25:
            base += 1.0   # cold tyres take slightly longer to fit
        if comp_on == 'INTERMEDIATE' or comp_on == 'WET':
            base += 0.5   # rain tyre changes are slightly slower
        duration = max(16.0, base + np.random.normal(0, 0.5))
        rows.append({
            'compound_off': comp_off,
            'compound_on': comp_on,
            'lap_number': lap,
            'total_laps': 70.0,
            'track_temp': temp,
            'duration': duration,
        })
    return rows


def build_features_labels(rows):
    """Encode and normalize into float32 arrays."""
    durations = np.array([r['duration'] for r in rows], dtype=np.float32)
    track_temps = np.array([r['track_temp'] for r in rows], dtype=np.float32)

    dur_min, dur_max = float(durations.min()), float(durations.max())
    temp_min, temp_max = float(track_temps.min()), float(track_temps.max())

    features, labels = [], []
    for r in rows:
        lap_norm = r['lap_number'] / r['total_laps']
        temp_norm = (r['track_temp'] - temp_min) / max(temp_max - temp_min, 1.0)
        feat = one_hot(r['compound_off']) + one_hot(r['compound_on']) + [lap_norm, temp_norm]
        dur_norm = (r['duration'] - dur_min) / max(dur_max - dur_min, 1.0)
        features.append(feat)
        labels.append([dur_norm])

    return (
        np.array(features, dtype=np.float32),
        np.array(labels, dtype=np.float32),
        {'dur_min': dur_min, 'dur_max': dur_max,
         'temp_min': temp_min, 'temp_max': temp_max},
    )


def he_init(in_dim, out_dim):
    scale = np.sqrt(2.0 / in_dim)
    return (np.random.randn(in_dim, out_dim) * scale).astype(np.float32)


def fmt_tensor(arr):
    return ','.join(f'{v:.6f}' for v in arr.flatten())


def write_nml_data(out_path, features, labels, norm_stats, predict_example):
    N = len(features)
    np.random.seed(0)

    weights = [(he_init(i, o), np.zeros(o, dtype=np.float32)) for i, o in LAYER_DIMS]

    lines = [
        f'; F1 Pit Stop Duration — {N} training samples',
        f'; Generated by f1_pitstop_prep.py',
        f'; Duration range: {norm_stats["dur_min"]:.2f}s – {norm_stats["dur_max"]:.2f}s',
        f'; Decode: predicted_s = value * {norm_stats["dur_max"] - norm_stats["dur_min"]:.4f} + {norm_stats["dur_min"]:.4f}',
        '',
        '; Architecture descriptor for TNDEEP in RV:',
        '; [n_layers=3, h1=16, act1=ReLU, h2=8, act2=ReLU, h3=1, act3=ReLU]',
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
        f'@training_data shape={N},12 dtype=f32 data={fmt_tensor(features)}',
        f'@training_labels shape={N},1 dtype=f32 data={fmt_tensor(labels)}',
        '',
        '; Prediction example — edit to test different scenarios',
        f'@predict_input shape=1,12 dtype=f32 data={fmt_tensor(predict_example)}',
    ]

    Path(out_path).write_text('\n'.join(lines) + '\n')
    print(f'\nWrote {out_path}')
    print(f'  {N} training samples  |  12 features  |  1 target (normalized duration)')
    print(f'  Duration range: {norm_stats["dur_min"]:.2f}s – {norm_stats["dur_max"]:.2f}s')


def make_predict_example(compound_off='SOFT', compound_on='HARD',
                         lap_number=28, total_laps=57, track_temp=38.0,
                         norm_stats=None):
    """Build a 1×12 feature vector for a single prediction query."""
    temp_min = norm_stats['temp_min'] if norm_stats else 20.0
    temp_max = norm_stats['temp_max'] if norm_stats else 55.0
    feat = (one_hot(compound_off) + one_hot(compound_on) +
            [lap_number / total_laps,
             (track_temp - temp_min) / max(temp_max - temp_min, 1.0)])
    print(f'\nPrediction example: {compound_off}→{compound_on}, '
          f'lap {lap_number}/{total_laps}, {track_temp}°C track')
    return np.array([feat], dtype=np.float32)


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
                all_rows.extend(fetch_race_pitstops(year, gp, stype))
            except Exception as e:
                print(f'  Warning — skipped {year} {gp}: {e}')

        if len(all_rows) < 20:
            print(f'Only {len(all_rows)} samples from FastF1, supplementing with synthetic data')
            all_rows.extend(make_synthetic_data(max(0, 60 - len(all_rows))))
    else:
        all_rows = make_synthetic_data(120)

    print(f'\nTotal pit stops: {len(all_rows)}')

    features, labels, norm_stats = build_features_labels(all_rows)
    predict_ex = make_predict_example(
        compound_off='SOFT', compound_on='HARD',
        lap_number=28, total_laps=57, track_temp=38.0,
        norm_stats=norm_stats,
    )

    nml_data_path = here / 'f1_pitstop.nml.data'
    write_nml_data(nml_data_path, features, labels, norm_stats, predict_ex)

    norm_path = here / 'f1_pitstop_norm.json'
    norm_path.write_text(json.dumps(norm_stats, indent=2))
    print(f'Wrote {norm_path}  (needed to decode predictions)')

    print('\nNext step:')
    print(f'  ./nml programs/f1_pitstop_regression.nml programs/f1_pitstop.nml.data')
