#!/usr/bin/env python3
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT / 'generated' / 'ml' / 'datasets'
OUT = DATASET_DIR / 'montana_runoff_training.parquet'
OUT_SAMPLE = DATASET_DIR / 'montana_runoff_training_sample.csv'
SCHEMA = '2026-04-20-static-v1'

best = {}
for path in sorted(DATASET_DIR.glob('station_*.parquet')):
    try:
        df = pd.read_parquet(path)
    except Exception:
        continue
    if df is None or df.empty or 'station_id' not in df.columns:
        continue
    versions = set(str(v) for v in df.get('station_dataset_schema_version', pd.Series(dtype=str)).dropna().unique())
    if SCHEMA not in versions:
        continue
    usable = int(df['target_discharge_cfs'].notna().sum()) if 'target_discharge_cfs' in df.columns else 0
    if usable < 10:
        continue
    sid = str(df['station_id'].iloc[0])
    date_max = pd.to_datetime(df['date']).max() if 'date' in df.columns else pd.Timestamp.min
    hydro = int(df.get('has_hydrobasins_static', pd.Series([0])).max()) if 'has_hydrobasins_static' in df.columns else 0
    camels = int(df.get('has_camels_static', pd.Series([0])).max()) if 'has_camels_static' in df.columns else 0
    score = (date_max, usable, hydro, camels, len(df))
    prev = best.get(sid)
    if prev is None or score > prev['score']:
        best[sid] = {'df': df, 'path': path.name, 'usable': usable, 'score': score}

frames = [item['df'] for item in best.values()]
if not frames:
    raise SystemExit('No corrected station datasets found')

merged = pd.concat(frames, ignore_index=True).sort_values(['date', 'station_id'])
merged.to_parquet(OUT, index=False)
merged.head(200).to_csv(OUT_SAMPLE, index=False)
print('stations', len(best))
print('rows', len(merged))
for sid, item in sorted(best.items())[:80]:
    print(sid, item['path'], item['usable'])
