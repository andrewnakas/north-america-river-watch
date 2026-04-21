#!/usr/bin/env python3
from pathlib import Path
import pandas as pd

root = Path('generated/ml/datasets')
by_station = {}
for path in sorted(root.glob('station_*.parquet')):
    try:
        df = pd.read_parquet(path, columns=['station_id','station_dataset_schema_version','target_discharge_cfs','has_camels_static','has_hydrobasins_static','date'])
    except Exception:
        continue
    if df is None or df.empty:
        continue
    sid = str(df['station_id'].iloc[0])
    version = sorted({str(v) for v in df.get('station_dataset_schema_version', pd.Series(dtype=str)).dropna().unique()})
    version = version[-1] if version else None
    usable = int(df['target_discharge_cfs'].notna().sum())
    if version != '2026-04-20-static-v1' or usable < 10:
        continue
    date_max = str(pd.to_datetime(df['date']).max().date())
    record = {
        'path': path.name,
        'usable': usable,
        'rows': len(df),
        'hydro': int(df.get('has_hydrobasins_static', pd.Series([0])).max()),
        'camels': int(df.get('has_camels_static', pd.Series([0])).max()),
        'date_max': date_max,
    }
    prev = by_station.get(sid)
    if prev is None or (record['date_max'], record['usable'], record['hydro'], record['camels']) > (prev['date_max'], prev['usable'], prev['hydro'], prev['camels']):
        by_station[sid] = record

print('deduped_stations', len(by_station))
print('full_window', sum(1 for r in by_station.values() if r['date_max'] == '2026-04-20'))
print('with_hydro', sum(r['hydro'] for r in by_station.values()))
print('with_camels', sum(r['camels'] for r in by_station.values()))
for sid, rec in sorted(by_station.items())[:50]:
    print(sid, rec)
