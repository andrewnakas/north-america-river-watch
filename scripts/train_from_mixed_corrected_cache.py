#!/usr/bin/env python3
from pathlib import Path
import json
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import train_montana_runoff_model as mt
DATASET_PATH = ROOT / 'generated' / 'ml' / 'datasets' / 'montana_runoff_training.parquet'

dataset = pd.read_parquet(DATASET_PATH)
model, report, features = mt.train_model(dataset)
station_links = []
for station_id, group in dataset.groupby('station_id'):
    first = group.iloc[0]
    station_links.append({
        'station_id': str(station_id),
        'station_name': first.get('station_name'),
        'state': first.get('state'),
        'snotel_station_id': first.get('snotel_station_id'),
        'snotel_distance_km': None if pd.isna(first.get('snotel_distance_km')) else float(first.get('snotel_distance_km')),
        'nohrsc_station_id': first.get('nohrsc_station_id'),
        'nohrsc_distance_km': None if pd.isna(first.get('nohrsc_distance_km')) else float(first.get('nohrsc_distance_km')),
        'basin_key': first.get('basin_key'),
        'snotel_basin_contributors': int(first.get('snotel_basin_contributors')) if not pd.isna(first.get('snotel_basin_contributors')) else None,
        'nohrsc_basin_contributors': int(first.get('nohrsc_basin_contributors')) if not pd.isna(first.get('nohrsc_basin_contributors')) else None,
        'snotel_basin_band': first.get('snotel_basin_band'),
        'nohrsc_basin_band': first.get('nohrsc_basin_band'),
        'has_camels_static': bool(first.get('has_camels_static')),
        'has_hydrobasins_static': bool(first.get('has_hydrobasins_static')),
    })
mt.save_model_bundle(model, features, report, station_links)
summary = {
    'generated_at': mt.datetime.now(mt.timezone.utc).isoformat(),
    'date_range': {
        'start': str(pd.to_datetime(dataset['date']).min().date()),
        'end': str(pd.to_datetime(dataset['date']).max().date()),
    },
    'target_group': None,
    'target_scope_label': 'mixed-corrected-cache',
    'target_scope': 'mixed-corrected-cache',
    'station_ids_filter': [],
    'station_candidates_considered': len(station_links),
    'stations_requested': len(station_links),
    'stations_used': len(station_links),
    'include_managed_stations': False,
    'static_descriptor_coverage': {
        'camels_station_count': sum(1 for link in station_links if link.get('has_camels_static')),
        'hydrobasins_station_count': sum(1 for link in station_links if link.get('has_hydrobasins_static')),
        'no_static_station_count': sum(1 for link in station_links if not link.get('has_camels_static') and not link.get('has_hydrobasins_static')),
    },
    'station_links': station_links,
    'stations_skipped': [],
    'rows': int(len(dataset)),
    'metrics': report['metrics'],
    'rolling_validation_summary': report['rolling_validation']['summary'],
    'experiment_flags': report.get('experiment_flags', {}),
    'data_provenance': 'merged corrected-schema station caches using best-per-station artifact while national refresh continued in background',
}
mt.write_json(mt.ML_DIR / 'latest_training_summary.json', summary)
print(json.dumps({
    'stations_used': summary['stations_used'],
    'rows': summary['rows'],
    'target_scope_label': summary['target_scope_label'],
    'ensemble_blend_test': report['metrics'].get('ensemble_blend_test', {}),
}, indent=2))
