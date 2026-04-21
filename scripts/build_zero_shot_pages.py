#!/usr/bin/env python3
from __future__ import annotations
import html
import json
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
ML = ROOT / 'generated' / 'ml'
MODELS = ML / 'models'
OUT = ROOT / 'src' / 'zero-shot.html'
DATA_OUT = ML / 'zero_shot_pages_summary.json'

summary = json.loads((ML / 'latest_training_summary.json').read_text())
report = json.loads((MODELS / 'montana_runoff_validation_report.json').read_text())

metrics = report.get('metrics', {})
rolling = report.get('rolling_validation', {}).get('summary', []) or []
stations = report.get('station_test_metrics', []) or []
station_links = summary.get('station_links', []) or []
link_by_id = {str(item.get('station_id')): item for item in station_links}

def fmt(x, digits=2):
    try:
        return f"{float(x):,.{digits}f}"
    except Exception:
        return 'n/a'

def metric_for(label):
    for item in rolling:
        if item.get('label') == label:
            return item
    return {}

selected = report.get('selected_model', {}).get('model', 'unknown')
ensemble = metrics.get('ensemble_blend_test', {})
runoff = metrics.get('runoff_only_test', {})
persist = metrics.get('benchmarks_test', {}).get('persistence_lag1', {})
full_model = metrics.get('full_model_test', {})
rolling_best = metric_for('ensemble_blend')
rolling_runoff = metric_for('runoff_only')
rolling_full = metric_for('full_model')
rolling_event = metric_for('event_gated_residual')

rows = []
for station in stations:
    sid = str(station.get('station_id'))
    link = link_by_id.get(sid, {})
    ens = station.get('ensemble_blend', {})
    runoff_m = station.get('runoff_only', {})
    rising = station.get('rising_flow_days', {}).get('ensemble_blend', {})
    high = station.get('high_flow_days', {}).get('ensemble_blend', {})
    rows.append({
        'station_id': sid,
        'station_name': station.get('station_name', ''),
        'state': link.get('state') or '',
        'has_camels_static': bool(link.get('has_camels_static')),
        'has_hydrobasins_static': bool(link.get('has_hydrobasins_static')),
        'mae': ens.get('mae'),
        'rmse': ens.get('rmse'),
        'r2': ens.get('r2'),
        'runoff_only_mae': runoff_m.get('mae'),
        'rising_mae': rising.get('mae'),
        'high_flow_mae': high.get('mae'),
    })
rows.sort(key=lambda r: (999999 if r['mae'] is None else r['mae'], r['station_id']))

states = sorted({link_by_id.get(str(s.get('station_id')), {}).get('state') for s in stations if link_by_id.get(str(s.get('station_id')), {}).get('state')})
summary_payload = {
    'generated_at': summary.get('generated_at'),
    'target_scope': summary.get('target_scope'),
    'target_scope_label': summary.get('target_scope_label'),
    'stations_used': summary.get('stations_used'),
    'station_candidates_considered': summary.get('station_candidates_considered'),
    'rows': summary.get('rows'),
    'date_range': summary.get('date_range'),
    'selected_model': selected,
    'metrics': {
        'ensemble_blend_test': ensemble,
        'runoff_only_test': runoff,
        'persistence_lag1_test': persist,
        'full_model_test': full_model,
    },
    'rolling_summary': rolling,
    'states_used': states,
    'station_rows': rows,
}
DATA_OUT.write_text(json.dumps(summary_payload, indent=2))

trs = []
for r in rows:
    trs.append(
        f"<tr><td><code>{html.escape(r['station_id'])}</code></td><td>{html.escape(r['station_name'])}</td><td>{html.escape(r['state'])}</td>"
        f"<td>{'yes' if r['has_camels_static'] else 'no'}</td><td>{'yes' if r['has_hydrobasins_static'] else 'no'}</td>"
        f"<td>{fmt(r['mae'])}</td><td>{fmt(r['rmse'])}</td><td>{fmt(r['r2'],3)}</td><td>{fmt(r['runoff_only_mae'])}</td><td>{fmt(r['rising_mae'])}</td><td>{fmt(r['high_flow_mae'])}</td></tr>"
    )

html_doc = f'''<!doctype html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>North America River Watch, zero-shot runoff benchmark</title>
    <style>
      body {{ font-family: Inter, system-ui, sans-serif; margin: 0; background: #0b1020; color: #e7ecf3; }}
      main {{ max-width: 1280px; margin: 0 auto; padding: 32px 20px 56px; }}
      a {{ color: #8ec5ff; }}
      .card {{ background: #131a2d; border: 1px solid #26314f; border-radius: 14px; padding: 18px 20px; margin: 16px 0; }}
      .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 16px; }}
      .pill {{ display: inline-block; padding: 4px 10px; border-radius: 999px; background: #1d2742; border: 1px solid #33446f; color: #cfe0ff; font-size: 12px; }}
      .stat {{ font-size: 28px; font-weight: 700; margin: 8px 0; }}
      table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
      th, td {{ text-align: left; padding: 8px 10px; border-bottom: 1px solid #26314f; vertical-align: top; }}
      th {{ position: sticky; top: 0; background: #131a2d; }}
      .table-wrap {{ overflow: auto; max-height: 70vh; }}
      code {{ background: #0d1426; padding: 2px 6px; border-radius: 6px; }}
    </style>
  </head>
  <body>
    <main>
      <p><a href="./index.html">← Back to map</a></p>
      <h1>National zero-shot runoff benchmark</h1>
      <p>This page tracks the latest broad zero-shot runoff benchmark across every station the pipeline could actually assemble into a usable dataset for the current date window. In practice that means stations with forecast metadata, non-managed filtering, enough target rows, and enough upstream weather and snow context to train and score.</p>

      <div class="grid">
        <div class="card"><div class="pill">Usable stations</div><div class="stat">{summary.get('stations_used','n/a')}</div><p>Out of {summary.get('station_candidates_considered','n/a')} eligible candidates considered.</p></div>
        <div class="card"><div class="pill">Test ensemble MAE</div><div class="stat">{fmt(ensemble.get('mae'))}</div><p>Ensemble blend on holdout rows.</p></div>
        <div class="card"><div class="pill">Test ensemble R²</div><div class="stat">{fmt(ensemble.get('r2'),3)}</div><p>Aggregate holdout fit.</p></div>
        <div class="card"><div class="pill">Rolling winner</div><div class="stat" style="font-size:20px">{html.escape(selected)}</div><p>Selected by lowest rolling-validation MAE.</p></div>
      </div>

      <div class="card">
        <div class="pill">Headline</div>
        <ul>
          <li><b>Scope:</b> {html.escape(str(summary.get('target_scope_label') or summary.get('target_scope') or 'current run'))}</li>
          <li><b>Date window:</b> {html.escape(summary.get('date_range',{}).get('start','n/a'))} to {html.escape(summary.get('date_range',{}).get('end','n/a'))}</li>
          <li><b>States represented:</b> {html.escape(', '.join(states) if states else 'n/a')}</li>
          <li><b>Ensemble blend:</b> MAE {fmt(ensemble.get('mae'))}, RMSE {fmt(ensemble.get('rmse'))}, R² {fmt(ensemble.get('r2'),3)}</li>
          <li><b>Runoff-only:</b> MAE {fmt(runoff.get('mae'))}, RMSE {fmt(runoff.get('rmse'))}, R² {fmt(runoff.get('r2'),3)}</li>
          <li><b>Persistence lag-1 baseline:</b> MAE {fmt(persist.get('mae'))}, RMSE {fmt(persist.get('rmse'))}, R² {fmt(persist.get('r2'),3)}</li>
          <li><b>Full feature model:</b> MAE {fmt(full_model.get('mae'))}, RMSE {fmt(full_model.get('rmse'))}, R² {fmt(full_model.get('r2'),3)}</li>
        </ul>
      </div>

      <div class="grid">
        <div class="card"><div class="pill">Rolling MAE</div><ul>
          <li>ensemble_blend: <b>{fmt(rolling_best.get('mae_mean'))}</b></li>
          <li>runoff_only: <b>{fmt(rolling_runoff.get('mae_mean'))}</b></li>
          <li>event_gated_residual: <b>{fmt(rolling_event.get('mae_mean'))}</b></li>
          <li>full_model: <b>{fmt(rolling_full.get('mae_mean'))}</b></li>
        </ul></div>
        <div class="card"><div class="pill">Static descriptors</div><ul>
          <li>CAMELS-covered stations: <b>{summary.get('static_descriptor_coverage',{}).get('camels_station_count','n/a')}</b></li>
          <li>HydroBASINS-covered stations: <b>{summary.get('static_descriptor_coverage',{}).get('hydrobasins_station_count','n/a')}</b></li>
          <li>No static descriptors: <b>{summary.get('static_descriptor_coverage',{}).get('no_static_station_count','n/a')}</b></li>
        </ul></div>
      </div>

      <div class="card">
        <div class="pill">Per-station test results</div>
        <div class="table-wrap">
          <table>
            <thead>
              <tr><th>Station</th><th>Name</th><th>State</th><th>CAMELS</th><th>HydroBASINS</th><th>Ensemble MAE</th><th>Ensemble RMSE</th><th>Ensemble R²</th><th>Runoff-only MAE</th><th>Rising-flow MAE</th><th>High-flow MAE</th></tr>
            </thead>
            <tbody>
              {''.join(trs)}
            </tbody>
          </table>
        </div>
      </div>

      <div class="card">
        <div class="pill">Artifacts</div>
        <ul>
          <li><code>generated/ml/latest_training_summary.json</code></li>
          <li><code>generated/ml/models/montana_runoff_validation_report.json</code></li>
          <li><code>generated/ml/zero_shot_pages_summary.json</code></li>
        </ul>
      </div>
    </main>
  </body>
</html>
'''
OUT.write_text(html_doc)
print(f'wrote {OUT}')
print(f'wrote {DATA_OUT}')
