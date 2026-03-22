# Montana runoff ML notes

This project now includes a first-pass **Montana runoff validation pipeline**.

## Goal

Predict **next-day runoff/discharge** for Montana USGS stream gauges using:

- **Historical runoff** from USGS NWIS daily discharge (`00060`)
- **Historical weather** from **Dynamical NOAA GEFS analysis**
- **Snowpack / mountain forcing** from nearby **NRCS SNOTEL** stations
- **Snow depth** from nearby **NOHRSC** daily station reports

## Why this shape

A good hydrologic forecast model needs more than river lags:

- recent runoff persistence
- precipitation forcing
- temperature / melt conditions
- snowpack state (SWE + snow depth)

This baseline is intentionally transparent and easy to audit before moving to a heavier model.

## Data wiring

### Runoff target
- USGS NWIS Daily Values API
- parameter `00060` = discharge
- target is `t + 1 day`

### Historical weather
- Dynamical catalog: `NOAA GEFS analysis`
- archive spans 2000-present and represents past weather reconstructed from historical forecasts
- script opens the Zarr archive and aggregates the nearest grid cell to daily features

### SNOTEL
- Uses NRCS report generator daily station CSVs
- Pulls:
  - `WTEQ`
  - `SNWD`
  - `PREC`
  - `TAVG`
  - `TMAX`
  - `TMIN`

### NOHRSC
- Uses daily station reports and extracts nearest station values
- Currently focuses on snow depth, plus SWE/snowfall when available

## Commands

```bash
npm run build:data
npm run ml:venv
npm run ml:train:montana
```

## Outputs

Written under `generated/ml/`:

- `datasets/montana_runoff_training.parquet`
- `datasets/montana_runoff_training_sample.csv`
- `models/montana_runoff_ridge.pkl`
- `models/montana_runoff_validation_report.json`
- `latest_training_summary.json`

## Current limitations

- nearest-station joins are simple geographic joins
- Montana only for now
- baseline model is `Ridge`, chosen for speed + interpretability
- no basin polygons / upstream area weighting yet
- no forecast inference path yet from real future weather forcings

## Next upgrades

1. basin-aware SNOTEL/NOHRSC weighting instead of nearest point
2. more gauges and longer training windows
3. add snowmelt features and degree-day accumulation
4. train per-basin or per-cluster models
5. add future inference using forecast forcings and publish forecast artifacts
