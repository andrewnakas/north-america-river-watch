# North America River Watch

Leaflet map of river sensors across the public sources I could reliably wire together for a static GitHub Pages deployment:

- **USGS Water Services** for U.S. stream gauges and live/historical series
- **NOAA National Water Prediction Service (NWPS)** for U.S. river forecast enrichment
- **Environment and Climate Change Canada GeoMet** for Canadian hydrometric stations and live/historical series

## What it does

- Loads a unified river-sensor index for the U.S. and Canada
- Displays all indexed sensors on a Leaflet map with clustering
- Lets you click a marker or click anywhere on the map to snap to the nearest river sensor
- Shows current and recent historical river data
- Shows NOAA forecast series when a U.S. gauge can be matched to NWPS
- Publishes automatically to GitHub Pages via Actions

## Why the data pipeline is split

USGS and Environment Canada expose browser-friendly CORS headers, so the Pages app can fetch live detail on demand.

NOAA NWPS does **not** expose permissive browser CORS headers for the endpoints needed here, so GitHub Actions prefetches forecast JSON into static files during each build.

## Run locally

```bash
npm install
npm run build
python3 -m http.server -d dist 8080
```

Then open <http://localhost:8080>.

## Montana runoff ML baseline

A first-pass Montana runoff training pipeline is included:

```bash
npm run build:data
npm run ml:venv
npm run ml:train:montana
```

See `ML_NOTES.md` for the modeling approach and output artifacts.

## Notes

- “North America” coverage is strongest for the U.S. and Canada because those are the comprehensive, public, API-accessible national sources available for a static site.
- Mexico can be added later if we identify a stable nationwide public API with usable licensing and browser/server access constraints.
