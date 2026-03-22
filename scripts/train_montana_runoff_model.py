#!/usr/bin/env python3
"""Build a first-pass Montana river runoff training set and train a validation model.

Data sources
- USGS NWIS daily discharge (target + lagged runoff history)
- Dynamical NOAA GEFS analysis archive (historical weather)
- NRCS SNOTEL daily station reports (SWE, snow depth, precip, temperature)
- NOAA NOHRSC daily station snow-depth reports

This is intentionally a pragmatic baseline pipeline:
- Montana only
- USGS stream gauges only
- one nearest SNOTEL and one nearest NOHRSC station per gauge
- 1-day-ahead runoff prediction
- regularized linear regression for a transparent baseline

Outputs land in generated/ml/.
"""

from __future__ import annotations

import csv
import json
import math
import os
import pathlib
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Dict, List, Optional, Sequence, Tuple
from urllib.parse import urljoin

import numpy as np
import pandas as pd
import requests
import xarray as xr
from bs4 import BeautifulSoup
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = pathlib.Path(__file__).resolve().parents[1]
GENERATED = ROOT / "generated"
ML_DIR = GENERATED / "ml"
CACHE_DIR = ML_DIR / "cache"
DATASET_DIR = ML_DIR / "datasets"
MODEL_DIR = ML_DIR / "models"

USER_AGENT = "north-america-river-watch-ml/0.1"
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": USER_AGENT})

DYNAMICAL_EMAIL = os.environ.get("DYNAMICAL_EMAIL", "treesixtyweather@gmail.com")
STATION_LIMIT = int(os.environ.get("MT_STATION_LIMIT", "8"))
LOOKBACK_DAYS = int(os.environ.get("MT_LOOKBACK_DAYS", "180"))
TARGET_STATE = os.environ.get("MT_TARGET_STATE", "MT")
TARGET_HORIZON_DAYS = int(os.environ.get("MT_TARGET_HORIZON_DAYS", "1"))
MAX_SNOTEL_KM = float(os.environ.get("MT_MAX_SNOTEL_KM", "80"))
MAX_NOHRSC_KM = float(os.environ.get("MT_MAX_NOHRSC_KM", "80"))


@dataclass
class Neighbor:
    station_id: str
    name: str
    distance_km: float
    latitude: float
    longitude: float
    meta: dict


def ensure_dirs() -> None:
    for path in [ML_DIR, CACHE_DIR, DATASET_DIR, MODEL_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def read_json(path: pathlib.Path):
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def write_json(path: pathlib.Path, data, pretty: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2 if pretty else None, default=str)


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def cache_json_path(name: str) -> pathlib.Path:
    return CACHE_DIR / name


def fetch_usgs_daily_discharge(site_no: str, start: date, end: date) -> pd.DataFrame:
    cache_path = cache_json_path(f"usgs_{site_no}_{start.isoformat()}_{end.isoformat()}.json")
    if cache_path.exists():
        payload = read_json(cache_path)
    else:
        url = (
            "https://waterservices.usgs.gov/nwis/dv/?format=json"
            f"&sites={site_no}&parameterCd=00060&startDT={start.isoformat()}&endDT={end.isoformat()}"
        )
        payload = SESSION.get(url, timeout=60).json()
        write_json(cache_path, payload, pretty=False)

    rows = []
    for ts in payload.get("value", {}).get("timeSeries", []):
        for values in ts.get("values", []):
            for item in values.get("value", []):
                try:
                    discharge = float(item.get("value"))
                except Exception:
                    continue
                rows.append({
                    "date": item.get("dateTime", "")[:10],
                    "discharge_cfs": discharge,
                    "qualifiers": ",".join(item.get("qualifiers", [])),
                })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"])
    return df.groupby("date", as_index=False).agg({"discharge_cfs": "mean"})


def load_mt_river_stations() -> List[dict]:
    stations_path = GENERATED / "stations.json"
    stations = read_json(stations_path)
    mt = [
        s for s in stations
        if s.get("network") == "USGS" and s.get("state") == TARGET_STATE and s.get("latitude") is not None and s.get("longitude") is not None
    ]
    mt.sort(key=lambda s: (not bool(s.get("noaaForecast")), s.get("stationId")))
    return mt[:STATION_LIMIT]


def load_snotel_metadata() -> List[dict]:
    cache_path = cache_json_path("snotel_metadata_mt.json")
    if cache_path.exists():
        return read_json(cache_path)

    url = 'https://wcc.sc.egov.usda.gov/reportGenerator/view_csv/customMultipleStationReport/daily/start_of_period/network=%22SNTL%22%7Cname/0,0/stationId,name,state,latitude,longitude,elevation?fitToScreen=false'
    resp = SESSION.get(url, timeout=60)
    resp.raise_for_status()
    lines = [ln for ln in resp.text.splitlines() if ln.strip() and not ln.startswith('#')]
    stations = []
    reader = csv.DictReader(lines)
    for row in reader:
        if row.get('state') != TARGET_STATE:
            continue
        try:
            stations.append({
                'id': str(row.get('stationId')).strip(),
                'name': row.get('name'),
                'state': row.get('state'),
                'latitude': float(row.get('latitude')) if row.get('latitude') else None,
                'longitude': float(row.get('longitude')) if row.get('longitude') else None,
                'elevation': row.get('elevation'),
            })
        except Exception:
            continue
    write_json(cache_path, stations)
    return stations


def load_nohrsc_metadata(anchor_day: date) -> List[dict]:
    stations = []
    seen = set()
    # Build a candidate list from recent daily reports so the repo is self-contained.
    for back in range(0, 7):
        day = anchor_day - timedelta(days=back)
        try:
            data = fetch_nohrsc_daily_region(day)
        except Exception:
            continue
        for s in data:
            sid = str(s.get('station_id'))
            if sid in seen:
                continue
            lat = s.get('lat')
            lon = s.get('lon')
            if lat is None or lon is None:
                continue
            if 44 <= lat <= 49.2 and -116.5 <= lon <= -103.8:
                seen.add(sid)
                stations.append(s)
    write_json(cache_json_path('nohrsc_metadata_mt.json'), stations)
    return stations


def nearest_station(lat: float, lon: float, stations: Sequence[dict], max_km: float, id_key: str = "id", lat_key: str = "latitude", lon_key: str = "longitude") -> Optional[Neighbor]:
    best = None
    best_dist = 1e18
    for s in stations:
        s_lat = s.get(lat_key)
        s_lon = s.get(lon_key)
        if s_lat is None or s_lon is None:
            continue
        d = haversine_km(lat, lon, s_lat, s_lon)
        if d < best_dist:
            best_dist = d
            best = s
    if not best or best_dist > max_km:
        return None
    return Neighbor(
        station_id=str(best.get(id_key)),
        name=best.get("name") or best.get("station_id") or str(best.get(id_key)),
        distance_km=best_dist,
        latitude=float(best.get(lat_key)),
        longitude=float(best.get(lon_key)),
        meta=best,
    )


def fetch_snotel_daily(triplet: str, start: date, end: date) -> pd.DataFrame:
    safe = triplet.replace(":", "_")
    cache_path = cache_json_path(f"snotel_{safe}_{start.isoformat()}_{end.isoformat()}.csv")
    if cache_path.exists():
        text = cache_path.read_text(encoding="utf-8")
    else:
        elements = "WTEQ::value,SNWD::value,PREC::value,TAVG::value,TMAX::value,TMIN::value"
        url = f"https://wcc.sc.egov.usda.gov/reportGenerator/view_csv/customSingleStationReport/daily/{triplet}%7Cname/{start.isoformat()},{end.isoformat()}/{elements}"
        resp = SESSION.get(url, timeout=60)
        resp.raise_for_status()
        text = resp.text
        cache_path.write_text(text, encoding="utf-8")

    lines = [ln for ln in text.splitlines() if ln.strip() and not ln.startswith("#")]
    if len(lines) < 2:
        return pd.DataFrame()

    reader = csv.DictReader(lines)
    rows = []
    for row in reader:
        normalized = {}
        for key, value in row.items():
            lk = (key or "").lower()
            val = value.strip() if isinstance(value, str) else value
            if "date" in lk:
                normalized["date"] = val
            elif "water equivalent" in lk or "wteq" in lk:
                normalized["snotel_wteq_in"] = float(val) if val not in (None, "") else np.nan
            elif "snow depth" in lk or "snwd" in lk:
                normalized["snotel_snwd_in"] = float(val) if val not in (None, "") else np.nan
            elif "precipitation" in lk or "prec" in lk:
                normalized["snotel_prec_in"] = float(val) if val not in (None, "") else np.nan
            elif "average air temperature" in lk or "tavg" in lk:
                normalized["snotel_tavg_f"] = float(val) if val not in (None, "") else np.nan
            elif "maximum air temperature" in lk or "tmax" in lk:
                normalized["snotel_tmax_f"] = float(val) if val not in (None, "") else np.nan
            elif "minimum air temperature" in lk or "tmin" in lk:
                normalized["snotel_tmin_f"] = float(val) if val not in (None, "") else np.nan
        if normalized.get("date"):
            rows.append(normalized)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date")


def _to_float(value):
    if value is None:
        return None
    text = str(value).strip().replace(",", "")
    filtered = "".join(ch for ch in text if ch.isdigit() or ch in ".-")
    if filtered in {"", ".", "-", "-."}:
        return None
    try:
        return float(filtered)
    except Exception:
        return None


def _inch_to_cm(value):
    return None if value is None else value * 2.54


def _inch_to_mm(value):
    return None if value is None else value * 25.4


def _feet_to_m(value):
    return None if value is None else value * 0.3048


def _find_nohrsc_report_links(html: str, base_url: str):
    soup = BeautifulSoup(html, "lxml")
    links = {"snowfall": None, "swe": None, "snowdepth": None}
    for a in soup.find_all("a"):
        text = (a.get_text(strip=True) or "").lower()
        href = a.get("href")
        if not href:
            continue
        full = urljoin(base_url, href)
        if "station" in text and "snowfall" in text:
            links["snowfall"] = full
        elif "station" in text and "snow water equivalent" in text:
            links["swe"] = full
        elif "station" in text and ("snowdepth" in text or "snow depth" in text):
            links["snowdepth"] = full
    return links


def _find_nohrsc_english_text_link(html: str, base_url: str):
    soup = BeautifulSoup(html, "lxml")
    for a in soup.find_all("a"):
        text = (a.get_text(strip=True) or "").lower()
        if "text file" in text and "english" in text:
            href = a.get("href")
            if href:
                return urljoin(base_url, href)
    return None


def _parse_nohrsc_text_rows(text: str):
    rows = []
    headers = None
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("!"):
            continue
        parts = [p.strip() for p in line.split("|")]
        if parts and parts[-1] == "":
            parts = parts[:-1]
        if not parts:
            continue
        if headers is None and "Station_Id" in parts[0]:
            headers = parts
            continue
        if headers is None:
            continue
        if len(parts) < len(headers):
            parts += [""] * (len(headers) - len(parts))
        elif len(parts) > len(headers):
            parts = parts[: len(headers)]
        rows.append(dict(zip(headers, parts)))
    return rows


def fetch_nohrsc_daily_region(day: date) -> List[dict]:
    cache_path = cache_json_path(f"nohrsc_{day.isoformat()}.json")
    if cache_path.exists():
        return read_json(cache_path)

    index_url = f"https://www.nohrsc.noaa.gov/nsa/index.html?region=National&var=swe&dy={day.year}&dm={day.month}&dd={day.day}&units=e"
    index_html = SESSION.get(index_url, timeout=60).text
    report_links = _find_nohrsc_report_links(index_html, index_url)

    fused = {}
    for kind in ["snowfall", "swe", "snowdepth"]:
        page_url = report_links.get(kind)
        if not page_url:
            continue
        page_html = SESSION.get(page_url, timeout=60).text
        text_url = _find_nohrsc_english_text_link(page_html, page_url)
        if not text_url:
            continue
        txt = SESSION.get(text_url, timeout=60).text
        for row in _parse_nohrsc_text_rows(txt):
            sid = str(row.get("Station_Id") or row.get("StationID") or "").strip()
            if not sid:
                continue
            item = fused.setdefault(sid, {
                "station_id": sid,
                "name": row.get("Name"),
                "lat": _to_float(row.get("Latitude")),
                "lon": _to_float(row.get("Longitude")),
                "elev_m": _feet_to_m(_to_float(row.get("Elevation"))),
                "timestamp": row.get("DateTime_Report(UTC)"),
                "snowfall_cm": None,
                "swe_mm": None,
                "snowdepth_cm": None,
                "source": "nohrsc",
            })
            amount_in = _to_float(row.get("Amount"))
            if kind == "snowfall":
                item["snowfall_cm"] = _inch_to_cm(amount_in)
            elif kind == "swe":
                item["swe_mm"] = _inch_to_mm(amount_in)
            elif kind == "snowdepth":
                item["snowdepth_cm"] = _inch_to_cm(amount_in)

    data = list(fused.values())
    write_json(cache_path, data, pretty=False)
    return data


def build_nohrsc_series(neighbor: Neighbor, start: date, end: date) -> pd.DataFrame:
    rows = []
    current = start
    while current <= end:
        try:
            daily = fetch_nohrsc_daily_region(current)
        except Exception:
            daily = []
        matched = None
        for row in daily:
            if str(row.get("station_id")) == str(neighbor.station_id):
                matched = row
                break
        if matched:
            rows.append({
                "date": current.isoformat(),
                "nohrsc_snowdepth_cm": matched.get("snowdepth_cm"),
                "nohrsc_swe_mm": matched.get("swe_mm"),
                "nohrsc_snowfall_cm": matched.get("snowfall_cm"),
            })
        current += timedelta(days=1)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date")


def nearest_grid_indices(values: np.ndarray, target: float) -> int:
    return int(np.abs(values - target).argmin())


def fetch_dynamical_weather(lat: float, lon: float, start: date, end: date) -> pd.DataFrame:
    cache_path = cache_json_path(f"dynamical_{lat:.4f}_{lon:.4f}_{start.isoformat()}_{end.isoformat()}.parquet")
    if cache_path.exists():
        return pd.read_parquet(cache_path)

    url = f"https://data.dynamical.org/noaa/gefs/analysis/latest.zarr?email={DYNAMICAL_EMAIL}"
    ds = xr.open_zarr(url, consolidated=False)
    lon_target = lon if lon >= -180 else lon + 360
    time_slice = slice(np.datetime64(start.isoformat()), np.datetime64((end + timedelta(days=1)).isoformat()))

    lat_idx = nearest_grid_indices(ds["latitude"].values, lat)
    lon_vals = ds["longitude"].values
    # dataset advertises -180..179.75; guard both conventions
    lon_idx = nearest_grid_indices(lon_vals, lon if np.nanmin(lon_vals) < 0 else lon_target)

    subset = ds[[
        "temperature_2m",
        "maximum_temperature_2m",
        "minimum_temperature_2m",
        "precipitation_surface",
        "categorical_snow_surface",
        "relative_humidity_2m",
        "total_cloud_cover_atmosphere",
    ]].isel(latitude=lat_idx, longitude=lon_idx).sel(time=time_slice).load()

    frame = subset.to_dataframe().reset_index().sort_values("time")
    if frame.empty:
        return frame
    frame["date"] = frame["time"].dt.floor("D")
    if "precipitation_surface" in frame:
        frame["precip_mm_step"] = frame["precipitation_surface"].astype(float).fillna(0) * 3 * 3600
    else:
        frame["precip_mm_step"] = 0.0

    daily = frame.groupby("date", as_index=False).agg({
        "temperature_2m": "mean",
        "maximum_temperature_2m": "max",
        "minimum_temperature_2m": "min",
        "precip_mm_step": "sum",
        "categorical_snow_surface": "max",
        "relative_humidity_2m": "mean",
        "total_cloud_cover_atmosphere": "mean",
    })
    daily = daily.rename(columns={
        "temperature_2m": "dyn_temp_c_mean",
        "maximum_temperature_2m": "dyn_temp_c_max",
        "minimum_temperature_2m": "dyn_temp_c_min",
        "categorical_snow_surface": "dyn_snow_flag",
        "relative_humidity_2m": "dyn_rh_pct_mean",
        "total_cloud_cover_atmosphere": "dyn_cloud_pct_mean",
        "precip_mm_step": "dyn_precip_mm",
    })
    daily.to_parquet(cache_path, index=False)
    return daily


def add_lag_features(df: pd.DataFrame, column: str, prefix: str) -> pd.DataFrame:
    out = df.copy()
    out[f"{prefix}_lag1"] = out[column].shift(1)
    out[f"{prefix}_lag2"] = out[column].shift(2)
    out[f"{prefix}_lag3"] = out[column].shift(3)
    out[f"{prefix}_lag7_mean"] = out[column].shift(1).rolling(7).mean()
    out[f"{prefix}_lag14_mean"] = out[column].shift(1).rolling(14).mean()
    return out


def assemble_station_dataset(station: dict, snotel_neighbor: Optional[Neighbor], nohrsc_neighbor: Optional[Neighbor], start: date, end: date) -> pd.DataFrame:
    runoff = fetch_usgs_daily_discharge(station["stationId"], start, end + timedelta(days=TARGET_HORIZON_DAYS))
    if runoff.empty:
        return runoff
    runoff = runoff.sort_values("date")

    weather = fetch_dynamical_weather(station["latitude"], station["longitude"], start, end)

    frames = [runoff, weather]
    if snotel_neighbor:
        triplet = f"{snotel_neighbor.station_id}:{TARGET_STATE}:SNTL"
        snotel = fetch_snotel_daily(triplet, start, end)
        if not snotel.empty:
            frames.append(snotel)
    if nohrsc_neighbor:
        nohrsc = build_nohrsc_series(nohrsc_neighbor, start, end)
        if not nohrsc.empty:
            frames.append(nohrsc)

    df = frames[0]
    for frame in frames[1:]:
        df = df.merge(frame, on="date", how="left")

    df = add_lag_features(df, "discharge_cfs", "q")
    if "dyn_precip_mm" in df:
        df = add_lag_features(df, "dyn_precip_mm", "dyn_precip")
    if "nohrsc_snowdepth_cm" in df:
        df = add_lag_features(df, "nohrsc_snowdepth_cm", "nohrsc_sd")
    if "snotel_wteq_in" in df:
        df = add_lag_features(df, "snotel_wteq_in", "snotel_wteq")

    df["target_discharge_cfs"] = df["discharge_cfs"].shift(-TARGET_HORIZON_DAYS)
    df["station_id"] = station["stationId"]
    df["station_name"] = station["name"]
    df["latitude"] = station["latitude"]
    df["longitude"] = station["longitude"]
    df["state"] = station["state"]
    df["snotel_station_id"] = snotel_neighbor.station_id if snotel_neighbor else None
    df["snotel_distance_km"] = round(snotel_neighbor.distance_km, 2) if snotel_neighbor else None
    df["nohrsc_station_id"] = nohrsc_neighbor.station_id if nohrsc_neighbor else None
    df["nohrsc_distance_km"] = round(nohrsc_neighbor.distance_km, 2) if nohrsc_neighbor else None
    return df


def train_model(dataset: pd.DataFrame) -> Tuple[Pipeline, dict, List[str]]:
    numeric_features = [
        c for c in dataset.columns
        if c not in {
            "date", "station_id", "station_name", "state", "target_discharge_cfs", "snotel_station_id", "nohrsc_station_id"
        }
        and pd.api.types.is_numeric_dtype(dataset[c])
    ]

    dataset = dataset.dropna(subset=["target_discharge_cfs"]).sort_values("date")
    if len(dataset) < 30:
        raise RuntimeError(f"Not enough rows to train after filtering: {len(dataset)}")

    split_idx = max(1, int(len(dataset) * 0.8))
    train = dataset.iloc[:split_idx].copy()
    test = dataset.iloc[split_idx:].copy()

    pre = ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]), numeric_features)
        ],
        remainder="drop",
    )

    model = Pipeline([
        ("pre", pre),
        ("reg", Ridge(alpha=1.0)),
    ])
    model.fit(train[numeric_features], train["target_discharge_cfs"])
    pred_train = model.predict(train[numeric_features])
    pred_test = model.predict(test[numeric_features])

    metrics = {
        "rows_total": int(len(dataset)),
        "rows_train": int(len(train)),
        "rows_test": int(len(test)),
        "target_horizon_days": TARGET_HORIZON_DAYS,
        "mae_train": float(mean_absolute_error(train["target_discharge_cfs"], pred_train)),
        "rmse_train": float(math.sqrt(mean_squared_error(train["target_discharge_cfs"], pred_train))),
        "r2_train": float(r2_score(train["target_discharge_cfs"], pred_train)),
        "mae_test": float(mean_absolute_error(test["target_discharge_cfs"], pred_test)),
        "rmse_test": float(math.sqrt(mean_squared_error(test["target_discharge_cfs"], pred_test))),
        "r2_test": float(r2_score(test["target_discharge_cfs"], pred_test)) if len(test) > 1 else None,
    }

    test_out = test[["date", "station_id", "station_name", "discharge_cfs", "target_discharge_cfs"]].copy()
    test_out["prediction_cfs"] = pred_test
    test_out["abs_error_cfs"] = (test_out["prediction_cfs"] - test_out["target_discharge_cfs"]).abs()
    return model, {"metrics": metrics, "test_predictions": test_out}, numeric_features


def save_model_bundle(model: Pipeline, features: Sequence[str], report: dict, station_links: List[dict]) -> None:
    import pickle

    with (MODEL_DIR / "montana_runoff_ridge.pkl").open("wb") as fh:
        pickle.dump(model, fh)

    report_payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "weather_source": "Dynamical NOAA GEFS analysis (historical archive)",
        "snow_sources": ["NRCS SNOTEL daily station reports", "NOHRSC daily station reports"],
        "runoff_source": "USGS NWIS daily values parameterCd=00060",
        "feature_columns": list(features),
        "station_links": station_links,
        "metrics": report["metrics"],
        "test_predictions": json.loads(report["test_predictions"].to_json(orient="records", date_format="iso")),
        "notes": [
            "Baseline 1-day-ahead model for Montana gauges.",
            "Nearest SNOTEL and nearest NOHRSC stations are used as proxy snowpack signals.",
            "Dynamical precipitation is aggregated from GEFS analysis precipitation rate to daily mm.",
            "This is a validation starter, not yet a production hydrologic forecast stack."
        ]
    }
    write_json(MODEL_DIR / "montana_runoff_validation_report.json", report_payload)


def main() -> None:
    ensure_dirs()
    end = date.today() - timedelta(days=1)
    start = end - timedelta(days=LOOKBACK_DAYS - 1)

    mt_stations = load_mt_river_stations()
    if not mt_stations:
        raise SystemExit("No Montana USGS stations found in generated/stations.json. Run npm run build:data first.")

    snotel_meta = load_snotel_metadata()
    nohrsc_meta = load_nohrsc_metadata(end)

    station_frames = []
    station_links = []
    for station in mt_stations:
        snotel_neighbor = nearest_station(station["latitude"], station["longitude"], snotel_meta, MAX_SNOTEL_KM)
        nohrsc_neighbor = nearest_station(station["latitude"], station["longitude"], nohrsc_meta, MAX_NOHRSC_KM, lat_key="lat", lon_key="lon")
        print(f"Station {station['stationId']} {station['name']}")
        print(f"  nearest SNOTEL: {snotel_neighbor.station_id if snotel_neighbor else 'none'}")
        print(f"  nearest NOHRSC: {nohrsc_neighbor.station_id if nohrsc_neighbor else 'none'}")
        frame = assemble_station_dataset(station, snotel_neighbor, nohrsc_neighbor, start, end)
        if frame.empty:
            continue
        station_frames.append(frame)
        station_links.append({
            "station_id": station["stationId"],
            "station_name": station["name"],
            "snotel_station_id": snotel_neighbor.station_id if snotel_neighbor else None,
            "snotel_distance_km": round(snotel_neighbor.distance_km, 2) if snotel_neighbor else None,
            "nohrsc_station_id": nohrsc_neighbor.station_id if nohrsc_neighbor else None,
            "nohrsc_distance_km": round(nohrsc_neighbor.distance_km, 2) if nohrsc_neighbor else None,
        })

    if not station_frames:
        raise SystemExit("No station datasets could be assembled.")

    dataset = pd.concat(station_frames, ignore_index=True).sort_values(["date", "station_id"])
    dataset_path = DATASET_DIR / "montana_runoff_training.parquet"
    dataset.to_parquet(dataset_path, index=False)
    dataset.head(100).to_csv(DATASET_DIR / "montana_runoff_training_sample.csv", index=False)

    model, report, features = train_model(dataset)
    save_model_bundle(model, features, report, station_links)

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "date_range": {"start": start.isoformat(), "end": end.isoformat()},
        "stations_used": len(station_frames),
        "rows": int(len(dataset)),
        "metrics": report["metrics"],
    }
    write_json(ML_DIR / "latest_training_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
