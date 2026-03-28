#!/usr/bin/env python3
"""Build a stronger Montana river runoff training set and train validation models.

Data sources
- USGS NWIS daily discharge (target + lagged runoff history)
- Dynamical NOAA GEFS analysis archive (historical weather)
- NRCS SNOTEL daily station reports (SWE, snow depth, precip, temperature)
- NOAA NOHRSC daily station snow-depth reports

This remains a pragmatic, inspectable pipeline, but now adds:
- snowmelt / runoff-response features
- benchmark checks versus simpler baselines
- rolling time-split validation
"""

from __future__ import annotations

import csv
import json
import math
import os
import pathlib
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from urllib.parse import urljoin
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import requests
import xarray as xr
from bs4 import BeautifulSoup

try:
    import s3fs
    import zarr
    from pyproj import Transformer
except Exception:  # optional for HRRR forecast forcing
    s3fs = None
    zarr = None
    Transformer = None
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = pathlib.Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = ROOT.parent
GENERATED = ROOT / "generated"
ML_DIR = GENERATED / "ml"
CACHE_DIR = ML_DIR / "cache"
DATASET_DIR = ML_DIR / "datasets"
MODEL_DIR = ML_DIR / "models"
TREESIXTY_SNOTEL_PATH = WORKSPACE_ROOT / "TreesixtyFirebase" / "public" / "data" / "snotel-stations.json"

USER_AGENT = "north-america-river-watch-ml/0.2"
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": USER_AGENT})

DYNAMICAL_EMAIL = os.environ.get("DYNAMICAL_EMAIL", "treesixtyweather@gmail.com")
STATION_LIMIT = int(os.environ.get("MT_STATION_LIMIT", "8"))
STATION_CANDIDATE_MULTIPLIER = int(os.environ.get("MT_STATION_CANDIDATE_MULTIPLIER", "5"))
LOOKBACK_DAYS = int(os.environ.get("MT_LOOKBACK_DAYS", "180"))
TARGET_STATE = os.environ.get("MT_TARGET_STATE", "MT")
TARGET_HORIZON_DAYS = int(os.environ.get("MT_TARGET_HORIZON_DAYS", "1"))
MAX_SNOTEL_KM = float(os.environ.get("MT_MAX_SNOTEL_KM", "80"))
MAX_NOHRSC_KM = float(os.environ.get("MT_MAX_NOHRSC_KM", "80"))
FORCE_OPENMETEO_WEATHER = os.environ.get("MT_FORCE_OPENMETEO_WEATHER", "0") == "1"
TARGET_GROUP = os.environ.get("MT_TARGET_GROUP", "").strip().lower()
STATION_IDS_FILTER = [sid.strip() for sid in os.environ.get("MT_STATION_IDS", "").split(",") if sid.strip()]
DEGREE_DAY_BASE_C = float(os.environ.get("MT_DEGREE_DAY_BASE_C", "0.0"))
ROLLING_VALIDATION_SPLITS = int(os.environ.get("MT_ROLLING_VALIDATION_SPLITS", "3"))
MIN_VALIDATION_DAYS = int(os.environ.get("MT_MIN_VALIDATION_DAYS", "21"))
TEMP_LAPSE_C_PER_KM = float(os.environ.get("MT_TEMP_LAPSE_C_PER_KM", "6.5"))
HRRR_PROJ4 = "+proj=lcc +a=6371200.0 +b=6371200.0 +lon_0=262.5 +lat_0=38.5 +lat_1=38.5 +lat_2=38.5"

TARGET_GROUP_STATION_IDS = {
    "yellowstone-gallatin": [
        "06043120",
        "06043500",
        "06048650",
        "06048700",
        "06050000",
        "06052500",
        "06191500",
        "06192500",
        "06195600",
    ]
}


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


def _to_meters(value):
    if value in (None, "", "nan"):
        return None
    if isinstance(value, (int, float)):
        val = float(value)
    else:
        raw = str(value).strip().lower().replace(",", "")
        for suffix in ["feet", "foot", "ft", "meters", "meter", "m"]:
            raw = raw.replace(suffix, "")
        try:
            val = float(raw)
        except Exception:
            return None
    if abs(val) > 10000:
        return None
    # SNOTEL metadata often comes in feet; NOHRSC already in meters.
    return val * 0.3048 if val > 2500 else val


def fetch_location_elevation_m(lat: float, lon: float) -> Optional[float]:
    cache_path = cache_json_path(f"elev_{lat:.4f}_{lon:.4f}.json")
    if cache_path.exists():
        cached = read_json(cache_path)
        return cached.get("elevation_m") if isinstance(cached, dict) else cached
    try:
        url = f"https://api.open-meteo.com/v1/elevation?latitude={lat}&longitude={lon}"
        resp = SESSION.get(url, timeout=30)
        resp.raise_for_status()
        payload = resp.json()
        elev = None
        if isinstance(payload.get("elevation"), list):
            elev = payload.get("elevation", [None])[0]
        else:
            elev = payload.get("elevation")
        elev = None if elev is None else float(elev)
        write_json(cache_path, {"elevation_m": elev}, pretty=False)
        return elev
    except Exception:
        return None


def adjust_temp_for_elevation(temp_c, source_elev_m, target_elev_m):
    if temp_c is None or pd.isna(temp_c) or source_elev_m is None or target_elev_m is None:
        return np.nan
    return float(temp_c) - ((float(target_elev_m) - float(source_elev_m)) / 1000.0) * TEMP_LAPSE_C_PER_KM


def snow_fraction_from_temp(temp_c):
    if temp_c is None or pd.isna(temp_c):
        return np.nan
    return float(np.clip((1.5 - float(temp_c)) / 3.0, 0.0, 1.0))


def _hrrr_cycle_candidates(now=None, long_only: bool = True):
    now = now or datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    for step in range(0, 36):
        cand = now - timedelta(hours=step)
        if long_only and cand.hour not in (0, 6, 12, 18):
            continue
        yield cand


def fetch_hrrr_forecast(lat: float, lon: float, days: int) -> pd.DataFrame:
    if s3fs is None or zarr is None or Transformer is None:
        raise RuntimeError("HRRR dependencies unavailable")
    cache_path = cache_json_path(f"hrrr_fc_{lat:.4f}_{lon:.4f}_{days}.parquet")
    if cache_path.exists():
        cached = pd.read_parquet(cache_path)
        if not cached.empty:
            return cached

    transformer = Transformer.from_crs("EPSG:4326", HRRR_PROJ4, always_xy=True)
    x_target, y_target = transformer.transform(lon, lat)
    fs = s3fs.S3FileSystem(anon=True)
    last_error = None
    for run_dt in _hrrr_cycle_candidates(long_only=True):
        ymd = run_dt.strftime('%Y%m%d')
        hh = run_dt.strftime('%H')
        store = f"s3://hrrrzarr/sfc/{ymd}/{ymd}_{hh}z_fcst.zarr"
        try:
            def open_var(path, level, var):
                zg = zarr.open_group(fs.get_mapper(store + '/' + path), mode='r')
                data = zg[level][var]
                x = np.asarray(zg['projection_x_coordinate'])
                y = np.asarray(zg['projection_y_coordinate'])
                t = np.asarray(zg['time'])
                return data, x, y, t

            tmp_arr, x_vals, y_vals, time_vals = open_var('2m_above_ground/TMP', '2m_above_ground', 'TMP')
            apcp_arr, _, _, _ = open_var('surface/APCP_1hr_acc_fcst', 'surface', 'APCP_1hr_acc_fcst')
            csnow_arr, _, _, _ = open_var('surface/CSNOW', 'surface', 'CSNOW')
            try:
                asnow_arr, _, _, _ = open_var('surface/ASNOW_1hr_acc_fcst', 'surface', 'ASNOW_1hr_acc_fcst')
            except Exception:
                asnow_arr = None

            x_idx = nearest_grid_indices(x_vals, x_target)
            y_idx = nearest_grid_indices(y_vals, y_target)
            times = pd.to_datetime(time_vals, unit='h', origin=pd.Timestamp('1970-01-01'))
            frame = pd.DataFrame({
                'time': times,
                'temp_c': np.asarray(tmp_arr[:, y_idx, x_idx], dtype='float32') - 273.15,
                'precip_mm': np.asarray(apcp_arr[:, y_idx, x_idx], dtype='float32'),
                'snow_flag': (np.asarray(csnow_arr[:, y_idx, x_idx], dtype='float32') > 0.5).astype(float),
            })
            if asnow_arr is not None:
                frame['snowfall_mm'] = np.asarray(asnow_arr[:, y_idx, x_idx], dtype='float32')
            else:
                frame['snowfall_mm'] = frame['precip_mm'] * frame['snow_flag']
            frame = frame.dropna(subset=['time']).sort_values('time')
            if frame.empty:
                raise RuntimeError('empty HRRR forecast frame')
            daily = frame.assign(date=frame['time'].dt.floor('D')).groupby('date', as_index=False).agg({
                'temp_c': ['mean', 'max', 'min'],
                'precip_mm': 'sum',
                'snowfall_mm': 'sum',
                'snow_flag': 'max',
            })
            daily.columns = ['date', 'dyn_temp_c_mean', 'dyn_temp_c_max', 'dyn_temp_c_min', 'dyn_precip_mm', 'dyn_snowfall_mm', 'dyn_snow_flag']
            daily['dyn_source'] = f'hrrr-fcst-{run_dt.strftime("%Y%m%d%H")}'
            daily = daily.sort_values('date').head(max(2, days))
            daily.to_parquet(cache_path, index=False)
            try:
                fs.close()
            except Exception:
                pass
            return daily
        except Exception as exc:
            last_error = exc
            continue
    try:
        fs.close()
    except Exception:
        pass
    raise RuntimeError(f'HRRR forecast unavailable: {last_error}')


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

                qualifiers = [str(q).strip() for q in item.get("qualifiers", []) if str(q).strip()]
                qualifiers_lower = {q.lower() for q in qualifiers}
                invalid = (
                    not math.isfinite(discharge)
                    or discharge <= -999000
                    or "ice" in qualifiers_lower
                    or "eqp" in qualifiers_lower
                    or "dis" in qualifiers_lower
                )
                rows.append({
                    "date": item.get("dateTime", "")[:10],
                    "discharge_cfs": np.nan if invalid else discharge,
                    "qualifiers": ",".join(qualifiers),
                    "is_valid": not invalid,
                })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"])
    grouped = df.groupby("date", as_index=False).agg({"discharge_cfs": "mean", "is_valid": "max"})
    grouped.loc[~grouped["is_valid"].fillna(False), "discharge_cfs"] = np.nan
    return grouped[["date", "discharge_cfs"]]


def load_mt_river_stations() -> List[dict]:
    stations_path = GENERATED / "stations.json"
    stations = read_json(stations_path)
    mt = [
        s for s in stations
        if s.get("network") == "USGS" and s.get("state") == TARGET_STATE and s.get("latitude") is not None and s.get("longitude") is not None
    ]

    requested_ids = []
    if TARGET_GROUP and TARGET_GROUP in TARGET_GROUP_STATION_IDS:
        requested_ids.extend(TARGET_GROUP_STATION_IDS[TARGET_GROUP])
    if STATION_IDS_FILTER:
        requested_ids.extend(STATION_IDS_FILTER)

    if requested_ids:
        wanted = {sid.strip() for sid in requested_ids if sid.strip()}
        selected = [s for s in mt if str(s.get("stationId")) in wanted]
        selected.sort(key=lambda s: requested_ids.index(str(s.get("stationId"))) if str(s.get("stationId")) in requested_ids else 10**9)
        return selected

    mt.sort(key=lambda s: (not bool(s.get("noaaForecast")), s.get("stationId")))
    if STATION_LIMIT <= 0:
        return mt
    candidate_limit = max(STATION_LIMIT, STATION_LIMIT * max(1, STATION_CANDIDATE_MULTIPLIER))
    return mt[:candidate_limit]


def load_snotel_metadata() -> List[dict]:
    cache_path = cache_json_path("snotel_metadata_mt.json")
    if cache_path.exists():
        cached = read_json(cache_path)
        if cached:
            return cached

    stations: List[dict] = []
    url = 'https://wcc.sc.egov.usda.gov/reportGenerator/view_csv/customMultipleStationReport/daily/start_of_period/network=%22SNTL%22%7Cname/0,0/stationId,name,state,latitude,longitude,elevation?fitToScreen=false'
    try:
        resp = SESSION.get(url, timeout=60)
        resp.raise_for_status()
        lines = [ln for ln in resp.text.splitlines() if ln.strip() and not ln.startswith('#')]
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
                    'source': 'nrcs-report-generator',
                })
            except Exception:
                continue
    except Exception:
        stations = []

    if not stations and TREESIXTY_SNOTEL_PATH.exists():
        try:
            payload = read_json(TREESIXTY_SNOTEL_PATH)
            tree_stations = payload.get('stations', []) if isinstance(payload, dict) else payload
            for row in tree_stations:
                if row.get('state') != TARGET_STATE:
                    continue
                lat = row.get('latitude')
                lon = row.get('longitude')
                if lat is None or lon is None:
                    continue
                stations.append({
                    'id': str(row.get('id')).strip(),
                    'name': row.get('name'),
                    'state': row.get('state'),
                    'latitude': float(lat),
                    'longitude': float(lon),
                    'elevation': row.get('elevation'),
                    'source': 'treesixty-catalog',
                    'is_active': row.get('is_active'),
                    'operation_period': row.get('operation_period'),
                })
        except Exception:
            stations = []

    write_json(cache_path, stations)
    return stations


def load_nohrsc_metadata(anchor_day: date) -> List[dict]:
    stations = []
    seen = set()
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


def fetch_openmeteo_fallback(lat: float, lon: float, start: date, end: date) -> pd.DataFrame:
    url = (
        "https://archive-api.open-meteo.com/v1/archive"
        f"?latitude={lat}&longitude={lon}"
        f"&start_date={start.isoformat()}&end_date={end.isoformat()}"
        "&daily=temperature_2m_mean,temperature_2m_max,temperature_2m_min,precipitation_sum,rain_sum,snowfall_sum"
        "&timezone=UTC"
    )
    resp = SESSION.get(url, timeout=60)
    resp.raise_for_status()
    payload = resp.json()
    daily = payload.get("daily", {})
    if not daily or not daily.get("time"):
        return pd.DataFrame()
    df = pd.DataFrame({
        "date": pd.to_datetime(daily.get("time", [])),
        "dyn_temp_c_mean": daily.get("temperature_2m_mean", []),
        "dyn_temp_c_max": daily.get("temperature_2m_max", []),
        "dyn_temp_c_min": daily.get("temperature_2m_min", []),
        "dyn_precip_mm": daily.get("precipitation_sum", []),
        "dyn_snow_flag": [1 if (v or 0) > 0 else 0 for v in daily.get("snowfall_sum", [])],
    })
    df["dyn_source"] = "open-meteo-fallback"
    return df


def fetch_dynamical_weather(lat: float, lon: float, start: date, end: date) -> pd.DataFrame:
    cache_path = cache_json_path(f"dynamical_{lat:.4f}_{lon:.4f}_{start.isoformat()}_{end.isoformat()}.parquet")
    if cache_path.exists():
        return pd.read_parquet(cache_path)

    if FORCE_OPENMETEO_WEATHER:
        daily = fetch_openmeteo_fallback(lat, lon, start, end)
        daily.to_parquet(cache_path, index=False)
        return daily

    try:
        url = f"https://data.dynamical.org/noaa/gefs/analysis/latest.zarr?email={DYNAMICAL_EMAIL}"
        ds = xr.open_zarr(url, consolidated=None)
        coord_names = set(ds.coords) | set(ds.variables)
        lat_name = "latitude" if "latitude" in coord_names else ("lat" if "lat" in coord_names else None)
        lon_name = "longitude" if "longitude" in coord_names else ("lon" if "lon" in coord_names else None)
        if not lat_name or not lon_name:
            raise RuntimeError("Dynamical dataset missing lat/lon coords")

        lon_target = lon if lon >= -180 else lon + 360
        time_slice = slice(np.datetime64(start.isoformat()), np.datetime64((end + timedelta(days=1)).isoformat()))
        lat_vals = ds[lat_name].values
        lon_vals = ds[lon_name].values
        lat_idx = nearest_grid_indices(lat_vals, lat)
        lon_idx = nearest_grid_indices(lon_vals, lon if np.nanmin(lon_vals) < 0 else lon_target)

        needed_vars = [
            "temperature_2m",
            "maximum_temperature_2m",
            "minimum_temperature_2m",
            "precipitation_surface",
            "categorical_snow_surface",
            "relative_humidity_2m",
            "total_cloud_cover_atmosphere",
        ]
        subset = ds[needed_vars].isel({lat_name: lat_idx, lon_name: lon_idx}).sel(time=time_slice).load()
        frame = subset.to_dataframe().reset_index().sort_values("time")
        if frame.empty:
            raise RuntimeError("Dynamical subset returned no rows")
        frame["date"] = frame["time"].dt.floor("D")
        frame["precip_mm_step"] = frame["precipitation_surface"].astype(float).fillna(0) * 3 * 3600

        daily = frame.groupby("date", as_index=False).agg({
            "temperature_2m": "mean",
            "maximum_temperature_2m": "max",
            "minimum_temperature_2m": "min",
            "precip_mm_step": "sum",
            "categorical_snow_surface": "max",
            "relative_humidity_2m": "mean",
            "total_cloud_cover_atmosphere": "mean",
        }).rename(columns={
            "temperature_2m": "dyn_temp_c_mean",
            "maximum_temperature_2m": "dyn_temp_c_max",
            "minimum_temperature_2m": "dyn_temp_c_min",
            "categorical_snow_surface": "dyn_snow_flag",
            "relative_humidity_2m": "dyn_rh_pct_mean",
            "total_cloud_cover_atmosphere": "dyn_cloud_pct_mean",
            "precip_mm_step": "dyn_precip_mm",
        })
        daily["dyn_source"] = "dynamical-gefs-analysis"
    except Exception as exc:
        print(f"Dynamical weather fallback for {lat},{lon}: {exc}")
        daily = fetch_openmeteo_fallback(lat, lon, start, end)

    daily.to_parquet(cache_path, index=False)
    return daily


def add_lag_features(df: pd.DataFrame, column: str, prefix: str) -> pd.DataFrame:
    out = df.copy()
    shifted = out[column].shift(1)
    out[f"{prefix}_lag1"] = out[column].shift(1)
    out[f"{prefix}_lag2"] = out[column].shift(2)
    out[f"{prefix}_lag3"] = out[column].shift(3)
    out[f"{prefix}_lag7_mean"] = shifted.rolling(7, min_periods=2).mean()
    out[f"{prefix}_lag14_mean"] = shifted.rolling(14, min_periods=3).mean()
    return out


def safe_divide(a, b):
    if isinstance(b, pd.Series):
        return a / b.replace({0: np.nan})
    return a / (np.nan if b == 0 else b)


def add_hydrology_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy().sort_values("date")
    temp_mean = out.get("dyn_temp_c_mean", pd.Series(np.nan, index=out.index))
    temp_max = out.get("dyn_temp_c_max", pd.Series(np.nan, index=out.index))
    temp_min = out.get("dyn_temp_c_min", pd.Series(np.nan, index=out.index))
    precip = out.get("dyn_precip_mm", pd.Series(np.nan, index=out.index)).fillna(0)
    snow_flag = out.get("dyn_snow_flag", pd.Series(np.nan, index=out.index)).fillna(0)

    out["temp_range_c"] = temp_max - temp_min
    out["degree_day_c"] = (temp_mean - DEGREE_DAY_BASE_C).clip(lower=0)
    out["freeze_degree_day_c"] = (DEGREE_DAY_BASE_C - temp_mean).clip(lower=0)
    out["degree_day_3d"] = out["degree_day_c"].rolling(3, min_periods=1).sum()
    out["degree_day_7d"] = out["degree_day_c"].rolling(7, min_periods=1).sum()
    out["freeze_degree_day_7d"] = out["freeze_degree_day_c"].rolling(7, min_periods=1).sum()
    out["warm_day_flag"] = (temp_max > 3.0).astype(float)
    out["hard_freeze_flag"] = (temp_min < -5.0).astype(float)

    out["precip_3d_mm"] = precip.rolling(3, min_periods=1).sum()
    out["precip_7d_mm"] = precip.rolling(7, min_periods=1).sum()
    out["precip_14d_mm"] = precip.rolling(14, min_periods=1).sum()
    out["precip_intensity_ratio"] = safe_divide(out["precip_3d_mm"], out["precip_14d_mm"])
    out["rain_on_warm_day_mm"] = precip * (temp_mean > 0).astype(float)
    out["rain_on_warm_3d_mm"] = out["rain_on_warm_day_mm"].rolling(3, min_periods=1).sum()
    out["snow_event_3d"] = snow_flag.rolling(3, min_periods=1).sum()

    snotel_wteq = out.get("snotel_wteq_in")
    snotel_snwd = out.get("snotel_snwd_in")
    snotel_prec = out.get("snotel_prec_in")
    snotel_tavg_f = out.get("snotel_tavg_f")
    nohrsc_sd = out.get("nohrsc_snowdepth_cm")
    nohrsc_swe = out.get("nohrsc_swe_mm")

    if snotel_tavg_f is not None:
        out["snotel_tavg_c"] = (snotel_tavg_f - 32.0) * (5.0 / 9.0)
        out["snotel_degree_day_c"] = out["snotel_tavg_c"].clip(lower=0)
        out["snotel_degree_day_3d"] = out["snotel_degree_day_c"].rolling(3, min_periods=1).sum()
    if snotel_wteq is not None:
        out["snotel_wteq_change_1d"] = snotel_wteq.diff(1)
        out["snotel_wteq_change_3d"] = snotel_wteq.diff(3)
        out["snotel_wteq_change_7d"] = snotel_wteq.diff(7)
        out["snotel_melt_proxy_in"] = (-out["snotel_wteq_change_1d"]).clip(lower=0)
        out["snotel_melt_proxy_3d"] = out["snotel_melt_proxy_in"].rolling(3, min_periods=1).sum()
        out["snotel_wteq_7d_mean"] = snotel_wteq.rolling(7, min_periods=2).mean()
        out["snotel_wteq_anom"] = snotel_wteq - out["snotel_wteq_7d_mean"]
    if snotel_snwd is not None:
        out["snotel_snwd_change_1d"] = snotel_snwd.diff(1)
    if snotel_prec is not None:
        out["snotel_prec_daily_in"] = snotel_prec.diff(1)
        out["snotel_prec_3d_in"] = out["snotel_prec_daily_in"].rolling(3, min_periods=1).sum()

    if nohrsc_sd is not None:
        out["nohrsc_snowdepth_change_1d"] = nohrsc_sd.diff(1)
        out["nohrsc_snowdepth_change_3d"] = nohrsc_sd.diff(3)
        out["nohrsc_melt_proxy_cm"] = (-out["nohrsc_snowdepth_change_1d"]).clip(lower=0)
        out["nohrsc_melt_proxy_3d"] = out["nohrsc_melt_proxy_cm"].rolling(3, min_periods=1).sum()
    if nohrsc_swe is not None:
        out["nohrsc_swe_change_1d"] = nohrsc_swe.diff(1)

    site_elev = out.get("site_elev_m", pd.Series(np.nan, index=out.index))
    snotel_elev = out.get("snotel_elev_m", pd.Series(np.nan, index=out.index))
    nohrsc_elev = out.get("nohrsc_elev_m", pd.Series(np.nan, index=out.index))
    out["temp_at_snotel_c"] = temp_mean - ((snotel_elev - site_elev) / 1000.0) * TEMP_LAPSE_C_PER_KM
    out["temp_at_nohrsc_c"] = temp_mean - ((nohrsc_elev - site_elev) / 1000.0) * TEMP_LAPSE_C_PER_KM
    out["temp_at_upper_basin_c"] = out[["temp_at_snotel_c", "temp_at_nohrsc_c"]].mean(axis=1, skipna=True)
    out["high_elev_snow_fraction"] = out["temp_at_upper_basin_c"].apply(snow_fraction_from_temp)
    out["basin_snow_mm_elev"] = precip * out["high_elev_snow_fraction"].fillna(snow_flag)
    out["basin_rain_mm_elev"] = precip - out["basin_snow_mm_elev"].fillna(0)
    out["basin_snow_3d_mm_elev"] = out["basin_snow_mm_elev"].rolling(3, min_periods=1).sum()
    out["basin_rain_3d_mm_elev"] = out["basin_rain_mm_elev"].rolling(3, min_periods=1).sum()
    out["warm_upper_basin_flag"] = (out["temp_at_upper_basin_c"] > 0.5).astype(float)
    out["high_elev_melt_pressure_c"] = out["temp_at_upper_basin_c"].clip(lower=0)
    out["high_elev_melt_3d_c"] = out["high_elev_melt_pressure_c"].rolling(3, min_periods=1).sum()

    if "degree_day_c" in out and "snotel_wteq_in" in out:
        out["degree_day_x_swe"] = out["degree_day_c"] * out["snotel_wteq_in"].fillna(0)
    if "rain_on_warm_day_mm" in out and "snotel_wteq_in" in out:
        out["rain_on_snow_proxy"] = out["rain_on_warm_day_mm"] * (out["snotel_wteq_in"].fillna(0) > 0.5).astype(float)
        out["rain_on_snow_proxy_elev"] = out["basin_rain_mm_elev"].fillna(0) * (out["snotel_wteq_in"].fillna(0) > 0.5).astype(float) * out["warm_upper_basin_flag"].fillna(0)
        out["degree_day_x_swe_elev"] = out["high_elev_melt_pressure_c"].fillna(0) * out["snotel_wteq_in"].fillna(0)
    if "degree_day_c" in out and "nohrsc_snowdepth_cm" in out:
        out["degree_day_x_nohrsc_sd"] = out["degree_day_c"] * out["nohrsc_snowdepth_cm"].fillna(0)
        out["degree_day_x_nohrsc_sd_elev"] = out["high_elev_melt_pressure_c"].fillna(0) * out["nohrsc_snowdepth_cm"].fillna(0)
    out["melt_risk_elev"] = out[[c for c in ["degree_day_x_swe_elev", "degree_day_x_nohrsc_sd_elev"] if c in out]].sum(axis=1, min_count=1)

    discharge = out.get("discharge_cfs")
    if discharge is not None:
        out["q_change_1d"] = discharge.diff(1)
        out["q_change_3d"] = discharge.diff(3)
        out["q_ratio_1d"] = safe_divide(discharge, discharge.shift(1))
        out["q_ratio_7d_mean"] = safe_divide(discharge, discharge.shift(1).rolling(7, min_periods=2).mean())
        out["runoff_response_index"] = safe_divide(out.get("q_lag1"), out.get("precip_7d_mm")) if "q_lag1" in out else np.nan

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

    site_elev_m = fetch_location_elevation_m(station["latitude"], station["longitude"])
    snotel_elev_m = _to_meters((snotel_neighbor.meta or {}).get("elevation")) if snotel_neighbor else None
    nohrsc_elev_m = _to_meters((nohrsc_neighbor.meta or {}).get("elev_m")) if nohrsc_neighbor else None
    df["site_elev_m"] = site_elev_m
    df["snotel_elev_m"] = snotel_elev_m
    df["nohrsc_elev_m"] = nohrsc_elev_m
    df["snotel_minus_site_elev_m"] = None if site_elev_m is None or snotel_elev_m is None else snotel_elev_m - site_elev_m
    df["nohrsc_minus_site_elev_m"] = None if site_elev_m is None or nohrsc_elev_m is None else nohrsc_elev_m - site_elev_m

    df = add_lag_features(df, "discharge_cfs", "q")
    if "dyn_precip_mm" in df:
        df = add_lag_features(df, "dyn_precip_mm", "dyn_precip")
    if "nohrsc_snowdepth_cm" in df:
        df = add_lag_features(df, "nohrsc_snowdepth_cm", "nohrsc_sd")
    if "snotel_wteq_in" in df:
        df = add_lag_features(df, "snotel_wteq_in", "snotel_wteq")

    df = add_hydrology_features(df)
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


def feature_columns_for_mode(dataset: pd.DataFrame, mode: str) -> List[str]:
    exclude = {"date", "station_id", "station_name", "state", "target_discharge_cfs", "snotel_station_id", "nohrsc_station_id"}
    numeric_features = [c for c in dataset.columns if c not in exclude and pd.api.types.is_numeric_dtype(dataset[c])]
    if mode == "runoff_only":
        numeric_features = [c for c in numeric_features if c.startswith("q_") or c in {"discharge_cfs", "latitude", "longitude"}]
    return sorted(set(numeric_features))


def build_model(features: Sequence[str]) -> Pipeline:
    pre = ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]), list(features))
        ],
        remainder="drop",
    )
    return Pipeline([
        ("pre", pre),
        ("reg", Ridge(alpha=1.0)),
    ])


def metric_summary(actual: pd.Series, pred: Sequence[float]) -> dict:
    actual_arr = pd.Series(actual).reset_index(drop=True).astype(float)
    pred_arr = pd.Series(pred).reset_index(drop=True).astype(float)
    mask = actual_arr.notna() & pred_arr.notna()
    actual_arr = actual_arr[mask]
    pred_arr = pred_arr[mask]
    if actual_arr.empty:
        return {"rows": 0, "mae": None, "rmse": None, "r2": None}
    rmse = float(math.sqrt(mean_squared_error(actual_arr, pred_arr)))
    return {
        "rows": int(len(actual_arr)),
        "mae": float(mean_absolute_error(actual_arr, pred_arr)),
        "rmse": rmse,
        "r2": float(r2_score(actual_arr, pred_arr)) if len(actual_arr) > 1 else None,
    }


def add_fold_metrics(records: List[dict], label: str, fold_id: str, station: Optional[str], actual: pd.Series, pred: Sequence[float]):
    m = metric_summary(actual, pred)
    if m["rows"] == 0:
        return
    records.append({
        "label": label,
        "fold_id": fold_id,
        "station_id": station,
        **m,
    })


def make_benchmark_predictions(test: pd.DataFrame, train: pd.DataFrame) -> Dict[str, np.ndarray]:
    q_lag1 = test.get("q_lag1", pd.Series(np.nan, index=test.index)).astype(float)
    q_lag7 = test.get("q_lag7_mean", pd.Series(np.nan, index=test.index)).astype(float)

    station_climo_map = train.groupby("station_id")["target_discharge_cfs"].median().to_dict() if not train.empty else {}
    global_climo = float(train["target_discharge_cfs"].median()) if not train.empty else np.nan
    station_climo = test["station_id"].map(station_climo_map).fillna(global_climo).astype(float).to_numpy()

    return {
        "persistence_lag1": q_lag1.to_numpy(),
        "persistence_lag7_mean": q_lag7.to_numpy(),
        "station_median_climatology": station_climo,
    }


def rolling_time_splits(dataset: pd.DataFrame, n_splits: int) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    dates = sorted(pd.to_datetime(dataset["date"]).dropna().unique())
    if len(dates) < max(40, MIN_VALIDATION_DAYS * 2):
        return []
    val_len = max(MIN_VALIDATION_DAYS, len(dates) // (n_splits + 2))
    splits = []
    for i in range(n_splits):
        val_end_idx = len(dates) - (n_splits - i - 1) * val_len - 1
        val_start_idx = max(1, val_end_idx - val_len + 1)
        if val_start_idx <= 0:
            continue
        train_end = dates[val_start_idx - 1]
        val_end = dates[val_end_idx]
        splits.append((pd.Timestamp(train_end), pd.Timestamp(val_end)))
    return splits


def train_model(dataset: pd.DataFrame) -> Tuple[Pipeline, dict, List[str]]:
    dataset = dataset.dropna(subset=["target_discharge_cfs", "discharge_cfs", "q_lag1"]).sort_values(["date", "station_id"]).copy()
    if len(dataset) < 30:
        raise RuntimeError(f"Not enough rows to train after filtering: {len(dataset)}")

    full_features = feature_columns_for_mode(dataset, "full")
    runoff_features = feature_columns_for_mode(dataset, "runoff_only")

    split_idx = max(1, int(len(dataset) * 0.8))
    train = dataset.iloc[:split_idx].copy()
    test = dataset.iloc[split_idx:].copy()

    full_model = build_model(full_features)
    full_model.fit(train[full_features], train["target_discharge_cfs"])
    pred_train = full_model.predict(train[full_features])
    pred_test = full_model.predict(test[full_features])

    runoff_only_model = build_model(runoff_features)
    runoff_only_model.fit(train[runoff_features], train["target_discharge_cfs"])
    pred_test_runoff_only = runoff_only_model.predict(test[runoff_features])

    benchmark_preds = make_benchmark_predictions(test, train)

    metrics = {
        "rows_total": int(len(dataset)),
        "rows_train": int(len(train)),
        "rows_test": int(len(test)),
        "target_horizon_days": TARGET_HORIZON_DAYS,
        "full_model_train": metric_summary(train["target_discharge_cfs"], pred_train),
        "full_model_test": metric_summary(test["target_discharge_cfs"], pred_test),
        "runoff_only_test": metric_summary(test["target_discharge_cfs"], pred_test_runoff_only),
        "benchmarks_test": {name: metric_summary(test["target_discharge_cfs"], values) for name, values in benchmark_preds.items()},
    }
    metrics["benchmark_wins"] = {
        name: {
            "mae_improvement_vs_full": None if metrics["benchmarks_test"][name]["mae"] is None else float(metrics["benchmarks_test"][name]["mae"] - metrics["full_model_test"]["mae"]),
            "rmse_improvement_vs_full": None if metrics["benchmarks_test"][name]["rmse"] is None else float(metrics["benchmarks_test"][name]["rmse"] - metrics["full_model_test"]["rmse"]),
        }
        for name in benchmark_preds
    }

    fold_records: List[dict] = []
    fold_summaries: List[dict] = []
    for idx, (train_end, val_end) in enumerate(rolling_time_splits(dataset, ROLLING_VALIDATION_SPLITS), start=1):
        fold_train = dataset[pd.to_datetime(dataset["date"]) <= train_end].copy()
        fold_test = dataset[(pd.to_datetime(dataset["date"]) > train_end) & (pd.to_datetime(dataset["date"]) <= val_end)].copy()
        if fold_train.empty or fold_test.empty:
            continue

        fold_full = build_model(full_features)
        fold_full.fit(fold_train[full_features], fold_train["target_discharge_cfs"])
        fold_pred_full = fold_full.predict(fold_test[full_features])
        add_fold_metrics(fold_records, "full_model", f"fold_{idx}", None, fold_test["target_discharge_cfs"], fold_pred_full)

        fold_runoff = build_model(runoff_features)
        fold_runoff.fit(fold_train[runoff_features], fold_train["target_discharge_cfs"])
        fold_pred_runoff = fold_runoff.predict(fold_test[runoff_features])
        add_fold_metrics(fold_records, "runoff_only", f"fold_{idx}", None, fold_test["target_discharge_cfs"], fold_pred_runoff)

        fold_bench = make_benchmark_predictions(fold_test, fold_train)
        for name, values in fold_bench.items():
            add_fold_metrics(fold_records, name, f"fold_{idx}", None, fold_test["target_discharge_cfs"], values)

        per_station = []
        for station_id, group in fold_test.groupby("station_id"):
            mask = group.index
            add_fold_metrics(fold_records, "full_model", f"fold_{idx}", str(station_id), group["target_discharge_cfs"], pd.Series(fold_pred_full, index=fold_test.index).loc[mask])
            per_station.append(str(station_id))
        fold_summaries.append({
            "fold_id": f"fold_{idx}",
            "train_end": pd.Timestamp(train_end).date().isoformat(),
            "validation_end": pd.Timestamp(val_end).date().isoformat(),
            "rows_train": int(len(fold_train)),
            "rows_validation": int(len(fold_test)),
            "stations_validation": sorted(set(per_station)),
        })

    fold_df = pd.DataFrame(fold_records)
    rolling_summary = []
    if not fold_df.empty:
        for label, group in fold_df[fold_df["station_id"].isna()].groupby("label"):
            rolling_summary.append({
                "label": label,
                "folds": int(len(group)),
                "mae_mean": float(group["mae"].mean()),
                "rmse_mean": float(group["rmse"].mean()),
                "r2_mean": float(group["r2"].dropna().mean()) if group["r2"].notna().any() else None,
            })

    test_out = test[["date", "station_id", "station_name", "discharge_cfs", "target_discharge_cfs"]].copy()
    test_out["prediction_cfs"] = pred_test
    test_out["runoff_only_prediction_cfs"] = pred_test_runoff_only
    for name, values in benchmark_preds.items():
        test_out[f"benchmark_{name}_cfs"] = values
    test_out["abs_error_cfs"] = (test_out["prediction_cfs"] - test_out["target_discharge_cfs"]).abs()
    test_out["benchmark_persistence_lag1_abs_error_cfs"] = (test_out["benchmark_persistence_lag1_cfs"] - test_out["target_discharge_cfs"]).abs()

    station_test_metrics = []
    for station_id, group in test_out.groupby("station_id"):
        station_test_metrics.append({
            "station_id": str(station_id),
            "station_name": group["station_name"].iloc[0],
            "full_model": metric_summary(group["target_discharge_cfs"], group["prediction_cfs"]),
            "runoff_only": metric_summary(group["target_discharge_cfs"], group["runoff_only_prediction_cfs"]),
            "persistence_lag1": metric_summary(group["target_discharge_cfs"], group["benchmark_persistence_lag1_cfs"]),
        })

    rolling_by_label = {item["label"]: item for item in rolling_summary}
    selected_name = "full_model"
    selected_pipeline = full_model
    selected_features = full_features
    if rolling_by_label.get("runoff_only") and rolling_by_label.get("full_model"):
        if rolling_by_label["runoff_only"]["mae_mean"] < rolling_by_label["full_model"]["mae_mean"]:
            selected_name = "runoff_only"
            selected_pipeline = runoff_only_model
            selected_features = runoff_features

    report = {
        "metrics": metrics,
        "rolling_validation": {
            "folds": fold_summaries,
            "summary": rolling_summary,
            "records": [] if fold_df.empty else json.loads(fold_df.to_json(orient="records")),
        },
        "station_test_metrics": station_test_metrics,
        "test_predictions": test_out,
        "runoff_only_feature_columns": runoff_features,
        "selected_model": {
            "model": selected_name,
            "feature_columns": selected_features,
            "pipeline": selected_pipeline,
            "selection_basis": "lowest rolling-validation mae among trainable models",
        },
    }
    return selected_pipeline, report, selected_features


def save_model_bundle(model: Pipeline, features: Sequence[str], report: dict, station_links: List[dict]) -> None:
    import pickle

    with (MODEL_DIR / "montana_runoff_ridge.pkl").open("wb") as fh:
        pickle.dump(model, fh)
    if "selected_model" in report and report["selected_model"].get("model") == "runoff_only":
        with (MODEL_DIR / "montana_runoff_ridge.pkl").open("wb") as fh:
            pickle.dump(report["selected_model"]["pipeline"], fh)
        features = report["runoff_only_feature_columns"]

    report_payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "weather_source": "Dynamical NOAA GEFS analysis (historical archive)",
        "snow_sources": ["NRCS SNOTEL daily station reports", "NOHRSC daily station reports"],
        "runoff_source": "USGS NWIS daily values parameterCd=00060",
        "feature_columns": list(features),
        "station_links": station_links,
        "metrics": report["metrics"],
        "rolling_validation": report["rolling_validation"],
        "station_test_metrics": report["station_test_metrics"],
        "runoff_only_feature_columns": report["runoff_only_feature_columns"],
        "selected_model": {
            "model": report["selected_model"]["model"],
            "feature_columns": list(features),
            "selection_basis": report["selected_model"]["selection_basis"],
        },
        "test_predictions": json.loads(report["test_predictions"].to_json(orient="records", date_format="iso")),
        "notes": [
            "1-day-ahead Montana gauge model with snow, melt, weather, and runoff-response features.",
            "Validation includes holdout metrics, rolling time-split validation, and explicit benchmark comparisons.",
            "Nearest SNOTEL and nearest NOHRSC stations remain proxy snowpack signals, not basin-weighted forcings.",
            "Forecast inference still uses forecast weather plus persistence-style snow state updates, so this is testable but still experimental."
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
    skipped_stations = []
    for station in mt_stations:
        if STATION_LIMIT > 0 and len(station_frames) >= STATION_LIMIT:
            break
        snotel_neighbor = nearest_station(station["latitude"], station["longitude"], snotel_meta, MAX_SNOTEL_KM)
        nohrsc_neighbor = nearest_station(station["latitude"], station["longitude"], nohrsc_meta, MAX_NOHRSC_KM, lat_key="lat", lon_key="lon")
        print(f"Station {station['stationId']} {station['name']}")
        print(f"  nearest SNOTEL: {snotel_neighbor.station_id if snotel_neighbor else 'none'}")
        print(f"  nearest NOHRSC: {nohrsc_neighbor.station_id if nohrsc_neighbor else 'none'}")
        frame = assemble_station_dataset(station, snotel_neighbor, nohrsc_neighbor, start, end)
        if frame.empty:
            skipped_stations.append({"station_id": station["stationId"], "reason": "empty_dataset"})
            print("  skipped: empty assembled dataset")
            continue
        usable = frame.dropna(subset=["target_discharge_cfs"])
        if usable.empty or len(usable) < 10:
            skipped_stations.append({"station_id": station["stationId"], "reason": "too_few_target_rows", "rows": int(len(usable))})
            print(f"  skipped: only {len(usable)} usable target rows")
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
    dataset.to_parquet(DATASET_DIR / "montana_runoff_training.parquet", index=False)
    dataset.head(100).to_csv(DATASET_DIR / "montana_runoff_training_sample.csv", index=False)

    model, report, features = train_model(dataset)
    save_model_bundle(model, features, report, station_links)

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "date_range": {"start": start.isoformat(), "end": end.isoformat()},
        "target_group": TARGET_GROUP or None,
        "station_ids_filter": STATION_IDS_FILTER,
        "snotel_metadata_count": len(snotel_meta),
        "nohrsc_metadata_count": len(nohrsc_meta),
        "stations_requested": (len(mt_stations) if STATION_LIMIT <= 0 else STATION_LIMIT) if not (TARGET_GROUP or STATION_IDS_FILTER) else len(mt_stations),
        "station_candidates_considered": len(mt_stations),
        "stations_used": len(station_frames),
        "station_links": station_links,
        "stations_skipped": skipped_stations,
        "rows": int(len(dataset)),
        "metrics": report["metrics"],
        "rolling_validation_summary": report["rolling_validation"]["summary"],
    }
    write_json(ML_DIR / "latest_training_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
