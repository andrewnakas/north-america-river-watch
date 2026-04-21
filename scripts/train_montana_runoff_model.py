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
import io
import json
import math
import os
import pathlib
import re
import struct
import zipfile
from collections import Counter
from dataclasses import dataclass, field
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
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.base import clone

ROOT = pathlib.Path(__file__).resolve().parents[1]
workspace_candidate = pathlib.Path.cwd().resolve()
if (workspace_candidate / "generated").exists() and (workspace_candidate / "scripts").exists():
    ROOT = workspace_candidate
elif not ROOT.exists() or not (ROOT / "generated").exists():
    candidate_root = pathlib.Path(__file__).resolve().parent.parent
    if candidate_root.exists() and (candidate_root / "generated").exists():
        ROOT = candidate_root
WORKSPACE_ROOT = ROOT.parent
GENERATED = ROOT / "generated"
ML_DIR = GENERATED / "ml"
CACHE_DIR = ML_DIR / "cache"
DATASET_DIR = ML_DIR / "datasets"
MODEL_DIR = ML_DIR / "models"
ZERO_SHOT_EXTERNAL_DIR = ROOT / "data_research" / "zero_shot_hydrology" / "external"
GEODAR_ARCHIVE_PATH = ZERO_SHOT_EXTERNAL_DIR / "GeoDAR_v10_v11.zip"
CAMELS_ATTRIBUTES_DIR = ZERO_SHOT_EXTERNAL_DIR / "camels_attributes_v2.0"
HYDROBASINS_DIR = ZERO_SHOT_EXTERNAL_DIR / "hydrobasins" / "hybas_na_lev12_v1c"
HYDROBASINS_SHP_PATH = HYDROBASINS_DIR / "hybas_na_lev12_v1c.shp"
HYDROBASINS_DBF_PATH = HYDROBASINS_DIR / "hybas_na_lev12_v1c.dbf"
TREESIXTY_SNOTEL_PATH = WORKSPACE_ROOT / "TreesixtyFirebase" / "public" / "data" / "snotel-stations.json"

USER_AGENT = "north-america-river-watch-ml/0.2"
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": USER_AGENT})


def get_json_with_retries(url: str, timeout: int = 60, attempts: int = 3) -> dict:
    last_error = None
    headers_variants = [None, {"Accept-Encoding": "identity"}]
    for attempt in range(attempts):
        for extra_headers in headers_variants:
            try:
                response = SESSION.get(url, timeout=timeout, headers=extra_headers)
                response.raise_for_status()
                return response.json()
            except (requests.exceptions.ContentDecodingError, requests.exceptions.ChunkedEncodingError, requests.exceptions.ConnectionError, requests.exceptions.Timeout, ValueError) as exc:
                last_error = exc
                continue
    if last_error is not None:
        raise last_error
    raise RuntimeError(f"Failed to fetch JSON from {url}")

DYNAMICAL_EMAIL = os.environ.get("DYNAMICAL_EMAIL", "treesixtyweather@gmail.com")
STATION_LIMIT = int(os.environ.get("MT_STATION_LIMIT", "8"))
STATION_CANDIDATE_MULTIPLIER = int(os.environ.get("MT_STATION_CANDIDATE_MULTIPLIER", "5"))
LOOKBACK_DAYS = int(os.environ.get("MT_LOOKBACK_DAYS", "180"))
LOOKBACK_START = os.environ.get("MT_LOOKBACK_START", "")
REQUIRE_3KM_FORECAST = os.environ.get("MT_REQUIRE_3KM_FORECAST", "1") not in {"0", "false", "False"}
TARGET_STATE = os.environ.get("MT_TARGET_STATE", "MT")
TARGET_STATES = [s.strip().upper() for s in os.environ.get("MT_TARGET_STATES", "").split(",") if s.strip()]
TARGET_SCOPE = os.environ.get("MT_TARGET_SCOPE", "state").strip().lower()
TARGET_HORIZON_DAYS = int(os.environ.get("MT_TARGET_HORIZON_DAYS", "1"))
MAX_SNOTEL_KM = float(os.environ.get("MT_MAX_SNOTEL_KM", "80"))
MAX_NOHRSC_KM = float(os.environ.get("MT_MAX_NOHRSC_KM", "80"))
BASIN_NEIGHBOR_LIMIT = int(os.environ.get("MT_BASIN_NEIGHBOR_LIMIT", "4"))
BASIN_PRIMARY_RATIO_CAP = float(os.environ.get("MT_BASIN_PRIMARY_RATIO_CAP", "2.5"))
FORCE_OPENMETEO_WEATHER = os.environ.get("MT_FORCE_OPENMETEO_WEATHER", "0") == "1"
TARGET_GROUP = os.environ.get("MT_TARGET_GROUP", "").strip().lower()
STATION_IDS_FILTER = [sid.strip() for sid in os.environ.get("MT_STATION_IDS", "").split(",") if sid.strip()]
INCLUDE_MANAGED_STATIONS = os.environ.get("MT_INCLUDE_MANAGED_STATIONS", "0") == "1"
DEGREE_DAY_BASE_C = float(os.environ.get("MT_DEGREE_DAY_BASE_C", "0.0"))
ROLLING_VALIDATION_SPLITS = int(os.environ.get("MT_ROLLING_VALIDATION_SPLITS", "3"))
MIN_VALIDATION_DAYS = int(os.environ.get("MT_MIN_VALIDATION_DAYS", "21"))
TEMP_LAPSE_C_PER_KM = float(os.environ.get("MT_TEMP_LAPSE_C_PER_KM", "6.5"))
EVENT_ONLY_RESIDUAL = os.environ.get("MT_EVENT_ONLY_RESIDUAL", "1") == "1"
TRAINING_WEATHER_MODE = os.environ.get("MT_TRAINING_WEATHER_MODE", "blend").strip().lower()
RESIDUAL_TRAIN_MASK_MODE = os.environ.get("MT_RESIDUAL_TRAIN_MASK_MODE", "event" if EVENT_ONLY_RESIDUAL else "all").strip().lower()
BLEND_GATE_MODE = os.environ.get("MT_BLEND_GATE_MODE", "event").strip().lower()
BLEND_BASE_WEIGHT = float(os.environ.get("MT_BLEND_BASE_WEIGHT", "0.15"))
BLEND_MAX_WEIGHT = float(os.environ.get("MT_BLEND_MAX_WEIGHT", "0.85"))
BLEND_EXTREME_BONUS = float(os.environ.get("MT_BLEND_EXTREME_BONUS", "0.12"))
FULL_FEATURE_SET_MODE = os.environ.get("MT_FULL_FEATURE_SET_MODE", "all").strip().lower()
RESIDUAL_CLIP_MODE = os.environ.get("MT_RESIDUAL_CLIP_MODE", "none").strip().lower()
ENSEMBLE_RUNOFF_WEIGHT = float(os.environ.get("MT_ENSEMBLE_RUNOFF_WEIGHT", "0.85"))
CATEGORICAL_MODE = os.environ.get("MT_CATEGORICAL_MODE", "station_id_and_cluster").strip().lower()
FEATURE_EXPERIMENT = os.environ.get("MT_FEATURE_EXPERIMENT", "baseline").strip().lower()
HRRR_PROJ4 = "+proj=lcc +a=6371200.0 +b=6371200.0 +lon_0=262.5 +lat_0=38.5 +lat_1=38.5 +lat_2=38.5"
STATION_DATASET_SCHEMA_VERSION = "2026-04-20-static-v1"

MOUNTAIN_WEST_STATES = ["MT", "ID", "WY", "CO", "UT", "NV", "NM", "AZ"]

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

BASIN_STOPWORDS = {
    'river', 'creek', 'fork', 'branch', 'near', 'at', 'above', 'below', 'nr', 'mt', 'montana',
    'gallatin', 'yellowstone', 'east', 'west', 'middle', 'south', 'north', 'little', 'big',
    'upper', 'lower', 'ab', 'bl', 'lake', 'reservoir'
}


def basin_key_for_station(station: dict) -> str:
    text = ' '.join(str(station.get(k) or '') for k in ['waterbody', 'name']).lower()
    tokens = re.findall(r'[a-z]{3,}', text)
    picked = [tok for tok in tokens if tok not in BASIN_STOPWORDS]
    if not picked:
        picked = [tok for tok in tokens if tok not in {'near', 'at', 'above', 'below', 'mt'}]
    if not picked:
        picked = ['unknown']
    return '-'.join(picked[:2])


def classify_elevation_band(site_elev_m: Optional[float], sensor_elev_m: Optional[float]) -> str:
    if site_elev_m is None or sensor_elev_m is None:
        return 'unknown'
    delta = float(sensor_elev_m) - float(site_elev_m)
    if delta < 300:
        return 'low'
    if delta < 900:
        return 'mid'
    return 'high'


@dataclass
class Neighbor:
    station_id: str
    name: str
    distance_km: float
    latitude: float
    longitude: float
    meta: dict


@dataclass
class BasinSignal:
    primary: Optional[Neighbor]
    contributors: List[Neighbor] = field(default_factory=list)
    weighted_elev_m: Optional[float] = None
    band_label: str = "unknown"
    basin_key: Optional[str] = None


@dataclass
class RegulationSignal:
    nearest_dam_distance_km: Optional[float] = None
    dams_within_25km: int = 0
    dams_within_50km: int = 0
    dams_within_100km: int = 0
    reservoir_storage_mcm_within_50km: Optional[float] = None
    reservoir_storage_mcm_within_100km: Optional[float] = None
    nearest_reservoir_distance_km: Optional[float] = None
    nearest_reservoir_storage_mcm: Optional[float] = None


@dataclass
class CamelsAttributes:
    values: Dict[str, object] = field(default_factory=dict)


@dataclass
class HydroBasinsAttributes:
    values: Dict[str, object] = field(default_factory=dict)


def ensure_dirs() -> None:
    for path in [ML_DIR, CACHE_DIR, DATASET_DIR, MODEL_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def station_dataset_cache_path(station_id: str, start: date, end: date) -> pathlib.Path:
    return DATASET_DIR / f"station_{station_id}_{start.isoformat()}_{end.isoformat()}.parquet"


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


def _to_float(value):
    if value in (None, "", "nan", "NaN", "-999", "-999.0", "-9999", "-9999.0"):
        return None
    try:
        val = float(value)
    except Exception:
        return None
    if not math.isfinite(val):
        return None
    return val


def load_geodar_records(kind: str = "dams") -> List[dict]:
    if kind not in {"dams", "reservoirs"}:
        raise ValueError(f"unsupported GeoDAR kind: {kind}")
    if not GEODAR_ARCHIVE_PATH.exists():
        return []
    cache_path = cache_json_path(f"geodar_{kind}.json")
    if cache_path.exists():
        cached = read_json(cache_path)
        if isinstance(cached, list):
            return cached
    member = f"GeoDAR_v10_v11/GeoDAR_v11_{kind}.csv"
    records: List[dict] = []
    with zipfile.ZipFile(GEODAR_ARCHIVE_PATH) as zf:
        if member not in zf.namelist():
            return []
        with zf.open(member) as fh:
            wrapper = io.TextIOWrapper(fh, encoding="utf-8", errors="ignore", newline="")
            reader = csv.DictReader(wrapper)
            for row in reader:
                lat = _to_float(row.get("lat"))
                lon = _to_float(row.get("lon"))
                if lat is None or lon is None:
                    continue
                rec = {
                    "lat": lat,
                    "lon": lon,
                    "qa_rank": row.get("qa_rank") or None,
                }
                for key in ("rv_mcm_v11", "rv_mcm_v10", "id_v11", "id_grd_v13"):
                    if key in row:
                        rec[key] = row.get(key)
                storage = _to_float(row.get("rv_mcm_v11"))
                if storage is None:
                    storage = _to_float(row.get("rv_mcm_v10"))
                rec["storage_mcm"] = storage
                records.append(rec)
    write_json(cache_path, records, pretty=False)
    return records


def build_regulation_signal(station: dict) -> RegulationSignal:
    lat = _to_float(station.get("latitude"))
    lon = _to_float(station.get("longitude"))
    if lat is None or lon is None:
        return RegulationSignal()

    dams = load_geodar_records("dams")
    reservoirs = load_geodar_records("reservoirs")
    signal = RegulationSignal()

    dam_distances = []
    for rec in dams:
        dist = haversine_km(lat, lon, rec["lat"], rec["lon"])
        dam_distances.append(dist)
    if dam_distances:
        signal.nearest_dam_distance_km = float(min(dam_distances))
        signal.dams_within_25km = int(sum(d <= 25 for d in dam_distances))
        signal.dams_within_50km = int(sum(d <= 50 for d in dam_distances))
        signal.dams_within_100km = int(sum(d <= 100 for d in dam_distances))

    res_distances = []
    storage50 = 0.0
    storage100 = 0.0
    nearest_storage = None
    nearest_dist = None
    for rec in reservoirs:
        dist = haversine_km(lat, lon, rec["lat"], rec["lon"])
        storage = _to_float(rec.get("storage_mcm"))
        res_distances.append((dist, storage))
        if storage is not None and dist <= 50:
            storage50 += storage
        if storage is not None and dist <= 100:
            storage100 += storage
        if nearest_dist is None or dist < nearest_dist:
            nearest_dist = dist
            nearest_storage = storage
    if res_distances:
        signal.nearest_reservoir_distance_km = float(min(d for d, _ in res_distances))
        signal.nearest_reservoir_storage_mcm = nearest_storage
        signal.reservoir_storage_mcm_within_50km = storage50
        signal.reservoir_storage_mcm_within_100km = storage100
    return signal


def load_camels_attribute_table(name: str) -> Optional[pd.DataFrame]:
    path = CAMELS_ATTRIBUTES_DIR / name
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path, sep=";")
    except Exception:
        return None
    if "gauge_id" not in df.columns:
        return None
    df = df.copy()
    df["gauge_id"] = df["gauge_id"].map(lambda x: str(int(float(x))).zfill(8) if _to_float(x) is not None else None)
    return df


def build_camels_lookup() -> Dict[str, dict]:
    cache_path = cache_json_path("camels_attributes_lookup.json")
    if cache_path.exists():
        cached = read_json(cache_path)
        if isinstance(cached, dict):
            return cached
    table_names = [
        "camels_topo.txt",
        "camels_clim.txt",
        "camels_soil.txt",
        "camels_geol.txt",
        "camels_vege.txt",
        "camels_hydro.txt",
        "camels_name.txt",
    ]
    merged = None
    for name in table_names:
        df = load_camels_attribute_table(name)
        if df is None or df.empty:
            continue
        merged = df if merged is None else merged.merge(df, on="gauge_id", how="outer")
    if merged is None or merged.empty:
        return {}
    keep_columns = {
        "gauge_id", "elev_mean", "slope_mean", "area_gages2", "area_geospa_fabric",
        "p_mean", "pet_mean", "p_seasonality", "high_prec_freq", "high_prec_dur", "high_prec_timing",
        "low_prec_freq", "low_prec_dur", "low_prec_timing", "frac_snow", "aridity",
        "soil_depth_pelletier", "soil_depth_statsgo", "soil_porosity", "soil_conductivity",
        "max_water_content", "sand_frac", "silt_frac", "clay_frac", "water_frac", "organic_frac",
        "other_frac", "geol_1st_class", "glim_1st_class_frac", "geol_2nd_class", "glim_2nd_class_frac",
        "carbonate_rocks_frac", "geol_porostiy", "geol_permeability", "frac_forest", "lai_max",
        "lai_diff", "gvf_max", "gvf_diff", "dom_land_cover_frac", "dom_land_cover",
        "root_depth_50", "root_depth_99", "q_mean", "runoff_ratio", "slope_fdc", "baseflow_index",
        "stream_elas", "q5", "q95", "high_q_freq", "high_q_dur", "low_q_freq", "low_q_dur",
        "zero_q_freq", "hfd_mean", "gauge_lat", "gauge_lon", "huc_02"
    }
    available = [c for c in merged.columns if c in keep_columns]
    merged = merged[available].copy()
    records = {}
    for row in merged.to_dict(orient="records"):
        gauge_id = row.pop("gauge_id", None)
        if gauge_id:
            records[gauge_id] = row
    write_json(cache_path, records, pretty=False)
    return records


def fetch_camels_attributes(site_no: str) -> CamelsAttributes:
    lookup = build_camels_lookup()
    key = str(site_no).strip().zfill(8)
    values = lookup.get(key, {}) if isinstance(lookup, dict) else {}
    return CamelsAttributes(values=values if isinstance(values, dict) else {})


def _dbf_fields(path: pathlib.Path) -> List[Tuple[str, str, int, int]]:
    with path.open("rb") as fh:
        header = fh.read(32)
        if len(header) < 32:
            return []
        fields = []
        while True:
            desc = fh.read(32)
            if not desc or desc[0] == 0x0D:
                break
            name = desc[:11].split(b"\x00", 1)[0].decode("ascii", "ignore")
            fields.append((name, chr(desc[11]), desc[16], desc[17]))
    return fields


def _dbf_parse_value(raw: bytes, field_type: str):
    text = raw.decode("latin1", "ignore").strip().replace("\x00", "")
    if text in {"", "*"}:
        return None
    if field_type in {"N", "F"}:
        try:
            num = float(text)
        except Exception:
            return None
        if not math.isfinite(num):
            return None
        if field_type == "N" and abs(num - round(num)) < 1e-9:
            return int(round(num))
        return num
    return text


def _load_hydrobasins_dbf_records() -> Dict[int, dict]:
    if not HYDROBASINS_DBF_PATH.exists():
        return {}
    cache_path = cache_json_path("hydrobasins_level12_records.json")
    if cache_path.exists():
        cached = read_json(cache_path)
        if isinstance(cached, dict):
            return {int(k): v for k, v in cached.items() if str(k).isdigit()}
    fields = _dbf_fields(HYDROBASINS_DBF_PATH)
    if not fields:
        return {}
    with HYDROBASINS_DBF_PATH.open("rb") as fh:
        header = fh.read(32)
        num_records = int.from_bytes(header[4:8], "little")
        header_len = int.from_bytes(header[8:10], "little")
        rec_len = int.from_bytes(header[10:12], "little")
        fh.seek(header_len)
        records: Dict[int, dict] = {}
        for _ in range(num_records):
            row = fh.read(rec_len)
            if not row or row[0] == 0x2A:
                continue
            pos = 1
            out = {}
            for name, field_type, length, _dec in fields:
                out[name] = _dbf_parse_value(row[pos:pos + length], field_type)
                pos += length
            basin_id = out.get("HYBAS_ID")
            if basin_id is None:
                continue
            records[int(basin_id)] = out
    write_json(cache_path, {str(k): v for k, v in records.items()}, pretty=False)
    return records


def _point_in_ring(x: float, y: float, ring: Sequence[Tuple[float, float]]) -> bool:
    inside = False
    n = len(ring)
    if n < 3:
        return False
    j = n - 1
    for i in range(n):
        xi, yi = ring[i]
        xj, yj = ring[j]
        intersects = ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-12) + xi)
        if intersects:
            inside = not inside
        j = i
    return inside


def _point_in_polygon(x: float, y: float, rings: Sequence[Sequence[Tuple[float, float]]]) -> bool:
    if not rings:
        return False
    if not _point_in_ring(x, y, rings[0]):
        return False
    for ring in rings[1:]:
        if _point_in_ring(x, y, ring):
            return False
    return True


def build_hydrobasins_lookup() -> Dict[str, dict]:
    if not HYDROBASINS_SHP_PATH.exists() or not HYDROBASINS_DBF_PATH.exists():
        return {}
    cache_path = cache_json_path("hydrobasins_station_lookup.json")
    if cache_path.exists():
        cached = read_json(cache_path)
        if isinstance(cached, dict) and cached:
            return cached

    dbf_records = _load_hydrobasins_dbf_records()
    if not dbf_records:
        return {}

    stations = read_json(GENERATED / "stations.json")
    station_points = []
    for station in stations:
        if station.get("network") != "USGS":
            continue
        lat = _to_float(station.get("latitude"))
        lon = _to_float(station.get("longitude"))
        sid = str(station.get("stationId") or "").strip()
        if lat is None or lon is None or not sid:
            continue
        station_points.append({"station_id": sid, "lat": lat, "lon": lon})

    with HYDROBASINS_SHP_PATH.open("rb") as fh:
        shp_header = fh.read(100)
        if len(shp_header) < 100:
            return {}
        global_bbox = struct.unpack("<4d", shp_header[36:68])
        minx, miny, maxx, maxy = global_bbox
        candidate_points = [p for p in station_points if minx <= p["lon"] <= maxx and miny <= p["lat"] <= maxy]
        matches: Dict[str, dict] = {}
        rec_index = 0
        while True:
            rec_header = fh.read(8)
            if not rec_header or len(rec_header) < 8:
                break
            rec_index += 1
            content_len = int.from_bytes(rec_header[4:8], "big") * 2
            content = fh.read(content_len)
            if len(content) < 44:
                continue
            shape_type = struct.unpack("<i", content[:4])[0]
            if shape_type != 5:
                continue
            bbox = struct.unpack("<4d", content[4:36])
            xmin, ymin, xmax, ymax = bbox
            local_points = [p for p in candidate_points if p["station_id"] not in matches and xmin <= p["lon"] <= xmax and ymin <= p["lat"] <= ymax]
            if not local_points:
                continue
            num_parts, num_points = struct.unpack("<2i", content[36:44])
            part_idx_offset = 44
            points_offset = part_idx_offset + 4 * num_parts
            parts = struct.unpack(f"<{num_parts}i", content[part_idx_offset:points_offset])
            coords_blob = content[points_offset:points_offset + 16 * num_points]
            coords = [struct.unpack("<2d", coords_blob[i * 16:(i + 1) * 16]) for i in range(num_points)]
            rings = []
            for i, start_idx in enumerate(parts):
                end_idx = parts[i + 1] if i + 1 < len(parts) else num_points
                rings.append(coords[start_idx:end_idx])
            basin_record = dbf_records.get(rec_index)
            if not basin_record:
                continue
            attrs = {
                "hybas_id": basin_record.get("HYBAS_ID"),
                "next_down": basin_record.get("NEXT_DOWN"),
                "main_bas": basin_record.get("MAIN_BAS"),
                "dist_sink_km": basin_record.get("DIST_SINK"),
                "dist_main_km": basin_record.get("DIST_MAIN"),
                "sub_area_km2": basin_record.get("SUB_AREA"),
                "up_area_km2": basin_record.get("UP_AREA"),
                "pfaf_id": basin_record.get("PFAF_ID"),
                "stream_order": basin_record.get("ORDER"),
                "endo": basin_record.get("ENDO"),
                "coast": basin_record.get("COAST"),
            }
            up_area = _to_float(attrs.get("up_area_km2"))
            sub_area = _to_float(attrs.get("sub_area_km2"))
            attrs["upstream_to_local_area_ratio"] = None if up_area is None or sub_area is None or sub_area <= 0 else up_area / sub_area
            attrs["is_endorheic"] = 1 if attrs.get("endo") else 0
            attrs["is_coastal_basin"] = 1 if attrs.get("coast") else 0
            for point in local_points:
                if _point_in_polygon(point["lon"], point["lat"], rings):
                    matches[point["station_id"]] = attrs.copy()

    if not matches:
        raise RuntimeError("hydrobasins lookup build produced zero station matches")
    write_json(cache_path, matches, pretty=False)
    return matches


def fetch_hydrobasins_attributes(site_no: str) -> HydroBasinsAttributes:
    lookup = build_hydrobasins_lookup()
    key = str(site_no).strip().zfill(8)
    values = lookup.get(key, {}) if isinstance(lookup, dict) else {}
    return HydroBasinsAttributes(values=values if isinstance(values, dict) else {})


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


def _normalize_nohrsc_elev_m(value, lat: Optional[float] = None, lon: Optional[float] = None):
    val = _to_float(value) if not isinstance(value, (int, float)) else float(value)
    if val is None:
        return None
    approx_site = fetch_location_elevation_m(lat, lon) if lat is not None and lon is not None else None
    if val > 2500:
        return val * 0.3048
    if approx_site is not None and val > approx_site * 1.8:
        return val * 0.3048
    return float(val)


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


def fetch_usgs_site_metadata(site_no: str) -> dict:
    cache_path = cache_json_path(f"usgs_site_meta_{site_no}.json")
    if cache_path.exists():
        payload = read_json(cache_path)
    else:
        url = (
            "https://waterservices.usgs.gov/nwis/site/?format=rdb&siteOutput=expanded"
            f"&sites={site_no}"
        )
        resp = SESSION.get(url, timeout=60)
        resp.raise_for_status()
        text = resp.text
        lines = [ln for ln in text.splitlines() if ln and not ln.startswith('#')]
        payload = {"raw": text, "lines": lines}
        write_json(cache_path, payload, pretty=False)

    lines = payload.get("lines") if isinstance(payload, dict) else None
    if not lines or len(lines) < 3:
        return {}
    reader = csv.DictReader(lines)
    row = next(reader, None)
    if not row:
        return {}

    def _float_or_none(value):
        if value in (None, "", "NaN", "nan"):
            return None
        try:
            return float(value)
        except Exception:
            return None

    drain_area_sqmi = _float_or_none(row.get("drain_area_va"))
    contrib_area_sqmi = _float_or_none(row.get("contrib_drain_area_va"))
    alt_ft = _float_or_none(row.get("alt_va"))
    return {
        "drain_area_sqmi": drain_area_sqmi,
        "contrib_drain_area_sqmi": contrib_area_sqmi,
        "drain_area_km2": None if drain_area_sqmi is None else drain_area_sqmi * 2.58999,
        "contrib_drain_area_km2": None if contrib_area_sqmi is None else contrib_area_sqmi * 2.58999,
        "site_altitude_ft": alt_ft,
        "site_altitude_m": None if alt_ft is None else alt_ft * 0.3048,
        "huc_cd": row.get("huc_cd") or None,
        "basin_cd": row.get("basin_cd") or None,
    }


def fetch_usgs_daily_discharge(site_no: str, start: date, end: date) -> pd.DataFrame:
    cache_path = cache_json_path(f"usgs_{site_no}_{start.isoformat()}_{end.isoformat()}.json")
    if cache_path.exists():
        payload = read_json(cache_path)
    else:
        url = (
            "https://waterservices.usgs.gov/nwis/dv/?format=json"
            f"&sites={site_no}&parameterCd=00060&startDT={start.isoformat()}&endDT={end.isoformat()}"
        )
        payload = get_json_with_retries(url, timeout=60)
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
                    or discharge < 0
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


def configured_target_states() -> List[str]:
    if TARGET_STATES:
        return sorted({s for s in TARGET_STATES if s})
    if TARGET_SCOPE == "mountain-west":
        return list(MOUNTAIN_WEST_STATES)
    if TARGET_SCOPE == "national":
        return []
    return [TARGET_STATE.upper()] if TARGET_STATE else []




def diversify_station_candidates(stations: List[dict], limit: int, preferred_states: Optional[List[str]] = None) -> List[dict]:
    if limit <= 0 or len(stations) <= limit:
        return stations
    preferred_states = [str(state or '').upper() for state in (preferred_states or []) if str(state or '').strip()]
    by_state: Dict[str, List[dict]] = {}
    for station in stations:
        by_state.setdefault(str(station.get('state') or 'UNK').upper(), []).append(station)
    for state in by_state:
        by_state[state].sort(key=lambda s: (
            not bool(s.get('naturalFlow', True)),
            not bool(s.get('noaaForecast')),
            s.get('stationId')
        ))
    remaining_states = [state for state in sorted(by_state.keys(), key=lambda st: (-len(by_state[st]), st)) if state not in preferred_states]
    ordered_states = [state for state in preferred_states if state in by_state] + remaining_states
    picked: List[dict] = []
    seen = set()
    while len(picked) < limit:
        progressed = False
        for state in ordered_states:
            bucket = by_state[state]
            while bucket and str(bucket[0].get('stationId')) in seen:
                bucket.pop(0)
            if not bucket:
                continue
            station = bucket.pop(0)
            sid = str(station.get('stationId'))
            if sid in seen:
                continue
            seen.add(sid)
            picked.append(station)
            progressed = True
            if len(picked) >= limit:
                break
        if not progressed:
            break
    return picked

def is_managed_or_problematic_station(station: dict) -> bool:
    sid = str(station.get("stationId") or "").strip()
    if sid in {"05017500", "05018500", "06016000", "06017000", "06018500", "06023100"}:
        return True
    if station.get("managedFlow") is True:
        return True
    name = str(station.get("name") or "").lower()
    waterbody = str(station.get("waterbody") or "").lower()
    text = f" {name} {waterbody} "
    managed_patterns = (
        r"\bcanal\b",
        r"\breservoir\b",
        r"\bdam\b",
        r"\bdiversion\b",
        r"\bpowerplant\b",
        r"\btailrace\b",
        r"\boutlet\b",
        r"\bfeeder\b",
        r"\bditch\b",
        r"\bspillway\b",
        r"\bpumped\b",
        r"\bregulated\b",
        r"\bcontrol\b",
        r"\bfish hatchery\b",
        r"\bbelow\b.*\bdam\b",
        r"\bbelow\b.*\blake\b",
        r"\babove\b.*\breservoir\b",
        r"\bat\b.*\boutlet\b",
    )
    return any(re.search(pattern, text) for pattern in managed_patterns)



def load_mt_river_stations() -> List[dict]:
    stations_path = GENERATED / "stations.json"
    stations = read_json(stations_path)
    wanted_states = set(configured_target_states())
    candidates = [
        s for s in stations
        if s.get("network") == "USGS"
        and s.get("latitude") is not None
        and s.get("longitude") is not None
        and (not wanted_states or str(s.get("state") or "").upper() in wanted_states)
        and (INCLUDE_MANAGED_STATIONS or not is_managed_or_problematic_station(s))
        and (not REQUIRE_3KM_FORECAST or bool(s.get("noaaForecast")))
    ]

    requested_ids = []
    if TARGET_GROUP and TARGET_GROUP in TARGET_GROUP_STATION_IDS:
        requested_ids.extend(TARGET_GROUP_STATION_IDS[TARGET_GROUP])
    if STATION_IDS_FILTER:
        requested_ids.extend(STATION_IDS_FILTER)

    if requested_ids:
        wanted = {sid.strip() for sid in requested_ids if sid.strip()}
        selected = [s for s in candidates if str(s.get("stationId")) in wanted]
        selected.sort(key=lambda s: requested_ids.index(str(s.get("stationId"))) if str(s.get("stationId")) in requested_ids else 10**9)
        return selected

    candidates.sort(key=lambda s: (
        not bool(s.get("naturalFlow", True)),
        not bool(s.get("noaaForecast")),
        str(s.get("state") or ""),
        s.get("stationId")
    ))
    if STATION_LIMIT <= 0:
        return candidates
    candidate_limit = max(STATION_LIMIT, STATION_LIMIT * max(1, STATION_CANDIDATE_MULTIPLIER))
    if TARGET_SCOPE == "national" and not configured_target_states():
        return diversify_station_candidates(candidates, candidate_limit)
    if TARGET_SCOPE == "mountain-west":
        return diversify_station_candidates(candidates, candidate_limit, preferred_states=['MT', 'ID', 'WY'])
    return candidates[:candidate_limit]


def load_snotel_metadata() -> List[dict]:
    state_token = "all" if TARGET_SCOPE == "national" or TARGET_STATES else (TARGET_STATE or "all").lower()
    cache_path = cache_json_path(f"snotel_metadata_{state_token}.json")
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
            if configured_target_states() and str(row.get('state') or '').upper() not in set(configured_target_states()):
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
                if configured_target_states() and str(row.get('state') or '').upper() not in set(configured_target_states()):
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
    state_token = "all" if TARGET_SCOPE == "national" or TARGET_STATES else (TARGET_STATE or "all").lower()
    cache_path = cache_json_path(f'nohrsc_metadata_v2_{state_token}.json')
    if cache_path.exists():
        cached = read_json(cache_path)
        if cached:
            return cached
    stations = []
    seen = set()
    wanted_states = set(configured_target_states())
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
            state = str((s.get('state') or '')).upper()
            if wanted_states and state and state not in wanted_states:
                continue
            seen.add(sid)
            stations.append(s)
    write_json(cache_path, stations)
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


def candidate_neighbors(lat: float, lon: float, stations: Sequence[dict], max_km: float, *, id_key: str = 'id', lat_key: str = 'latitude', lon_key: str = 'longitude', limit: int = 6) -> List[Neighbor]:
    out: List[Neighbor] = []
    for s in stations:
        s_lat = s.get(lat_key)
        s_lon = s.get(lon_key)
        if s_lat is None or s_lon is None:
            continue
        sid = s.get(id_key)
        if sid in (None, ''):
            continue
        d = haversine_km(lat, lon, s_lat, s_lon)
        if d > max_km:
            continue
        out.append(Neighbor(
            station_id=str(sid),
            name=s.get('name') or s.get('station_id') or str(s.get(id_key)),
            distance_km=d,
            latitude=float(s_lat),
            longitude=float(s_lon),
            meta=s,
        ))
    out.sort(key=lambda n: n.distance_km)
    return out[:limit]


def _neighbor_elev_m(neighbor: Neighbor) -> Optional[float]:
    meta = neighbor.meta or {}
    if meta.get('source') == 'nohrsc' or 'elev_m' in meta:
        return _normalize_nohrsc_elev_m(meta.get('elev_raw', meta.get('elev_m')), meta.get('lat') or meta.get('latitude'), meta.get('lon') or meta.get('longitude'))
    return _to_meters(meta.get('elevation') or meta.get('elev_m'))


def basin_signal_for_station(station: dict, candidates: Sequence[Neighbor], site_elev_m: Optional[float], preferred_band: str = 'high') -> BasinSignal:
    if not candidates:
        return BasinSignal(primary=None, contributors=[], weighted_elev_m=None, band_label='unknown', basin_key=basin_key_for_station(station))
    weights = []
    chosen: List[Neighbor] = []
    primary = candidates[0]
    primary_elev = _neighbor_elev_m(primary)
    for neighbor in list(candidates)[: max(1, BASIN_NEIGHBOR_LIMIT)]:
        elev_m = _neighbor_elev_m(neighbor)
        band = classify_elevation_band(site_elev_m, elev_m)
        band_bonus = 1.4 if band == preferred_band else (1.1 if band == 'mid' else 0.7)
        similarity = 1.0
        if primary_elev is not None and elev_m is not None:
            similarity = max(0.3, 1.0 - min(abs(elev_m - primary_elev), 1400.0) / 1800.0)
        w = (band_bonus * similarity) / max(5.0, neighbor.distance_km)
        chosen.append(neighbor)
        weights.append(w)
    weighted_elev_m = None
    elev_pairs = [(w, _neighbor_elev_m(n)) for w, n in zip(weights, chosen) if _neighbor_elev_m(n) is not None]
    if elev_pairs:
        weighted_elev_m = float(sum(w * e for w, e in elev_pairs) / sum(w for w, _ in elev_pairs))
    band_counts = Counter(classify_elevation_band(site_elev_m, _neighbor_elev_m(n)) for n in chosen)
    band_label = band_counts.most_common(1)[0][0] if band_counts else 'unknown'
    return BasinSignal(primary=primary, contributors=list(chosen), weighted_elev_m=weighted_elev_m, band_label=band_label, basin_key=basin_key_for_station(station))


def fetch_snotel_daily(triplet: str, start: date, end: date) -> pd.DataFrame:
    safe = triplet.replace(":", "_")
    cache_path = cache_json_path(f"snotel_{safe}_{start.isoformat()}_{end.isoformat()}.csv")
    text = None
    if cache_path.exists():
        text = cache_path.read_text(encoding="utf-8")
    else:
        elements = "WTEQ::value,SNWD::value,PREC::value,TAVG::value,TMAX::value,TMIN::value"
        url = f"https://wcc.sc.egov.usda.gov/reportGenerator/view_csv/customSingleStationReport/daily/{triplet}%7Cname/{start.isoformat()},{end.isoformat()}/{elements}"
        last_exc = None
        for attempt in range(3):
            try:
                resp = SESSION.get(url, timeout=60)
                resp.raise_for_status()
                text = resp.text
                cache_path.write_text(text, encoding="utf-8")
                break
            except Exception as exc:
                last_exc = exc
                if attempt < 2:
                    print(f"SNOTEL retry {attempt + 1}/3 failed for {triplet}: {exc}")
                    continue
        if text is None:
            print(f"SNOTEL fetch failed for {triplet}; continuing without SNOTEL data: {last_exc}")
            return pd.DataFrame()

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
        href_l = href.lower()
        if "station" in text and "snowfall" in text:
            links["snowfall"] = full
        elif "station" in text and ("snow water equivalent" in text or "swe" in text):
            links["swe"] = full
        elif "station" in text and ("snowdepth" in text or "snow depth" in text):
            links["snowdepth"] = full
        elif "station" in href_l and "snowfall" in href_l and not links["snowfall"]:
            links["snowfall"] = full
        elif "station" in href_l and ("swe" in href_l or "equivalent" in href_l) and not links["swe"]:
            links["swe"] = full
        elif "station" in href_l and ("snowdepth" in href_l or "snow_depth" in href_l) and not links["snowdepth"]:
            links["snowdepth"] = full

    if any(v is None for v in links.values()):
        patterns = {
            'snowfall': r'href=["\']([^"\']*snowfall[^"\']*station[^"\']*)["\']',
            'swe': r'href=["\']([^"\']*(?:swe|equivalent)[^"\']*station[^"\']*)["\']',
            'snowdepth': r'href=["\']([^"\']*snow[^"\']*depth[^"\']*station[^"\']*)["\']',
        }
        for key, pattern in patterns.items():
            if links[key]:
                continue
            m = re.search(pattern, html, flags=re.I)
            if m:
                links[key] = urljoin(base_url, m.group(1))
    return links


def _find_nohrsc_english_text_link(html: str, base_url: str):
    soup = BeautifulSoup(html, "lxml")
    for a in soup.find_all("a"):
        text = (a.get_text(strip=True) or "").lower()
        href = a.get("href")
        if not href:
            continue
        href_l = href.lower()
        if ("text file" in text and "english" in text) or href_l.endswith('.txt') or 'english' in href_l:
            return urljoin(base_url, href)
    m = re.search(r'href=["\']([^"\']+\.txt(?:\?[^"\']*)?)["\']', html, flags=re.I)
    if m:
        return urljoin(base_url, m.group(1))
    return None


def _parse_nohrsc_text_rows(text: str):
    rows = []
    headers = None
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("!"):
            continue
        delim = '|' if '|' in line else None
        parts = [p.strip() for p in (line.split(delim) if delim else re.split(r'\s{2,}', line))]
        if parts and parts[-1] == "":
            parts = parts[:-1]
        if not parts:
            continue
        first = parts[0].lower().replace(' ', '_')
        if headers is None and ('station_id' in first or 'stationid' in first):
            headers = [re.sub(r'[^A-Za-z0-9_()]+', '_', p).strip('_') for p in parts]
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


def weighted_merge_frames(frames: List[Tuple[pd.DataFrame, float]], prefix: str, columns: Sequence[str], primary_frame: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    usable = [(frame.copy(), weight) for frame, weight in frames if frame is not None and not frame.empty and weight > 0]
    if not usable:
        return pd.DataFrame()
    merged = None
    for idx, (frame, weight) in enumerate(usable, start=1):
        renamed = frame.copy()
        suffix_map = {c: f'{c}__{idx}' for c in columns if c in renamed.columns}
        renamed = renamed.rename(columns=suffix_map)
        renamed[f'__weight__{idx}'] = weight
        merged = renamed if merged is None else merged.merge(renamed, on='date', how='outer')
    if merged is None or merged.empty:
        return pd.DataFrame()
    out = merged[['date']].copy()
    weight_sum = pd.Series(0.0, index=merged.index)
    for col in columns:
        numer = pd.Series(0.0, index=merged.index)
        denom = pd.Series(0.0, index=merged.index)
        for idx in range(1, len(usable) + 1):
            data_col = f'{col}__{idx}'
            weight_col = f'__weight__{idx}'
            if data_col not in merged.columns:
                continue
            vals = pd.to_numeric(merged[data_col], errors='coerce')
            w = merged[weight_col].where(vals.notna(), 0.0)
            numer = numer.add(vals.fillna(0.0) * w, fill_value=0.0)
            denom = denom.add(w, fill_value=0.0)
        out[f'{prefix}_{col}'] = numer / denom.replace({0.0: np.nan})
        weight_sum = weight_sum.add(denom.fillna(0.0), fill_value=0.0)
    if primary_frame is not None and not primary_frame.empty:
        primary = primary_frame[['date'] + [c for c in columns if c in primary_frame.columns]].copy()
        primary['date'] = pd.to_datetime(primary['date'])
        out = out.merge(primary, on='date', how='left', suffixes=('', '__primary'))
        for col in columns:
            pcol = f'{col}__primary'
            tcol = f'{prefix}_{col}'
            if pcol in out.columns and tcol in out.columns:
                primary_vals = pd.to_numeric(out[pcol], errors='coerce')
                agg_vals = pd.to_numeric(out[tcol], errors='coerce')
                hi = primary_vals * BASIN_PRIMARY_RATIO_CAP
                lo = primary_vals / BASIN_PRIMARY_RATIO_CAP
                adjusted = agg_vals.where(primary_vals.isna(), np.minimum(np.maximum(agg_vals, lo), hi))
                out[tcol] = adjusted.where(primary_vals.notna(), agg_vals)
                out.drop(columns=[pcol], inplace=True)
    out[f'{prefix}_contributors'] = len(usable)
    out[f'{prefix}_signal_weight'] = weight_sum.replace({0.0: np.nan})
    out['date'] = pd.to_datetime(out['date'])
    return out.sort_values('date')


def build_snotel_basin_series(signal: BasinSignal, start: date, end: date) -> pd.DataFrame:
    frames = []
    for neighbor in signal.contributors:
        try:
            triplet = f"{neighbor.station_id}:{TARGET_STATE}:SNTL"
            frame = fetch_snotel_daily(triplet, start, end)
        except Exception:
            frame = pd.DataFrame()
        weight = 1.0 / max(5.0, neighbor.distance_km)
        if not frame.empty:
            frames.append((frame, weight))
    primary_frame = pd.DataFrame()
    if signal.primary is not None:
        try:
            triplet = f"{signal.primary.station_id}:{signal.primary.meta.get('state', TARGET_STATE)}:SNTL"
            primary_frame = fetch_snotel_daily(triplet, start, end)
        except Exception:
            primary_frame = pd.DataFrame()
    return weighted_merge_frames(frames, 'basin_snotel', ['snotel_wteq_in', 'snotel_snwd_in', 'snotel_prec_in', 'snotel_tavg_f', 'snotel_tmax_f', 'snotel_tmin_f'], primary_frame=primary_frame)


def build_nohrsc_basin_series(signal: BasinSignal, start: date, end: date) -> pd.DataFrame:
    frames = []
    for neighbor in signal.contributors:
        try:
            frame = build_nohrsc_series(neighbor, start, end)
        except Exception:
            frame = pd.DataFrame()
        weight = 1.0 / max(5.0, neighbor.distance_km)
        if not frame.empty:
            frames.append((frame, weight))
    primary_frame = pd.DataFrame()
    if signal.primary is not None:
        try:
            primary_frame = build_nohrsc_series(signal.primary, start, end)
        except Exception:
            primary_frame = pd.DataFrame()
    return weighted_merge_frames(frames, 'basin_nohrsc', ['nohrsc_snowdepth_cm', 'nohrsc_swe_mm', 'nohrsc_snowfall_cm'], primary_frame=primary_frame)


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
    last_exc = None
    for attempt in range(3):
        try:
            resp = SESSION.get(url, timeout=60)
            resp.raise_for_status()
            payload = resp.json()
            break
        except Exception as exc:
            last_exc = exc
            if attempt < 2:
                print(f"Open-Meteo retry {attempt + 1}/3 failed for {lat},{lon}: {exc}")
                continue
            raise
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


def fetch_dynamical_weather_raw(lat: float, lon: float, start: date, end: date) -> pd.DataFrame:
    if FORCE_OPENMETEO_WEATHER:
        return fetch_openmeteo_fallback(lat, lon, start, end)

    def _open_and_extract(url: str, needed_vars: list[str], source_name: str, precip_transform=None) -> pd.DataFrame:
        ds = xr.open_zarr(url, consolidated=None)
        coord_names = set(ds.coords) | set(ds.variables)
        lat_name = "latitude" if "latitude" in coord_names else ("lat" if "lat" in coord_names else None)
        lon_name = "longitude" if "longitude" in coord_names else ("lon" if "lon" in coord_names else None)
        if not lat_name or not lon_name:
            raise RuntimeError(f"{source_name} dataset missing lat/lon coords")

        lon_target = lon if lon >= -180 else lon + 360
        time_slice = slice(np.datetime64(start.isoformat()), np.datetime64((end + timedelta(days=1)).isoformat()))
        lat_vals = ds[lat_name].values
        lon_vals = ds[lon_name].values

        if getattr(lat_vals, 'ndim', 1) == 2 and getattr(lon_vals, 'ndim', 1) == 2:
            lon_comp = lon if np.nanmin(lon_vals) < 0 else lon_target
            dist = (lat_vals - lat) ** 2 + (lon_vals - lon_comp) ** 2
            y_idx, x_idx = np.unravel_index(np.nanargmin(dist), dist.shape)
            indexers = {'y': int(y_idx), 'x': int(x_idx)}
        else:
            lat_idx = nearest_grid_indices(lat_vals, lat)
            lon_idx = nearest_grid_indices(lon_vals, lon if np.nanmin(lon_vals) < 0 else lon_target)
            if 'y' in ds.dims and 'x' in ds.dims and lat_name in ds.coords and lon_name in ds.coords and getattr(ds[lat_name].values, 'ndim', 1) == 2:
                dist = (ds[lat_name].values - lat) ** 2 + (ds[lon_name].values - (lon if np.nanmin(ds[lon_name].values) < 0 else lon_target)) ** 2
                y_idx, x_idx = np.unravel_index(np.nanargmin(dist), dist.shape)
                indexers = {'y': int(y_idx), 'x': int(x_idx)}
            else:
                indexers = {lat_name: int(lat_idx), lon_name: int(lon_idx)}

        available = [v for v in needed_vars if v in ds.variables]
        if not available:
            raise RuntimeError(f"{source_name} missing needed vars: {needed_vars}")
        subset = ds[available].isel(indexers).sel(time=time_slice).load()
        frame = subset.to_dataframe().reset_index().sort_values("time")
        if frame.empty:
            raise RuntimeError(f"{source_name} subset returned no rows")
        frame["date"] = frame["time"].dt.floor("D")
        if precip_transform and precip_transform in frame.columns:
            frame["precip_mm_step"] = pd.to_numeric(frame[precip_transform], errors='coerce').fillna(0.0)
        elif "precipitation_surface" in frame.columns:
            frame["precip_mm_step"] = frame["precipitation_surface"].astype(float).fillna(0) * 3 * 3600
        elif "precipitation" in frame.columns:
            frame["precip_mm_step"] = frame["precipitation"].astype(float).fillna(0)
        else:
            frame["precip_mm_step"] = 0.0
        return frame

    try:
        hrrr_url = f"https://data.dynamical.org/noaa/hrrr/analysis/latest.zarr?email={DYNAMICAL_EMAIL}"
        hrrr_needed = [
            "temperature_2m",
            "maximum_temperature_2m",
            "minimum_temperature_2m",
            "precipitation",
            "categorical_snow_surface",
            "relative_humidity_2m",
            "total_cloud_cover_atmosphere",
        ]
        hrrr = _open_and_extract(hrrr_url, hrrr_needed, "dynamical-hrrr-analysis", precip_transform="precipitation")
        daily = hrrr.groupby("date", as_index=False).agg({
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

        try:
            mrms_url = f"https://data.dynamical.org/noaa/mrms/analysis/latest.zarr?email={DYNAMICAL_EMAIL}"
            mrms = _open_and_extract(mrms_url, ["precipitation"], "dynamical-mrms-analysis", precip_transform="precipitation")
            mrms_daily = mrms.groupby("date", as_index=False).agg({"precip_mm_step": "sum"}).rename(columns={"precip_mm_step": "mrms_precip_mm"})
            daily = daily.merge(mrms_daily, on="date", how="left")
            daily["dyn_precip_mm"] = pd.concat([
                pd.to_numeric(daily.get("mrms_precip_mm"), errors='coerce'),
                pd.to_numeric(daily.get("dyn_precip_mm"), errors='coerce')
            ], axis=1).max(axis=1, skipna=True)
            daily = daily.drop(columns=[c for c in ["mrms_precip_mm"] if c in daily.columns])
            daily["dyn_source"] = "dynamical-hrrr-analysis+mrms"
        except Exception as mrms_exc:
            print(f"MRMS overlay fallback for {lat},{lon}: {mrms_exc}")
            daily["dyn_source"] = "dynamical-hrrr-analysis"
        return daily
    except Exception as hrrr_exc:
        print(f"HRRR analysis fallback for {lat},{lon}: {hrrr_exc}")

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
        return daily
    except Exception as exc:
        print(f"Dynamical weather fallback for {lat},{lon}: {exc}")
        return fetch_openmeteo_fallback(lat, lon, start, end)


def blend_training_weather_frames(primary: pd.DataFrame, secondary: pd.DataFrame) -> pd.DataFrame:
    if primary.empty:
        return secondary
    if secondary.empty:
        return primary
    merged = primary.merge(secondary, on="date", how="outer", suffixes=("_p", "_s"))
    out = pd.DataFrame({"date": merged["date"]})

    def _series(name: str) -> pd.Series:
        if name in merged.columns:
            return pd.to_numeric(merged[name], errors="coerce")
        return pd.Series(np.nan, index=merged.index, dtype=float)

    numeric_cols = ["dyn_temp_c_mean", "dyn_temp_c_max", "dyn_temp_c_min", "dyn_precip_mm", "dyn_rh_pct_mean", "dyn_cloud_pct_mean"]
    for col in numeric_cols:
        p = _series(f"{col}_p")
        s = _series(f"{col}_s")
        out[col] = pd.concat([p, s], axis=1).mean(axis=1, skipna=True)
    p_snow = _series("dyn_snow_flag_p")
    s_snow = _series("dyn_snow_flag_s")
    out["dyn_snow_flag"] = pd.concat([p_snow, s_snow], axis=1).max(axis=1, skipna=True).fillna(0)
    out["dyn_source"] = "blend:dynamical+openmeteo"
    return out.sort_values("date")


def fetch_dynamical_weather(lat: float, lon: float, start: date, end: date) -> pd.DataFrame:
    cache_path = cache_json_path(f"training_weather_{TRAINING_WEATHER_MODE}_{lat:.4f}_{lon:.4f}_{start.isoformat()}_{end.isoformat()}.parquet")
    if cache_path.exists():
        return pd.read_parquet(cache_path)

    if TRAINING_WEATHER_MODE == "openmeteo":
        daily = fetch_openmeteo_fallback(lat, lon, start, end)
    elif TRAINING_WEATHER_MODE == "dynamical":
        daily = fetch_dynamical_weather_raw(lat, lon, start, end)
    else:
        daily = blend_training_weather_frames(
            fetch_dynamical_weather_raw(lat, lon, start, end),
            fetch_openmeteo_fallback(lat, lon, start, end),
        )

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
    out["event_gate_flag"] = 0.0
    area_sqmi = pd.to_numeric(out.get("drain_area_sqmi", pd.Series(np.nan, index=out.index)), errors="coerce")
    area_km2 = pd.to_numeric(out.get("drain_area_km2", pd.Series(np.nan, index=out.index)), errors="coerce")
    discharge = pd.to_numeric(out.get("discharge_cfs", pd.Series(np.nan, index=out.index)), errors="coerce")
    target_discharge = pd.to_numeric(out.get("target_discharge_cfs", pd.Series(np.nan, index=out.index)), errors="coerce")
    out["log_drain_area_sqmi"] = np.log1p(area_sqmi)
    out["log_drain_area_km2"] = np.log1p(area_km2)
    out["specific_discharge_cfs_per_sqmi"] = safe_divide(discharge, area_sqmi)
    out["specific_target_discharge_cfs_per_sqmi"] = safe_divide(target_discharge, area_sqmi)
    out["unit_runoff_mm_day"] = safe_divide(discharge * 0.0283168 * 86400.0 * 1000.0, area_km2 * 1_000_000.0)
    out["target_unit_runoff_mm_day"] = safe_divide(target_discharge * 0.0283168 * 86400.0 * 1000.0, area_km2 * 1_000_000.0)
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
    out["precip_30d_mm"] = precip.rolling(30, min_periods=3).sum()
    out["precip_intensity_ratio"] = safe_divide(out["precip_3d_mm"], out["precip_14d_mm"])
    out["antecedent_wetness_ratio"] = safe_divide(out["precip_7d_mm"], out["precip_30d_mm"])
    out["rain_on_warm_day_mm"] = precip * (temp_mean > 0).astype(float)
    out["rain_on_warm_3d_mm"] = out["rain_on_warm_day_mm"].rolling(3, min_periods=1).sum()
    out["snow_event_3d"] = snow_flag.rolling(3, min_periods=1).sum()
    out["dryness_index"] = 1.0 - np.clip(out["antecedent_wetness_ratio"].fillna(0), 0.0, 1.0)
    out["warm_after_freeze_flag"] = ((out["hard_freeze_flag"].shift(1).fillna(0) > 0) & (out["warm_day_flag"] > 0)).astype(float)
    out["wet_after_dry_flag"] = ((out["precip_3d_mm"] > 5.0) & (out["precip_30d_mm"].fillna(0) < 12.0)).astype(float)
    out["rain_after_freeze_mm"] = precip * out["warm_after_freeze_flag"]
    out["wetness_intensity_interaction"] = out["antecedent_wetness_ratio"].fillna(0) * out["precip_intensity_ratio"].fillna(0)

    snotel_wteq = out.get("snotel_wteq_in")
    snotel_snwd = out.get("snotel_snwd_in")
    snotel_prec = out.get("snotel_prec_in")
    snotel_tavg_f = out.get("snotel_tavg_f")
    nohrsc_sd = out.get("nohrsc_snowdepth_cm")
    nohrsc_swe = out.get("nohrsc_swe_mm")
    basin_snotel_wteq = out.get("basin_snotel_snotel_wteq_in")
    basin_nohrsc_sd = out.get("basin_nohrsc_nohrsc_snowdepth_cm")

    if snotel_tavg_f is not None:
        snotel_tavg_f = pd.to_numeric(snotel_tavg_f, errors="coerce")
        out["snotel_tavg_c"] = (snotel_tavg_f - 32.0) * (5.0 / 9.0)
        out["snotel_degree_day_c"] = out["snotel_tavg_c"].clip(lower=0)
        out["snotel_degree_day_3d"] = out["snotel_degree_day_c"].rolling(3, min_periods=1).sum()
    if snotel_wteq is not None:
        snotel_wteq = pd.to_numeric(snotel_wteq, errors="coerce")
        out["snotel_wteq_change_1d"] = snotel_wteq.diff(1)
        out["snotel_wteq_change_3d"] = snotel_wteq.diff(3)
        out["snotel_wteq_change_7d"] = snotel_wteq.diff(7)
        out["snotel_melt_proxy_in"] = (-out["snotel_wteq_change_1d"]).clip(lower=0)
        out["snotel_melt_proxy_3d"] = out["snotel_melt_proxy_in"].rolling(3, min_periods=1).sum()
        out["snotel_wteq_7d_mean"] = snotel_wteq.rolling(7, min_periods=2).mean()
        out["snotel_wteq_anom"] = snotel_wteq - out["snotel_wteq_7d_mean"]
    if snotel_snwd is not None:
        snotel_snwd = pd.to_numeric(snotel_snwd, errors="coerce")
        out["snotel_snwd_change_1d"] = snotel_snwd.diff(1)
    if snotel_prec is not None:
        snotel_prec = pd.to_numeric(snotel_prec, errors="coerce")
        out["snotel_prec_daily_in"] = snotel_prec.diff(1)
        out["snotel_prec_3d_in"] = out["snotel_prec_daily_in"].rolling(3, min_periods=1).sum()

    if nohrsc_sd is not None:
        nohrsc_sd = pd.to_numeric(nohrsc_sd, errors="coerce")
        out["nohrsc_snowdepth_change_1d"] = nohrsc_sd.diff(1)
        out["nohrsc_snowdepth_change_3d"] = nohrsc_sd.diff(3)
        out["nohrsc_melt_proxy_cm"] = (-out["nohrsc_snowdepth_change_1d"]).clip(lower=0)
        out["nohrsc_melt_proxy_3d"] = out["nohrsc_melt_proxy_cm"].rolling(3, min_periods=1).sum()
    if nohrsc_swe is not None:
        nohrsc_swe = pd.to_numeric(nohrsc_swe, errors="coerce")
        out["nohrsc_swe_change_1d"] = nohrsc_swe.diff(1)

    site_elev = out.get("site_elev_m", pd.Series(np.nan, index=out.index))
    snotel_elev = out.get("snotel_elev_m", pd.Series(np.nan, index=out.index))
    nohrsc_elev = out.get("nohrsc_elev_m", pd.Series(np.nan, index=out.index))
    out["temp_at_snotel_c"] = temp_mean - ((snotel_elev - site_elev) / 1000.0) * TEMP_LAPSE_C_PER_KM
    out["temp_at_nohrsc_c"] = temp_mean - ((nohrsc_elev - site_elev) / 1000.0) * TEMP_LAPSE_C_PER_KM
    if 'basin_snotel_elev_m' in out:
        out['temp_at_basin_snotel_c'] = temp_mean - ((out['basin_snotel_elev_m'] - site_elev) / 1000.0) * TEMP_LAPSE_C_PER_KM
    if 'basin_nohrsc_elev_m' in out:
        out['temp_at_basin_nohrsc_c'] = temp_mean - ((out['basin_nohrsc_elev_m'] - site_elev) / 1000.0) * TEMP_LAPSE_C_PER_KM
    upper_cols = [c for c in ["temp_at_snotel_c", "temp_at_nohrsc_c", 'temp_at_basin_snotel_c', 'temp_at_basin_nohrsc_c'] if c in out]
    out["temp_at_upper_basin_c"] = out[upper_cols].mean(axis=1, skipna=True)
    out["high_elev_snow_fraction"] = out["temp_at_upper_basin_c"].apply(snow_fraction_from_temp)
    out["basin_snow_mm_elev"] = precip * out["high_elev_snow_fraction"].fillna(snow_flag)
    out["basin_rain_mm_elev"] = precip - out["basin_snow_mm_elev"].fillna(0)
    out["basin_snow_3d_mm_elev"] = out["basin_snow_mm_elev"].rolling(3, min_periods=1).sum()
    out["basin_rain_3d_mm_elev"] = out["basin_rain_mm_elev"].rolling(3, min_periods=1).sum()
    out["warm_upper_basin_flag"] = (out["temp_at_upper_basin_c"] > 0.5).astype(float)
    out["high_elev_melt_pressure_c"] = out["temp_at_upper_basin_c"].clip(lower=0)
    out["high_elev_melt_3d_c"] = out["high_elev_melt_pressure_c"].rolling(3, min_periods=1).sum()

    snowpack_wteq = basin_snotel_wteq if basin_snotel_wteq is not None else snotel_wteq
    snowpack_sd = basin_nohrsc_sd if basin_nohrsc_sd is not None else nohrsc_sd
    if snowpack_wteq is not None:
        out['basin_swe_storage_in'] = snowpack_wteq
        out['basin_swe_storage_3d_mean_in'] = snowpack_wteq.rolling(3, min_periods=1).mean()
    if snowpack_sd is not None:
        out['basin_snowdepth_storage_cm'] = snowpack_sd
        out['basin_snowdepth_storage_3d_mean_cm'] = snowpack_sd.rolling(3, min_periods=1).mean()
    if "degree_day_c" in out and snowpack_wteq is not None:
        out["degree_day_x_swe"] = out["degree_day_c"] * snowpack_wteq.fillna(0)
    if "rain_on_warm_day_mm" in out and snowpack_wteq is not None:
        out["rain_on_snow_proxy"] = out["rain_on_warm_day_mm"] * (snowpack_wteq.fillna(0) > 0.5).astype(float)
        out["rain_on_snow_proxy_elev"] = out["basin_rain_mm_elev"].fillna(0) * (snowpack_wteq.fillna(0) > 0.5).astype(float) * out["warm_upper_basin_flag"].fillna(0)
        out["degree_day_x_swe_elev"] = out["high_elev_melt_pressure_c"].fillna(0) * snowpack_wteq.fillna(0)
    if "degree_day_c" in out and snowpack_sd is not None:
        out["degree_day_x_nohrsc_sd"] = out["degree_day_c"] * snowpack_sd.fillna(0)
        out["degree_day_x_nohrsc_sd_elev"] = out["high_elev_melt_pressure_c"].fillna(0) * snowpack_sd.fillna(0)
    out["melt_risk_elev"] = out[[c for c in ["degree_day_x_swe_elev", "degree_day_x_nohrsc_sd_elev"] if c in out]].sum(axis=1, min_count=1)
    out["melt_risk_elev"] = np.clip(out["melt_risk_elev"], 0.0, 250.0)
    if 'basin_snotel_signal_weight' in out or 'basin_nohrsc_signal_weight' in out:
        cols = [c for c in ['basin_snotel_signal_weight', 'basin_nohrsc_signal_weight'] if c in out]
        out['basin_signal_weight'] = out[cols].sum(axis=1, min_count=1)

    discharge = out.get("discharge_cfs")
    if discharge is not None:
        out["q_change_1d"] = discharge.diff(1)
        out["q_change_3d"] = discharge.diff(3)
        out["q_ratio_1d"] = safe_divide(discharge, discharge.shift(1))
        out["q_ratio_7d_mean"] = safe_divide(discharge, discharge.shift(1).rolling(7, min_periods=2).mean())
        out["runoff_response_index"] = safe_divide(out.get("q_lag1"), out.get("precip_7d_mm")) if "q_lag1" in out else np.nan
        out["rising_flow_flag"] = ((out["q_change_1d"] > 0) & (out["q_ratio_1d"] > 1.02)).astype(float)
    else:
        out["rising_flow_flag"] = 0.0

    warm_upper = out.get("warm_upper_basin_flag", pd.Series(0.0, index=out.index)).fillna(0)
    melt_risk = out.get("melt_risk_elev", pd.Series(0.0, index=out.index)).fillna(0)
    ros = out.get("rain_on_snow_proxy_elev", pd.Series(0.0, index=out.index)).fillna(0)
    basin_rain = out.get("basin_rain_3d_mm_elev", pd.Series(0.0, index=out.index)).fillna(0)
    wetness = out.get("antecedent_wetness_ratio", pd.Series(0.0, index=out.index)).fillna(0)
    q_ratio = out.get("q_ratio_1d", pd.Series(1.0, index=out.index)).fillna(1.0)
    out["event_gate_flag"] = ((out["rising_flow_flag"] > 0) | (warm_upper > 0) | (melt_risk > 15.0) | (ros > 2.0) | (basin_rain > 8.0) | (wetness > 0.45)).astype(float)
    out["primed_basin_flag"] = ((wetness > 0.45) | (out["warm_after_freeze_flag"] > 0) | (out["wet_after_dry_flag"] > 0)).astype(float)
    out["primed_rain_interaction"] = out["primed_basin_flag"] * np.clip(basin_rain, 0.0, None)
    out["primed_melt_interaction"] = out["primed_basin_flag"] * np.clip(melt_risk, 0.0, None)
    out["event_gate_strength"] = np.clip(
        0.16 * out["rising_flow_flag"].fillna(0)
        + 0.15 * warm_upper
        + 0.20 * np.clip(melt_risk / 75.0, 0.0, 1.0)
        + 0.15 * np.clip(ros / 10.0, 0.0, 1.0)
        + 0.11 * np.clip(basin_rain / 20.0, 0.0, 1.0)
        + 0.10 * np.clip(wetness, 0.0, 1.0)
        + 0.07 * out["warm_after_freeze_flag"].fillna(0)
        + 0.06 * out["wet_after_dry_flag"].fillna(0),
        0.0,
        1.0,
    )
    out["extreme_event_flag"] = ((q_ratio > 1.08) | (melt_risk > 35.0) | (ros > 5.0) | (basin_rain > 16.0)).astype(float)
    out["extreme_event_strength"] = np.clip(
        0.30 * np.clip((q_ratio - 1.0) / 0.20, 0.0, 1.0)
        + 0.25 * np.clip(melt_risk / 120.0, 0.0, 1.0)
        + 0.20 * np.clip(ros / 15.0, 0.0, 1.0)
        + 0.15 * np.clip(basin_rain / 30.0, 0.0, 1.0)
        + 0.10 * np.clip(wetness, 0.0, 1.0),
        0.0,
        1.0,
    )

    return out


def assemble_station_dataset(station: dict, snotel_signal: BasinSignal, nohrsc_signal: BasinSignal, start: date, end: date) -> pd.DataFrame:
    runoff = fetch_usgs_daily_discharge(station["stationId"], start, end + timedelta(days=TARGET_HORIZON_DAYS))
    if runoff.empty:
        return runoff
    runoff = runoff.sort_values("date")
    weather = fetch_dynamical_weather(station["latitude"], station["longitude"], start, end)

    frames = [runoff, weather]
    snotel_neighbor = snotel_signal.primary
    nohrsc_neighbor = nohrsc_signal.primary
    if snotel_neighbor:
        triplet = f"{snotel_neighbor.station_id}:{snotel_neighbor.meta.get('state', TARGET_STATE)}:SNTL"
        snotel = fetch_snotel_daily(triplet, start, end)
        if not snotel.empty:
            frames.append(snotel)
    basin_snotel = build_snotel_basin_series(snotel_signal, start, end) if snotel_signal.contributors else pd.DataFrame()
    if not basin_snotel.empty:
        frames.append(basin_snotel)
    if nohrsc_neighbor:
        nohrsc = build_nohrsc_series(nohrsc_neighbor, start, end)
        if not nohrsc.empty:
            frames.append(nohrsc)
    basin_nohrsc = build_nohrsc_basin_series(nohrsc_signal, start, end) if nohrsc_signal.contributors else pd.DataFrame()
    if not basin_nohrsc.empty:
        frames.append(basin_nohrsc)

    df = frames[0]
    for frame in frames[1:]:
        df = df.merge(frame, on="date", how="left")

    site_elev_m = fetch_location_elevation_m(station["latitude"], station["longitude"])
    snotel_elev_m = _to_meters((snotel_neighbor.meta or {}).get("elevation")) if snotel_neighbor else None
    nohrsc_elev_m = _neighbor_elev_m(nohrsc_neighbor) if nohrsc_neighbor else None
    df["site_elev_m"] = site_elev_m
    df["snotel_elev_m"] = snotel_elev_m
    df["nohrsc_elev_m"] = nohrsc_elev_m
    df['basin_snotel_elev_m'] = snotel_signal.weighted_elev_m
    df['basin_nohrsc_elev_m'] = nohrsc_signal.weighted_elev_m
    df["snotel_minus_site_elev_m"] = None if site_elev_m is None or snotel_elev_m is None else snotel_elev_m - site_elev_m
    df["nohrsc_minus_site_elev_m"] = None if site_elev_m is None or nohrsc_elev_m is None else nohrsc_elev_m - site_elev_m

    df = add_lag_features(df, "discharge_cfs", "q")
    if "dyn_precip_mm" in df:
        df = add_lag_features(df, "dyn_precip_mm", "dyn_precip")
    if "nohrsc_snowdepth_cm" in df:
        df = add_lag_features(df, "nohrsc_snowdepth_cm", "nohrsc_sd")
    if "snotel_wteq_in" in df:
        df = add_lag_features(df, "snotel_wteq_in", "snotel_wteq")

    df["target_discharge_cfs"] = df["discharge_cfs"].shift(-TARGET_HORIZON_DAYS)
    meta = fetch_usgs_site_metadata(station["stationId"])
    regulation = build_regulation_signal(station)
    camels = fetch_camels_attributes(station["stationId"]).values
    hydrobasins = fetch_hydrobasins_attributes(station["stationId"]).values
    df["drain_area_sqmi"] = meta.get("drain_area_sqmi")
    df["drain_area_km2"] = meta.get("drain_area_km2")
    df["contrib_drain_area_sqmi"] = meta.get("contrib_drain_area_sqmi")
    df["contrib_drain_area_km2"] = meta.get("contrib_drain_area_km2")
    df["site_altitude_ft_usgs"] = meta.get("site_altitude_ft")
    df["site_altitude_m_usgs"] = meta.get("site_altitude_m")
    df["huc_cd"] = meta.get("huc_cd")
    df["basin_cd_usgs"] = meta.get("basin_cd")
    df["nearest_dam_distance_km"] = regulation.nearest_dam_distance_km
    df["dams_within_25km"] = regulation.dams_within_25km
    df["dams_within_50km"] = regulation.dams_within_50km
    df["dams_within_100km"] = regulation.dams_within_100km
    df["nearest_reservoir_distance_km"] = regulation.nearest_reservoir_distance_km
    df["nearest_reservoir_storage_mcm"] = regulation.nearest_reservoir_storage_mcm
    df["reservoir_storage_mcm_within_50km"] = regulation.reservoir_storage_mcm_within_50km
    df["reservoir_storage_mcm_within_100km"] = regulation.reservoir_storage_mcm_within_100km
    for key, value in camels.items():
        df[f"camels_{key}"] = value
    for key, value in hydrobasins.items():
        df[f"hydrobasins_{key}"] = value
    descriptor_source = []
    if camels:
        descriptor_source.append("camels")
    if hydrobasins:
        descriptor_source.append("hydrobasins")
    df["static_descriptor_source"] = "+".join(descriptor_source) if descriptor_source else "none"
    df["has_camels_static"] = 1 if camels else 0
    df["has_hydrobasins_static"] = 1 if hydrobasins else 0
    df = add_hydrology_features(df)
    df["station_id"] = station["stationId"]
    df["station_name"] = station["name"]
    df["latitude"] = station["latitude"]
    df["longitude"] = station["longitude"]
    df["state"] = station["state"]
    df["snotel_station_id"] = snotel_neighbor.station_id if snotel_neighbor else None
    df["snotel_distance_km"] = round(snotel_neighbor.distance_km, 2) if snotel_neighbor else None
    df["nohrsc_station_id"] = nohrsc_neighbor.station_id if nohrsc_neighbor else None
    df["nohrsc_distance_km"] = round(nohrsc_neighbor.distance_km, 2) if nohrsc_neighbor else None
    df['basin_key'] = snotel_signal.basin_key or nohrsc_signal.basin_key or basin_key_for_station(station)
    df['snotel_basin_band'] = snotel_signal.band_label
    df['nohrsc_basin_band'] = nohrsc_signal.band_label
    df['snotel_basin_contributors'] = len(snotel_signal.contributors)
    df['nohrsc_basin_contributors'] = len(nohrsc_signal.contributors)
    station_area = (station["name"] or "").lower()
    if "yellowstone" in station_area or "gallatin river at logan" in station_area:
        df["station_cluster"] = "mainstem"
    elif "creek" in station_area or "shields" in station_area or "hyalite" in station_area:
        df["station_cluster"] = "tributary"
    else:
        df["station_cluster"] = "corridor"
    return df


def load_or_assemble_station_dataset(station: dict, snotel_signal: BasinSignal, nohrsc_signal: BasinSignal, start: date, end: date) -> pd.DataFrame:
    cache_path = station_dataset_cache_path(station["stationId"], start, end)
    if cache_path.exists():
        try:
            cached = pd.read_parquet(cache_path)
            if cached is not None and not cached.empty:
                required_columns = {
                    "static_descriptor_source",
                    "has_camels_static",
                    "has_hydrobasins_static",
                    "station_dataset_schema_version",
                }
                if required_columns.issubset(set(cached.columns)):
                    versions = set(str(v) for v in cached["station_dataset_schema_version"].dropna().unique()) if "station_dataset_schema_version" in cached.columns else set()
                    if STATION_DATASET_SCHEMA_VERSION in versions:
                        return cached
        except Exception:
            pass
    frame = assemble_station_dataset(station, snotel_signal, nohrsc_signal, start, end)
    if frame is not None and not frame.empty:
        frame["station_dataset_schema_version"] = STATION_DATASET_SCHEMA_VERSION
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(cache_path, index=False)
    return frame


def merge_station_dataset_cache(start: date, end: date) -> Optional[pd.DataFrame]:
    pattern = f"station_*_{start.isoformat()}_{end.isoformat()}.parquet"
    frames = []
    for path in sorted(DATASET_DIR.glob(pattern)):
        try:
            frame = pd.read_parquet(path)
        except Exception:
            continue
        if frame is None or frame.empty or 'station_id' not in frame.columns:
            continue
        usable = frame.dropna(subset=["target_discharge_cfs"]) if "target_discharge_cfs" in frame.columns else pd.DataFrame()
        if usable.empty or len(usable) < 10:
            continue
        frames.append(frame)
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True).sort_values(["date", "station_id"])


def feature_columns_for_mode(dataset: pd.DataFrame, mode: str) -> List[str]:
    exclude = {"date", "station_name", "state", "target_discharge_cfs", "snotel_station_id", "nohrsc_station_id"}
    numeric_features = [c for c in dataset.columns if c not in exclude and pd.api.types.is_numeric_dtype(dataset[c])]
    if mode == "runoff_only":
        numeric_features = [c for c in numeric_features if c.startswith("q_") or c in {"discharge_cfs", "latitude", "longitude", "drain_area_sqmi", "drain_area_km2", "log_drain_area_sqmi", "log_drain_area_km2", "specific_discharge_cfs_per_sqmi", "unit_runoff_mm_day"}]
        return sorted(set(numeric_features))

    zero_shot_prefixes = (
        "camels_",
        "hydrobasins_",
        "nearest_dam_",
        "nearest_reservoir_",
        "reservoir_storage_",
        "dams_within_",
    )

    if FULL_FEATURE_SET_MODE == "core":
        core_prefixes = (
            "q_",
            "dyn_precip",
            "precip_",
            "degree_day",
            "freeze_degree_day",
            "snotel_wteq",
            "snotel_melt",
            "nohrsc_snowdepth",
            "nohrsc_melt",
            "basin_swe",
            "basin_snowdepth",
            "basin_rain",
            "basin_snow",
            "high_elev_melt",
            "rain_on_snow",
            "rain_on_warm",
            "antecedent_wetness",
            "runoff_response",
            "event_gate",
            "extreme_event",
            "temp_at_upper_basin",
            "warm_upper_basin",
            "melt_risk",
            "temp_range",
        )
        core_exact = {
            "discharge_cfs", "latitude", "longitude", "drain_area_sqmi", "drain_area_km2", "log_drain_area_sqmi", "log_drain_area_km2", "specific_discharge_cfs_per_sqmi", "unit_runoff_mm_day",
            "dyn_temp_c_mean", "dyn_temp_c_max", "dyn_temp_c_min", "dyn_snow_flag",
            "rising_flow_flag", "warm_day_flag", "hard_freeze_flag",
        }
        numeric_features = [
            c for c in numeric_features
            if c in core_exact or c.startswith(core_prefixes) or c.startswith(zero_shot_prefixes)
        ]
    elif FULL_FEATURE_SET_MODE in {"core_no_geo", "exp2_transition"}:
        core_prefixes = (
            "q_", "dyn_precip", "precip_", "degree_day", "freeze_degree_day", "snotel_wteq", "snotel_melt",
            "nohrsc_snowdepth", "nohrsc_melt", "basin_swe", "basin_snowdepth", "basin_rain", "basin_snow",
            "high_elev_melt", "rain_on_snow", "rain_on_warm", "antecedent_wetness", "runoff_response",
            "event_gate", "extreme_event", "temp_at_upper_basin", "warm_upper_basin", "melt_risk", "temp_range",
        )
        core_exact = {"discharge_cfs", "dyn_temp_c_mean", "dyn_temp_c_max", "dyn_temp_c_min", "dyn_snow_flag", "rising_flow_flag", "warm_day_flag", "hard_freeze_flag", "drain_area_sqmi", "drain_area_km2", "log_drain_area_sqmi", "log_drain_area_km2", "specific_discharge_cfs_per_sqmi", "unit_runoff_mm_day"}
        numeric_features = [c for c in numeric_features if c in core_exact or c.startswith(core_prefixes) or c.startswith(zero_shot_prefixes)]
        if FULL_FEATURE_SET_MODE == "exp2_transition" or FEATURE_EXPERIMENT == "true_exp2":
            exp2_exact = {
                "dryness_index", "warm_after_freeze_flag", "wet_after_dry_flag", "rain_after_freeze_mm",
                "wetness_intensity_interaction", "primed_basin_flag", "primed_rain_interaction", "primed_melt_interaction",
            }
            numeric_features = sorted(set(numeric_features).union(c for c in dataset.columns if c in exp2_exact))
    features = sorted(set(numeric_features))
    categorical_candidates = []
    if CATEGORICAL_MODE == "none":
        categorical_candidates = []
    elif CATEGORICAL_MODE == "cluster_only":
        categorical_candidates = ["station_cluster"]
    elif CATEGORICAL_MODE == "station_id_only":
        categorical_candidates = ["station_id"]
    elif CATEGORICAL_MODE == "basin_only":
        categorical_candidates = ["basin_key"]
    elif CATEGORICAL_MODE == "station_and_basin":
        categorical_candidates = ["station_id", "basin_key"]
    else:
        categorical_candidates = ["station_id", "station_cluster", "basin_key"]
    categorical = [col for col in categorical_candidates if col in dataset.columns]
    return features + categorical


def build_model(features: Sequence[str], *, log_target: bool = True) -> Pipeline:
    categorical_feature_names = {"station_id", "station_cluster", "basin_key"}
    numeric_features = [f for f in features if f not in categorical_feature_names]
    categorical_features = [f for f in features if f in categorical_feature_names]
    transformers = []
    if numeric_features:
        transformers.append(("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), list(numeric_features)))
    if categorical_features:
        transformers.append(("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), list(categorical_features)))
    pre = ColumnTransformer(transformers=transformers, remainder="drop")
    reg = Ridge(alpha=1.0)
    pipe = Pipeline([
        ("pre", pre),
        ("reg", reg),
    ])
    if categorical_features and log_target:
        return TransformedTargetRegressor(regressor=pipe, func=np.log1p, inverse_func=np.expm1)
    return pipe


def metric_summary(actual: pd.Series, pred: Sequence[float]) -> dict:
    actual_arr = pd.Series(actual).reset_index(drop=True).astype(float)
    pred_arr = pd.Series(pred).reset_index(drop=True).astype(float)
    finite_mask = np.isfinite(actual_arr.to_numpy(dtype=float, na_value=np.nan)) & np.isfinite(pred_arr.to_numpy(dtype=float, na_value=np.nan))
    mask = (actual_arr.notna() & pred_arr.notna() & pd.Series(finite_mask, index=actual_arr.index))
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


def slice_metric_summary(actual: pd.Series, pred: Sequence[float], mask: pd.Series) -> dict:
    mask = pd.Series(mask).reset_index(drop=True).fillna(False).astype(bool)
    actual_arr = pd.Series(actual).reset_index(drop=True)
    pred_arr = pd.Series(pred).reset_index(drop=True)
    if len(mask) != len(actual_arr):
        mask = mask.reindex(range(len(actual_arr)), fill_value=False)
    return metric_summary(actual_arr[mask], pred_arr[mask])


def residual_mask_from_mode(frame: pd.DataFrame, mode: str) -> pd.Series:
    mode = (mode or "all").strip().lower()
    if mode == "event":
        return (frame.get("event_gate_flag", pd.Series(0.0, index=frame.index)).fillna(0) > 0)
    if mode == "rising":
        return (frame.get("rising_flow_flag", pd.Series(0.0, index=frame.index)).fillna(0) > 0)
    if mode == "extreme":
        return (frame.get("extreme_event_strength", pd.Series(0.0, index=frame.index)).fillna(0.0) >= 0.35)
    if mode == "highflow":
        q = pd.to_numeric(frame.get("q_lag1", pd.Series(np.nan, index=frame.index)), errors="coerce")
        thresh = q.quantile(0.8)
        return q >= thresh
    return pd.Series(True, index=frame.index)


def clip_residuals(values: Sequence[float], mode: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    mode = (mode or "none").strip().lower()
    if mode == "positive":
        return np.clip(arr, 0.0, None)
    if mode == "negative":
        return np.clip(arr, None, 0.0)
    if mode == "small":
        return np.clip(arr, -100.0, 100.0)
    return arr


def gate_strength_from_mode(frame: pd.DataFrame, mode: str) -> pd.Series:
    mode = (mode or "event").strip().lower()
    event = frame.get("event_gate_strength", pd.Series(0.0, index=frame.index)).fillna(0.0)
    rising = frame.get("rising_flow_flag", pd.Series(0.0, index=frame.index)).fillna(0.0)
    extreme = frame.get("extreme_event_strength", pd.Series(0.0, index=frame.index)).fillna(0.0)
    if mode == "rising":
        return np.clip(rising, 0.0, 1.0)
    if mode == "extreme":
        return np.clip(extreme, 0.0, 1.0)
    if mode == "max":
        return np.clip(np.maximum(np.asarray(event, dtype=float), np.asarray(extreme, dtype=float)), 0.0, 1.0)
    if mode == "rising_plus_extreme":
        return np.clip(0.55 * np.asarray(rising, dtype=float) + 0.45 * np.asarray(extreme, dtype=float), 0.0, 1.0)
    return np.clip(event, 0.0, 1.0)


def blend_residual_prediction(runoff_pred: Sequence[float], residual_pred: Sequence[float], gate_strength: Sequence[float], extreme_strength: Optional[Sequence[float]] = None, base_weight: float = BLEND_BASE_WEIGHT, max_weight: float = BLEND_MAX_WEIGHT, extreme_bonus: float = BLEND_EXTREME_BONUS) -> np.ndarray:
    runoff_arr = np.asarray(runoff_pred, dtype=float)
    residual_arr = np.asarray(residual_pred, dtype=float)
    gate_arr = np.clip(np.asarray(gate_strength, dtype=float), 0.0, 1.0)
    extreme_arr = np.zeros_like(gate_arr) if extreme_strength is None else np.clip(np.asarray(extreme_strength, dtype=float), 0.0, 1.0)
    residual_weight = np.clip(base_weight + (max_weight - base_weight) * gate_arr + extreme_bonus * extreme_arr, 0.0, 0.97)
    return np.maximum(0.0, runoff_arr + residual_arr * residual_weight)


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


def train_model(dataset: pd.DataFrame) -> Tuple[dict, dict, List[str]]:
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
    pred_train_full = full_model.predict(train[full_features])
    pred_test_full = full_model.predict(test[full_features])

    runoff_only_model = build_model(runoff_features)
    runoff_only_model.fit(train[runoff_features], train["target_discharge_cfs"])
    pred_train_runoff_only = runoff_only_model.predict(train[runoff_features])
    pred_test_runoff_only = runoff_only_model.predict(test[runoff_features])

    residual_target_train = train["target_discharge_cfs"].to_numpy() - pred_train_runoff_only
    residual_model = build_model(full_features, log_target=False)
    residual_train_mask = residual_mask_from_mode(train, RESIDUAL_TRAIN_MASK_MODE)
    residual_fit_x = train.loc[residual_train_mask, full_features]
    residual_fit_y = residual_target_train[residual_train_mask.to_numpy()]
    if residual_fit_x.empty:
        residual_train_mask = pd.Series(True, index=train.index)
        residual_fit_x = train.loc[residual_train_mask, full_features]
        residual_fit_y = residual_target_train[residual_train_mask.to_numpy()]
    residual_model.fit(residual_fit_x, residual_fit_y)
    pred_train_residual_component = clip_residuals(residual_model.predict(train[full_features]), RESIDUAL_CLIP_MODE)
    pred_test_residual_component = clip_residuals(residual_model.predict(test[full_features]), RESIDUAL_CLIP_MODE)
    pred_train_residual_hybrid = np.maximum(0.0, pred_train_runoff_only + pred_train_residual_component)
    pred_test_residual_hybrid = np.maximum(0.0, pred_test_runoff_only + pred_test_residual_component)
    pred_test_event_gated = blend_residual_prediction(
        pred_test_runoff_only,
        pred_test_residual_component,
        gate_strength_from_mode(test, BLEND_GATE_MODE),
        test.get("extreme_event_strength", pd.Series(0.0, index=test.index)).fillna(0.0),
    )
    pred_test_ensemble_blend = np.maximum(
        0.0,
        ENSEMBLE_RUNOFF_WEIGHT * np.asarray(pred_test_runoff_only, dtype=float)
        + (1.0 - ENSEMBLE_RUNOFF_WEIGHT) * np.asarray(pred_test_event_gated, dtype=float),
    )

    benchmark_preds = make_benchmark_predictions(test, train)
    rising_mask = test.get("rising_flow_flag", pd.Series(0.0, index=test.index)).fillna(0) > 0
    event_mask = test.get("event_gate_flag", pd.Series(0.0, index=test.index)).fillna(0) > 0
    high_flow_threshold = float(train["target_discharge_cfs"].quantile(0.9)) if not train.empty else float(test["target_discharge_cfs"].quantile(0.9))
    high_flow_mask = test["target_discharge_cfs"].fillna(-np.inf) >= high_flow_threshold

    model_predictions = {
        "full_model": pred_test_full,
        "runoff_only": pred_test_runoff_only,
        "residual_hybrid": pred_test_residual_hybrid,
        "event_gated_residual": pred_test_event_gated,
        "ensemble_blend": pred_test_ensemble_blend,
    }

    metrics = {
        "rows_total": int(len(dataset)),
        "rows_train": int(len(train)),
        "rows_test": int(len(test)),
        "target_horizon_days": TARGET_HORIZON_DAYS,
        "full_model_train": metric_summary(train["target_discharge_cfs"], pred_train_full),
        "runoff_only_train": metric_summary(train["target_discharge_cfs"], pred_train_runoff_only),
        "residual_hybrid_train": metric_summary(train["target_discharge_cfs"], pred_train_residual_hybrid),
        "full_model_test": metric_summary(test["target_discharge_cfs"], pred_test_full),
        "runoff_only_test": metric_summary(test["target_discharge_cfs"], pred_test_runoff_only),
        "residual_hybrid_test": metric_summary(test["target_discharge_cfs"], pred_test_residual_hybrid),
        "event_gated_residual_test": metric_summary(test["target_discharge_cfs"], pred_test_event_gated),
        "ensemble_blend_test": metric_summary(test["target_discharge_cfs"], pred_test_ensemble_blend),
        "benchmarks_test": {name: metric_summary(test["target_discharge_cfs"], values) for name, values in benchmark_preds.items()},
        "slice_metrics": {
            "rising_flow_days": {
                label: slice_metric_summary(test["target_discharge_cfs"], pred, rising_mask)
                for label, pred in model_predictions.items()
            },
            "event_gate_days": {
                label: slice_metric_summary(test["target_discharge_cfs"], pred, event_mask)
                for label, pred in model_predictions.items()
            },
            "high_flow_days": {
                label: slice_metric_summary(test["target_discharge_cfs"], pred, high_flow_mask)
                for label, pred in model_predictions.items()
            },
        },
        "high_flow_threshold_cfs": high_flow_threshold,
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

        fold_residual = build_model(full_features, log_target=False)
        fold_residual_target = fold_train["target_discharge_cfs"].to_numpy() - fold_runoff.predict(fold_train[runoff_features])
        fold_residual_mask = residual_mask_from_mode(fold_train, RESIDUAL_TRAIN_MASK_MODE)
        fold_residual_x = fold_train.loc[fold_residual_mask, full_features]
        fold_residual_y = fold_residual_target[fold_residual_mask.to_numpy()]
        if fold_residual_x.empty:
            fold_residual_x = fold_train[full_features]
            fold_residual_y = fold_residual_target
        fold_residual.fit(fold_residual_x, fold_residual_y)
        fold_pred_residual_component = clip_residuals(fold_residual.predict(fold_test[full_features]), RESIDUAL_CLIP_MODE)
        fold_pred_residual = np.maximum(0.0, fold_pred_runoff + fold_pred_residual_component)
        add_fold_metrics(fold_records, "residual_hybrid", f"fold_{idx}", None, fold_test["target_discharge_cfs"], fold_pred_residual)
        fold_pred_event = blend_residual_prediction(
            fold_pred_runoff,
            fold_pred_residual_component,
            gate_strength_from_mode(fold_test, BLEND_GATE_MODE),
            fold_test.get("extreme_event_strength", pd.Series(0.0, index=fold_test.index)).fillna(0.0),
        )
        add_fold_metrics(fold_records, "event_gated_residual", f"fold_{idx}", None, fold_test["target_discharge_cfs"], fold_pred_event)
        fold_pred_ensemble = np.maximum(
            0.0,
            ENSEMBLE_RUNOFF_WEIGHT * np.asarray(fold_pred_runoff, dtype=float)
            + (1.0 - ENSEMBLE_RUNOFF_WEIGHT) * np.asarray(fold_pred_event, dtype=float),
        )
        add_fold_metrics(fold_records, "ensemble_blend", f"fold_{idx}", None, fold_test["target_discharge_cfs"], fold_pred_ensemble)

        fold_bench = make_benchmark_predictions(fold_test, fold_train)
        for name, values in fold_bench.items():
            add_fold_metrics(fold_records, name, f"fold_{idx}", None, fold_test["target_discharge_cfs"], values)

        per_station = []
        full_series = pd.Series(fold_pred_full, index=fold_test.index)
        runoff_series = pd.Series(fold_pred_runoff, index=fold_test.index)
        residual_series = pd.Series(fold_pred_residual, index=fold_test.index)
        event_series = pd.Series(fold_pred_event, index=fold_test.index)
        ensemble_series = pd.Series(fold_pred_ensemble, index=fold_test.index)
        for station_id, group in fold_test.groupby("station_id"):
            mask = group.index
            add_fold_metrics(fold_records, "full_model", f"fold_{idx}", str(station_id), group["target_discharge_cfs"], full_series.loc[mask])
            add_fold_metrics(fold_records, "runoff_only", f"fold_{idx}", str(station_id), group["target_discharge_cfs"], runoff_series.loc[mask])
            add_fold_metrics(fold_records, "residual_hybrid", f"fold_{idx}", str(station_id), group["target_discharge_cfs"], residual_series.loc[mask])
            add_fold_metrics(fold_records, "event_gated_residual", f"fold_{idx}", str(station_id), group["target_discharge_cfs"], event_series.loc[mask])
            add_fold_metrics(fold_records, "ensemble_blend", f"fold_{idx}", str(station_id), group["target_discharge_cfs"], ensemble_series.loc[mask])
            per_station.append(str(station_id))
        fold_summaries.append({
            "fold_id": f"fold_{idx}",
            "train_end": pd.Timestamp(train_end).date().isoformat(),
            "validation_end": pd.Timestamp(val_end).date().isoformat(),
            "rows_train": int(len(fold_train)),
            "rows_validation": int(len(fold_test)),
            "rows_rising_flow": int((fold_test.get("rising_flow_flag", pd.Series(0.0, index=fold_test.index)).fillna(0) > 0).sum()),
            "rows_event_gate": int((fold_test.get("event_gate_flag", pd.Series(0.0, index=fold_test.index)).fillna(0) > 0).sum()),
            "rows_high_flow": int((fold_test["target_discharge_cfs"].fillna(-np.inf) >= high_flow_threshold).sum()),
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

    test_out = test[["date", "station_id", "station_name", "discharge_cfs", "target_discharge_cfs", "rising_flow_flag", "event_gate_flag", "event_gate_strength"]].copy()
    test_out["prediction_cfs"] = pred_test_full
    test_out["runoff_only_prediction_cfs"] = pred_test_runoff_only
    test_out["residual_hybrid_prediction_cfs"] = pred_test_residual_hybrid
    test_out["event_gated_residual_prediction_cfs"] = pred_test_event_gated
    test_out["ensemble_blend_prediction_cfs"] = pred_test_ensemble_blend
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
            "residual_hybrid": metric_summary(group["target_discharge_cfs"], group["residual_hybrid_prediction_cfs"]),
            "event_gated_residual": metric_summary(group["target_discharge_cfs"], group["event_gated_residual_prediction_cfs"]),
            "ensemble_blend": metric_summary(group["target_discharge_cfs"], group["ensemble_blend_prediction_cfs"]),
            "persistence_lag1": metric_summary(group["target_discharge_cfs"], group["benchmark_persistence_lag1_cfs"]),
            "rising_flow_days": {
                "full_model": slice_metric_summary(group["target_discharge_cfs"], group["prediction_cfs"], group["rising_flow_flag"] > 0),
                "runoff_only": slice_metric_summary(group["target_discharge_cfs"], group["runoff_only_prediction_cfs"], group["rising_flow_flag"] > 0),
                "residual_hybrid": slice_metric_summary(group["target_discharge_cfs"], group["residual_hybrid_prediction_cfs"], group["rising_flow_flag"] > 0),
                "event_gated_residual": slice_metric_summary(group["target_discharge_cfs"], group["event_gated_residual_prediction_cfs"], group["rising_flow_flag"] > 0),
                "ensemble_blend": slice_metric_summary(group["target_discharge_cfs"], group["ensemble_blend_prediction_cfs"], group["rising_flow_flag"] > 0),
            },
            "high_flow_days": {
                "full_model": slice_metric_summary(group["target_discharge_cfs"], group["prediction_cfs"], group["target_discharge_cfs"] >= high_flow_threshold),
                "runoff_only": slice_metric_summary(group["target_discharge_cfs"], group["runoff_only_prediction_cfs"], group["target_discharge_cfs"] >= high_flow_threshold),
                "residual_hybrid": slice_metric_summary(group["target_discharge_cfs"], group["residual_hybrid_prediction_cfs"], group["target_discharge_cfs"] >= high_flow_threshold),
                "event_gated_residual": slice_metric_summary(group["target_discharge_cfs"], group["event_gated_residual_prediction_cfs"], group["target_discharge_cfs"] >= high_flow_threshold),
                "ensemble_blend": slice_metric_summary(group["target_discharge_cfs"], group["ensemble_blend_prediction_cfs"], group["target_discharge_cfs"] >= high_flow_threshold),
            },
        })

    rolling_by_label = {item["label"]: item for item in rolling_summary}
    candidate_order = ["ensemble_blend", "event_gated_residual", "residual_hybrid", "runoff_only", "full_model"]
    selected_name = min(
        [name for name in candidate_order if name in rolling_by_label],
        key=lambda name: rolling_by_label[name]["mae_mean"],
    )

    model_bundle = {
        "model_type": selected_name,
        "full_model": full_model,
        "runoff_only_model": runoff_only_model,
        "residual_model": residual_model,
        "ensemble_runoff_weight": ENSEMBLE_RUNOFF_WEIGHT,
        "event_only_residual": EVENT_ONLY_RESIDUAL,
    }
    selected_features = full_features if selected_name in {"full_model", "residual_hybrid", "event_gated_residual"} else runoff_features

    report = {
        "metrics": metrics,
        "rolling_validation": {
            "folds": fold_summaries,
            "summary": rolling_summary,
            "records": [] if fold_df.empty else json.loads(fold_df.to_json(orient="records")),
        },
        "station_test_metrics": station_test_metrics,
        "test_predictions": test_out,
        "full_feature_columns": full_features,
        "runoff_only_feature_columns": runoff_features,
        "selected_model": {
            "model": selected_name,
            "feature_columns": selected_features,
            "selection_basis": "lowest rolling-validation mae among runoff_only, full_model, residual_hybrid, and event_gated_residual",
        },
        "experiment_flags": {
            "event_only_residual": EVENT_ONLY_RESIDUAL,
            "training_weather_mode": TRAINING_WEATHER_MODE,
            "categorical_mode": CATEGORICAL_MODE,
            "full_feature_set_mode": FULL_FEATURE_SET_MODE,
            "blend_gate_mode": BLEND_GATE_MODE,
            "feature_experiment": FEATURE_EXPERIMENT,
        },
    }
    return model_bundle, report, selected_features


def save_model_bundle(model_bundle: dict, features: Sequence[str], report: dict, station_links: List[dict]) -> None:
    import pickle

    with (MODEL_DIR / "montana_runoff_ridge.pkl").open("wb") as fh:
        pickle.dump(model_bundle, fh)
    if "selected_model" in report and report["selected_model"].get("model") == "runoff_only":
        features = report["runoff_only_feature_columns"]

    report_payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "weather_source": "Dynamical NOAA GEFS analysis (historical archive)",
        "snow_sources": ["NRCS SNOTEL daily station reports", "NOHRSC daily station reports"],
        "runoff_source": "USGS NWIS daily values parameterCd=00060",
        "feature_columns": list(features),
        "full_feature_columns": report["full_feature_columns"],
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
        "experiment_flags": report.get("experiment_flags", {}),
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
    if LOOKBACK_START:
        start = date.fromisoformat(LOOKBACK_START)
    else:
        start = end - timedelta(days=LOOKBACK_DAYS - 1)

    mt_stations = load_mt_river_stations()
    if not mt_stations:
        raise SystemExit("No target USGS stations found in generated/stations.json. Run npm run build:data first.")

    snotel_meta = load_snotel_metadata()
    nohrsc_meta = load_nohrsc_metadata(end)

    station_frames = []
    station_links = []
    skipped_stations = []
    for station in mt_stations:
        if STATION_LIMIT > 0 and len(station_frames) >= STATION_LIMIT:
            break
        site_elev_m = fetch_location_elevation_m(station["latitude"], station["longitude"])
        snotel_candidates = candidate_neighbors(station["latitude"], station["longitude"], snotel_meta, MAX_SNOTEL_KM)
        nohrsc_candidates = candidate_neighbors(station["latitude"], station["longitude"], nohrsc_meta, MAX_NOHRSC_KM, id_key='station_id', lat_key="lat", lon_key="lon")
        snotel_signal = basin_signal_for_station(station, snotel_candidates, site_elev_m, preferred_band='high')
        nohrsc_signal = basin_signal_for_station(station, nohrsc_candidates, site_elev_m, preferred_band='high')
        snotel_neighbor = snotel_signal.primary
        nohrsc_neighbor = nohrsc_signal.primary
        print(f"Station {station['stationId']} {station['name']}")
        print(f"  nearest SNOTEL: {snotel_neighbor.station_id if snotel_neighbor else 'none'} | basin contributors={len(snotel_signal.contributors)} band={snotel_signal.band_label}")
        print(f"  nearest NOHRSC: {nohrsc_neighbor.station_id if nohrsc_neighbor else 'none'} | basin contributors={len(nohrsc_signal.contributors)} band={nohrsc_signal.band_label}")
        frame = load_or_assemble_station_dataset(station, snotel_signal, nohrsc_signal, start, end)
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
            "basin_key": snotel_signal.basin_key or nohrsc_signal.basin_key,
            "snotel_basin_contributors": len(snotel_signal.contributors),
            "nohrsc_basin_contributors": len(nohrsc_signal.contributors),
            "snotel_basin_band": snotel_signal.band_label,
            "nohrsc_basin_band": nohrsc_signal.band_label,
            "has_camels_static": bool(fetch_camels_attributes(station["stationId"]).values),
            "has_hydrobasins_static": bool(fetch_hydrobasins_attributes(station["stationId"]).values),
        })

    dataset = pd.concat(station_frames, ignore_index=True).sort_values(["date", "station_id"]) if station_frames else merge_station_dataset_cache(start, end)
    if dataset is None or dataset.empty:
        raise SystemExit("No station datasets could be assembled.")

    dataset.to_parquet(DATASET_DIR / "montana_runoff_training.parquet", index=False)
    dataset.head(100).to_csv(DATASET_DIR / "montana_runoff_training_sample.csv", index=False)

    model, report, features = train_model(dataset)
    save_model_bundle(model, features, report, station_links)

    target_states = configured_target_states()
    target_scope_label = (
        TARGET_GROUP or
        ('mountain-west' if TARGET_SCOPE == 'mountain-west' else None) or
        ('all-us' if TARGET_SCOPE == 'national' and not target_states else None) or
        ('multi-state' if len(target_states) > 1 else None) or
        (target_states[0].lower() if len(target_states) == 1 else None)
    )
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "date_range": {"start": start.isoformat(), "end": end.isoformat()},
        "target_group": TARGET_GROUP or None,
        "target_scope_label": target_scope_label,
        "station_ids_filter": STATION_IDS_FILTER,
        "snotel_metadata_count": len(snotel_meta),
        "nohrsc_metadata_count": len(nohrsc_meta),
        "target_scope": TARGET_SCOPE,
        "target_states": target_states,
        "stations_requested": (len(mt_stations) if STATION_LIMIT <= 0 else STATION_LIMIT) if not (TARGET_GROUP or STATION_IDS_FILTER) else len(mt_stations),
        "station_candidates_considered": len(mt_stations),
        "stations_used": len(station_frames),
        "include_managed_stations": INCLUDE_MANAGED_STATIONS,
        "static_descriptor_coverage": {
            "camels_station_count": sum(1 for link in station_links if link.get("has_camels_static")),
            "hydrobasins_station_count": sum(1 for link in station_links if link.get("has_hydrobasins_static")),
            "no_static_station_count": sum(1 for link in station_links if not link.get("has_camels_static") and not link.get("has_hydrobasins_static")),
        },
        "station_links": station_links,
        "stations_skipped": skipped_stations,
        "rows": int(len(dataset)),
        "metrics": report["metrics"],
        "rolling_validation_summary": report["rolling_validation"]["summary"],
        "experiment_flags": report.get("experiment_flags", {}),
    }
    write_json(ML_DIR / "latest_training_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
