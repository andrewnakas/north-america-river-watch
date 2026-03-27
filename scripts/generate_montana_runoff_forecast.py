#!/usr/bin/env python3
"""Generate forward Montana runoff forecasts from the trained model."""

from __future__ import annotations

import json
import pickle
from datetime import date, datetime, timedelta, timezone
from typing import List, Optional

import numpy as np
import pandas as pd

import train_montana_runoff_model as mt

FORECAST_DAYS = int(mt.os.environ.get("MT_FORECAST_DAYS", "7"))
MODEL_PATH = mt.MODEL_DIR / "montana_runoff_ridge.pkl"
REPORT_PATH = mt.MODEL_DIR / "montana_runoff_validation_report.json"
FORECAST_DIR = mt.ML_DIR / "forecasts"


def load_model_bundle():
    if not MODEL_PATH.exists() or not REPORT_PATH.exists():
        raise SystemExit("Missing trained Montana model artifacts. Run npm run ml:train:montana first.")
    with MODEL_PATH.open("rb") as fh:
        model = pickle.load(fh)
    report = mt.read_json(REPORT_PATH)
    features = report.get("feature_columns") or []
    if not features:
        raise SystemExit("Model report is missing feature_columns.")
    return model, report, features


def fetch_openmeteo_forecast(lat: float, lon: float, days: int) -> pd.DataFrame:
    end = date.today() + timedelta(days=days)
    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        f"&start_date={date.today().isoformat()}&end_date={end.isoformat()}"
        "&daily=temperature_2m_mean,temperature_2m_max,temperature_2m_min,precipitation_sum,snowfall_sum"
        "&timezone=UTC"
    )
    resp = mt.SESSION.get(url, timeout=60)
    resp.raise_for_status()
    daily = resp.json().get("daily", {})
    if not daily or not daily.get("time"):
        return pd.DataFrame()
    return pd.DataFrame({
        "date": pd.to_datetime(daily.get("time", [])),
        "dyn_temp_c_mean": daily.get("temperature_2m_mean", []),
        "dyn_temp_c_max": daily.get("temperature_2m_max", []),
        "dyn_temp_c_min": daily.get("temperature_2m_min", []),
        "dyn_precip_mm": daily.get("precipitation_sum", []),
        "dyn_snow_flag": [1 if (v or 0) > 0 else 0 for v in daily.get("snowfall_sum", [])],
        "dyn_source": "open-meteo-forecast",
    }).sort_values("date")


def latest_non_null(series: Optional[pd.Series]):
    if series is None:
        return None
    valid = series.dropna()
    return None if valid.empty else valid.iloc[-1]


def rolling_mean(history: List[float], window: int) -> Optional[float]:
    if len(history) < window:
        return None
    vals = history[-window:]
    return None if not vals else float(sum(vals) / len(vals))


def build_station_forecast_rows(station: dict, features: List[str]) -> Optional[dict]:
    today = date.today()
    history_start = today - timedelta(days=max(mt.LOOKBACK_DAYS, 30))
    history_end = today - timedelta(days=1)

    snotel_meta = build_station_forecast_rows.snotel_meta
    nohrsc_meta = build_station_forecast_rows.nohrsc_meta
    snotel_neighbor = mt.nearest_station(station["latitude"], station["longitude"], snotel_meta, mt.MAX_SNOTEL_KM)
    nohrsc_neighbor = mt.nearest_station(station["latitude"], station["longitude"], nohrsc_meta, mt.MAX_NOHRSC_KM, lat_key="lat", lon_key="lon")

    runoff = mt.fetch_usgs_daily_discharge(station["stationId"], history_start, history_end).sort_values("date")
    runoff = runoff.dropna(subset=["discharge_cfs"])
    if runoff.empty or len(runoff) < 14:
        return None

    weather_fc = fetch_openmeteo_forecast(station["latitude"], station["longitude"], FORECAST_DAYS)
    if weather_fc.empty:
        return None

    history_weather = mt.fetch_dynamical_weather(station["latitude"], station["longitude"], history_start, history_end)
    if history_weather.empty:
        return None

    base = runoff.merge(history_weather, on="date", how="left")
    if base.empty or "date" not in base.columns:
        return None

    latest_snotel = {}
    if snotel_neighbor:
        triplet = f"{snotel_neighbor.station_id}:{mt.TARGET_STATE}:SNTL"
        snotel_df = mt.fetch_snotel_daily(triplet, history_start, history_end)
        if not snotel_df.empty:
            base = base.merge(snotel_df, on="date", how="left")
            latest_snotel = {
                "snotel_wteq_in": latest_non_null(snotel_df.get("snotel_wteq_in")),
                "snotel_snwd_in": latest_non_null(snotel_df.get("snotel_snwd_in")),
                "snotel_prec_in": latest_non_null(snotel_df.get("snotel_prec_in")),
                "snotel_tavg_f": latest_non_null(snotel_df.get("snotel_tavg_f")),
                "snotel_tmax_f": latest_non_null(snotel_df.get("snotel_tmax_f")),
                "snotel_tmin_f": latest_non_null(snotel_df.get("snotel_tmin_f")),
            }

    latest_nohrsc = {}
    if nohrsc_neighbor:
        nohrsc_df = mt.build_nohrsc_series(nohrsc_neighbor, history_start, history_end)
        if not nohrsc_df.empty:
            base = base.merge(nohrsc_df, on="date", how="left")
            latest_nohrsc = {
                "nohrsc_snowdepth_cm": latest_non_null(nohrsc_df.get("nohrsc_snowdepth_cm")),
                "nohrsc_swe_mm": latest_non_null(nohrsc_df.get("nohrsc_swe_mm")),
                "nohrsc_snowfall_cm": latest_non_null(nohrsc_df.get("nohrsc_snowfall_cm")),
            }

    base = mt.add_lag_features(base, "discharge_cfs", "q")
    if "dyn_precip_mm" in base:
        base = mt.add_lag_features(base, "dyn_precip_mm", "dyn_precip")
    if "nohrsc_snowdepth_cm" in base:
        base = mt.add_lag_features(base, "nohrsc_snowdepth_cm", "nohrsc_sd")
    if "snotel_wteq_in" in base:
        base = mt.add_lag_features(base, "snotel_wteq_in", "snotel_wteq")
    base = mt.add_hydrology_features(base)

    return {
        "station": station,
        "base": base.sort_values("date").reset_index(drop=True),
        "weather_fc": weather_fc.sort_values("date").reset_index(drop=True),
        "snotel_neighbor": snotel_neighbor,
        "nohrsc_neighbor": nohrsc_neighbor,
        "latest_snotel": latest_snotel,
        "latest_nohrsc": latest_nohrsc,
        "feature_columns": features,
    }


def evolve_snow_state(prev_row: pd.Series, wx: pd.Series, state: dict) -> dict:
    temp_mean = float(wx.get("dyn_temp_c_mean") or 0.0)
    precip_mm = float(wx.get("dyn_precip_mm") or 0.0)
    snowfall_factor = 1.0 if float(wx.get("dyn_snow_flag") or 0.0) > 0 else 0.0
    degree_day = max(0.0, temp_mean - mt.DEGREE_DAY_BASE_C)

    next_state = dict(state)

    snotel_wteq = next_state.get("snotel_wteq_in")
    if snotel_wteq is not None:
        snow_gain_in = (precip_mm / 25.4) * snowfall_factor
        melt_in = min(float(snotel_wteq), degree_day * 0.06)
        next_state["snotel_wteq_in"] = max(0.0, float(snotel_wteq) + snow_gain_in - melt_in)
        if next_state.get("snotel_snwd_in") is not None:
            snwd = float(next_state["snotel_snwd_in"])
            snowdepth_gain = (precip_mm / 25.4) * snowfall_factor * 8.0
            snowdepth_loss = degree_day * 0.4
            next_state["snotel_snwd_in"] = max(0.0, snwd + snowdepth_gain - snowdepth_loss)
        next_state["snotel_tavg_f"] = temp_mean * 9.0 / 5.0 + 32.0
        next_state["snotel_tmax_f"] = float(wx.get("dyn_temp_c_max") or temp_mean) * 9.0 / 5.0 + 32.0
        next_state["snotel_tmin_f"] = float(wx.get("dyn_temp_c_min") or temp_mean) * 9.0 / 5.0 + 32.0
        prev_prec = next_state.get("snotel_prec_in")
        next_state["snotel_prec_in"] = (float(prev_prec) if prev_prec is not None else 0.0) + (precip_mm / 25.4)

    nohrsc_sd = next_state.get("nohrsc_snowdepth_cm")
    if nohrsc_sd is not None:
        gain_cm = precip_mm * snowfall_factor
        loss_cm = degree_day * 1.2
        next_state["nohrsc_snowdepth_cm"] = max(0.0, float(nohrsc_sd) + gain_cm - loss_cm)
    nohrsc_swe = next_state.get("nohrsc_swe_mm")
    if nohrsc_swe is not None:
        gain_mm = precip_mm * snowfall_factor
        loss_mm = degree_day * 1.5
        next_state["nohrsc_swe_mm"] = max(0.0, float(nohrsc_swe) + gain_mm - loss_mm)
    next_state["nohrsc_snowfall_cm"] = precip_mm * snowfall_factor
    return next_state


def predict_station(model, features: List[str], station_bundle: dict) -> dict:
    history = station_bundle["base"].copy()
    weather_fc = station_bundle["weather_fc"]
    station = station_bundle["station"]
    snow_state = {**station_bundle.get("latest_snotel", {}), **station_bundle.get("latest_nohrsc", {})}

    recent_q = [float(v) for v in history["discharge_cfs"].dropna().tail(30).tolist()]
    recent_q7 = [float(v) for v in history["discharge_cfs"].dropna().tail(7).tolist()]
    cap_hi = max(recent_q) * 1.75 if recent_q else None
    if recent_q7:
        cap_hi = max(cap_hi or 0.0, (sum(recent_q7) / len(recent_q7)) * 2.25)
    cap_lo = 0.0

    predictions = []
    for _, wx in weather_fc.iterrows():
        snow_state = evolve_snow_state(history.iloc[-1], wx, snow_state)
        row = {
            "date": pd.Timestamp(wx["date"]),
            "discharge_cfs": np.nan,
            "dyn_temp_c_mean": wx.get("dyn_temp_c_mean"),
            "dyn_temp_c_max": wx.get("dyn_temp_c_max"),
            "dyn_temp_c_min": wx.get("dyn_temp_c_min"),
            "dyn_precip_mm": wx.get("dyn_precip_mm"),
            "dyn_snow_flag": wx.get("dyn_snow_flag"),
            "dyn_source": wx.get("dyn_source", "open-meteo-forecast"),
            "latitude": station["latitude"],
            "longitude": station["longitude"],
            "snotel_distance_km": round(station_bundle["snotel_neighbor"].distance_km, 2) if station_bundle.get("snotel_neighbor") else np.nan,
            "nohrsc_distance_km": round(station_bundle["nohrsc_neighbor"].distance_km, 2) if station_bundle.get("nohrsc_neighbor") else np.nan,
            **snow_state,
        }

        history = pd.concat([history, pd.DataFrame([row])], ignore_index=True)
        history = mt.add_lag_features(history, "discharge_cfs", "q")
        history = mt.add_lag_features(history, "dyn_precip_mm", "dyn_precip")
        if "nohrsc_snowdepth_cm" in history:
            history = mt.add_lag_features(history, "nohrsc_snowdepth_cm", "nohrsc_sd")
        if "snotel_wteq_in" in history:
            history = mt.add_lag_features(history, "snotel_wteq_in", "snotel_wteq")
        history = mt.add_hydrology_features(history)

        feature_row = history.iloc[-1].copy()
        frame = pd.DataFrame([{feature: feature_row.get(feature, np.nan) for feature in features}])
        pred = max(0.0, float(model.predict(frame)[0]))
        if cap_hi is not None:
            pred = min(pred, cap_hi)
        history.loc[history.index[-1], "discharge_cfs"] = pred
        history = mt.add_lag_features(history, "discharge_cfs", "q")
        history = mt.add_hydrology_features(history)

        predictions.append({
            "date": pd.Timestamp(wx["date"]).date().isoformat(),
            "predicted_discharge_cfs": round(pred, 2),
            "weather_source": row.get("dyn_source", "open-meteo-forecast"),
            "precip_mm": None if pd.isna(row.get("dyn_precip_mm", np.nan)) else round(float(row.get("dyn_precip_mm")), 2),
            "temp_c_mean": None if pd.isna(row.get("dyn_temp_c_mean", np.nan)) else round(float(row.get("dyn_temp_c_mean")), 2),
            "snow_flag": int(row.get("dyn_snow_flag") or 0),
            "degree_day_c": None if pd.isna(feature_row.get("degree_day_c", np.nan)) else round(float(feature_row.get("degree_day_c")), 2),
            "rain_on_snow_proxy": None if pd.isna(feature_row.get("rain_on_snow_proxy", np.nan)) else round(float(feature_row.get("rain_on_snow_proxy")), 3),
            "snotel_wteq_in": None if pd.isna(feature_row.get("snotel_wteq_in", np.nan)) else round(float(feature_row.get("snotel_wteq_in")), 3),
            "nohrsc_snowdepth_cm": None if pd.isna(feature_row.get("nohrsc_snowdepth_cm", np.nan)) else round(float(feature_row.get("nohrsc_snowdepth_cm")), 2),
        })

    pred_values = [item["predicted_discharge_cfs"] for item in predictions]
    peak_idx = int(np.argmax(pred_values)) if pred_values else None
    peak = predictions[peak_idx] if peak_idx is not None else None
    latest_obs = history.iloc[len(station_bundle["base"]) - 1]["discharge_cfs"] if not station_bundle["base"].empty else None
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "target_group": mt.TARGET_GROUP or None,
        "station_id": station["stationId"],
        "station_name": station["name"],
        "state": station["state"],
        "latitude": station["latitude"],
        "longitude": station["longitude"],
        "forecast_days": FORECAST_DAYS,
        "latest_observed_discharge_cfs": None if latest_obs is None or pd.isna(latest_obs) else round(float(latest_obs), 2),
        "peak_predicted_discharge_cfs": peak["predicted_discharge_cfs"] if peak else None,
        "peak_date": peak["date"] if peak else None,
        "snotel_station_id": station_bundle["snotel_neighbor"].station_id if station_bundle.get("snotel_neighbor") else None,
        "nohrsc_station_id": station_bundle["nohrsc_neighbor"].station_id if station_bundle.get("nohrsc_neighbor") else None,
        "predictions": predictions,
        "notes": [
            "Experimental Montana runoff forecast from a ridge model with snow, melt, weather, and runoff-response features.",
            "Future weather forcing currently comes from Open-Meteo daily forecast.",
            "Snow states are advanced with simple persistence-plus-melt heuristics rather than basin-scale forecast snow physics.",
            "Use for iteration and testing, not safety-critical decisions."
        ]
    }


def main() -> None:
    mt.ensure_dirs()
    FORECAST_DIR.mkdir(parents=True, exist_ok=True)
    model, report, features = load_model_bundle()

    stations = mt.load_mt_river_stations()
    build_station_forecast_rows.snotel_meta = mt.load_snotel_metadata()
    build_station_forecast_rows.nohrsc_meta = mt.load_nohrsc_metadata(date.today() - timedelta(days=1))

    outputs = []
    for station in stations:
        try:
            bundle = build_station_forecast_rows(station, features)
            if not bundle:
                print(f"Forecast skipped for {station['stationId']}: insufficient history/feature assembly")
                continue
            result = predict_station(model, features, bundle)
            outputs.append(result)
            mt.write_json(FORECAST_DIR / f"{station['stationId']}.json", result)
            print(f"Forecasted {station['stationId']} {station['name']}")
        except Exception as exc:
            print(f"Forecast failed for {station['stationId']}: {exc}")

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "target_group": mt.TARGET_GROUP or None,
        "forecast_days": FORECAST_DAYS,
        "stations_forecasted": len(outputs),
        "model_metrics": report.get("metrics", {}),
        "rolling_validation": report.get("rolling_validation", {}).get("summary", []),
        "stations": [
            {
                "station_id": item["station_id"],
                "station_name": item["station_name"],
                "peak_predicted_discharge_cfs": item["peak_predicted_discharge_cfs"],
                "peak_date": item["peak_date"],
            }
            for item in outputs
        ]
    }
    mt.write_json(FORECAST_DIR / "montana_runoff_forecast.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
