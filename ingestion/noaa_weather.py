"""
NOAA Climate Data Online (CDO) weather ingestion.

Pulls 10 years of hourly ASOS data from RDU airport, computes a wind rose
for fire-weather conditions, and implements the Canadian Fire Weather Index.

Usage:
    python ingestion/noaa_weather.py

Colab:
    Set COLAB_MODE = True before running on Google Colab.
    Set NOAA_CDO_TOKEN environment variable with your CDO API token.
"""

from __future__ import annotations

# ── Colab mode flag ──────────────────────────────────────────────────────────
COLAB_MODE = False  # Set to True when running on Google Colab
# ─────────────────────────────────────────────────────────────────────────────

import json
import math
import os
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from loguru import logger

from ingestion.config_loader import get_paths, load_config


_CDO_BASE = "https://www.ncdc.noaa.gov/cdo-web/api/v2"


def fetch_hourly_data(
    station_id: str,
    years: int = 10,
    out_dir: Path | None = None,
    token: str | None = None,
) -> pd.DataFrame:
    """
    Pull hourly ASOS data from NOAA CDO for the past `years` years.

    Parameters
    ----------
    station_id:
        GHCND station identifier, e.g. "USW00013722" for RDU.
    years:
        Number of years of historical data to retrieve.
    out_dir:
        If provided, cache raw JSON responses here.
    token:
        NOAA CDO API token (or set NOAA_CDO_TOKEN env var).

    Returns
    -------
    DataFrame with columns: datetime, wind_speed_mph, wind_dir_deg,
    temp_f, rh_pct, precip_in.
    """
    token = token or os.environ.get("NOAA_CDO_TOKEN", "")
    if not token:
        logger.warning("NOAA_CDO_TOKEN not set. Requests may be rate-limited.")

    end_date = datetime.now()
    start_date = end_date - timedelta(days=365 * years)

    all_records: list[dict] = []
    # CDO API max range is 1 year per request; chunk by year
    chunk_start = start_date
    while chunk_start < end_date:
        chunk_end = min(chunk_start + timedelta(days=364), end_date)
        records = _fetch_cdo_chunk(station_id, chunk_start, chunk_end, token, out_dir)
        all_records.extend(records)
        chunk_start = chunk_end + timedelta(days=1)

    if not all_records:
        logger.warning("No weather records retrieved — using synthetic Durham NC climatology.")
        return _synthetic_durham_weather(years)

    df = _parse_cdo_records(all_records)
    logger.info(f"Retrieved {len(df)} hourly records from {station_id}")
    return df


def build_wind_rose(
    df: pd.DataFrame,
    fire_weather_only: bool = True,
    rh_max: float = 40.0,
    wind_min_mph: float = 8.0,
    temp_min_f: float = 85.0,
    n_sectors: int = 16,
) -> dict:
    """
    Compute a directional wind frequency distribution from daily weather data.

    Thresholds are calibrated for daily GHCND summaries (TMAX, mean wind),
    not hourly obs — daily averages are inherently moderated so the classic
    hourly thresholds (RH≤25%, wind≥15mph, temp≥90°F) yield zero matches.
    These daily equivalents capture the same high fire-danger days.

    Parameters
    ----------
    df:
        Daily weather DataFrame from fetch_hourly_data().
    fire_weather_only:
        If True, filter to days meeting all three fire weather thresholds.
    rh_max, wind_min_mph, temp_min_f:
        Fire weather condition thresholds (daily values).
    n_sectors:
        Number of compass sectors (default 16 = 22.5° each).

    Returns
    -------
    dict with keys:
        sectors: list of sector center azimuths (degrees)
        frequency: fraction of days in each sector
        pct_90: 90th percentile wind speed (mph) by sector
        pct_95: 95th percentile wind speed (mph) by sector
        pct_99: 99th percentile wind speed (mph) by sector
        n_fire_weather_hours: count of qualifying days
    """
    work = df.copy()
    temp_col = "temp_f" if "temp_f" in work.columns else None
    if fire_weather_only:
        mask = pd.Series(True, index=work.index)
        if "rh_pct" in work.columns:
            mask &= work["rh_pct"] <= rh_max
        if "wind_speed_mph" in work.columns:
            mask &= work["wind_speed_mph"] >= wind_min_mph
        if temp_col:
            mask &= work[temp_col] >= temp_min_f
        work = work[mask]
    if work.empty:
        logger.warning("No fire-weather hours found with given thresholds.")
        return {}

    sector_width = 360.0 / n_sectors
    sector_centers = [i * sector_width for i in range(n_sectors)]

    def sector_idx(deg: float) -> int:
        return int((deg + sector_width / 2) % 360 / sector_width)

    work = work.dropna(subset=["wind_dir_deg", "wind_speed_mph"])
    work["sector"] = work["wind_dir_deg"].apply(sector_idx)

    freq = work.groupby("sector").size() / len(work)
    p90 = work.groupby("sector")["wind_speed_mph"].quantile(0.90)
    p95 = work.groupby("sector")["wind_speed_mph"].quantile(0.95)
    p99 = work.groupby("sector")["wind_speed_mph"].quantile(0.99)

    rose = {
        "sectors": sector_centers,
        "frequency": [freq.get(i, 0.0) for i in range(n_sectors)],
        "pct_90": [p90.get(i, 0.0) for i in range(n_sectors)],
        "pct_95": [p95.get(i, 0.0) for i in range(n_sectors)],
        "pct_99": [p99.get(i, 0.0) for i in range(n_sectors)],
        "n_fire_weather_hours": len(work),
        "dominant_direction_deg": float(work["wind_dir_deg"].mode().iloc[0]),
    }
    logger.info(
        f"Wind rose built from {rose['n_fire_weather_hours']} fire-weather hours. "
        f"Dominant direction: {rose['dominant_direction_deg']:.0f}°"
    )
    return rose


def compute_fire_weather_index(daily_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the Canadian Fire Weather Index (FWI) system components.

    Implements all six sub-indices:
        FFMC  Fine Fuel Moisture Code
        DMC   Duff Moisture Code
        DC    Drought Code
        ISI   Initial Spread Index
        BUI   Buildup Index
        FWI   Fire Weather Index

    Parameters
    ----------
    daily_df:
        Daily weather DataFrame with columns:
        date, temp_c, rh_pct, wind_kmh, precip_mm.

    Returns
    -------
    Input DataFrame with FWI columns appended.
    """
    df = daily_df.copy().reset_index(drop=True)
    n = len(df)

    ffmc = np.zeros(n)
    dmc = np.zeros(n)
    dc = np.zeros(n)
    isi = np.zeros(n)
    bui = np.zeros(n)
    fwi = np.zeros(n)

    # Initial conditions (standard start-of-season defaults)
    ffmc_prev, dmc_prev, dc_prev = 85.0, 6.0, 15.0

    for i, row in df.iterrows():
        T = float(row["temp_c"])
        H = float(row["rh_pct"])
        W = float(row["wind_kmh"])
        r = float(row["precip_mm"])

        # --- FFMC ---
        mo = 147.2 * (101 - ffmc_prev) / (59.5 + ffmc_prev)
        if r > 0.5:
            rf = r - 0.5
            mo = mo + 42.5 * rf * math.exp(-100 / (251 - mo)) * (1 - math.exp(-6.93 / rf))
            if mo > 250:
                mo = 250
        ed = 0.942 * H ** 0.679 + 11 * math.exp((H - 100) / 10) + 0.18 * (21.1 - T) * (1 - math.exp(-0.115 * H))
        ew = 0.618 * H ** 0.753 + 10 * math.exp((H - 100) / 10) + 0.18 * (21.1 - T) * (1 - math.exp(-0.115 * H))
        if mo > ed:
            ko = 0.424 * (1 - (H / 100) ** 1.7) + 0.0694 * W ** 0.5 * (1 - (H / 100) ** 8)
            kd = ko * 0.581 * math.exp(0.0365 * T)
            m = ed + (mo - ed) * 10 ** (-kd)
        elif mo < ew:
            ko = 0.424 * (1 - ((100 - H) / 100) ** 1.7) + 0.0694 * W ** 0.5 * (1 - ((100 - H) / 100) ** 8)
            kw = ko * 0.581 * math.exp(0.0365 * T)
            m = ew - (ew - mo) * 10 ** (-kw)
        else:
            m = mo
        ffmc[i] = 59.5 * (250 - m) / (147.2 + m)

        # --- DMC ---
        le = max(9.0, 11.5 + 4.5 * math.sin(math.radians((i % 365 + 1 - 15) * 360 / 365)))
        k = max(0, 1.894 * (T + 1.1) * (100 - H) * le * 1e-4)
        if r > 1.5:
            re = 0.92 * r - 1.27
            mo = 20 + math.exp(5.6348 - dmc_prev / 43.43)
            b = (65.91 + 11.2 * math.log(mo)) if dmc_prev <= 65 else 200 / (0.999 + 0.23 * math.log(dmc_prev))
            mr = mo + 1000 * re / (48.77 + b * re)
            pr = 244.72 - 43.43 * math.log(mr - 20) if mr > 20 else 0
            dmc[i] = max(0, pr + k)
        else:
            dmc[i] = dmc_prev + k

        # --- DC ---
        if r > 2.8:
            rd = 0.83 * r - 1.27
            qo = 800 * math.exp(-dc_prev / 400)
            qr = qo + 3.937 * rd
            dr = 400 * math.log(800 / qr) if qr > 0 else 0
        else:
            dr = dc_prev
        lf = max(-1.6, 0.36 * (T + 2.8) + 0.5)
        dc[i] = dr + 0.5 * lf

        # --- ISI ---
        fw = math.exp(0.05039 * W)
        fm = 147.2 * (101 - ffmc[i]) / (59.5 + ffmc[i])
        sf = 91.9 * math.exp(-0.1386 * fm) * (1 + fm ** 5.31 / (4.93e7))
        isi[i] = 0.208 * fw * sf

        # --- BUI ---
        bui[i] = (
            0.8 * dmc[i] * dc[i] / (dmc[i] + 0.4 * dc[i])
            if dmc[i] > 0 else 0
        )

        # --- FWI ---
        if bui[i] <= 80:
            bb = 0.1 * isi[i] * (0.626 * bui[i] ** 0.809 + 2)
        else:
            bb = 0.1 * isi[i] * (1000 / (25 + 108.64 * math.exp(-0.023 * bui[i])))
        fwi[i] = math.exp(2.72 * (0.434 * math.log(bb)) ** 0.647) if bb > 1 else bb

        ffmc_prev, dmc_prev, dc_prev = ffmc[i], dmc[i], dc[i]

    df["FFMC"] = ffmc
    df["DMC"] = dmc
    df["DC"] = dc
    df["ISI"] = isi
    df["BUI"] = bui
    df["FWI"] = fwi
    return df


# ── Private helpers ──────────────────────────────────────────────────────────

def _fetch_cdo_chunk(
    station_id: str,
    start: datetime,
    end: datetime,
    token: str,
    out_dir: Path | None,
) -> list[dict]:
    headers = {"token": token}
    params = {
        "datasetid": "GHCND",
        "stationid": f"GHCND:{station_id}",
        "startdate": start.strftime("%Y-%m-%d"),
        "enddate": end.strftime("%Y-%m-%d"),
        "limit": 1000,
        "offset": 1,
    }
    records: list[dict] = []
    while True:
        try:
            resp = requests.get(f"{_CDO_BASE}/data", headers=headers, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            logger.warning(f"CDO API error: {e}")
            break

        chunk = data.get("results", [])
        records.extend(chunk)
        meta = data.get("metadata", {}).get("resultset", {})
        count = meta.get("count", 0)
        limit = meta.get("limit", 1000)
        offset = meta.get("offset", 1)
        if offset + limit > count:
            break
        params["offset"] = offset + limit

    if out_dir:
        cache = out_dir / f"cdo_{station_id}_{start.year}.json"
        with open(cache, "w") as f:
            json.dump(records, f)
    return records


def _parse_cdo_records(records: list[dict]) -> pd.DataFrame:
    """Convert raw CDO/GHCND records to a clean daily DataFrame.

    GHCND records are pivoted: one row per (date, datatype) pair.
    We pivot them into one row per date with weather columns.
    """
    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])

    # Pivot: one row per date, columns are datatypes
    pivoted = df.pivot_table(index="date", columns="datatype", values="value", aggfunc="first")

    result = pd.DataFrame(index=pivoted.index)
    # GHCND stores temps in tenths of degree C, wind in tenths of m/s, precip in tenths of mm
    if "TMAX" in pivoted.columns:
        result["temp_f"] = pivoted["TMAX"] / 10.0 * 9 / 5 + 32  # tenths C → F
    if "TMIN" in pivoted.columns:
        result["temp_min_f"] = pivoted["TMIN"] / 10.0 * 9 / 5 + 32
    if "AWND" in pivoted.columns:
        result["wind_speed_mph"] = pivoted["AWND"] / 10.0 * 2.23694  # tenths m/s → mph
    if "WSF2" in pivoted.columns:
        result["wind_gust_mph"] = pivoted["WSF2"] / 10.0 * 2.23694
    if "WDF2" in pivoted.columns:
        result["wind_dir_deg"] = pivoted["WDF2"]
    if "PRCP" in pivoted.columns:
        result["precip_in"] = pivoted["PRCP"] / 10.0 / 25.4  # tenths mm → inches
    # RH is not in GHCND; estimate from TMAX/TMIN using Magnus formula approximation
    if "TMAX" in pivoted.columns and "TMIN" in pivoted.columns:
        tmax_c = pivoted["TMAX"] / 10.0
        tmin_c = pivoted["TMIN"] / 10.0
        # Approximate daily mean RH from dewpoint ≈ Tmin (common assumption)
        e_sat = 6.108 * np.exp(17.27 * tmax_c / (tmax_c + 237.3))
        e_dew = 6.108 * np.exp(17.27 * tmin_c / (tmin_c + 237.3))
        result["rh_pct"] = np.clip(100.0 * e_dew / e_sat, 0, 100)

    result.index.name = "datetime"
    return result.sort_index()


def _synthetic_durham_weather(years: int = 10) -> pd.DataFrame:
    """
    Generate synthetic daily weather for Durham NC based on NOAA climatological
    normals (1991-2020) when the CDO API is unreachable.

    Monthly means: temp from NCEI normals, RH from Piedmont summer ~65%,
    wind from RDU surface obs ~8 mph prevailing SW.
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365 * years)
    dates = pd.date_range(start_date, end_date, freq="D")

    # Durham NC monthly temperature normals (°F)
    tmax_normals = [50, 54, 63, 72, 79, 87, 90, 88, 82, 71, 62, 52]
    tmin_normals = [30, 32, 40, 49, 58, 67, 71, 70, 63, 51, 41, 32]
    precip_normals_in = [3.6, 3.3, 4.1, 3.0, 3.5, 3.8, 4.5, 4.2, 3.9, 3.4, 3.0, 3.1]

    rng = np.random.default_rng(42)
    months = dates.month - 1  # 0-indexed

    tmax = np.array([tmax_normals[m] for m in months]) + rng.normal(0, 5, len(dates))
    tmin = np.array([tmin_normals[m] for m in months]) + rng.normal(0, 5, len(dates))
    precip_monthly_rate = np.array([precip_normals_in[m] / 30 for m in months])
    precip = rng.exponential(precip_monthly_rate)

    # RH estimated from Tmin/Tmax (Tmin ≈ dewpoint approximation)
    tmax_c = (tmax - 32) * 5 / 9
    tmin_c = (tmin - 32) * 5 / 9
    e_sat = 6.108 * np.exp(17.27 * tmax_c / (tmax_c + 237.3))
    e_dew = 6.108 * np.exp(17.27 * tmin_c / (tmin_c + 237.3))
    rh = np.clip(100.0 * e_dew / e_sat, 20, 95)

    # Wind: prevailing SW at RDU, mean 8 mph
    wind_speed = rng.gamma(2, 4, len(dates))  # mean ~8 mph
    wind_dir = rng.choice([180, 225, 270], size=len(dates))  # S/SW/W prevailing

    df = pd.DataFrame({
        "temp_f": tmax,
        "temp_min_f": tmin,
        "rh_pct": rh,
        "wind_speed_mph": wind_speed,
        "wind_dir_deg": wind_dir.astype(float),
        "precip_in": precip,
    }, index=dates)
    df.index.name = "datetime"
    logger.info(f"Generated {len(df)} days of synthetic Durham NC weather.")
    return df


# ── Entry point ──────────────────────────────────────────────────────────────

def main():
    paths = get_paths(COLAB_MODE)
    cfg = load_config("data_sources.yaml")
    w_cfg = cfg["weather"]

    out_dir = paths["raw_weather"]
    out_dir.mkdir(parents=True, exist_ok=True)

    df = fetch_hourly_data(
        station_id=w_cfg["station_id"],
        years=w_cfg["years_history"],
        out_dir=out_dir,
    )

    if not df.empty:
        df.to_parquet(out_dir / "daily_weather.parquet")
        rose = build_wind_rose(df)
        with open(out_dir / "wind_rose.json", "w") as f:
            json.dump(rose, f, indent=2)

        # Compute FWI — data is already daily from GHCND
        daily = pd.DataFrame(index=df.index)
        daily["temp_c"] = (df["temp_f"] - 32) * 5 / 9 if "temp_f" in df.columns else np.nan
        daily["rh_pct"] = df["rh_pct"] if "rh_pct" in df.columns else 50.0
        daily["wind_kmh"] = df["wind_speed_mph"] * 1.60934 if "wind_speed_mph" in df.columns else 0.0
        daily["precip_mm"] = df["precip_in"] * 25.4 if "precip_in" in df.columns else 0.0
        daily = daily.dropna()
        daily = compute_fire_weather_index(daily.reset_index())
        daily.to_parquet(out_dir / "daily_fwi.parquet")
        logger.info(f"Weather data saved to {out_dir}")


if __name__ == "__main__":
    main()
