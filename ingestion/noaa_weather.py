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
        logger.warning("No weather records retrieved.")
        return pd.DataFrame()

    df = _parse_cdo_records(all_records)
    logger.info(f"Retrieved {len(df)} hourly records from {station_id}")
    return df


def build_wind_rose(
    df: pd.DataFrame,
    fire_weather_only: bool = True,
    rh_max: float = 25.0,
    wind_min_mph: float = 15.0,
    temp_min_f: float = 90.0,
    n_sectors: int = 16,
) -> dict:
    """
    Compute a directional wind frequency distribution from hourly data.

    Parameters
    ----------
    df:
        Hourly weather DataFrame from fetch_hourly_data().
    fire_weather_only:
        If True, filter to rows meeting all three fire weather thresholds.
    rh_max, wind_min_mph, temp_min_f:
        Fire weather condition thresholds.
    n_sectors:
        Number of compass sectors (default 16 = 22.5° each).

    Returns
    -------
    dict with keys:
        sectors: list of sector center azimuths (degrees)
        frequency: fraction of hours in each sector
        pct_90: 90th percentile wind speed (mph) by sector
        pct_95: 95th percentile wind speed (mph) by sector
        pct_99: 99th percentile wind speed (mph) by sector
        n_fire_weather_hours: count of qualifying hours
    """
    work = df.copy()
    if fire_weather_only:
        work = work[
            (work["rh_pct"] <= rh_max)
            & (work["wind_speed_mph"] >= wind_min_mph)
            & (work["temp_f"] >= temp_min_f)
        ]
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
        "datasetid": "LCD",
        "stationid": f"WBAN:{station_id[-5:]}",
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
    """Convert raw CDO records to a clean hourly DataFrame."""
    rows = []
    for r in records:
        rows.append({
            "datetime": r.get("date"),
            "wind_speed_mph": r.get("HourlyWindSpeed"),
            "wind_dir_deg": r.get("HourlyWindDirection"),
            "temp_f": r.get("HourlyDryBulbTemperature"),
            "rh_pct": r.get("HourlyRelativeHumidity"),
            "precip_in": r.get("HourlyPrecipitation"),
        })
    df = pd.DataFrame(rows)
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    for col in ["wind_speed_mph", "wind_dir_deg", "temp_f", "rh_pct", "precip_in"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=["datetime"]).set_index("datetime").sort_index()


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
        df.to_parquet(out_dir / "hourly_weather.parquet")
        rose = build_wind_rose(df)
        with open(out_dir / "wind_rose.json", "w") as f:
            json.dump(rose, f, indent=2)

        # Compute FWI for daily summaries
        daily = df.resample("D").agg(
            temp_c=("temp_f", lambda x: (x.mean() - 32) * 5 / 9),
            rh_pct=("rh_pct", "mean"),
            wind_kmh=("wind_speed_mph", lambda x: x.max() * 1.60934),
            precip_mm=("precip_in", lambda x: x.sum() * 25.4),
        ).dropna()
        daily = compute_fire_weather_index(daily.reset_index())
        daily.to_parquet(out_dir / "daily_fwi.parquet")
        logger.info(f"Weather data saved to {out_dir}")


if __name__ == "__main__":
    main()
