"""
Vegetation index computation from 4-band NAIP imagery.

Computes NDVI, NDWI, EVI, and spectral mixture fractions at native 0.6m
resolution and aggregates to parcel-level statistics.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
import xarray as xr
from rasterio.mask import mask as rio_mask
from loguru import logger


def compute_vegetation_indices(naip_path: Path) -> xr.Dataset:
    """
    Compute vegetation indices from a 4-band NAIP GeoTIFF.

    Band order expected: (1=Red, 2=Green, 3=Blue, 4=NIR).

    Returns
    -------
    xr.Dataset with variables: ndvi, ndwi, evi, dry_veg_fraction.
    """
    with rasterio.open(naip_path) as src:
        # Read all bands as float32, scaling from uint8 [0,255] to [0,1]
        r, g, b, nir = [src.read(i).astype("float32") / 255.0 for i in range(1, 5)]
        profile = src.profile
        transform = src.transform
        crs = src.crs

    eps = 1e-6

    # NDVI
    ndvi = (nir - r) / (nir + r + eps)
    ndvi = np.clip(ndvi, -1, 1)

    # NDWI (Gao 1996) - sensitive to vegetation water content
    ndwi = (g - nir) / (g + nir + eps)
    ndwi = np.clip(ndwi, -1, 1)

    # EVI (Huete et al. 2002) - less saturation than NDVI in dense canopy
    evi = 2.5 * (nir - r) / (nir + 6 * r - 7.5 * b + 1 + eps)
    evi = np.clip(evi, -1, 5)

    # Spectral Mixture Analysis (simplified 3-endmember linear unmixing)
    # Endmembers: green vegetation (gv), dry vegetation (npv), soil
    gv_frac, npv_frac, soil_frac = _spectral_mixture(r, g, b, nir)

    ds = xr.Dataset(
        {
            "ndvi": (["y", "x"], ndvi.astype("float32")),
            "ndwi": (["y", "x"], ndwi.astype("float32")),
            "evi": (["y", "x"], evi.astype("float32")),
            "green_veg_fraction": (["y", "x"], gv_frac.astype("float32")),
            "dry_veg_fraction": (["y", "x"], npv_frac.astype("float32")),
            "soil_fraction": (["y", "x"], soil_frac.astype("float32")),
        },
        attrs={
            "crs": crs.to_string(),
            "transform": str(transform),
            "source": str(naip_path),
        },
    )

    logger.info(
        f"Vegetation indices computed. NDVI: [{ndvi.min():.3f}, {ndvi.max():.3f}], "
        f"mean dry_veg={npv_frac.mean():.3f}"
    )
    return ds


def compute_parcel_veg_stats(
    veg_ds: xr.Dataset,
    parcels: gpd.GeoDataFrame,
    naip_path: Path,
) -> gpd.GeoDataFrame:
    """
    Aggregate per-pixel vegetation indices to parcel-level summary statistics.

    For each parcel, computes:
        ndvi_mean, ndvi_p90, ndvi_std (heterogeneity),
        ndwi_mean, evi_mean,
        dry_veg_fraction_mean (direct flammability indicator),
        canopy_cover_pct (pixels with NDVI > 0.4).

    Parameters
    ----------
    veg_ds:
        Vegetation index Dataset from compute_vegetation_indices().
    parcels:
        GeoDataFrame with parcel polygons (must be in same CRS as naip_path).
    naip_path:
        Original NAIP GeoTIFF (for CRS/transform reference in masking).

    Returns
    -------
    parcels GeoDataFrame with vegetation stat columns appended.
    """
    results: list[dict] = []

    with rasterio.open(naip_path) as src:
        parcel_crs = parcels.crs
        naip_crs = src.crs

        if parcel_crs != naip_crs:
            parcels_reproj = parcels.to_crs(naip_crs)
        else:
            parcels_reproj = parcels

        ndvi_arr = veg_ds["ndvi"].values
        ndwi_arr = veg_ds["ndwi"].values
        evi_arr = veg_ds["evi"].values
        dry_arr = veg_ds["dry_veg_fraction"].values

        for idx, (geom, row) in enumerate(zip(parcels_reproj.geometry, parcels_reproj.itertuples())):
            try:
                masked, _ = rio_mask(src, [geom.__geo_interface__], crop=True, nodata=0)
                valid_mask = masked[0] > 0  # Use red band as valid-pixel proxy

                def stat(arr: np.ndarray) -> dict:
                    vals = arr[valid_mask[:arr.shape[0], :arr.shape[1]]]
                    if len(vals) == 0:
                        return {"mean": np.nan, "p90": np.nan, "std": np.nan}
                    return {"mean": float(vals.mean()), "p90": float(np.percentile(vals, 90)),
                            "std": float(vals.std())}

                ndvi_s = stat(ndvi_arr)
                results.append({
                    "ndvi_mean": ndvi_s["mean"],
                    "ndvi_p90": ndvi_s["p90"],
                    "ndvi_std": ndvi_s["std"],
                    "ndwi_mean": stat(ndwi_arr)["mean"],
                    "evi_mean": stat(evi_arr)["mean"],
                    "dry_veg_fraction_mean": stat(dry_arr)["mean"],
                    "canopy_cover_pct": float(np.mean(ndvi_arr[
                        valid_mask[:ndvi_arr.shape[0], :ndvi_arr.shape[1]]
                    ] > 0.4) * 100) if valid_mask.any() else 0.0,
                })
            except Exception as e:
                logger.debug(f"Parcel {idx} vegetation extraction error: {e}")
                results.append({
                    "ndvi_mean": np.nan, "ndvi_p90": np.nan, "ndvi_std": np.nan,
                    "ndwi_mean": np.nan, "evi_mean": np.nan,
                    "dry_veg_fraction_mean": np.nan, "canopy_cover_pct": np.nan,
                })

    import pandas as pd
    stats_df = pd.DataFrame(results)
    for col in stats_df.columns:
        parcels[col] = stats_df[col].values

    return parcels


def classify_vegetation_flammability(ndvi: np.ndarray, ndwi: np.ndarray) -> np.ndarray:
    """
    Classify vegetation flammability on a 0–4 scale:
        0 = Non-vegetated / impervious
        1 = Low (healthy dense vegetation, high moisture)
        2 = Moderate
        3 = High (dry sparse vegetation)
        4 = Extreme (dead/cured grass or dry brush)

    Based on NDVI and NDWI thresholds aligned with fire behavior literature.
    """
    flam = np.zeros_like(ndvi, dtype="uint8")
    flam[ndvi > 0.5] = 1                                    # Dense healthy
    flam[(ndvi > 0.3) & (ndvi <= 0.5)] = 2                 # Moderate
    flam[(ndvi > 0.1) & (ndvi <= 0.3) & (ndwi < 0.0)] = 3  # Dry moderate
    flam[(ndvi <= 0.1) & (ndwi < -0.2)] = 4                # Extreme (bare/dead)
    flam[ndvi < 0.05] = 0                                   # Impervious
    return flam


# ── Private helpers ──────────────────────────────────────────────────────────

def _spectral_mixture(
    r: np.ndarray, g: np.ndarray, b: np.ndarray, nir: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simplified 3-endmember linear spectral unmixing.

    Endmembers (Roberts et al. spectral library, normalized):
        GV  (green vegetation): high NIR, moderate Red
        NPV (non-photosynthetic vegetation / dry): high Red, low NIR
        Soil: moderate all bands

    Returns fractions for each endmember. Fractions are clipped to [0,1]
    and normalized to sum to 1.
    """
    # Endmember spectra in [R, G, B, NIR] order (normalized 0-1 scale)
    em_gv  = np.array([0.05, 0.10, 0.05, 0.45])
    em_npv = np.array([0.22, 0.20, 0.18, 0.12])
    em_soil = np.array([0.30, 0.28, 0.25, 0.25])

    em_matrix = np.stack([em_gv, em_npv, em_soil], axis=0)  # (3, 4)
    # Pixel matrix: (4, H*W)
    px = np.stack([r, g, b, nir], axis=0).reshape(4, -1)

    # Least-squares unmixing: E @ f = px  →  f = pinv(E) @ px
    pinv_em = np.linalg.pinv(em_matrix)  # (4, 3)
    fracs = pinv_em.T @ px               # (3, H*W)
    fracs = np.clip(fracs, 0, 1)
    fracs /= fracs.sum(axis=0, keepdims=True) + 1e-6

    h, w = r.shape
    gv_frac  = fracs[0].reshape(h, w).astype("float32")
    npv_frac = fracs[1].reshape(h, w).astype("float32")
    soil_frac = fracs[2].reshape(h, w).astype("float32")
    return gv_frac, npv_frac, soil_frac
