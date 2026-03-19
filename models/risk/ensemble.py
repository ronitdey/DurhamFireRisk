"""
Multi-peril risk aggregation.

Combines wildfire and flood risk scores into a composite score,
with configurable weights and correlation adjustments.
"""

from __future__ import annotations

import numpy as np


def compute_composite_score(
    wildfire_score: float,
    flood_score: float,
    wildfire_weight: float = 0.65,
    flood_weight: float = 0.35,
    correlation_penalty: float = 0.0,
) -> float:
    """
    Aggregate wildfire and flood risk into a composite score.

    Uses a weighted combination that can be adjusted for:
        - Multi-peril correlation (e.g., dry conditions → wildfire AND
          subsequent runoff flooding after rain on burned slopes)
        - Regulatory requirements (some insurers weight flood separately)

    Parameters
    ----------
    wildfire_score, flood_score:
        Individual peril scores in [0, 100].
    wildfire_weight, flood_weight:
        Weights for each peril (should sum to ~1.0).
    correlation_penalty:
        Additional risk premium for compound event scenarios (0-10 pts).

    Returns
    -------
    Composite risk score in [0, 100].
    """
    composite = (wildfire_score * wildfire_weight + flood_score * flood_weight) + correlation_penalty
    return float(np.clip(composite, 0, 100))


def compute_annual_loss_estimate(
    composite_score: float,
    assessed_value: float,
    vulnerability_curve: str = "residential",
) -> dict[str, float]:
    """
    Estimate expected annual loss from composite risk score.

    Maps risk score to an expected annual loss fraction using
    empirical vulnerability curves derived from FEMA historical claims
    and NC fire incident structural loss data.

    Parameters
    ----------
    composite_score:
        Composite risk score [0, 100].
    assessed_value:
        Property replacement value (USD).
    vulnerability_curve:
        "residential", "educational", or "medical".

    Returns
    -------
    dict with:
        eal_fraction: Expected annual loss as fraction of value
        eal_usd: Expected annual loss in USD
        eal_pct: EAL as percentage of assessed value
    """
    # Empirical mapping: score → EAL fraction
    # Derived from NC fire/flood historical loss ratios
    curves: dict[str, list[tuple[float, float]]] = {
        "residential":  [(0, 0.0001), (25, 0.001), (50, 0.005), (75, 0.02), (100, 0.08)],
        "educational":  [(0, 0.0001), (25, 0.0008), (50, 0.004), (75, 0.015), (100, 0.06)],
        "medical":      [(0, 0.0001), (25, 0.0005), (50, 0.003), (75, 0.012), (100, 0.05)],
    }
    curve = curves.get(vulnerability_curve, curves["residential"])
    eal_fraction = float(np.interp(composite_score, [p[0] for p in curve], [p[1] for p in curve]))
    eal_usd = eal_fraction * assessed_value
    return {
        "eal_fraction": eal_fraction,
        "eal_usd": eal_usd,
        "eal_pct": eal_fraction * 100,
    }
