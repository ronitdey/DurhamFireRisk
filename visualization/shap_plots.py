"""
SHAP attribution visualization: waterfall charts, summary plots,
and before/after mitigation comparison views.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from models.attribution.shap_explainer import RiskExplanation, FeatureAttribution


CATEGORY_COLORS = {
    "structure": "#e74c3c",    # Red — controllable (structure)
    "vegetation": "#e67e22",   # Orange — partially controllable
    "terrain": "#3498db",      # Blue — uncontrollable
    "exposure": "#9b59b6",     # Purple — partially controllable
}

CONTROLLABLE_HATCH = ""
UNCONTROLLABLE_HATCH = "///"


def plot_waterfall(
    explanation: RiskExplanation,
    output_path: Optional[Path] = None,
    max_features: int = 12,
    figsize: tuple = (10, 7),
) -> plt.Figure:
    """
    Plot a SHAP waterfall chart for a single property's risk explanation.

    Shows how each feature contributes positively or negatively to the
    final risk score from the population baseline. Controllable features
    are shown with solid fill; uncontrollable with hatching.

    Parameters
    ----------
    explanation:
        RiskExplanation from WildfireRiskExplainer.explain_property().
    output_path:
        If provided, save the figure to this path.
    max_features:
        Maximum number of features to display (others grouped as "Other").
    figsize:
        Figure size in inches.

    Returns
    -------
    matplotlib Figure.
    """
    attrs = explanation.top_risks[:max_features]
    labels = [a.label for a in attrs]
    values = [a.shap_value for a in attrs]
    categories = [a.category for a in attrs]
    controllable = [a.controllable for a in attrs]

    # Waterfall: compute running sum
    base = explanation.base_value
    running = base
    lefts: list[float] = []
    for v in values:
        lefts.append(running)
        running += v

    fig, ax = plt.subplots(figsize=figsize)

    for i, (label, val, cat, ctrl, left) in enumerate(
        zip(labels, values, categories, controllable, lefts)
    ):
        color = CATEGORY_COLORS.get(cat, "#95a5a6")
        hatch = CONTROLLABLE_HATCH if ctrl else UNCONTROLLABLE_HATCH
        bar_color = color if val > 0 else "#2ecc71"

        ax.barh(
            i, val, left=left, color=bar_color, hatch=hatch,
            edgecolor="white", linewidth=0.5, height=0.7,
        )
        # Value label
        x_text = left + val + (0.5 if val > 0 else -0.5)
        ha = "left" if val > 0 else "right"
        ax.text(x_text, i, f"{val:+.1f}", va="center", ha=ha, fontsize=9, color="#2c3e50")

    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=10)
    ax.invert_yaxis()

    ax.axvline(base, color="#95a5a6", linestyle="--", linewidth=1, label=f"Baseline: {base:.0f}")
    ax.axvline(explanation.risk_score, color="#c0392b", linestyle="-", linewidth=2,
               label=f"Score: {explanation.risk_score:.0f}")

    ax.set_xlabel("Risk Score Contribution", fontsize=11)
    ax.set_title(
        f"Risk Attribution — {explanation.parcel_id}\n"
        f"Score: {explanation.risk_score:.0f}/100 | "
        f"Controllable: {explanation.controllable_risk_points:.0f} pts | "
        f"Terrain: {explanation.uncontrollable_risk_points:.0f} pts",
        fontsize=12, fontweight="bold",
    )

    # Legend for categories and controllability
    legend_elements = [
        mpatches.Patch(facecolor=c, label=cat.title()) for cat, c in CATEGORY_COLORS.items()
    ] + [
        mpatches.Patch(facecolor="#bdc3c7", label="Controllable (solid)"),
        mpatches.Patch(facecolor="#bdc3c7", hatch="///", label="Not controllable (hatched)"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=8, ncol=2)
    ax.grid(axis="x", alpha=0.3)

    fig.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    return fig


def plot_mitigation_comparison(
    original_score: float,
    mitigated_score: float,
    action_results: list[dict],
    property_name: str = "",
    output_path: Optional[Path] = None,
    figsize: tuple = (10, 5),
) -> plt.Figure:
    """
    Side-by-side before/after risk score comparison with action breakdown.
    """
    fig, (ax_gauge, ax_bar) = plt.subplots(1, 2, figsize=figsize, gridspec_kw={"width_ratios": [1, 2]})

    # Left: gauge/score comparison
    _draw_score_gauge(ax_gauge, original_score, mitigated_score, property_name)

    # Right: action contribution bars
    actions = action_results[:8]
    reductions = [a.get("risk_reduction_pts", 0) for a in actions]
    names = [a.get("description", a.get("action", ""))[:35] for a in actions]
    costs = [a.get("cost_estimate_usd", (0, 0)) for a in actions]

    bars = ax_bar.barh(range(len(actions)), reductions, color="#e74c3c", alpha=0.8, height=0.6)
    ax_bar.set_yticks(range(len(actions)))
    ax_bar.set_yticklabels(names, fontsize=9)
    ax_bar.invert_yaxis()

    for i, (bar, cost) in enumerate(zip(bars, costs)):
        cost_str = f"${cost[0]:,}–${cost[1]:,}"
        ax_bar.text(bar.get_width() + 0.3, i, f"-{bar.get_width():.1f} pts | {cost_str}",
                    va="center", fontsize=8, color="#2c3e50")

    ax_bar.set_xlabel("Risk Score Reduction (points)", fontsize=10)
    ax_bar.set_title("Mitigation Actions by Impact", fontsize=11, fontweight="bold")
    ax_bar.grid(axis="x", alpha=0.3)

    fig.tight_layout()
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    return fig


def _draw_score_gauge(ax, original: float, mitigated: float, name: str) -> None:
    """Draw a simple before/after score visual."""
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 1)
    ax.axis("off")

    def _color(score: float) -> str:
        if score < 30: return "#2ecc71"
        if score < 55: return "#f39c12"
        if score < 75: return "#e74c3c"
        return "#8e44ad"

    def _label(score: float) -> str:
        if score < 30: return "LOW"
        if score < 55: return "MODERATE"
        if score < 75: return "HIGH"
        return "VERY HIGH"

    ax.text(50, 0.95, name, ha="center", va="top", fontsize=10, fontweight="bold")
    ax.text(25, 0.75, "BEFORE", ha="center", fontsize=9, color="#7f8c8d")
    ax.text(75, 0.75, "AFTER", ha="center", fontsize=9, color="#7f8c8d")

    # Score circles
    for x, score in [(25, original), (75, mitigated)]:
        circle = plt.Circle((x / 100, 0.45), 0.18, color=_color(score), transform=ax.transAxes)
        ax.add_patch(circle)
        ax.text(x / 100, 0.45, f"{score:.0f}", ha="center", va="center",
                fontsize=18, fontweight="bold", color="white", transform=ax.transAxes)
        ax.text(x / 100, 0.20, _label(score), ha="center", va="center",
                fontsize=9, color=_color(score), transform=ax.transAxes)

    reduction = original - mitigated
    ax.text(0.5, 0.05,
            f"Risk reduction: {reduction:.0f} pts ({reduction / max(original, 1) * 100:.0f}%)",
            ha="center", va="bottom", fontsize=9, color="#27ae60",
            fontweight="bold", transform=ax.transAxes)
