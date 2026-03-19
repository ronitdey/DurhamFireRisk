"""
Counterfactual mitigation scenario runner.

Clones a PropertyTwin, applies mitigation actions, re-runs the risk model,
and quantifies the risk reduction and cost-effectiveness of each action.

This implements the core of Stand's value proposition:
    "Not just your risk score — but exactly what to change, and what each change buys you."
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Optional

from loguru import logger

from twin.property_twin import PropertyTwin


@dataclass
class MitigationAction:
    """Definition of a single mitigation action."""
    key: str
    description: str
    modifies: dict                              # {field: new_value}
    cost_estimate_range: tuple[int, int]        # (low, high) in USD
    category: str = "structure"                 # structure | vegetation | exposure


@dataclass
class ActionResult:
    """Result of applying a single mitigation action."""
    action: MitigationAction
    original_score: float
    mitigated_score: float
    risk_reduction_pts: float                   # Absolute reduction
    risk_reduction_pct: float                   # Percentage reduction
    fire_arrival_time_gained_min: float         # Minutes of additional warning
    structural_survival_improvement: float      # Delta in survival probability
    priority_rank: int = 0

    @property
    def cost_low(self) -> int:
        return self.action.cost_estimate_range[0]

    @property
    def cost_high(self) -> int:
        return self.action.cost_estimate_range[1]

    @property
    def cost_per_risk_point_low(self) -> str:
        if self.risk_reduction_pts <= 0:
            return "N/A"
        return f"${self.cost_low / self.risk_reduction_pts:,.0f}"

    @property
    def cost_per_risk_point_high(self) -> str:
        if self.risk_reduction_pts <= 0:
            return "N/A"
        return f"${self.cost_high / self.risk_reduction_pts:,.0f}"

    def to_dict(self) -> dict:
        return {
            "action": self.action.key,
            "description": self.action.description,
            "original_risk_score": round(self.original_score, 2),
            "mitigated_risk_score": round(self.mitigated_score, 2),
            "risk_reduction_pts": round(self.risk_reduction_pts, 2),
            "risk_reduction_pct": round(self.risk_reduction_pct, 1),
            "fire_arrival_time_gained_min": round(self.fire_arrival_time_gained_min, 1),
            "structural_survival_improvement": round(self.structural_survival_improvement, 3),
            "cost_estimate_usd": self.action.cost_estimate_range,
            "cost_per_risk_point": f"{self.cost_per_risk_point_low}–{self.cost_per_risk_point_high}",
            "priority_rank": self.priority_rank,
        }


@dataclass
class MitigationResult:
    """Result of applying a set of mitigation actions together."""
    original_risk_score: float
    mitigated_risk_score: float
    action_results: list[ActionResult] = field(default_factory=list)

    @property
    def risk_reduction_pct(self) -> float:
        if self.original_risk_score <= 0:
            return 0.0
        return (self.original_risk_score - self.mitigated_risk_score) / self.original_risk_score * 100

    def to_dict(self) -> dict:
        return {
            "original_risk_score": round(self.original_risk_score, 2),
            "mitigated_risk_score": round(self.mitigated_risk_score, 2),
            "risk_reduction_pct": round(self.risk_reduction_pct, 1),
            "actions": [ar.to_dict() for ar in self.action_results],
        }


# ── Mitigation action library ─────────────────────────────────────────────────

MITIGATION_ACTIONS: dict[str, MitigationAction] = {
    "replace_wood_roof_with_metal": MitigationAction(
        key="replace_wood_roof_with_metal",
        description="Replace wood shake/shingle roof with metal standing seam",
        modifies={"roof_material": "metal_standing_seam"},
        cost_estimate_range=(15_000, 45_000),
        category="structure",
    ),
    "screen_all_vents": MitigationAction(
        key="screen_all_vents",
        description="Install 1/16\" mesh screens on all vents and soffits",
        modifies={"vent_screening_status": "screened"},
        cost_estimate_range=(500, 2_000),
        category="structure",
    ),
    "clear_zone1_combustibles": MitigationAction(
        key="clear_zone1_combustibles",
        description="Remove all combustible material within 5ft of structure",
        modifies={"zone1_fuel_load": 0.0},
        cost_estimate_range=(200, 800),
        category="vegetation",
    ),
    "replace_zone1_mulch_with_gravel": MitigationAction(
        key="replace_zone1_mulch_with_gravel",
        description="Replace organic mulch with decomposed granite in Zone 1",
        modifies={"zone1_fuel_load": 0.02},
        cost_estimate_range=(300, 1_200),
        category="vegetation",
    ),
    "remove_ladder_fuels": MitigationAction(
        key="remove_ladder_fuels",
        description="Remove shrubs and low branches creating vertical fuel continuity",
        modifies={"ladder_fuel_present": False, "zone2_fuel_load_pct_reduction": 0.40},
        cost_estimate_range=(1_500, 5_000),
        category="vegetation",
    ),
    "reduce_zone2_fuels": MitigationAction(
        key="reduce_zone2_fuels",
        description="Thin vegetation and remove dead material in 5-30ft zone",
        modifies={"zone2_fuel_load_pct_reduction": 0.50, "zone3_fuel_continuity": 0.4},
        cost_estimate_range=(2_000, 6_000),
        category="vegetation",
    ),
    "replace_wood_deck": MitigationAction(
        key="replace_wood_deck",
        description="Replace wood deck with composite or concrete decking",
        modifies={"deck_material": "composite_fire_resistant"},
        cost_estimate_range=(8_000, 25_000),
        category="structure",
    ),
    "install_ember_resistant_fence": MitigationAction(
        key="install_ember_resistant_fence",
        description="Replace wood fence within 30ft with metal or masonry",
        modifies={"zone2_fuel_load_pct_reduction": 0.20},
        cost_estimate_range=(3_000, 12_000),
        category="vegetation",
    ),
}


class MitigationScenarioRunner:
    """
    Runs counterfactual scenarios to quantify risk reduction from mitigation actions.

    Requires a callable `scorer` that accepts a PropertyTwin and returns
    a risk score in [0, 100]. Typically wraps WildfireScorer.score_twin().
    """

    def __init__(self, scorer: Any):
        """
        Parameters
        ----------
        scorer:
            Callable that takes a dict (PropertyTwin attributes) and returns
            a float risk score in [0, 100].
        """
        self.scorer = scorer

    def run_counterfactual(
        self,
        twin: PropertyTwin,
        action_keys: list[str],
    ) -> MitigationResult:
        """
        Apply a set of mitigation actions and compute the combined risk reduction.

        Parameters
        ----------
        twin:
            Original PropertyTwin to analyze.
        action_keys:
            List of action keys from MITIGATION_ACTIONS.

        Returns
        -------
        MitigationResult with before/after scores and per-action breakdown.
        """
        original_score = self.scorer(twin.to_dict()) if not twin.wildfire_risk_score else twin.wildfire_risk_score

        modified_twin = copy.deepcopy(twin)
        action_results: list[ActionResult] = []

        for key in action_keys:
            action = MITIGATION_ACTIONS.get(key)
            if action is None:
                logger.warning(f"Unknown mitigation action: {key}")
                continue

            # Apply modifications to twin clone (sequential, cumulative)
            before_score = self.scorer(modified_twin.to_dict())
            _apply_action(modified_twin, action)
            after_score = self.scorer(modified_twin.to_dict())

            reduction = before_score - after_score
            reduction_pct = (reduction / max(before_score, 0.01)) * 100

            # Estimate time-of-arrival benefit: each risk point ≈ 0.3 minutes
            toa_gain = reduction * 0.3

            # Structural survival improvement: sigmoid mapping
            survival_improvement = min(reduction / 30.0, 0.5)

            action_results.append(ActionResult(
                action=action,
                original_score=before_score,
                mitigated_score=after_score,
                risk_reduction_pts=reduction,
                risk_reduction_pct=reduction_pct,
                fire_arrival_time_gained_min=toa_gain,
                structural_survival_improvement=survival_improvement,
            ))

        final_score = self.scorer(modified_twin.to_dict())

        result = MitigationResult(
            original_risk_score=original_score,
            mitigated_risk_score=final_score,
            action_results=action_results,
        )
        logger.info(
            f"Counterfactual complete: {original_score:.1f} → {final_score:.1f} "
            f"({result.risk_reduction_pct:.1f}% reduction)"
        )
        return result

    def rank_all_mitigations(self, twin: PropertyTwin) -> list[ActionResult]:
        """
        Run every available single-action counterfactual and rank by
        risk reduction per dollar (cost-effectiveness).

        Returns list of ActionResult sorted by risk_reduction_pts descending.
        """
        original_score = self.scorer(twin.to_dict())
        results: list[ActionResult] = []

        for key, action in MITIGATION_ACTIONS.items():
            modified_twin = copy.deepcopy(twin)
            _apply_action(modified_twin, action)
            after_score = self.scorer(modified_twin.to_dict())

            reduction = original_score - after_score
            if reduction <= 0:
                continue

            reduction_pct = (reduction / max(original_score, 0.01)) * 100
            results.append(ActionResult(
                action=action,
                original_score=original_score,
                mitigated_score=after_score,
                risk_reduction_pts=reduction,
                risk_reduction_pct=reduction_pct,
                fire_arrival_time_gained_min=reduction * 0.3,
                structural_survival_improvement=min(reduction / 30.0, 0.5),
            ))

        # Sort by risk_reduction_pts (impact), then annotate ranks
        results.sort(key=lambda r: r.risk_reduction_pts, reverse=True)
        for i, r in enumerate(results):
            r.priority_rank = i + 1

        if results:
            logger.info(
                f"Top mitigation: '{results[0].action.description}' "
                f"→ {results[0].risk_reduction_pts:.1f} pts reduction"
            )
        return results


def _apply_action(twin: PropertyTwin, action: MitigationAction) -> None:
    """Apply a MitigationAction's modifications to a twin in-place."""
    for field_name, new_value in action.modifies.items():
        if field_name == "zone2_fuel_load_pct_reduction":
            twin.zone2_fuel_load *= (1 - new_value)
        elif field_name == "zone3_fuel_continuity":
            twin.zone3_fuel_continuity = min(twin.zone3_fuel_continuity, new_value)
        elif hasattr(twin, field_name):
            setattr(twin, field_name, new_value)
