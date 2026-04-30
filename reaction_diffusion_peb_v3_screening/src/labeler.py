"""Map an FD output dict to one of the v3 status labels.

Precedence (top wins):
    numerical_invalid > merged > under_exposed > roughness_degraded
                                                   > margin_risk
                                                   > robust_valid
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class LabelThresholds:
    P_space_max: float = 0.50
    P_line_min: float = 0.65
    contrast_min: float = 0.15
    area_frac_max: float = 0.90
    CD_pitch_frac_max: float = 0.85
    P_line_margin_robust: float = 0.05
    roughness_excess_nm: float = 1.5

    @classmethod
    def from_yaml(cls, schema_yaml_path: str) -> "LabelThresholds":
        import yaml
        from pathlib import Path
        cfg = yaml.safe_load(Path(schema_yaml_path).read_text())
        t = cfg.get("thresholds", {})
        return cls(
            P_space_max=float(t.get("P_space_max", 0.50)),
            P_line_min=float(t.get("P_line_min", 0.65)),
            contrast_min=float(t.get("contrast_min", 0.15)),
            area_frac_max=float(t.get("area_frac_max", 0.90)),
            CD_pitch_frac_max=float(t.get("CD_pitch_frac_max", 0.85)),
            P_line_margin_robust=float(t.get("P_line_margin_robust", 0.05)),
            roughness_excess_nm=float(t.get("roughness_excess_nm", 1.5)),
        )


def _is_finite(x) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(float(x))


def label_one(r: dict, t: LabelThresholds | None = None) -> str:
    """r is the helper output dict (run_one_with_overrides). Returns one
    of the six v3 label strings."""
    t = t or LabelThresholds()

    # numerical_invalid
    if not (
        _is_finite(r.get("P_max")) and _is_finite(r.get("P_min"))
        and _is_finite(r.get("LER_after_PEB_P_nm"))
    ):
        return "numerical_invalid"
    if r.get("H_min", 0.0) < -1e-6:
        return "numerical_invalid"
    if r.get("P_min", 0.0) < -1e-6 or r.get("P_max", 0.0) > 1.0 + 1e-6:
        return "numerical_invalid"

    # merged
    cd_pitch = r.get("CD_pitch_frac")
    if (
        r.get("P_space_center_mean", 0.0) >= t.P_space_max
        or r.get("area_frac", 0.0) >= t.area_frac_max
        or (_is_finite(cd_pitch) and cd_pitch >= t.CD_pitch_frac_max)
    ):
        return "merged"

    # under_exposed
    if r.get("P_line_center_mean", 0.0) < t.P_line_min:
        return "under_exposed"

    # below: passes basic interior gate (P_space, P_line, area, CD/pitch).
    # contrast guard is folded under margin_risk if it fails.
    if r.get("contrast", 0.0) <= t.contrast_min:
        return "margin_risk"

    # roughness_degraded — locked LER exceeds the σ-independent design baseline.
    ler_design = r.get("LER_design_initial_nm")
    ler_locked = r.get("LER_CD_locked_nm")
    if (
        _is_finite(ler_design)
        and _is_finite(ler_locked)
        and (ler_locked - ler_design) > t.roughness_excess_nm
    ):
        return "roughness_degraded"

    # margin_risk vs robust_valid by margin.
    margin = r.get("P_line_margin", 0.0)
    if margin >= t.P_line_margin_robust:
        return "robust_valid"
    return "margin_risk"


def label_batch(rows: list[dict], t: LabelThresholds | None = None) -> list[str]:
    return [label_one(r, t=t) for r in rows]


# Defect-class grouping (used by metric reports).
DEFECT_GROUP = {
    "robust_valid":      "normal",
    "margin_risk":       "marginal",
    "under_exposed":     "defect",
    "merged":            "defect",
    "roughness_degraded": "defect",
    "numerical_invalid": "defect",
}

LABEL_ORDER = [
    "robust_valid",
    "margin_risk",
    "roughness_degraded",
    "under_exposed",
    "merged",
    "numerical_invalid",
]
