"""Stage 06A — bounded process-variation sampling.

For a given base recipe (a dict carrying every surrogate feature), draw
N independent variations. Each varied knob uses an independent
Uniform(-w, +w) draw (relative or absolute per spec) and is clipped to
the candidate-space bound of that knob.

`line_cd ±0.5 nm` is applied through `line_cd_ratio` because the
surrogate feature set carries the ratio, not the absolute CD. Width per
recipe = 0.5 / pitch_nm, then clipped into the ratio range spanned by
the candidate-space choice list.

Knobs not listed in the spec (e.g. kdep, kq, abs_len, pitch_nm) get zero
variation — they pass through unchanged. This matches the user
specification.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .candidate_sampler import CandidateSpace
from .metrics_io import FEATURE_KEYS


@dataclass
class KnobSpec:
    name: str
    relative: bool   # True → width interpreted as fraction of base value
    absolute: bool   # True → width interpreted as absolute units
    width: float

    def __post_init__(self) -> None:
        if self.relative == self.absolute:
            raise ValueError(
                f"KnobSpec[{self.name}]: exactly one of relative/absolute must be True"
            )


@dataclass
class VariationSpec:
    knobs: list[KnobSpec]
    line_cd_abs_nm: float = 0.5  # absolute width on the line_cd channel

    @classmethod
    def from_yaml_dict(cls, d: dict) -> "VariationSpec":
        ks = []
        for k in d.get("knobs", []):
            ks.append(KnobSpec(
                name=str(k["name"]),
                relative=bool(k.get("relative", False)),
                absolute=bool(k.get("absolute", False)),
                width=float(k["width"]),
            ))
        return cls(
            knobs=ks,
            line_cd_abs_nm=float(d.get("line_cd_abs_nm", 0.5)),
        )


def _bounds_for(space: CandidateSpace, name: str) -> tuple[float, float] | None:
    """Return (lo, hi) for a uniform parameter, or (min, max) of the choice
    list for a choice parameter. None if the knob is not in the space."""
    for p in space.parameters:
        if p["name"] != name:
            continue
        if p["type"] == "uniform":
            return float(p["low"]), float(p["high"])
        if p["type"] == "choice":
            vals = [float(v) for v in p["values"]]
            return min(vals), max(vals)
    return None


def sample_variations(
    base: dict,
    spec: VariationSpec,
    n: int,
    space: CandidateSpace,
    rng: np.random.Generator | None = None,
) -> list[dict]:
    """Return n independent variations of `base`. Knobs in `spec` get
    Uniform(-w, +w) noise (relative or absolute), then are clipped to the
    candidate-space bound.

    `line_cd_ratio` gets a Uniform(-Δr, +Δr) draw with
    Δr = spec.line_cd_abs_nm / pitch_nm. Bounds = (min, max) of the
    line_cd_ratio choice list in candidate_space.
    """
    if rng is None:
        rng = np.random.default_rng()

    pitch = float(base["pitch_nm"])
    out_rows: list[dict] = []

    # Pre-compute width and bounds per listed knob.
    plans: list[tuple[str, float, tuple[float, float]]] = []
    for k in spec.knobs:
        b = _bounds_for(space, k.name)
        if b is None:
            continue
        if k.relative:
            half_width = float(k.width) * float(base[k.name])
        else:
            half_width = float(k.width)
        plans.append((k.name, half_width, b))

    # line_cd channel — applied through line_cd_ratio.
    ratio_bounds = _bounds_for(space, "line_cd_ratio")
    ratio_half_width = float(spec.line_cd_abs_nm) / pitch if ratio_bounds is not None else 0.0

    for _ in range(n):
        v = dict(base)
        for name, half, (lo, hi) in plans:
            if half <= 0.0:
                continue
            delta = float(rng.uniform(-half, half))
            v[name] = float(np.clip(float(base[name]) + delta, lo, hi))
        if ratio_bounds is not None and ratio_half_width > 0.0:
            lo, hi = ratio_bounds
            delta = float(rng.uniform(-ratio_half_width, ratio_half_width))
            v["line_cd_ratio"] = float(np.clip(float(base["line_cd_ratio"]) + delta, lo, hi))
        # Recompute the derived line_cd_nm so that downstream consumers
        # (e.g. an FD-cfg builder) see a self-consistent recipe — the
        # surrogate itself does not consume line_cd_nm so this is
        # belt-and-braces.
        if "line_cd_nm" in base:
            v["line_cd_nm"] = float(v["line_cd_ratio"]) * float(v["pitch_nm"])
        out_rows.append(v)
    return out_rows


def feature_matrix_from_recipes(
    recipes: Iterable[dict],
    feature_keys: list[str] = FEATURE_KEYS,
) -> np.ndarray:
    """Build the (N, n_features) matrix consumed by the surrogate. Same
    column order as `metrics_io.build_feature_matrix` so the trained model
    feature semantics line up exactly."""
    rows = list(recipes)
    X = np.zeros((len(rows), len(feature_keys)), dtype=np.float64)
    for i, r in enumerate(rows):
        for j, k in enumerate(feature_keys):
            X[i, j] = float(r[k])
    return X
