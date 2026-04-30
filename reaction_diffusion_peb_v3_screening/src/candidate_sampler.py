"""Latin-Hypercube / Sobol sampling for the v3 candidate space.

The candidate space mixes discrete (`type: choice`) and continuous
(`type: uniform`) parameters. We draw a single low-discrepancy sample in
[0,1)^d and then map each axis to its declared parameter type.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from scipy.stats import qmc


@dataclass
class CandidateSpace:
    parameters: list[dict]
    derived: list[dict]
    fixed: dict[str, Any]

    @classmethod
    def from_yaml(cls, path: str | Path) -> "CandidateSpace":
        cfg = yaml.safe_load(Path(path).read_text())
        return cls(
            parameters=cfg["parameters"],
            derived=cfg.get("derived", []),
            fixed=cfg.get("fixed", {}),
        )


def _map_axis(u: np.ndarray, spec: dict) -> np.ndarray:
    """Map a [0,1) column to one parameter axis."""
    if spec["type"] == "choice":
        values = list(spec["values"])
        idx = np.minimum(np.floor(u * len(values)).astype(int), len(values) - 1)
        return np.array([values[i] for i in idx])
    if spec["type"] == "uniform":
        lo = float(spec["low"])
        hi = float(spec["high"])
        return lo + u * (hi - lo)
    raise ValueError(f"unsupported parameter type: {spec['type']}")


def _eval_derived(row: dict, derived: list[dict]) -> dict:
    """Evaluate derived expressions safely with the row's named values."""
    env = dict(row)
    out = {}
    for d in derived:
        out[d["name"]] = float(eval(d["formula"], {"__builtins__": {}}, env))  # noqa: S307
        env[d["name"]] = out[d["name"]]
    return out


def sample_candidates(
    space: CandidateSpace,
    n: int,
    method: str = "sobol",
    seed: int | None = 7,
) -> list[dict]:
    """Return n candidates as dicts. Each dict carries every parameter +
    derived value + fixed value, so it can be plugged directly into the
    FD batch runner."""
    d = len(space.parameters)
    if method == "sobol":
        sampler = qmc.Sobol(d=d, scramble=True, seed=seed)
        u = sampler.random(n)
    elif method == "latin_hypercube":
        sampler = qmc.LatinHypercube(d=d, seed=seed)
        u = sampler.random(n)
    else:
        raise ValueError(f"unsupported sampling method: {method}")

    cols = {}
    for j, spec in enumerate(space.parameters):
        cols[spec["name"]] = _map_axis(u[:, j], spec)

    candidates = []
    for i in range(n):
        row = {name: (col[i].item() if hasattr(col[i], "item") else col[i])
               for name, col in cols.items()}
        # Discrete-cast where appropriate so downstream code sees ints, not floats.
        if "pitch_nm" in row and isinstance(cols["pitch_nm"][i], (np.integer, int)):
            row["pitch_nm"] = int(row["pitch_nm"])
        derived = _eval_derived(row, space.derived)
        candidate = {**row, **derived, **space.fixed, "_id": i}
        candidates.append(candidate)
    return candidates


def _apply_bias_to_parameter(param: dict, bias: dict) -> dict:
    """Return a copy of `param` with `bias` overrides applied. `bias` may carry
    `low`/`high` for uniform params or `choice` for choice params."""
    p = dict(param)
    if param["type"] == "uniform" and ("low" in bias or "high" in bias):
        p["low"] = float(bias.get("low", param["low"]))
        p["high"] = float(bias.get("high", param["high"]))
    elif param["type"] == "choice" and "choice" in bias:
        p["values"] = list(bias["choice"])
    elif param["type"] == "uniform" and "values" in bias:
        # allow biasing a uniform into a discrete short list (e.g. pitch)
        p["type"] = "choice"
        p["values"] = list(bias["values"])
    return p


def sample_with_bias(
    space: CandidateSpace,
    bias_spec: dict,
    n: int,
    method: str = "latin_hypercube",
    seed: int | None = 7,
) -> list[dict]:
    """Sample n candidates from a biased version of `space`. Parameters listed
    under `bias_spec["parameters"]` get their ranges narrowed; everything else
    keeps the full range."""
    biased_params = []
    bias_per_param = bias_spec.get("parameters", {})
    for p in space.parameters:
        b = bias_per_param.get(p["name"], {})
        if not b:
            biased_params.append(p)
        else:
            biased_params.append(_apply_bias_to_parameter(p, b))
    biased_space = CandidateSpace(
        parameters=biased_params,
        derived=space.derived,
        fixed=space.fixed,
    )
    return sample_candidates(biased_space, n=n, method=method, seed=seed)


def perturb_candidate(
    base: dict,
    perturb_keys: list[str],
    relative_amplitude: float,
    bounds_from_space: CandidateSpace,
    rng: np.random.Generator,
) -> dict:
    """Return a perturbed copy of `base`. Only parameters listed in
    `perturb_keys` are changed; the perturbation is uniform within
    ±relative_amplitude × range_span and clipped to the original range.
    Discrete parameters (choice) are left unchanged."""
    out = dict(base)
    bounds = {p["name"]: p for p in bounds_from_space.parameters}
    for k in perturb_keys:
        if k not in bounds or bounds[k]["type"] != "uniform":
            continue
        lo = float(bounds[k]["low"])
        hi = float(bounds[k]["high"])
        span = hi - lo
        delta = float(rng.uniform(-relative_amplitude, relative_amplitude)) * span
        v = float(out[k]) + delta
        out[k] = float(np.clip(v, lo, hi))
    # Recompute derived values.
    derived = _eval_derived(out, bounds_from_space.derived)
    out.update(derived)
    return out


def sample_margin_perturbation(
    space: CandidateSpace,
    bias_spec: dict,
    seed_rows: list[dict],
    n: int,
    seed: int | None = 7,
) -> list[dict]:
    """Pick rows from `seed_rows` that are inside the margin band, perturb
    them, and return n candidates. Falls back to plain LHS over the full
    space if no seed row is in the band."""
    perturb_cfg = bias_spec.get("perturbation", {})
    band = perturb_cfg.get("margin_band", [0.0, 0.05])
    keys = perturb_cfg.get("perturb_keys", [])
    rel = float(perturb_cfg.get("perturb_relative", 0.10))

    boundary = []
    for r in seed_rows:
        try:
            m = float(r.get("P_line_margin", float("nan")))
        except (TypeError, ValueError):
            continue
        if not np.isfinite(m):
            continue
        if band[0] <= m <= band[1]:
            boundary.append(r)
    if not boundary:
        return sample_with_bias(space, bias_spec={}, n=n, method="latin_hypercube", seed=seed)

    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        base = boundary[int(rng.integers(0, len(boundary)))]
        # the seed row carries CSV strings; coerce known numeric params back to floats
        base_clean = dict(base)
        for k in keys + ["pitch_nm", "line_cd_ratio"]:
            if k in base_clean:
                try:
                    base_clean[k] = float(base_clean[k])
                except (TypeError, ValueError):
                    pass
        # ensure required fixed-keys are present in the candidate
        for fk, fv in space.fixed.items():
            base_clean.setdefault(fk, fv)
        # ensure derived keys are present
        if "line_cd_nm" not in base_clean and "pitch_nm" in base_clean and "line_cd_ratio" in base_clean:
            base_clean["line_cd_nm"] = float(base_clean["pitch_nm"]) * float(base_clean["line_cd_ratio"])
        if "domain_x_nm" not in base_clean and "pitch_nm" in base_clean:
            base_clean["domain_x_nm"] = float(base_clean["pitch_nm"]) * 5.0
        if "dose_norm" not in base_clean and "dose_mJ_cm2" in base_clean and "reference_dose_mJ_cm2" in base_clean:
            base_clean["dose_norm"] = float(base_clean["dose_mJ_cm2"]) / float(base_clean["reference_dose_mJ_cm2"])

        c = perturb_candidate(base_clean, keys, rel, space, rng)
        c["_id"] = f"perturb_{i}"
        out.append(c)
    return out


def write_jsonl(candidates: list[dict], out_path: str | Path) -> None:
    import json
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for c in candidates:
            f.write(json.dumps(c) + "\n")


def read_jsonl(in_path: str | Path) -> list[dict]:
    import json
    out = []
    with open(in_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out
