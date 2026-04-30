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
