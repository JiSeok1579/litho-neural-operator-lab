"""PEB Phase-9 demo: validate the generated FD datasets.

Run:
    python reaction_diffusion_peb/experiments/09_dataset_generation/validate_dataset.py

For both the safe and stiff Phase-9 archives, asserts:

  - input/output field shapes are consistent across all samples
  - every per-sample scalar is finite
  - P stays in [0, 1] sample-by-sample
  - R is binary
  - splits are non-overlapping and cover every sample exactly once
  - metadata records the regime, solver, seed, parameter ranges

Exits non-zero on first failure. Writes a small CSV summary alongside.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np

from reaction_diffusion_peb.src.dataset_builder import (
    AERIAL_KIND_CODES,
    DATASET_FIELD_NAMES,
    DATASET_SCALAR_NAMES,
    load_dataset,
)

OUT_DATA = Path("reaction_diffusion_peb/outputs/datasets")
OUT_LOG = Path("reaction_diffusion_peb/outputs/logs")

CASES = [
    ("safe", OUT_DATA / "peb_phase9_safe_dataset.npz"),
    ("stiff", OUT_DATA / "peb_phase9_stiff_dataset.npz"),
]


def validate_one(regime: str, path: Path, summary_rows: list[list[str]]) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"missing dataset {path} — run generate_fd_dataset.py first"
        )
    arrays, meta = load_dataset(path)
    n = arrays["I"].shape[0]
    G = arrays["I"].shape[1]
    assert arrays["I"].shape == (n, G, G)

    # 1. shapes
    for name in DATASET_FIELD_NAMES:
        assert name in arrays, f"missing field {name}"
        assert arrays[name].shape == (n, G, G), \
            f"{name} shape {arrays[name].shape} != {(n, G, G)}"
        assert arrays[name].dtype == np.float32

    # 2. scalars finite
    for name in DATASET_SCALAR_NAMES:
        assert name in arrays, f"missing scalar {name}"
        assert arrays[name].shape == (n,), f"{name} shape mismatch"
        assert np.all(np.isfinite(arrays[name])), f"{name} has NaN/Inf"

    # 3. aerial codes valid
    codes = arrays["aerial_kind_code"]
    valid_codes = set(AERIAL_KIND_CODES.values())
    bad = [int(c) for c in codes if int(c) not in valid_codes]
    assert not bad, f"unknown aerial_kind_code(s): {bad}"

    # 4. physical bounds
    assert np.all(arrays["I"] >= 0.0) and np.all(arrays["I"] <= 1.0 + 1e-5)
    assert np.all(arrays["H0"] >= 0.0)
    assert np.all(arrays["H_final"] >= 0.0)
    assert np.all(arrays["Q_final"] >= 0.0)
    assert np.all(arrays["P_final"] >= 0.0)
    assert np.all(arrays["P_final"] <= 1.0)
    R = arrays["R"]
    uniq = np.unique(R)
    assert set(uniq.tolist()).issubset({0.0, 1.0}), \
        f"R is not binary: {uniq.tolist()}"

    # 5. splits
    splits = meta.get("splits", {})
    assert set(splits.keys()) == {"train", "val", "test"}
    train, val, test = splits["train"], splits["val"], splits["test"]
    all_idx = sorted(train + val + test)
    assert all_idx == list(range(n)), \
        "splits do not cover every sample exactly once"
    assert not (set(train) & set(val))
    assert not (set(train) & set(test))
    assert not (set(val) & set(test))

    # 6. metadata
    assert meta.get("solver") == "fd"
    assert meta.get("regime") == regime
    assert "seed" in meta
    assert "parameter_ranges" in meta

    # 7. regime-specific kq range
    kq = arrays["kq_ref_s_inv"]
    if regime == "safe":
        assert kq.min() >= 0.5 and kq.max() <= 5.0, \
            f"safe regime kq out of range: [{kq.min()}, {kq.max()}]"
    else:
        assert kq.min() >= 100.0 and kq.max() <= 1000.0, \
            f"stiff regime kq out of range: [{kq.min()}, {kq.max()}]"

    print(f"[{regime}] OK   n={n}  grid={G}x{G}  "
          f"P_max range=[{arrays['P_final'].max(axis=(1,2)).min():.3f},"
          f" {arrays['P_final'].max(axis=(1,2)).max():.3f}]  "
          f"R_pixels range=[{int(R.sum(axis=(1,2)).min())}, "
          f"{int(R.sum(axis=(1,2)).max())}]")
    summary_rows.append([
        regime, str(path.name), str(n), f"{G}x{G}",
        f"{kq.min():.3g}", f"{kq.max():.3g}",
        f"{arrays['P_final'].max(axis=(1,2)).mean():.4f}",
        f"{int(R.sum(axis=(1,2)).mean())}",
        str(len(train)), str(len(val)), str(len(test)),
    ])


def main() -> None:
    OUT_LOG.mkdir(parents=True, exist_ok=True)
    rows = [["regime", "file", "n_samples", "grid",
             "kq_min", "kq_max", "mean_P_max", "mean_R_px",
             "train", "val", "test"]]
    for regime, path in CASES:
        validate_one(regime, path, rows)
    log = OUT_LOG / "peb_phase9_dataset_summary.csv"
    with open(log, "w", newline="") as f:
        csv.writer(f).writerows(rows)
    print(f"\nwrote {log}")


if __name__ == "__main__":
    main()
