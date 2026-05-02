"""Stage 06R -- build the feature-engineered training dataset.

Reads:
    outputs/labels/stage06P_training_dataset.csv

Adds derived features (raw 06P columns are kept untouched):
    diffusion_length_nm = sqrt(2 * DH_nm2_s * time_s)
    reaction_budget     = Hmax_mol_dm3 * kdep_s_inv * time_s
    quencher_budget     = Q0_mol_dm3   * kq_s_inv   * time_s
    blur_to_pitch       = sigma_nm / pitch_nm
    line_cd_nm_derived  = pitch_nm * line_cd_ratio
    DH_x_time           = DH_nm2_s * time_s
    Hmax_kdep_ratio     = Hmax_mol_dm3 / kdep_s_inv
    Q0_kq_ratio         = Q0_mol_dm3   / kq_s_inv
    dose_x_blur         = dose_mJ_cm2  * sigma_nm

(line_cd_nm_derived is named to avoid colliding with any existing
`line_cd_nm` column that already exists in some FD CSVs.)

Divisions are guarded with epsilon. Non-finite values are written as
empty cells, which downstream readers coerce to NaN.

Writes:
    outputs/labels/stage06R_training_dataset.csv
    outputs/models/stage06R_feature_list.json

Closed Stage 04C / 04D / 06C / 06L / 06P artefacts are not modified.

Policy
    v2_OP_frozen           = true
    published_data_loaded  = false
    No external calibration.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reaction_diffusion_peb_v3_screening.src.metrics_io import (
    FEATURE_KEYS,
    read_labels_csv,
)


V3_DIR = ROOT / "reaction_diffusion_peb_v3_screening"
EPS = 1e-12

DERIVED_FEATURE_KEYS: list[str] = [
    "diffusion_length_nm",
    "reaction_budget",
    "quencher_budget",
    "blur_to_pitch",
    "line_cd_nm_derived",
    "DH_x_time",
    "Hmax_kdep_ratio",
    "Q0_kq_ratio",
    "dose_x_blur",
]

# Full feature list used at training time (raw 06P features first, then
# the 9 derived features in the order above). 06R training and analysis
# scripts read this list from stage06R_feature_list.json so they share
# the same feature ordering.
STAGE06R_FEATURE_KEYS: list[str] = list(FEATURE_KEYS) + DERIVED_FEATURE_KEYS


def _safe_float(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def _safe(x: float) -> float:
    """Return x if finite, else NaN."""
    return x if math.isfinite(x) else float("nan")


def derive_features(row: dict) -> dict:
    """Compute the 9 derived features from one row's raw values.

    Divisions are guarded with epsilon so a zero divisor returns
    inf, then `_safe` converts inf to NaN; downstream code treats
    NaN cells as missing."""
    p   = _safe_float(row.get("pitch_nm"))
    r   = _safe_float(row.get("line_cd_ratio"))
    dose= _safe_float(row.get("dose_mJ_cm2"))
    sg  = _safe_float(row.get("sigma_nm"))
    dh  = _safe_float(row.get("DH_nm2_s"))
    t   = _safe_float(row.get("time_s"))
    hm  = _safe_float(row.get("Hmax_mol_dm3"))
    kd  = _safe_float(row.get("kdep_s_inv"))
    q0  = _safe_float(row.get("Q0_mol_dm3"))
    kq  = _safe_float(row.get("kq_s_inv"))

    out = {
        "diffusion_length_nm": _safe(math.sqrt(2.0 * dh * t))
                                    if (dh >= 0.0 and t >= 0.0) else float("nan"),
        "reaction_budget":     _safe(hm * kd * t),
        "quencher_budget":     _safe(q0 * kq * t),
        "blur_to_pitch":       _safe(sg / max(p, EPS)),
        "line_cd_nm_derived":  _safe(p * r),
        "DH_x_time":           _safe(dh * t),
        "Hmax_kdep_ratio":     _safe(hm / max(kd, EPS)),
        "Q0_kq_ratio":         _safe(q0 / max(kq, EPS)),
        "dose_x_blur":         _safe(dose * sg),
    }
    return out


def augment_rows(rows: list[dict]) -> list[dict]:
    """Return new dicts with the 9 derived features merged in. Caller
    owns the returned list; original rows are not mutated."""
    out: list[dict] = []
    for r in rows:
        new = dict(r)
        new.update(derive_features(r))
        out.append(new)
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--stage06p_csv", type=str,
                   default=str(V3_DIR / "outputs" / "labels"
                               / "stage06P_training_dataset.csv"))
    p.add_argument("--out_csv", type=str,
                   default=str(V3_DIR / "outputs" / "labels"
                               / "stage06R_training_dataset.csv"))
    p.add_argument("--out_feature_list_json", type=str,
                   default=str(V3_DIR / "outputs" / "models"
                               / "stage06R_feature_list.json"))
    args = p.parse_args()

    rows_in = read_labels_csv(args.stage06p_csv)
    rows_out = augment_rows(rows_in)

    # Compose canonical column order: existing 06P canonical + derived.
    if rows_in:
        in_cols = list(rows_in[0].keys())
    else:
        in_cols = []
    cols = list(in_cols)
    for k in DERIVED_FEATURE_KEYS:
        if k not in cols:
            cols.append(k)

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows_out:
            w.writerow(r)

    # Feature-list JSON. Stored under outputs/models so train + analyse
    # scripts can both read it from the same canonical place.
    feat_payload = {
        "stage": "06R",
        "raw_feature_keys":     list(FEATURE_KEYS),
        "derived_feature_keys": list(DERIVED_FEATURE_KEYS),
        "feature_keys":         list(STAGE06R_FEATURE_KEYS),
        "epsilon":              EPS,
        "derived_feature_definitions": {
            "diffusion_length_nm": "sqrt(2 * DH_nm2_s * time_s)",
            "reaction_budget":     "Hmax_mol_dm3 * kdep_s_inv * time_s",
            "quencher_budget":     "Q0_mol_dm3 * kq_s_inv * time_s",
            "blur_to_pitch":       "sigma_nm / pitch_nm",
            "line_cd_nm_derived":  "pitch_nm * line_cd_ratio",
            "DH_x_time":           "DH_nm2_s * time_s",
            "Hmax_kdep_ratio":     "Hmax_mol_dm3 / kdep_s_inv",
            "Q0_kq_ratio":         "Q0_mol_dm3 / kq_s_inv",
            "dose_x_blur":         "dose_mJ_cm2 * sigma_nm",
        },
        "policy": {
            "v2_OP_frozen": True,
            "published_data_loaded": False,
            "external_calibration": "none",
        },
    }
    fpath = Path(args.out_feature_list_json)
    fpath.parent.mkdir(parents=True, exist_ok=True)
    fpath.write_text(json.dumps(feat_payload, indent=2))

    # Console summary.
    n_finite_per_derived = Counter()
    for r in rows_out:
        for k in DERIVED_FEATURE_KEYS:
            v = r.get(k)
            try:
                fv = float(v)
                if math.isfinite(fv):
                    n_finite_per_derived[k] += 1
            except (TypeError, ValueError):
                pass

    print(f"  06P input rows:              {len(rows_in)}")
    print(f"  06R output rows:             {len(rows_out)}")
    print(f"  raw feature count:           {len(FEATURE_KEYS)}")
    print(f"  derived feature count:       {len(DERIVED_FEATURE_KEYS)}")
    print(f"  total feature count:         {len(STAGE06R_FEATURE_KEYS)}")
    for k in DERIVED_FEATURE_KEYS:
        n = n_finite_per_derived[k]
        print(f"    {k:<22} finite_rows = {n:>5} / {len(rows_out)}")
    print(f"  -> {out_path}")
    print(f"  -> {fpath}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
