"""Stage 06P-B -- merge targeted G_4867 AL FD rows into the 06R
training pool to produce stage06PB_training_dataset.csv.

Inputs
    outputs/labels/stage06R_training_dataset.csv     (14,194 rows;
                                                       derived features
                                                       already present)
    outputs/labels/stage06PB_targeted_fd_rows.csv    (~702 rows; raw
                                                       features only;
                                                       derived features
                                                       computed here)

Adds, for the 06PB rows only:
    source           = stage06PB_G4867_targeted_AL  (kept verbatim from
                                                       run_stage06pb_targeted_fd.py)
    mode             = mode_a (pitch=24, ratio=0.52)
    recipe_family    = G4867_family
    boundary_tags    = ;-joined subset of
        time_boundary, cd_boundary, ler_boundary,
        margin_boundary, strict_boundary,
        reaction_budget_boundary, diffusion_boundary,
        quencher_boundary, cd_drift_boundary,
        strict_pass_residual_target
    strict_score_per_row -- one-hot strict_score under the active 06G
                            strict config (no std penalties).
    9 derived process-budget features (build_stage06r_dataset.derive_features).

Closed Stage 04C / 04D / 06C / 06L / 06P / 06R datasets are not
mutated. Output: outputs/labels/stage06PB_training_dataset.csv.

Dedup: keep the first occurrence of each rounded feature tuple in
source order (06R first, then 06PB).

Policy
    v2_OP_frozen           = true
    published_data_loaded  = false
    No external calibration.
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reaction_diffusion_peb_v3_screening.src.metrics_io import (
    FEATURE_KEYS,
    read_labels_csv,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_stage06l_dataset import per_row_strict_score  # noqa: E402
from build_stage06r_dataset import (
    DERIVED_FEATURE_KEYS, derive_features,
)


V3_DIR = ROOT / "reaction_diffusion_peb_v3_screening"
CD_TARGET_NM = 15.0


def _safe_float(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def _round_key(row: dict, ndigits: int = 6) -> tuple:
    out = []
    for k in FEATURE_KEYS:
        try:
            out.append(round(float(row[k]), ndigits))
        except (TypeError, ValueError, KeyError):
            out.append(None)
    return tuple(out)


def _tag_06pb_boundary(row: dict, *, strict_yaml: dict,
                            ref_g4867: dict) -> str:
    """Boundary tags for a 06PB targeted FD row."""
    th = strict_yaml["thresholds"]
    cd_tol  = float(th["cd_tol_nm"])
    ler_cap = float(th["ler_cap_nm"])

    cd  = _safe_float(row.get("CD_final_nm"))
    ler = _safe_float(row.get("LER_CD_locked_nm"))
    mar = _safe_float(row.get("P_line_margin"))
    sc  = _safe_float(row.get("strict_score_per_row"))
    off = _safe_float(row.get("time_offset_s"))

    diffusion_length = _safe_float(row.get("diffusion_length_nm"))
    reaction_budget  = _safe_float(row.get("reaction_budget"))
    quencher_budget  = _safe_float(row.get("quencher_budget"))
    ref_diff = _safe_float(ref_g4867.get("diffusion_length_nm"))
    ref_react = _safe_float(ref_g4867.get("reaction_budget"))
    ref_quench = _safe_float(ref_g4867.get("quencher_budget"))

    tags: list[str] = []
    if np.isfinite(off) and abs(off) > 0.0:
        tags.append("time_boundary")
    if np.isfinite(cd):
        cd_err = abs(cd - CD_TARGET_NM)
        if abs(cd_err - cd_tol) <= 0.10:
            tags.append("cd_boundary")
        if cd_err > cd_tol:
            tags.append("cd_drift_boundary")
    if np.isfinite(ler) and abs(ler - ler_cap) <= 0.15:
        tags.append("ler_boundary")
    if np.isfinite(mar) and 0.18 <= mar <= 0.22:
        tags.append("margin_boundary")
    if np.isfinite(sc) and -0.5 <= sc <= 0.5:
        tags.append("strict_boundary")
    if (np.isfinite(diffusion_length) and np.isfinite(ref_diff)
            and abs(diffusion_length - ref_diff) / max(ref_diff, 1e-12) >= 0.10):
        tags.append("diffusion_boundary")
    if (np.isfinite(reaction_budget) and np.isfinite(ref_react)
            and abs(reaction_budget - ref_react) / max(ref_react, 1e-12) >= 0.15):
        tags.append("reaction_budget_boundary")
    if (np.isfinite(quencher_budget) and np.isfinite(ref_quench)
            and abs(quencher_budget - ref_quench) / max(ref_quench, 1e-12) >= 0.20):
        tags.append("quencher_boundary")
    # All 06PB rows are explicitly part of the strict_pass-residual
    # AL set, so always carry that intent tag.
    tags.append("strict_pass_residual_target")
    return ";".join(tags)


def _stamp_06pb(row: dict, *, strict_yaml: dict,
                  ref_g4867: dict) -> dict:
    out = dict(row)
    out.update(derive_features(row))
    out["mode"]          = "mode_a"
    out["recipe_family"] = "G4867_family"
    if not out.get("source"):
        out["source"]    = "stage06PB_G4867_targeted_AL"
    out["strict_score_per_row"] = per_row_strict_score(out, strict_yaml)
    out["boundary_tags"] = _tag_06pb_boundary(out, strict_yaml=strict_yaml,
                                                    ref_g4867=ref_g4867)
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--stage06r_csv", type=str,
                   default=str(V3_DIR / "outputs" / "labels"
                               / "stage06R_training_dataset.csv"))
    p.add_argument("--stage06pb_targeted_csv", type=str,
                   default=str(V3_DIR / "outputs" / "labels"
                               / "stage06PB_targeted_fd_rows.csv"))
    p.add_argument("--strict_cfg_yaml", type=str,
                   default=str(V3_DIR / "outputs" / "yield_optimization"
                               / "stage06G_strict_score_config.yaml"))
    p.add_argument("--mode_a_recipes_yaml", type=str,
                   default=str(V3_DIR / "outputs" / "yield_optimization"
                               / "stage06I_mode_a_final_recipes.yaml"))
    p.add_argument("--out_csv", type=str,
                   default=str(V3_DIR / "outputs" / "labels"
                               / "stage06PB_training_dataset.csv"))
    args = p.parse_args()

    strict_yaml = yaml.safe_load(Path(args.strict_cfg_yaml).read_text())

    # G_4867 reference for derived-budget boundary thresholds.
    g4867_yaml = next(rep for rep in yaml.safe_load(
        Path(args.mode_a_recipes_yaml).read_text())["representatives"]
                        if rep["recipe_id"] == "G_4867")
    g4867_params = {k: float(v) for k, v in g4867_yaml["parameters"].items()}
    g4867_derived = derive_features(g4867_params)

    seen: set = set()
    out_rows: list[dict] = []

    rows_06r = read_labels_csv(args.stage06r_csv)
    n_dup_06r = 0
    for r in rows_06r:
        key = _round_key(r)
        if key in seen:
            n_dup_06r += 1
            continue
        seen.add(key)
        out_rows.append(r)
    n_from_06r = len(out_rows)

    rows_06pb = read_labels_csv(args.stage06pb_targeted_csv)
    n_dup_06pb = 0
    n_added_06pb = 0
    for r in rows_06pb:
        key = _round_key(r)
        if key in seen:
            n_dup_06pb += 1
            continue
        seen.add(key)
        out_rows.append(_stamp_06pb(r, strict_yaml=strict_yaml,
                                       ref_g4867=g4867_derived))
        n_added_06pb += 1

    print(f"  06R base rows kept:                {n_from_06r}  "
          f"(within-source dedup drop {n_dup_06r})")
    print(f"  06PB targeted FD rows kept:        {n_added_06pb}  "
          f"(drop {n_dup_06pb})")
    print(f"  total 06PB training rows:          {len(out_rows)}")

    by_mode = Counter(r.get("mode", "?") for r in out_rows)
    by_label = Counter(str(r.get("label", "?")) for r in out_rows)
    by_family = Counter(r.get("recipe_family", "?") for r in out_rows)
    by_source = Counter(r.get("source", "?") for r in out_rows)
    print(f"  by mode:                            {dict(by_mode)}")
    print(f"  by label:")
    for k in sorted(by_label):
        print(f"    {k:<22} {by_label[k]:>6}")
    print(f"  by recipe_family:")
    for k in sorted(by_family):
        print(f"    {k:<22} {by_family[k]:>6}")
    print(f"  by source (top 10):")
    for k, v in by_source.most_common(10):
        print(f"    {k:<54} {v:>6}")

    # Boundary-tag tally on 06PB rows specifically.
    new_tags: Counter[str] = Counter()
    for r in out_rows:
        if str(r.get("source", "")).startswith("stage06PB_"):
            for t in str(r.get("boundary_tags", "")).split(";"):
                if t:
                    new_tags[t] += 1
    print(f"  boundary_tags on 06PB rows (rows can carry several):")
    for k in sorted(new_tags):
        print(f"    {k:<28} {new_tags[k]:>6}")

    # Compose canonical column order: 06R columns first, then 06PB
    # extras (sub_phase / source_recipe_id / scenario / time_offset_s /
    # variation_idx) so the CSV stays readable.
    if rows_06r:
        in_cols = list(rows_06r[0].keys())
    else:
        in_cols = []
    cols = list(in_cols)
    for k in DERIVED_FEATURE_KEYS:
        if k not in cols:
            cols.append(k)
    extra: list[str] = []
    for r in out_rows:
        for k in r.keys():
            if k not in cols and k not in extra:
                extra.append(k)
    cols += extra

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in out_rows:
            w.writerow(r)
    print(f"  -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
