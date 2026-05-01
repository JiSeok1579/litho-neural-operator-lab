"""Stage 06L -- build the strict-score-head training dataset.

Concatenate:
    1. Stage 06H training dataset (9,385 rows: 04C + 06B + 06E
       disagreement + 06H FD top-100 / top-10 MC / 6 reps MC).
    2. Stage 06J Mode B FD rows: 33 nominal + 500 MC = 533 rows.

Add columns:
    - source     -- the prior source string for 06H rows; for 06J
                     rows: stage06J_mode_b_nominal | stage06J_mode_b_mc
    - mode       -- mode_a if (pitch=24 AND line_cd_ratio=0.52),
                     mode_b otherwise. (This is the geometric mode
                     tag rather than the recipe-pool tag, so 04C
                     rows split correctly between the two modes.)
    - strict_score_per_row -- per-row strict_score using the active
                     Stage 06G strict config (one-hot probs, no std
                     penalties). This is the supervised target for
                     the new strict_score regressor head.

Closed Stage 04C / 04D / 06C dataset CSVs are NOT mutated. Output goes
to outputs/labels/stage06L_training_dataset.csv.

Dedup: keep the first occurrence of each rounded feature tuple in
source order (06H first, then 06J nominal, then 06J MC).

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


def per_row_strict_score(row: dict, strict_yaml: dict) -> float:
    """One-hot strict_score on a single FD row (no std penalties)."""
    label = str(row.get("label", ""))
    cw = strict_yaml["class_weights"]
    sp = strict_yaml["strict_penalties"]
    th = strict_yaml["thresholds"]
    cd_tol  = float(th["cd_tol_nm"])
    ler_cap = float(th["ler_cap_nm"])
    cd = _safe_float(row.get("CD_final_nm"))
    ler = _safe_float(row.get("LER_CD_locked_nm"))
    margin = _safe_float(row.get("P_line_margin", 0.0))
    if np.isfinite(cd):
        cd_pen = max(0.0, abs(cd - CD_TARGET_NM) - cd_tol) / max(cd_tol, 1e-12)
    else:
        cd_pen = 0.0
    if np.isfinite(ler):
        ler_pen = max(0.0, ler - ler_cap) / max(float(sp["ler_std_norm_nm"]), 1e-12)
    else:
        ler_pen = 0.0
    return float(
        cw.get(label, 0.0)
        - float(sp["cd_strict_weight"])  * cd_pen
        - float(sp["ler_strict_weight"]) * ler_pen
        + float(sp["margin_bonus"])      * (margin if np.isfinite(margin) else 0.0)
    )


def _tag_mode(row: dict) -> str:
    p = _safe_float(row.get("pitch_nm"))
    r = _safe_float(row.get("line_cd_ratio"))
    if abs(p - 24.0) < 1e-6 and abs(r - 0.52) < 1e-6:
        return "mode_a"
    return "mode_b"


def _stamp(row: dict, *, source: str, strict_yaml: dict) -> dict:
    out = dict(row)
    out["source"] = source
    out["mode"]   = _tag_mode(out)
    out["strict_score_per_row"] = per_row_strict_score(out, strict_yaml)
    return out


def _add_with_source(rows: list[dict], source: str, *,
                       seen: set, n_dup: list[int],
                       strict_yaml: dict) -> list[dict]:
    out = []
    for r in rows:
        key = _round_key(r)
        if key in seen:
            n_dup[0] += 1
            continue
        seen.add(key)
        out.append(_stamp(r, source=source, strict_yaml=strict_yaml))
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--stage06h_csv", type=str,
                   default=str(V3_DIR / "outputs" / "labels"
                               / "stage06H_training_dataset.csv"))
    p.add_argument("--stage06j_nom_csv", type=str,
                   default=str(V3_DIR / "outputs" / "labels"
                               / "stage06J_mode_b_fd_sanity.csv"))
    p.add_argument("--stage06j_mc_csv", type=str,
                   default=str(V3_DIR / "outputs" / "labels"
                               / "stage06J_mode_b_fd_mc_optional.csv"))
    p.add_argument("--strict_cfg_yaml", type=str,
                   default=str(V3_DIR / "outputs" / "yield_optimization"
                               / "stage06G_strict_score_config.yaml"))
    p.add_argument("--out_csv", type=str,
                   default=str(V3_DIR / "outputs" / "labels"
                               / "stage06L_training_dataset.csv"))
    args = p.parse_args()

    strict_yaml = yaml.safe_load(Path(args.strict_cfg_yaml).read_text())

    seen: set = set()
    out_rows: list[dict] = []

    rows_06h = read_labels_csv(args.stage06h_csv)
    n_dup_h = 0
    for r in rows_06h:
        key = _round_key(r)
        if key in seen:
            n_dup_h += 1
            continue
        seen.add(key)
        # Preserve 06H's per-row source tag in a sub-source.
        prev_source = str(r.get("source", "")) or "stage06H_unknown"
        out_rows.append(_stamp(r, source=f"stage06H/{prev_source}",
                                  strict_yaml=strict_yaml))

    n_dup = [0]
    rows_06j_nom = read_labels_csv(args.stage06j_nom_csv)
    add_jnom = _add_with_source(rows_06j_nom, "stage06J_mode_b_nominal",
                                  seen=seen, n_dup=n_dup, strict_yaml=strict_yaml)
    n_dup_jnom = n_dup[0]; n_dup = [0]; out_rows.extend(add_jnom)

    rows_06j_mc = read_labels_csv(args.stage06j_mc_csv)
    add_jmc = _add_with_source(rows_06j_mc, "stage06J_mode_b_mc",
                                 seen=seen, n_dup=n_dup, strict_yaml=strict_yaml)
    n_dup_jmc = n_dup[0]; out_rows.extend(add_jmc)

    print(f"  06H rows kept:                {sum(1 for r in out_rows if str(r['source']).startswith('stage06H'))}")
    print(f"  06J nominal kept:             {len(add_jnom)} (drop {n_dup_jnom} as duplicate)")
    print(f"  06J MC kept:                  {len(add_jmc)} (drop {n_dup_jmc} as duplicate)")
    print(f"  06H within-source dedup drop: {n_dup_h}")
    print(f"  total 06L rows:               {len(out_rows)}")

    by_mode = Counter(r["mode"] for r in out_rows)
    by_source = Counter(r["source"] for r in out_rows)
    by_label  = Counter(str(r.get("label", "?")) for r in out_rows)
    print(f"  by mode:")
    for k, v in by_mode.items():
        print(f"    {k:<10} {v:>6}")
    print(f"  by source (top 10):")
    for k, v in by_source.most_common(10):
        print(f"    {k:<48} {v:>6}")
    print(f"  by label:")
    for k in sorted(by_label):
        print(f"    {k:<22} {by_label[k]:>6}")

    # Build column union.
    canonical = ["_id", "label", "source", "mode", "strict_score_per_row"] \
                + list(FEATURE_KEYS) + [
                    "CD_locked_nm", "LER_CD_locked_nm", "area_frac",
                    "P_line_margin", "CD_final_nm",
                ]
    extra: list[str] = []
    for r in out_rows:
        for k in r.keys():
            if k not in canonical and k not in extra:
                extra.append(k)
    cols = canonical + extra

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
