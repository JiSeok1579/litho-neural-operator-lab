"""Stage 06C — build the refresh training dataset.

Concatenate the closed Stage 04C training rows and the 1,100 Stage 06B
AL-additions rows into a new CSV. Two new columns are stamped:

    source     stage04C  | stage06B_yield_opt
    top_tier   0          | 1   (only the 06B rows that came from
                                 Stage 06A's top-100 surrogate picks)

Dedup is applied on a rounded FEATURE_KEYS tuple so an exact-parameter
collision does not get double-counted. We dedup *across* sources only
(no within-04C dedup); Stage 06B nominal rows have parameter centres
identical to Stage 06A's top-100 candidates, but those candidates were
*not* in the closed 04C training pool, so dedup will rarely fire.

The closed Stage 04C dataset CSV is **not** mutated. Output goes to
outputs/labels/stage06C_training_dataset.csv.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reaction_diffusion_peb_v3_screening.src.metrics_io import (
    FEATURE_KEYS,
    read_labels_csv,
)


V3_DIR = ROOT / "reaction_diffusion_peb_v3_screening"


def _round_key(row: dict, ndigits: int = 6) -> tuple:
    out = []
    for k in FEATURE_KEYS:
        try:
            out.append(round(float(row[k]), ndigits))
        except (TypeError, ValueError, KeyError):
            out.append(None)
    return tuple(out)


def _stamp(row: dict, source: str, top_tier: int) -> dict:
    out = dict(row)
    out["source"] = source
    out["top_tier"] = str(int(top_tier))
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--stage04c_csv", type=str,
                   default=str(V3_DIR / "outputs" / "labels"
                               / "stage04C_training_dataset.csv"))
    p.add_argument("--al_additions_csv", type=str,
                   default=str(V3_DIR / "outputs" / "labels"
                               / "06_yield_optimization_al_additions.csv"))
    p.add_argument("--out_csv", type=str,
                   default=str(V3_DIR / "outputs" / "labels"
                               / "stage06C_training_dataset.csv"))
    args = p.parse_args()

    rows_04c = read_labels_csv(args.stage04c_csv)
    rows_06b = read_labels_csv(args.al_additions_csv)

    print(f"  04C rows:      {len(rows_04c)}")
    print(f"  06B AL rows:   {len(rows_06b)}")

    seen: set = set()
    out_rows: list[dict] = []

    for r in rows_04c:
        key = _round_key(r)
        if key in seen:
            continue
        seen.add(key)
        out_rows.append(_stamp(r, source="stage04C", top_tier=0))

    n_dup = 0
    for r in rows_06b:
        key = _round_key(r)
        if key in seen:
            n_dup += 1
            continue
        seen.add(key)
        out_rows.append(_stamp(r, source="stage06B_yield_opt", top_tier=1))

    print(f"  06B dedup overlap with 04C: {n_dup} row(s) dropped")
    print(f"  06C dataset:   {len(out_rows)} rows")

    # Build column union, prepending the canonical training columns so a
    # downstream reader always finds them at the front.
    canonical = ["_id", "label", "source", "top_tier"] + list(FEATURE_KEYS) + [
        "CD_locked_nm", "LER_CD_locked_nm", "area_frac", "P_line_margin",
        "CD_final_nm",
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

    by_source: dict[str, int] = {}
    by_top_tier: dict[str, int] = {"0": 0, "1": 0}
    by_label: dict[str, int] = {}
    for r in out_rows:
        by_source[r["source"]] = by_source.get(r["source"], 0) + 1
        by_top_tier[r["top_tier"]] = by_top_tier.get(r["top_tier"], 0) + 1
        by_label[r.get("label", "?")] = by_label.get(r.get("label", "?"), 0) + 1
    print(f"  by source:    {by_source}")
    print(f"  by top_tier:  {by_top_tier}")
    print(f"  by label:")
    for k in sorted(by_label):
        print(f"    {k:<22} {by_label[k]:>5}")
    print(f"  → {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
