"""Stage 06H -- build the surrogate-refresh training dataset.

Concatenate:
    1. Stage 06C training rows  (closed: 04C-train + 06B AL).
    2. Stage 06E disagreement rows (17 high-06C / low-06A FD rows).
    3. Stage 06H FD rows from Parts 1-3:
         - top-100 nominal FD             (100 rows)
         - top-10 x 100 MC FD             (1,000 rows)
         - 6 representative x 300 MC FD   (1,800 rows)

Each row carries a `source` column from the spec:
    stage06C
    stage06E_disagreement
    stage06H_top100_nominal
    stage06H_top10_mc
    stage06H_representative_mc

Closed Stage 04C / 04D / 06C dataset CSVs are NOT mutated. Output goes
to outputs/labels/stage06H_training_dataset.csv.

Dedup: an exact (rounded) feature tuple appearing across sources keeps
its first occurrence in the order above. This means a 06H row with the
same parameters as a 06C row is dropped (06C has the prior label and
metrics already).
"""
from __future__ import annotations

import argparse
import csv
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


def _round_key(row: dict, ndigits: int = 6) -> tuple:
    out = []
    for k in FEATURE_KEYS:
        try:
            out.append(round(float(row[k]), ndigits))
        except (TypeError, ValueError, KeyError):
            out.append(None)
    return tuple(out)


def _stamp(row: dict, source: str) -> dict:
    out = dict(row)
    out["source"] = source
    return out


def _load_with_source(path: str, source: str,
                      seen: set, n_dup: list[int]) -> list[dict]:
    rows = read_labels_csv(path)
    out: list[dict] = []
    for r in rows:
        key = _round_key(r)
        if key in seen:
            n_dup[0] += 1
            continue
        seen.add(key)
        out.append(_stamp(r, source=source))
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--stage06c_csv", type=str,
                   default=str(V3_DIR / "outputs" / "labels"
                               / "stage06C_training_dataset.csv"))
    p.add_argument("--stage06e_disagree_csv", type=str,
                   default=str(V3_DIR / "outputs" / "labels"
                               / "stage06E_fd_disagreement.csv"))
    p.add_argument("--stage06h_top100_csv", type=str,
                   default=str(V3_DIR / "outputs" / "labels"
                               / "stage06H_fd_top100_nominal.csv"))
    p.add_argument("--stage06h_top10_mc_csv", type=str,
                   default=str(V3_DIR / "outputs" / "labels"
                               / "stage06H_fd_top10_mc.csv"))
    p.add_argument("--stage06h_rep_mc_csv", type=str,
                   default=str(V3_DIR / "outputs" / "labels"
                               / "stage06H_fd_representative_mc.csv"))
    p.add_argument("--out_csv", type=str,
                   default=str(V3_DIR / "outputs" / "labels"
                               / "stage06H_training_dataset.csv"))
    args = p.parse_args()

    seen: set = set()
    out_rows: list[dict] = []

    # 06C rows -- carry a per-row sub-source if available.
    rows_06c = read_labels_csv(args.stage06c_csv)
    n_06c_dup = 0
    for r in rows_06c:
        key = _round_key(r)
        if key in seen:
            n_06c_dup += 1
            continue
        seen.add(key)
        prev_source = str(r.get("source", "")) or "stage06C_unknown"
        out_rows.append(_stamp(r, source=f"stage06C/{prev_source}"))

    n_dup = [0]
    add_e = _load_with_source(args.stage06e_disagree_csv,
                                "stage06E_disagreement", seen, n_dup)
    n_06e_dup = n_dup[0]; n_dup = [0]
    out_rows.extend(add_e)

    add_h1 = _load_with_source(args.stage06h_top100_csv,
                                 "stage06H_top100_nominal", seen, n_dup)
    n_h1_dup = n_dup[0]; n_dup = [0]
    out_rows.extend(add_h1)

    add_h2 = _load_with_source(args.stage06h_top10_mc_csv,
                                 "stage06H_top10_mc", seen, n_dup)
    n_h2_dup = n_dup[0]; n_dup = [0]
    out_rows.extend(add_h2)

    add_h3 = _load_with_source(args.stage06h_rep_mc_csv,
                                 "stage06H_representative_mc", seen, n_dup)
    n_h3_dup = n_dup[0]
    out_rows.extend(add_h3)

    print(f"  source dedup summary (rows dropped because a prior source "
          f"already carries that exact feature tuple):")
    print(f"    within 06C input:                 {n_06c_dup}")
    print(f"    06E disagreement vs prior:        {n_06e_dup}")
    print(f"    06H top-100 nominal vs prior:     {n_h1_dup}")
    print(f"    06H top-10 MC vs prior:           {n_h2_dup}")
    print(f"    06H representative MC vs prior:   {n_h3_dup}")

    # Build column union, prepending canonical training columns.
    canonical = ["_id", "label", "source"] + list(FEATURE_KEYS) + [
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

    by_source: Counter = Counter()
    by_label:  Counter = Counter()
    for r in out_rows:
        by_source[str(r.get("source", "?"))] += 1
        by_label[str(r.get("label", "?"))] += 1
    print(f"\n  06H dataset:   {len(out_rows)} rows  -> {out_path}")
    print(f"  by source:")
    for k, v in by_source.most_common():
        print(f"    {k:<42} {v:>6}")
    print(f"  by label:")
    for k in sorted(by_label):
        print(f"    {k:<22} {by_label[k]:>6}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
