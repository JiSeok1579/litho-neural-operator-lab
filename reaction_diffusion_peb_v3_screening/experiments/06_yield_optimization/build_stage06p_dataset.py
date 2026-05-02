"""Stage 06P -- build the AL-refresh training dataset.

Concatenate, in source order:
    1. Stage 06L training dataset
       (06H union 06J Mode B nominal / MC, already tagged with
       source / mode / strict_score_per_row).
    2. Stage 06J-B FD additions (Mode B FD verification at scale):
         - top-100 nominal              (~100 rows)
         - top-10 MC                    (~1,000 rows)
         - representative MC (6 kinds x 50 rows x ~6 reps) (~1,800 rows)
       totalling ~2,900 FD rows.
    3. Stage 06M-B J_1453 time-window FD additions:
         - deterministic time offsets   (~1,100 rows)
         - Gaussian time-smearing       (~300 rows)
       totalling ~1,400 FD rows.
    4. Stage 06L tagged AL candidates if FD labels exist anywhere in
       the FD CSVs above (most 06L AL targets are surrogate-only; this
       step is a no-op when no FD evidence is found).

Add columns (in addition to the 06L source / mode / strict_score_per_row
columns that survive untouched):
    - recipe_family : G4867_family | J1453_family | mode_b_other |
                       false_promise | general
    - boundary_tags : ;-separated subset of
                       {time_boundary, strict_boundary, cd_boundary,
                        ler_boundary, margin_boundary,
                        false_pass_candidate, disagreement_candidate}

Closed Stage 04C / 04D / 06C / 06L dataset CSVs are NOT mutated. The
06L training-dataset CSV is read but only used as input.

Dedup: keep the first occurrence of each rounded feature tuple in
source order (06L first, then 06J-B nominal, then 06J-B top10 MC, then
06J-B representative MC, then 06M-B det_offset, then 06M-B Gaussian).

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


V3_DIR = ROOT / "reaction_diffusion_peb_v3_screening"
CD_TARGET_NM = 15.0

# Recipes that 06L's false-PASS demotion analysis flagged as
# nominally-PASS-but-FD-fails. Rows from these recipe_ids are tagged
# `false_pass_candidate` so 06P sees them weighted as failure-zone.
FALSE_PASS_RECIPES_06L: frozenset[str] = frozenset(
    {"J_175", "J_3780", "J_3447", "J_4516"}
)


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


def _tag_mode(row: dict) -> str:
    p = _safe_float(row.get("pitch_nm"))
    r = _safe_float(row.get("line_cd_ratio"))
    if abs(p - 24.0) < 1e-6 and abs(r - 0.52) < 1e-6:
        return "mode_a"
    return "mode_b"


def _tag_recipe_family(row: dict) -> str:
    rid = str(row.get("source_recipe_id", "")) or ""
    if rid in FALSE_PASS_RECIPES_06L:
        return "false_promise"
    if rid == "G_4867":
        return "G4867_family"
    if rid == "J_1453":
        return "J1453_family"
    src = str(row.get("source", ""))
    if rid.startswith("J_") or "mode_b" in src or "stage06J" in src:
        return "mode_b_other"
    return "general"


def _tag_boundary(row: dict, *, strict_yaml: dict) -> str:
    """Return a `;`-joined subset of the 7 boundary tags."""
    th = strict_yaml["thresholds"]
    cd_tol  = float(th["cd_tol_nm"])
    ler_cap = float(th["ler_cap_nm"])

    src = str(row.get("source", ""))
    cd  = _safe_float(row.get("CD_final_nm"))
    ler = _safe_float(row.get("LER_CD_locked_nm"))
    mar = _safe_float(row.get("P_line_margin"))
    sc  = _safe_float(row.get("strict_score_per_row"))
    rid = str(row.get("source_recipe_id", "")) or ""

    tags: list[str] = []
    if "time_deep_mc" in src or "gaussian_time_smearing" in src:
        tags.append("time_boundary")
    if np.isfinite(cd):
        cd_err = abs(cd - CD_TARGET_NM)
        # within +/- 0.10 nm of the strict CD tolerance shoulder
        if abs(cd_err - cd_tol) <= 0.10:
            tags.append("cd_boundary")
    if np.isfinite(ler) and abs(ler - ler_cap) <= 0.15:
        tags.append("ler_boundary")
    if np.isfinite(mar) and 0.18 <= mar <= 0.22:
        tags.append("margin_boundary")
    if np.isfinite(sc) and -0.5 <= sc <= 0.5:
        tags.append("strict_boundary")
    if rid in FALSE_PASS_RECIPES_06L:
        tags.append("false_pass_candidate")
    if "disagreement" in src:
        tags.append("disagreement_candidate")
    return ";".join(tags)


def _stamp_new(row: dict, *, source: str, strict_yaml: dict) -> dict:
    """Apply 06P stamp (source / mode / strict_score / family / tags)
    to a freshly-read FD row."""
    out = dict(row)
    out["source"] = source
    out["mode"]   = _tag_mode(out)
    out["strict_score_per_row"] = per_row_strict_score(out, strict_yaml)
    out["recipe_family"] = _tag_recipe_family(out)
    out["boundary_tags"] = _tag_boundary(out, strict_yaml=strict_yaml)
    return out


def _stamp_existing(row: dict, *, strict_yaml: dict) -> dict:
    """Apply 06P-only fields (recipe_family / boundary_tags) to an
    already-stamped 06L row, leaving its existing source / mode /
    strict_score_per_row untouched."""
    out = dict(row)
    out["recipe_family"] = _tag_recipe_family(out)
    out["boundary_tags"] = _tag_boundary(out, strict_yaml=strict_yaml)
    return out


def _split_06m_b_by_scenario(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    det, gau = [], []
    for r in rows:
        s = str(r.get("scenario", ""))
        if s == "det_offset":
            det.append(r)
        elif s == "gaussian_time":
            gau.append(r)
    return det, gau


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
        out.append(_stamp_new(r, source=source, strict_yaml=strict_yaml))
    return out


def _maybe_add_06l_al_candidates(al_csv: Path, fd_pool: list[dict],
                                       *, seen: set, strict_yaml: dict
                                       ) -> list[dict]:
    """For each 06L AL-target recipe, add any matching FD row from the
    accumulated FD pool. Most 06L AL targets are surrogate-only so this
    routine is typically a no-op; we still run it because the spec
    asks for it."""
    if not al_csv.exists():
        return []
    al_rows = read_labels_csv(al_csv)
    al_recipes = {str(r.get("recipe_id", "")) for r in al_rows
                    if str(r.get("recipe_id", ""))}
    if not al_recipes:
        return []
    out = []
    for r in fd_pool:
        rid = str(r.get("source_recipe_id", "")) or ""
        if rid not in al_recipes:
            continue
        key = _round_key(r)
        if key in seen:
            continue
        seen.add(key)
        out.append(_stamp_new(r, source="stage06L_tagged_al",
                                  strict_yaml=strict_yaml))
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--stage06l_csv", type=str,
                   default=str(V3_DIR / "outputs" / "labels"
                               / "stage06L_training_dataset.csv"))
    p.add_argument("--stage06j_b_top100_csv", type=str,
                   default=str(V3_DIR / "outputs" / "labels"
                               / "stage06J_B_fd_top100_nominal.csv"))
    p.add_argument("--stage06j_b_top10mc_csv", type=str,
                   default=str(V3_DIR / "outputs" / "labels"
                               / "stage06J_B_fd_top10_mc.csv"))
    p.add_argument("--stage06j_b_repmc_csv", type=str,
                   default=str(V3_DIR / "outputs" / "labels"
                               / "stage06J_B_fd_representative_mc.csv"))
    p.add_argument("--stage06m_b_csv", type=str,
                   default=str(V3_DIR / "outputs" / "yield_optimization"
                               / "stage06M_B_J1453_time_deep_mc.csv"))
    p.add_argument("--stage06l_al_csv", type=str,
                   default=str(V3_DIR / "outputs" / "yield_optimization"
                               / "stage06L_al_targets.csv"))
    p.add_argument("--strict_cfg_yaml", type=str,
                   default=str(V3_DIR / "outputs" / "yield_optimization"
                               / "stage06G_strict_score_config.yaml"))
    p.add_argument("--out_csv", type=str,
                   default=str(V3_DIR / "outputs" / "labels"
                               / "stage06P_training_dataset.csv"))
    args = p.parse_args()

    strict_yaml = yaml.safe_load(Path(args.strict_cfg_yaml).read_text())

    seen: set = set()
    out_rows: list[dict] = []

    # ---- 1. 06L base ---------------------------------------------------
    rows_06l = read_labels_csv(args.stage06l_csv)
    n_dup_06l = 0
    for r in rows_06l:
        key = _round_key(r)
        if key in seen:
            n_dup_06l += 1
            continue
        seen.add(key)
        out_rows.append(_stamp_existing(r, strict_yaml=strict_yaml))
    n_06l = len(out_rows)

    # ---- 2. 06J-B FD additions ----------------------------------------
    n_dup = [0]
    rows_jb_nom = read_labels_csv(args.stage06j_b_top100_csv)
    add_jb_nom = _add_with_source(rows_jb_nom, "stage06J_B_top100_nominal",
                                       seen=seen, n_dup=n_dup,
                                       strict_yaml=strict_yaml)
    n_dup_jb_nom = n_dup[0]; n_dup = [0]
    out_rows.extend(add_jb_nom)

    rows_jb_top10mc = read_labels_csv(args.stage06j_b_top10mc_csv)
    add_jb_top10mc = _add_with_source(rows_jb_top10mc,
                                            "stage06J_B_top10_mc",
                                            seen=seen, n_dup=n_dup,
                                            strict_yaml=strict_yaml)
    n_dup_jb_top10mc = n_dup[0]; n_dup = [0]
    out_rows.extend(add_jb_top10mc)

    rows_jb_repmc = read_labels_csv(args.stage06j_b_repmc_csv)
    # Sub-source by rep_kind so the manifest preserves which
    # representative each row came from.
    jb_repmc_sub: list[dict] = []
    for r in rows_jb_repmc:
        rk = str(r.get("rep_kind", "")) or "unknown"
        key = _round_key(r)
        if key in seen:
            n_dup[0] += 1
            continue
        seen.add(key)
        jb_repmc_sub.append(_stamp_new(
            r, source=f"stage06J_B_representative_mc/{rk}",
            strict_yaml=strict_yaml))
    n_dup_jb_repmc = n_dup[0]; n_dup = [0]
    out_rows.extend(jb_repmc_sub)

    # ---- 3. 06M-B J_1453 time-window FD additions ---------------------
    rows_06m_b_all = read_labels_csv(args.stage06m_b_csv)
    rows_06m_b_det, rows_06m_b_gau = _split_06m_b_by_scenario(rows_06m_b_all)
    add_06m_b_det = _add_with_source(
        rows_06m_b_det, "stage06M_B_J1453_time_deep_mc",
        seen=seen, n_dup=n_dup, strict_yaml=strict_yaml,
    )
    n_dup_06m_b_det = n_dup[0]; n_dup = [0]
    out_rows.extend(add_06m_b_det)

    add_06m_b_gau = _add_with_source(
        rows_06m_b_gau, "stage06M_B_gaussian_time_smearing",
        seen=seen, n_dup=n_dup, strict_yaml=strict_yaml,
    )
    n_dup_06m_b_gau = n_dup[0]
    out_rows.extend(add_06m_b_gau)

    # ---- 4. 06L tagged AL candidates with FD --------------------------
    fd_pool = (rows_jb_nom + rows_jb_top10mc + rows_jb_repmc
                + rows_06m_b_det + rows_06m_b_gau)
    add_06l_al = _maybe_add_06l_al_candidates(
        Path(args.stage06l_al_csv), fd_pool,
        seen=seen, strict_yaml=strict_yaml,
    )
    out_rows.extend(add_06l_al)

    # ---- 5. Console summary -------------------------------------------
    print(f"  06L base rows kept:                    {n_06l} "
            f"(within-source dedup drop {n_dup_06l})")
    print(f"  06J-B top100 nominal kept:             {len(add_jb_nom):>5} "
            f"(drop {n_dup_jb_nom})")
    print(f"  06J-B top10 MC kept:                   {len(add_jb_top10mc):>5} "
            f"(drop {n_dup_jb_top10mc})")
    print(f"  06J-B representative MC kept:          {len(jb_repmc_sub):>5} "
            f"(drop {n_dup_jb_repmc})")
    print(f"  06M-B det_offset kept:                 {len(add_06m_b_det):>5} "
            f"(drop {n_dup_06m_b_det})")
    print(f"  06M-B Gaussian time-smearing kept:     {len(add_06m_b_gau):>5} "
            f"(drop {n_dup_06m_b_gau})")
    print(f"  06L tagged AL with FD kept:            {len(add_06l_al):>5}")
    print(f"  total 06P rows:                        {len(out_rows):>5}")

    by_mode = Counter(r.get("mode", "?") for r in out_rows)
    by_label = Counter(str(r.get("label", "?")) for r in out_rows)
    by_family = Counter(r.get("recipe_family", "?") for r in out_rows)
    by_source = Counter(r.get("source", "?") for r in out_rows)
    print(f"  by mode:")
    for k, v in by_mode.items():
        print(f"    {k:<10} {v:>6}")
    print(f"  by label:")
    for k in sorted(by_label):
        print(f"    {k:<22} {by_label[k]:>6}")
    print(f"  by recipe_family:")
    for k in sorted(by_family):
        print(f"    {k:<22} {by_family[k]:>6}")
    print(f"  by source (top 10):")
    for k, v in by_source.most_common(10):
        print(f"    {k:<54} {v:>6}")

    # Boundary-tag tallies (multi-tag rows are counted under each).
    tag_counter: Counter[str] = Counter()
    for r in out_rows:
        for t in str(r.get("boundary_tags", "")).split(";"):
            if t:
                tag_counter[t] += 1
    print(f"  boundary_tags (rows can carry several):")
    for k in sorted(tag_counter):
        print(f"    {k:<24} {tag_counter[k]:>6}")

    # ---- 6. Write CSV --------------------------------------------------
    canonical = ["_id", "label", "source", "mode",
                  "recipe_family", "boundary_tags",
                  "strict_score_per_row"] + list(FEATURE_KEYS) + [
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
