"""Stage 06J-B -- Mode B FD verification at scale.

Mirrors Stage 06H but applied to the Stage 06L-rescored Mode B
candidate pool (the 06J top-100 re-ranked by the 06L direct
strict_score head).

Three FD parts:
    Part 1 -- top-100 Mode B nominal FD                       (100 runs)
    Part 2 -- top-10 Mode B FD MC, 100 variations each        (1,000 runs)
    Part 3 -- Mode B representative MC, 300 variations each   (up to 1,800 runs)

Representative roles for Part 3 (deduplicated):
    1. J_1453  (06L Mode B strict-best, hard-coded for comparison)
    2. strict-best  (06L Mode B top-1 by strict_score; J_1453 in practice)
    3. CD-best       (06L Mode B with lowest |mean_cd_fixed_06l - 15|)
    4. LER-best      (06L Mode B with lowest mean_ler_locked_06l)
    5. balanced-best (06J top-100 balanced_score_06j argmin)
    6. margin-best   (06L Mode B with highest mean_p_line_margin_06l)
If two roles collapse to the same recipe, the next best non-duplicate
candidate is added so that 6 distinct recipes get representative MC.

Policy
    v2_OP_frozen           = true
    published_data_loaded  = false
    No external calibration. Closed Stage 04C / 04D / 06C / 06H / 06L
    joblibs and label datasets are read-only. New FD rows go to
    Stage 06J-B-only CSVs.
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reaction_diffusion_peb_v3_screening.src.candidate_sampler import (
    CandidateSpace,
)
from reaction_diffusion_peb_v3_screening.src.fd_batch_runner import (
    run_one_candidate,
)
from reaction_diffusion_peb_v3_screening.src.labeler import LabelThresholds
from reaction_diffusion_peb_v3_screening.src.metrics_io import (
    FEATURE_KEYS, LABEL_CSV_COLUMNS, read_labels_csv,
)
from reaction_diffusion_peb_v3_screening.src.process_variation import (
    VariationSpec, sample_variations,
)


V3_DIR = ROOT / "reaction_diffusion_peb_v3_screening"
CD_TARGET_NM = 15.0


EXTRA_COLS = [
    "source_recipe_id",
    "rank_06l",
    "variation_idx",
    "phase",
    "rep_kind",
    "strict_score_06l_surrogate",
    "yield_score_06l_surrogate",
]
WRITE_COLS = LABEL_CSV_COLUMNS + EXTRA_COLS


def _safe_float(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def _row_to_candidate(row: dict, space: CandidateSpace) -> dict:
    out = {}
    for k in FEATURE_KEYS:
        out[k] = float(row[k])
    out["pitch_nm"]    = float(out["pitch_nm"])
    out["line_cd_nm"]  = out["pitch_nm"] * out["line_cd_ratio"]
    out["domain_x_nm"] = out["pitch_nm"] * 5.0
    out["dose_norm"]   = out["dose_mJ_cm2"] / float(space.fixed["reference_dose_mJ_cm2"])
    for fk, fv in space.fixed.items():
        out.setdefault(fk, fv)
    out["_id"] = row.get("recipe_id", "?")
    return out


def _open_csv(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    f = path.open("w", newline="")
    w = csv.DictWriter(f, fieldnames=WRITE_COLS, extrasaction="ignore")
    w.writeheader()
    return f, w


def _write_fd_row(writer, fd_row: dict, *, source_recipe_id: str,
                    rank_06l: int, variation_idx: int, phase: str,
                    rep_kind: str, sur: dict) -> None:
    out = dict(fd_row)
    out["source_recipe_id"] = source_recipe_id
    out["rank_06l"]          = int(rank_06l)
    out["variation_idx"]     = int(variation_idx)
    out["phase"]              = phase
    out["rep_kind"]           = str(rep_kind)
    out["strict_score_06l_surrogate"] = float(sur.get("strict_score_06l_direct_mean",
                                                            float("nan")))
    out["yield_score_06l_surrogate"]  = float(sur.get("yield_score_06l", float("nan")))
    writer.writerow(out)


# --------------------------------------------------------------------------
# Part 1 -- top-100 nominal FD.
# --------------------------------------------------------------------------
def run_part1_nominal(top_rows: list[dict], space: CandidateSpace,
                       thresholds: LabelThresholds, out_path: Path,
                       n_top: int, progress_every: int = 10) -> None:
    f, w = _open_csv(out_path)
    t0 = time.time()
    try:
        for rank, row in enumerate(top_rows[:n_top], start=1):
            cand = _row_to_candidate(row, space)
            cand["_id"] = f"{row.get('recipe_id', '?')}__nom_06jb"
            fd_row = run_one_candidate(cand, thresholds)
            _write_fd_row(w, fd_row,
                            source_recipe_id=str(row["recipe_id"]),
                            rank_06l=rank, variation_idx=0, phase="nominal",
                            rep_kind="", sur=row)
            f.flush()
            if rank % progress_every == 0:
                elapsed = time.time() - t0
                rate = rank / max(elapsed, 1e-9)
                print(f"  Part1: {rank}/{n_top} done  ({elapsed:.1f}s, {rate:.2f} runs/s)")
    finally:
        f.close()
    print(f"  Part1 -> {out_path} ({n_top} runs in {time.time()-t0:.1f}s)")


# --------------------------------------------------------------------------
# Part 2 -- top-10 FD MC.
# --------------------------------------------------------------------------
def run_part2_top10_mc(top_rows: list[dict], space: CandidateSpace,
                       thresholds: LabelThresholds, var_spec: VariationSpec,
                       n_top: int, n_var: int, out_path: Path,
                       seed: int, progress_every: int = 25) -> None:
    f, w = _open_csv(out_path)
    t0 = time.time()
    base_rng = np.random.default_rng(seed)
    total = 0
    try:
        for rank, row in enumerate(top_rows[:n_top], start=1):
            base = _row_to_candidate(row, space)
            sub_seed = int(base_rng.integers(0, 2**31 - 1))
            sub_rng = np.random.default_rng(sub_seed)
            variations = sample_variations(base, var_spec, n_var, space, rng=sub_rng)
            for j, v in enumerate(variations, start=1):
                v["_id"] = f"{row.get('recipe_id', '?')}__mc{j:03d}_06jb"
                for fk, fv in space.fixed.items():
                    v.setdefault(fk, fv)
                v["domain_x_nm"] = v["pitch_nm"] * 5.0
                v["dose_norm"]   = v["dose_mJ_cm2"] / float(space.fixed["reference_dose_mJ_cm2"])
                fd_row = run_one_candidate(v, thresholds)
                _write_fd_row(w, fd_row,
                                source_recipe_id=str(row["recipe_id"]),
                                rank_06l=rank, variation_idx=j, phase="mc_top10",
                                rep_kind="", sur=row)
                f.flush()
                total += 1
                if total % progress_every == 0:
                    elapsed = time.time() - t0
                    rate = total / max(elapsed, 1e-9)
                    eta = (n_top * n_var - total) / max(rate, 1e-9)
                    print(f"  Part2: {total}/{n_top*n_var} done  "
                          f"({elapsed:.1f}s, {rate:.2f} runs/s, ETA {eta:.0f}s)")
            print(f"  Part2: rank {rank}/{n_top} ({row.get('recipe_id','?')}) done")
    finally:
        f.close()
    print(f"  Part2 -> {out_path} ({total} runs in {time.time()-t0:.1f}s)")


# --------------------------------------------------------------------------
# Part 3 -- Mode B representatives FD MC (300 variations each).
# --------------------------------------------------------------------------
def run_part3_representative_mc(rep_rows: list[tuple[dict, str]],
                                  space: CandidateSpace,
                                  thresholds: LabelThresholds,
                                  var_spec: VariationSpec,
                                  n_var: int, out_path: Path,
                                  seed: int, progress_every: int = 50) -> None:
    f, w = _open_csv(out_path)
    t0 = time.time()
    base_rng = np.random.default_rng(seed)
    total = 0
    try:
        for k_idx, (row, kind) in enumerate(rep_rows, start=1):
            base = _row_to_candidate(row, space)
            sub_seed = int(base_rng.integers(0, 2**31 - 1))
            sub_rng = np.random.default_rng(sub_seed)
            variations = sample_variations(base, var_spec, n_var, space, rng=sub_rng)
            for j, v in enumerate(variations, start=1):
                v["_id"] = f"{row.get('recipe_id', '?')}__{kind}_mc{j:03d}_06jb"
                for fk, fv in space.fixed.items():
                    v.setdefault(fk, fv)
                v["domain_x_nm"] = v["pitch_nm"] * 5.0
                v["dose_norm"]   = v["dose_mJ_cm2"] / float(space.fixed["reference_dose_mJ_cm2"])
                fd_row = run_one_candidate(v, thresholds)
                _write_fd_row(w, fd_row,
                                source_recipe_id=str(row["recipe_id"]),
                                rank_06l=int(_safe_float(row.get("rank_06l", k_idx))),
                                variation_idx=j, phase="mc_representative",
                                rep_kind=kind, sur=row)
                f.flush()
                total += 1
                if total % progress_every == 0:
                    elapsed = time.time() - t0
                    rate = total / max(elapsed, 1e-9)
                    eta = (len(rep_rows) * n_var - total) / max(rate, 1e-9)
                    print(f"  Part3: {total}/{len(rep_rows)*n_var} done  "
                          f"({elapsed:.1f}s, {rate:.2f} runs/s, ETA {eta:.0f}s)")
            print(f"  Part3: rep {k_idx}/{len(rep_rows)} ({kind}: "
                  f"{row.get('recipe_id','?')}) done")
    finally:
        f.close()
    print(f"  Part3 -> {out_path} ({total} runs in {time.time()-t0:.1f}s)")


# --------------------------------------------------------------------------
# Representative selection (deduplicated).
# --------------------------------------------------------------------------
def select_representatives(rows: list[dict], top_06j_csv: Path) -> list[tuple[dict, str]]:
    """Pick 6 distinct Mode B representatives covering the spec roles."""
    by_id = {r["recipe_id"]: r for r in rows}
    sorted_strict = sorted(
        rows, key=lambda r: -_safe_float(r.get("strict_score_06l_direct_mean")))
    sorted_cd = sorted(
        rows, key=lambda r: abs(_safe_float(r.get("mean_cd_fixed_06l")) - CD_TARGET_NM))
    sorted_ler = sorted(
        rows, key=lambda r: _safe_float(r.get("mean_ler_locked_06l")))
    sorted_margin = sorted(
        rows, key=lambda r: -_safe_float(r.get("mean_p_line_margin_06l")))

    # Read 06J top-100 to recover the 06J balanced_score_06j ordering.
    if Path(top_06j_csv).exists():
        rows_06j_top = read_labels_csv(top_06j_csv)
        for r in rows_06j_top:
            for k in ("balanced_score_06j",):
                if k in r:
                    r[k] = _safe_float(r[k])
        sorted_balanced_pool = sorted(
            rows_06j_top,
            key=lambda r: _safe_float(r.get("balanced_score_06j", float("inf"))))
        sorted_balanced = []
        for r in sorted_balanced_pool:
            if r["recipe_id"] in by_id:
                sorted_balanced.append(by_id[r["recipe_id"]])
    else:
        sorted_balanced = sorted_strict

    targets: list[tuple[dict, str]] = []
    seen: set[str] = set()

    def _add_first(pool, kind):
        for r in pool:
            if r["recipe_id"] in seen:
                continue
            targets.append((r, kind))
            seen.add(r["recipe_id"])
            return

    # Always include J_1453 first if present; otherwise the strict-best.
    j1453 = by_id.get("J_1453")
    if j1453 is not None:
        targets.append((j1453, "j1453_pinned"))
        seen.add("J_1453")
    _add_first(sorted_strict, "strict_best")
    _add_first(sorted_cd, "cd_best")
    _add_first(sorted_ler, "ler_best")
    _add_first(sorted_balanced, "balanced_best")
    _add_first(sorted_margin, "margin_best")
    return targets


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str,
                   default=str(V3_DIR / "configs" / "yield_optimization.yaml"))
    p.add_argument("--rescored_csv", type=str,
                   default=str(V3_DIR / "outputs" / "yield_optimization"
                               / "stage06L_mode_b_rescored_candidates.csv"))
    p.add_argument("--top_06j_csv", type=str,
                   default=str(V3_DIR / "outputs" / "yield_optimization"
                               / "stage06J_mode_b_top_recipes.csv"))
    p.add_argument("--n_top_nominal",   type=int, default=100)
    p.add_argument("--n_top_mc",         type=int, default=10)
    p.add_argument("--n_mc_per_top",    type=int, default=100)
    p.add_argument("--n_mc_per_rep",    type=int, default=300)
    p.add_argument("--seed", type=int, default=8989,
                   help="Distinct from prior FD MC seeds (06H 6262, 06J 7373).")
    p.add_argument("--skip_part1", action="store_true")
    p.add_argument("--skip_part2", action="store_true")
    p.add_argument("--skip_part3", action="store_true")
    p.add_argument("--label_schema",
                   default=str(V3_DIR / "configs" / "label_schema.yaml"))
    args = p.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    space = CandidateSpace.from_yaml(cfg["candidate_space_yaml"])
    var_spec = VariationSpec.from_yaml_dict(cfg["process_variation"])
    thresholds = LabelThresholds.from_yaml(args.label_schema)

    # Read 06L Mode B rescored candidates (100 rows) and sort by 06L strict_score.
    rescored = read_labels_csv(args.rescored_csv)
    num_keys = ["strict_score_06l_eval", "strict_score_06l_direct_mean",
                 "strict_score_06l_direct_std",
                 "yield_score_06l", "p_robust_valid_06l",
                 "mean_cd_fixed_06l", "std_cd_fixed_06l",
                 "mean_ler_locked_06l", "std_ler_locked_06l",
                 "mean_p_line_margin_06l"] + FEATURE_KEYS
    for r in rescored:
        for k in num_keys:
            if k in r:
                r[k] = _safe_float(r.get(k))
    rows_b = [r for r in rescored if str(r.get("mode", "")) == "mode_b"]
    rows_b.sort(key=lambda r: -float(r["strict_score_06l_direct_mean"]))
    for i, r in enumerate(rows_b, start=1):
        r["rank_06l"] = i

    print(f"Stage 06J-B -- Mode B FD verification at scale")
    print(f"  policy: v2_OP_frozen={cfg['policy']['v2_OP_frozen']}, "
          f"published_data_loaded={cfg['policy']['published_data_loaded']}")
    print(f"  Mode B rescored pool size:    {len(rows_b)}")
    print(f"  top-{args.n_top_nominal} for Part 1, top-{args.n_top_mc} for Part 2")
    print(f"  Part 3 representatives:")
    reps = select_representatives(rows_b, args.top_06j_csv)
    for r, kind in reps:
        print(f"    {kind:>16}  {r['recipe_id']}  "
              f"strict={float(r['strict_score_06l_direct_mean']):.4f}  "
              f"pitch={float(r['pitch_nm']):.0f}  ratio={float(r['line_cd_ratio']):.2f}")

    labels_dir = V3_DIR / "outputs" / "labels"
    p1_path = labels_dir / "stage06J_B_fd_top100_nominal.csv"
    p2_path = labels_dir / "stage06J_B_fd_top10_mc.csv"
    p3_path = labels_dir / "stage06J_B_fd_representative_mc.csv"

    if args.skip_part1:
        print("  Part1: skipped (--skip_part1)")
    elif p1_path.exists():
        print(f"  Part1: skipped -- {p1_path} already exists")
    else:
        print(f"\n  Part1 -- top-{args.n_top_nominal} Mode B nominal FD")
        run_part1_nominal(rows_b, space, thresholds, p1_path,
                            n_top=args.n_top_nominal)

    if args.skip_part2:
        print("  Part2: skipped (--skip_part2)")
    elif p2_path.exists():
        print(f"  Part2: skipped -- {p2_path} already exists")
    else:
        print(f"\n  Part2 -- top-{args.n_top_mc} Mode B FD MC "
              f"(x{args.n_mc_per_top} variations each)")
        run_part2_top10_mc(rows_b, space, thresholds, var_spec,
                            n_top=args.n_top_mc, n_var=args.n_mc_per_top,
                            out_path=p2_path, seed=args.seed)

    if args.skip_part3:
        print("  Part3: skipped (--skip_part3)")
    elif p3_path.exists():
        print(f"  Part3: skipped -- {p3_path} already exists")
    elif not reps:
        print("  Part3: skipped -- no representatives")
    else:
        print(f"\n  Part3 -- {len(reps)} Mode B representatives FD MC "
              f"(x{args.n_mc_per_rep} variations each)")
        run_part3_representative_mc(reps, space, thresholds, var_spec,
                                       n_var=args.n_mc_per_rep,
                                       out_path=p3_path,
                                       seed=args.seed + 1)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
