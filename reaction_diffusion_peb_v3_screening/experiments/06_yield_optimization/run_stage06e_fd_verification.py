"""Stage 06E -- FD verification + AL update for Stage 06D recipes.

Five parts:
    Part 0 -- single nominal FD on the v2 frozen OP (nominal baseline).
    Part 1 -- top-100 nominal FD on Stage 06D fixed-design recipes.
    Part 2 -- top-10 FD Monte-Carlo (100 process variations each).
    Part 3 -- nominal FD on the 17 Stage 06D disagreement candidates.
    Part 4 -- 100-variation FD MC on the v2 frozen OP (MC baseline). The
              v2 nominal FD is robust_valid -> score 1.0, which caps the
              nominal-FD comparison; the MC baseline gives us a fair
              MC-vs-MC reference point for the top-10 06D recipes.

All four parts stream rows to dedicated CSVs so a crash mid-batch leaves
a partial CSV on disk. The script is idempotent: if a CSV already
exists, the corresponding part is skipped (delete the CSV to re-run).

Stage 06A's process-variation widths are loaded straight from
configs/yield_optimization.yaml so the FD MC uses the same statistical
setup as the surrogate optimisation it is verifying.

Policy
    v2_OP_frozen           = true
    published_data_loaded  = false
    No external calibration. Closed Stage 04C / 04D / 06B training rows
    are not mutated. New FD rows go to Stage 06E-only CSVs and are
    concatenated into a stage06E_al_additions.csv by analyze_stage06e.py.
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
    FEATURE_KEYS,
    LABEL_CSV_COLUMNS,
    read_labels_csv,
)
from reaction_diffusion_peb_v3_screening.src.process_variation import (
    VariationSpec,
    sample_variations,
)


V3_DIR = ROOT / "reaction_diffusion_peb_v3_screening"


EXTRA_COLS = [
    "source_recipe_id",
    "rank_surrogate",
    "variation_idx",
    "phase",
    "yield_score_surrogate",
    "yield_score_06a_rescore",
    "score_gap",
]
WRITE_COLS = LABEL_CSV_COLUMNS + EXTRA_COLS


def _safe_float(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


# --------------------------------------------------------------------------
# CSV row -> full candidate dict.
# --------------------------------------------------------------------------
def _row_to_candidate(row: dict, space: CandidateSpace) -> dict:
    out = {}
    for k in FEATURE_KEYS:
        try:
            out[k] = float(row[k])
        except (TypeError, ValueError, KeyError):
            raise ValueError(f"recipe row missing or non-numeric {k}: {row.get(k)!r}")
    out["pitch_nm"]    = float(out["pitch_nm"])
    out["line_cd_nm"]  = out["pitch_nm"] * out["line_cd_ratio"]
    out["domain_x_nm"] = out["pitch_nm"] * 5.0
    out["dose_norm"]   = out["dose_mJ_cm2"] / float(space.fixed["reference_dose_mJ_cm2"])
    for fk, fv in space.fixed.items():
        out.setdefault(fk, fv)
    out["_id"] = row.get("recipe_id", "?")
    return out


def _v2_frozen_op_candidate(space: CandidateSpace, op: dict) -> dict:
    base = {p["name"]: (p["values"][0] if p["type"] == "choice" else p["low"])
            for p in space.parameters}
    base.update({k: float(v) if not isinstance(v, int) else int(v)
                 for k, v in op.items()})
    base["pitch_nm"]    = float(base["pitch_nm"])
    base["line_cd_nm"]  = base["pitch_nm"] * float(base["line_cd_ratio"])
    base["domain_x_nm"] = base["pitch_nm"] * 5.0
    base["dose_norm"]   = base["dose_mJ_cm2"] / float(space.fixed["reference_dose_mJ_cm2"])
    for fk, fv in space.fixed.items():
        base.setdefault(fk, fv)
    base["_id"] = "v2_frozen_op"
    return base


# --------------------------------------------------------------------------
# CSV streaming.
# --------------------------------------------------------------------------
def _open_csv(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    f = path.open("w", newline="")
    w = csv.DictWriter(f, fieldnames=WRITE_COLS, extrasaction="ignore")
    w.writeheader()
    return f, w


def _write_fd_row(writer, row: dict, *, source_recipe_id: str, rank: int,
                  variation_idx: int, phase: str,
                  yield_score_surrogate: float,
                  yield_score_06a_rescore: float = float("nan"),
                  score_gap: float = float("nan")) -> None:
    out = dict(row)
    out["source_recipe_id"]        = source_recipe_id
    out["rank_surrogate"]          = int(rank)
    out["variation_idx"]           = int(variation_idx)
    out["phase"]                   = phase
    out["yield_score_surrogate"]   = float(yield_score_surrogate)
    out["yield_score_06a_rescore"] = float(yield_score_06a_rescore)
    out["score_gap"]               = float(score_gap)
    writer.writerow(out)


# --------------------------------------------------------------------------
# Part 0 -- v2 frozen OP nominal FD baseline.
# --------------------------------------------------------------------------
def run_part0_baseline(space: CandidateSpace,
                        op_cfg: dict,
                        thresholds: LabelThresholds,
                        out_path: Path) -> None:
    f, w = _open_csv(out_path)
    try:
        cand = _v2_frozen_op_candidate(space, op_cfg)
        cand["_id"] = "v2_frozen_op__nom"
        fd_row = run_one_candidate(cand, thresholds)
        _write_fd_row(
            w, fd_row,
            source_recipe_id="v2_frozen_op",
            rank=0, variation_idx=0, phase="baseline_nominal",
            yield_score_surrogate=float("nan"),
        )
        f.flush()
    finally:
        f.close()
    print(f"  Part0 -> {out_path} (v2 frozen OP nominal FD)")


def run_part4_baseline_mc(space: CandidateSpace,
                            op_cfg: dict,
                            thresholds: LabelThresholds,
                            var_spec: VariationSpec,
                            n_var: int,
                            out_path: Path,
                            seed: int) -> None:
    f, w = _open_csv(out_path)
    t0 = time.time()
    try:
        base = _v2_frozen_op_candidate(space, op_cfg)
        rng = np.random.default_rng(seed)
        variations = sample_variations(base, var_spec, n_var, space, rng=rng)
        for j, v in enumerate(variations, start=1):
            v["_id"] = f"v2_frozen_op__mc{j:03d}"
            for fk, fv in space.fixed.items():
                v.setdefault(fk, fv)
            v["domain_x_nm"] = v["pitch_nm"] * 5.0
            v["dose_norm"]   = v["dose_mJ_cm2"] / float(space.fixed["reference_dose_mJ_cm2"])
            fd_row = run_one_candidate(v, thresholds)
            _write_fd_row(
                w, fd_row,
                source_recipe_id="v2_frozen_op",
                rank=0, variation_idx=j, phase="baseline_mc",
                yield_score_surrogate=float("nan"),
            )
            f.flush()
    finally:
        f.close()
    print(f"  Part4 -> {out_path} ({n_var} MC runs in {time.time()-t0:.1f}s)")


# --------------------------------------------------------------------------
# Part 1 -- top-100 nominal FD.
# --------------------------------------------------------------------------
def run_part1_top100_nominal(top_rows: list[dict],
                              space: CandidateSpace,
                              thresholds: LabelThresholds,
                              out_path: Path,
                              n_top: int,
                              progress_every: int = 10) -> None:
    f, w = _open_csv(out_path)
    t0 = time.time()
    try:
        for rank, row in enumerate(top_rows[:n_top], start=1):
            cand = _row_to_candidate(row, space)
            cand["_id"] = f"{row.get('recipe_id', '?')}__nom"
            fd_row = run_one_candidate(cand, thresholds)
            _write_fd_row(
                w, fd_row,
                source_recipe_id=str(row.get("recipe_id", "?")),
                rank=rank, variation_idx=0, phase="nominal",
                yield_score_surrogate=_safe_float(row.get("yield_score_06c")),
                yield_score_06a_rescore=_safe_float(row.get("yield_score_06a")),
            )
            f.flush()
            if rank % progress_every == 0:
                elapsed = time.time() - t0
                rate = rank / max(elapsed, 1e-9)
                print(f"  Part1: {rank}/{n_top} done  "
                      f"({elapsed:.1f}s, {rate:.2f} runs/s)")
    finally:
        f.close()
    print(f"  Part1 -> {out_path} ({n_top} runs in {time.time()-t0:.1f}s)")


# --------------------------------------------------------------------------
# Part 2 -- top-10 FD Monte-Carlo.
# --------------------------------------------------------------------------
def run_part2_top10_mc(top_rows: list[dict],
                       space: CandidateSpace,
                       thresholds: LabelThresholds,
                       var_spec: VariationSpec,
                       n_top: int,
                       n_var: int,
                       out_path: Path,
                       seed: int,
                       progress_every: int = 25) -> None:
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
                v["_id"] = f"{row.get('recipe_id', '?')}__mc{j:03d}"
                for fk, fv in space.fixed.items():
                    v.setdefault(fk, fv)
                v["domain_x_nm"] = v["pitch_nm"] * 5.0
                v["dose_norm"]   = v["dose_mJ_cm2"] / float(space.fixed["reference_dose_mJ_cm2"])
                fd_row = run_one_candidate(v, thresholds)
                _write_fd_row(
                    w, fd_row,
                    source_recipe_id=str(row.get("recipe_id", "?")),
                    rank=rank, variation_idx=j, phase="mc",
                    yield_score_surrogate=_safe_float(row.get("yield_score_06c")),
                    yield_score_06a_rescore=_safe_float(row.get("yield_score_06a")),
                )
                f.flush()
                total += 1
                if total % progress_every == 0:
                    elapsed = time.time() - t0
                    rate = total / max(elapsed, 1e-9)
                    eta = (n_top * n_var - total) / max(rate, 1e-9)
                    print(f"  Part2: {total}/{n_top*n_var} done  "
                          f"({elapsed:.1f}s, {rate:.2f} runs/s, ETA {eta:.0f}s)")
            print(f"  Part2: rank {rank}/{n_top} ({row.get('recipe_id','?')}) done  "
                  f"({n_var} MC runs)")
    finally:
        f.close()
    print(f"  Part2 -> {out_path} ({total} runs in {time.time()-t0:.1f}s)")


# --------------------------------------------------------------------------
# Part 3 -- 17 disagreement candidates nominal FD.
# --------------------------------------------------------------------------
def run_part3_disagreement(rows: list[dict],
                            space: CandidateSpace,
                            thresholds: LabelThresholds,
                            out_path: Path) -> None:
    f, w = _open_csv(out_path)
    t0 = time.time()
    try:
        for i, row in enumerate(rows, start=1):
            cand = _row_to_candidate(row, space)
            cand["_id"] = f"{row.get('recipe_id', '?')}__disagree"
            fd_row = run_one_candidate(cand, thresholds)
            _write_fd_row(
                w, fd_row,
                source_recipe_id=str(row.get("recipe_id", "?")),
                rank=int(_safe_float(row.get("rank_06c", 0))) or i,
                variation_idx=0, phase="disagreement_nominal",
                yield_score_surrogate=_safe_float(row.get("yield_score_06c")),
                yield_score_06a_rescore=_safe_float(row.get("yield_score_06a")),
                score_gap=_safe_float(row.get("score_gap")),
            )
            f.flush()
        print(f"  Part3 -> {out_path} ({len(rows)} runs in {time.time()-t0:.1f}s)")
    finally:
        f.close()


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str,
                   default=str(V3_DIR / "configs" / "yield_optimization.yaml"))
    p.add_argument("--top_recipes_06d_csv", type=str,
                   default=str(V3_DIR / "outputs" / "yield_optimization"
                               / "stage06D_top_recipes.csv"))
    p.add_argument("--disagreement_06d_csv", type=str,
                   default=str(V3_DIR / "outputs" / "yield_optimization"
                               / "stage06D_disagreement_candidates.csv"))
    p.add_argument("--n_top_nominal", type=int, default=100)
    p.add_argument("--n_top_mc", type=int, default=10)
    p.add_argument("--n_mc_per_recipe", type=int, default=100)
    p.add_argument("--seed", type=int, default=4143,
                   help="MC seed -- different from Stage 06B's 2026.")
    p.add_argument("--skip_part0", action="store_true")
    p.add_argument("--skip_part1", action="store_true")
    p.add_argument("--skip_part2", action="store_true")
    p.add_argument("--skip_part3", action="store_true")
    p.add_argument("--skip_part4", action="store_true")
    p.add_argument("--label_schema",
                   default=str(V3_DIR / "configs" / "label_schema.yaml"))
    args = p.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    space = CandidateSpace.from_yaml(cfg["candidate_space_yaml"])
    var_spec = VariationSpec.from_yaml_dict(cfg["process_variation"])
    thresholds = LabelThresholds.from_yaml(args.label_schema)

    top_rows = read_labels_csv(args.top_recipes_06d_csv)
    # Already sorted by 06C rank in the CSV; sort defensively.
    top_rows.sort(key=lambda r: _safe_float(r.get("rank_06c", 1e9)))

    disagreement_rows = read_labels_csv(args.disagreement_06d_csv)

    print(f"Stage 06E -- FD verification + AL update")
    print(f"  config:                 {args.config}")
    print(f"  06D top recipes:        {len(top_rows)}  src={args.top_recipes_06d_csv}")
    print(f"  06D disagreement set:   {len(disagreement_rows)}  src={args.disagreement_06d_csv}")
    print(f"  policy:                 {cfg['policy']}")

    labels_dir = V3_DIR / "outputs" / "labels"
    p0_path = labels_dir / "stage06E_fd_baseline_v2_op.csv"
    p1_path = labels_dir / "stage06E_fd_top100_nominal.csv"
    p2_path = labels_dir / "stage06E_fd_top10_mc.csv"
    p3_path = labels_dir / "stage06E_fd_disagreement.csv"
    p4_path = labels_dir / "stage06E_fd_baseline_v2_op_mc.csv"

    # ----- Part 0 -----
    if args.skip_part0:
        print("  Part0: skipped (--skip_part0)")
    elif p0_path.exists():
        print(f"  Part0: skipped -- {p0_path} already exists")
    else:
        print(f"\n  Part0 -- v2 frozen OP nominal FD baseline")
        run_part0_baseline(space, cfg["v2_frozen_op"], thresholds, p0_path)

    # ----- Part 1 -----
    if args.skip_part1:
        print("  Part1: skipped (--skip_part1)")
    elif p1_path.exists():
        print(f"  Part1: skipped -- {p1_path} already exists")
    else:
        print(f"\n  Part1 -- top-{args.n_top_nominal} nominal FD")
        run_part1_top100_nominal(top_rows, space, thresholds, p1_path,
                                  n_top=args.n_top_nominal)

    # ----- Part 2 -----
    if args.skip_part2:
        print("  Part2: skipped (--skip_part2)")
    elif p2_path.exists():
        print(f"  Part2: skipped -- {p2_path} already exists")
    else:
        print(f"\n  Part2 -- top-{args.n_top_mc} FD Monte-Carlo "
              f"(x {args.n_mc_per_recipe} variations each)")
        run_part2_top10_mc(top_rows, space, thresholds, var_spec,
                           n_top=args.n_top_mc,
                           n_var=args.n_mc_per_recipe,
                           out_path=p2_path,
                           seed=args.seed)

    # ----- Part 3 -----
    if args.skip_part3:
        print("  Part3: skipped (--skip_part3)")
    elif p3_path.exists():
        print(f"  Part3: skipped -- {p3_path} already exists")
    elif not disagreement_rows:
        print("  Part3: skipped -- 0 disagreement candidates supplied")
    else:
        print(f"\n  Part3 -- disagreement candidates ({len(disagreement_rows)} nominal FD)")
        run_part3_disagreement(disagreement_rows, space, thresholds, p3_path)

    # ----- Part 4 -----
    if args.skip_part4:
        print("  Part4: skipped (--skip_part4)")
    elif p4_path.exists():
        print(f"  Part4: skipped -- {p4_path} already exists")
    else:
        print(f"\n  Part4 -- v2 frozen OP FD MC baseline "
              f"({args.n_mc_per_recipe} variations)")
        run_part4_baseline_mc(space, cfg["v2_frozen_op"], thresholds, var_spec,
                                n_var=args.n_mc_per_recipe, out_path=p4_path,
                                seed=args.seed)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
