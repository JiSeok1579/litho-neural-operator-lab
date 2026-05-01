"""Stage 06B — FD verification of Stage 06A surrogate-driven recipes.

Two parts:
    Part 1 — top-100 Mode A nominal FD (single FD per recipe, no MC).
    Part 2 — top-10 Mode A FD Monte-Carlo (100 process variations each).

Both parts stream rows to CSV so a crash mid-batch leaves a partial CSV
on disk. The script is idempotent: if a CSV already exists, the
corresponding part is skipped (delete the CSV to re-run).

Stage 06A's process-variation widths are loaded straight from
configs/yield_optimization.yaml so the FD MC uses the same statistical
setup as the surrogate optimisation it is verifying.

Policy
    v2_OP_frozen           = true
    published_data_loaded  = false
    No external calibration. The closed Stage 04D / 04C training dataset
    is *not* mutated. New FD rows go to a separate AL-additions CSV
    written by analyze_fd_verification.py.
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


# Extra columns we keep on every FD row so analyses can group by base
# recipe and back-trace to the surrogate prediction.
EXTRA_COLS = [
    "source_recipe_id",     # Stage 06A `_id` (e.g. "A_4856")
    "rank_surrogate",       # 1..N within Mode A surrogate top-N
    "variation_idx",        # 0 = nominal, 1..n_var = MC sample
    "phase",                # "nominal" | "mc"
    "yield_score_surrogate",
]
WRITE_COLS = LABEL_CSV_COLUMNS + EXTRA_COLS


# --------------------------------------------------------------------------
# Recipe row → full candidate dict (rebuild geometry / fixed / derived).
# --------------------------------------------------------------------------
def _row_to_candidate(row: dict, space: CandidateSpace) -> dict:
    """Take a CSV row from top_recipes_fixed_design_surrogate.csv and
    return a candidate dict carrying every field the FD batch runner
    expects."""
    out = {}
    for k in FEATURE_KEYS:
        try:
            out[k] = float(row[k])
        except (TypeError, ValueError, KeyError):
            raise ValueError(f"recipe row missing or non-numeric {k}: {row.get(k)!r}")
    # pitch_nm is stored as float in the CSV but the candidate-space
    # treats it as an integer choice. Keep both — runner casts as float.
    out["pitch_nm"] = float(out["pitch_nm"])
    out["line_cd_nm"]  = out["pitch_nm"] * out["line_cd_ratio"]
    out["domain_x_nm"] = out["pitch_nm"] * 5.0
    out["dose_norm"]   = out["dose_mJ_cm2"] / float(space.fixed["reference_dose_mJ_cm2"])
    for fk, fv in space.fixed.items():
        out.setdefault(fk, fv)
    out["_id"] = row.get("recipe_id", "?")
    return out


# --------------------------------------------------------------------------
# Stream FD rows to a CSV.
# --------------------------------------------------------------------------
def _open_csv(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    f = path.open("w", newline="")
    w = csv.DictWriter(f, fieldnames=WRITE_COLS, extrasaction="ignore")
    w.writeheader()
    return f, w


def _write_fd_row(writer, row: dict, source_recipe_id: str, rank: int,
                  variation_idx: int, phase: str,
                  yield_score_surrogate: float) -> None:
    out = dict(row)
    out["source_recipe_id"]      = source_recipe_id
    out["rank_surrogate"]        = int(rank)
    out["variation_idx"]         = int(variation_idx)
    out["phase"]                 = phase
    out["yield_score_surrogate"] = float(yield_score_surrogate)
    writer.writerow(out)


# --------------------------------------------------------------------------
# Part 1 — top-100 nominal FD.
# --------------------------------------------------------------------------
def run_part1_top100_nominal(
    top_rows: list[dict],
    space: CandidateSpace,
    thresholds: LabelThresholds,
    out_path: Path,
    n_top: int,
    progress_every: int = 10,
) -> None:
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
                rank=rank,
                variation_idx=0,
                phase="nominal",
                yield_score_surrogate=float(row.get("yield_score", "nan")),
            )
            f.flush()
            if rank % progress_every == 0:
                elapsed = time.time() - t0
                rate = rank / max(elapsed, 1e-9)
                print(f"  Part1: {rank}/{n_top} done  "
                      f"({elapsed:.1f}s, {rate:.2f} runs/s)")
    finally:
        f.close()
    print(f"  Part1 → {out_path} ({n_top} runs in {time.time()-t0:.1f}s)")


# --------------------------------------------------------------------------
# Part 2 — top-10 FD Monte-Carlo.
# --------------------------------------------------------------------------
def run_part2_top10_mc(
    top_rows: list[dict],
    space: CandidateSpace,
    thresholds: LabelThresholds,
    var_spec: VariationSpec,
    n_top: int,
    n_var: int,
    out_path: Path,
    seed: int,
    progress_every: int = 25,
) -> None:
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
                # Re-fill fixed keys (sample_variations preserves base
                # keys, but be defensive).
                for fk, fv in space.fixed.items():
                    v.setdefault(fk, fv)
                v["domain_x_nm"] = v["pitch_nm"] * 5.0
                v["dose_norm"]   = v["dose_mJ_cm2"] / float(space.fixed["reference_dose_mJ_cm2"])
                fd_row = run_one_candidate(v, thresholds)
                _write_fd_row(
                    w, fd_row,
                    source_recipe_id=str(row.get("recipe_id", "?")),
                    rank=rank,
                    variation_idx=j,
                    phase="mc",
                    yield_score_surrogate=float(row.get("yield_score", "nan")),
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
    print(f"  Part2 → {out_path} ({total} runs in {time.time()-t0:.1f}s)")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str,
                   default=str(V3_DIR / "configs" / "yield_optimization.yaml"))
    p.add_argument("--top_recipes_csv", type=str,
                   default=str(V3_DIR / "outputs" / "labels"
                               / "06_top_recipes_fixed_design_surrogate.csv"))
    p.add_argument("--n_top_nominal", type=int, default=100)
    p.add_argument("--n_top_mc", type=int, default=10)
    p.add_argument("--n_mc_per_recipe", type=int, default=100)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--skip_part1", action="store_true")
    p.add_argument("--skip_part2", action="store_true")
    p.add_argument("--label_schema",
                   default=str(V3_DIR / "configs" / "label_schema.yaml"))
    args = p.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    space = CandidateSpace.from_yaml(cfg["candidate_space_yaml"])
    var_spec = VariationSpec.from_yaml_dict(cfg["process_variation"])
    thresholds = LabelThresholds.from_yaml(args.label_schema)

    top_rows = read_labels_csv(args.top_recipes_csv)
    # Sort by surrogate yield_score descending — the CSV already is, but be safe.
    top_rows = sorted(top_rows, key=lambda r: -float(r["yield_score"]))

    print(f"Stage 06B — FD verification")
    print(f"  config:        {args.config}")
    print(f"  recipes:       {len(top_rows)} (sorted by surrogate yield_score)")
    print(f"  top-100 src:   {args.top_recipes_csv}")
    print(f"  policy:        {cfg['policy']}")

    labels_dir = V3_DIR / "outputs" / "labels"
    p1_path = labels_dir / "fd_top100_nominal_verification.csv"
    p2_path = labels_dir / "fd_top10_mc_verification.csv"

    # ----- Part 1 -----
    if args.skip_part1:
        print("  Part1: skipped (--skip_part1)")
    elif p1_path.exists():
        print(f"  Part1: skipped — {p1_path} already exists")
    else:
        print(f"\n  Part1 — top-{args.n_top_nominal} nominal FD")
        run_part1_top100_nominal(
            top_rows, space, thresholds, p1_path,
            n_top=args.n_top_nominal,
        )

    # ----- Part 2 -----
    if args.skip_part2:
        print("  Part2: skipped (--skip_part2)")
    elif p2_path.exists():
        print(f"  Part2: skipped — {p2_path} already exists")
    else:
        print(f"\n  Part2 — top-{args.n_top_mc} FD Monte-Carlo "
              f"(× {args.n_mc_per_recipe} variations each)")
        run_part2_top10_mc(
            top_rows, space, thresholds, var_spec,
            n_top=args.n_top_mc,
            n_var=args.n_mc_per_recipe,
            out_path=p2_path,
            seed=args.seed,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
