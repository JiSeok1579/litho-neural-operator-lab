"""Stage 06H -- FD verification of Stage 06G strict-score candidates.

Three parts (Parts 1-3 of the spec). Surrogate refresh + validation
live in build_stage06h_dataset.py / train_stage06h_models.py /
analyze_stage06h.py.

    Part 1 -- top-100 06G nominal FD (100 runs).
    Part 2 -- top-10 06G FD MC (100 variations each, 1,000 runs).
    Part 3 -- 6 06G representative recipes FD MC (300 variations each,
              up to 1,800 runs).

All three parts stream rows to dedicated CSVs so a crash mid-batch
leaves a partial CSV on disk. Idempotent: if a CSV exists, that part
is skipped (delete the CSV to re-run).

Policy
    v2_OP_frozen           = true
    published_data_loaded  = false
    No external calibration. Closed Stage 04C / 04D / 06C joblibs are
    read-only. New FD rows are written to Stage 06H-specific CSVs and
    flow into the 06H training dataset via build_stage06h_dataset.py.
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
    "rank_strict",
    "variation_idx",
    "phase",
    "strict_score_surrogate",
    "yield_score_surrogate",
    "rep_kind",
]
WRITE_COLS = LABEL_CSV_COLUMNS + EXTRA_COLS


def _safe_float(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


# --------------------------------------------------------------------------
# Recipe row -> full candidate dict.
# --------------------------------------------------------------------------
def _row_to_candidate(row: dict, space: CandidateSpace) -> dict:
    out = {}
    for k in FEATURE_KEYS:
        try:
            out[k] = float(row[k])
        except (TypeError, ValueError, KeyError):
            raise ValueError(f"recipe row missing or non-numeric {k}: "
                             f"{row.get(k)!r}")
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


def _write_fd_row(writer, row: dict, *, source_recipe_id: str, rank: int,
                  variation_idx: int, phase: str,
                  strict_score_surrogate: float = float("nan"),
                  yield_score_surrogate: float = float("nan"),
                  rep_kind: str = "") -> None:
    out = dict(row)
    out["source_recipe_id"]       = source_recipe_id
    out["rank_strict"]            = int(rank)
    out["variation_idx"]          = int(variation_idx)
    out["phase"]                  = phase
    out["strict_score_surrogate"] = float(strict_score_surrogate)
    out["yield_score_surrogate"]  = float(yield_score_surrogate)
    out["rep_kind"]               = str(rep_kind)
    writer.writerow(out)


# --------------------------------------------------------------------------
# Part 1 -- top-100 nominal FD.
# --------------------------------------------------------------------------
def run_part1_nominal(top_rows: list[dict],
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
            cand["_id"] = f"{row.get('recipe_id', '?')}__nom_06h"
            fd_row = run_one_candidate(cand, thresholds)
            _write_fd_row(
                w, fd_row,
                source_recipe_id=str(row.get("recipe_id", "?")),
                rank=rank, variation_idx=0, phase="nominal",
                strict_score_surrogate=_safe_float(row.get("strict_score")),
                yield_score_surrogate=_safe_float(row.get("yield_score")),
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
# Part 2 -- top-10 FD Monte-Carlo (100 variations each).
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
                v["_id"] = f"{row.get('recipe_id', '?')}__mc{j:03d}_06h"
                for fk, fv in space.fixed.items():
                    v.setdefault(fk, fv)
                v["domain_x_nm"] = v["pitch_nm"] * 5.0
                v["dose_norm"]   = v["dose_mJ_cm2"] / float(space.fixed["reference_dose_mJ_cm2"])
                fd_row = run_one_candidate(v, thresholds)
                _write_fd_row(
                    w, fd_row,
                    source_recipe_id=str(row.get("recipe_id", "?")),
                    rank=rank, variation_idx=j, phase="mc_top10",
                    strict_score_surrogate=_safe_float(row.get("strict_score")),
                    yield_score_surrogate=_safe_float(row.get("yield_score")),
                )
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
# Part 3 -- representative recipes FD MC (300 variations each).
# --------------------------------------------------------------------------
def run_part3_representative_mc(rep_rows: list[dict],
                                  space: CandidateSpace,
                                  thresholds: LabelThresholds,
                                  var_spec: VariationSpec,
                                  n_var: int,
                                  out_path: Path,
                                  seed: int,
                                  progress_every: int = 50) -> None:
    f, w = _open_csv(out_path)
    t0 = time.time()
    base_rng = np.random.default_rng(seed)
    total = 0
    try:
        for k_idx, row in enumerate(rep_rows, start=1):
            base = _row_to_candidate(row, space)
            kind = str(row.get("kind", f"rep{k_idx}"))
            sub_seed = int(base_rng.integers(0, 2**31 - 1))
            sub_rng = np.random.default_rng(sub_seed)
            variations = sample_variations(base, var_spec, n_var, space, rng=sub_rng)
            for j, v in enumerate(variations, start=1):
                v["_id"] = f"{row.get('recipe_id', '?')}__{kind}_mc{j:03d}_06h"
                for fk, fv in space.fixed.items():
                    v.setdefault(fk, fv)
                v["domain_x_nm"] = v["pitch_nm"] * 5.0
                v["dose_norm"]   = v["dose_mJ_cm2"] / float(space.fixed["reference_dose_mJ_cm2"])
                fd_row = run_one_candidate(v, thresholds)
                _write_fd_row(
                    w, fd_row,
                    source_recipe_id=str(row.get("recipe_id", "?")),
                    rank=int(_safe_float(row.get("rank_strict", 0))),
                    variation_idx=j, phase="mc_representative",
                    strict_score_surrogate=_safe_float(row.get("strict_score")),
                    yield_score_surrogate=_safe_float(row.get("yield_score")),
                    rep_kind=kind,
                )
                f.flush()
                total += 1
                if total % progress_every == 0:
                    elapsed = time.time() - t0
                    rate = total / max(elapsed, 1e-9)
                    eta = (len(rep_rows) * n_var - total) / max(rate, 1e-9)
                    print(f"  Part3: {total}/{len(rep_rows)*n_var} done  "
                          f"({elapsed:.1f}s, {rate:.2f} runs/s, ETA {eta:.0f}s)")
            print(f"  Part3: representative {k_idx}/{len(rep_rows)} "
                  f"({kind}: {row.get('recipe_id','?')}) done "
                  f"({n_var} MC runs)")
    finally:
        f.close()
    print(f"  Part3 -> {out_path} ({total} runs in {time.time()-t0:.1f}s)")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str,
                   default=str(V3_DIR / "configs" / "yield_optimization.yaml"))
    p.add_argument("--top_06g_csv", type=str,
                   default=str(V3_DIR / "outputs" / "yield_optimization"
                               / "stage06G_top_recipes.csv"))
    p.add_argument("--reps_06g_csv", type=str,
                   default=str(V3_DIR / "outputs" / "yield_optimization"
                               / "stage06G_representative_recipes.csv"))
    p.add_argument("--n_top_nominal",  type=int, default=100)
    p.add_argument("--n_top_mc",       type=int, default=10)
    p.add_argument("--n_mc_per_top",   type=int, default=100)
    p.add_argument("--n_mc_per_rep",   type=int, default=300)
    p.add_argument("--seed",           type=int, default=6262,
                   help="MC seed -- distinct from 06B's 2026 and 06E's 4143.")
    p.add_argument("--skip_part1",     action="store_true")
    p.add_argument("--skip_part2",     action="store_true")
    p.add_argument("--skip_part3",     action="store_true")
    p.add_argument("--label_schema",
                   default=str(V3_DIR / "configs" / "label_schema.yaml"))
    args = p.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    space = CandidateSpace.from_yaml(cfg["candidate_space_yaml"])
    var_spec = VariationSpec.from_yaml_dict(cfg["process_variation"])
    thresholds = LabelThresholds.from_yaml(args.label_schema)

    top_rows = read_labels_csv(args.top_06g_csv)
    top_rows.sort(key=lambda r: _safe_float(r.get("rank_strict", 1e9)))

    rep_rows = read_labels_csv(args.reps_06g_csv)

    print(f"Stage 06H -- FD verification of 06G strict-score recipes")
    print(f"  config:               {args.config}")
    print(f"  06G top recipes:      {len(top_rows)}  src={args.top_06g_csv}")
    print(f"  06G representatives:  {len(rep_rows)}  src={args.reps_06g_csv}")
    print(f"  policy:               {cfg['policy']}")

    labels_dir = V3_DIR / "outputs" / "labels"
    p1_path = labels_dir / "stage06H_fd_top100_nominal.csv"
    p2_path = labels_dir / "stage06H_fd_top10_mc.csv"
    p3_path = labels_dir / "stage06H_fd_representative_mc.csv"

    if args.skip_part1:
        print("  Part1: skipped (--skip_part1)")
    elif p1_path.exists():
        print(f"  Part1: skipped -- {p1_path} already exists")
    else:
        print(f"\n  Part1 -- top-{args.n_top_nominal} 06G nominal FD")
        run_part1_nominal(top_rows, space, thresholds, p1_path,
                            n_top=args.n_top_nominal)

    if args.skip_part2:
        print("  Part2: skipped (--skip_part2)")
    elif p2_path.exists():
        print(f"  Part2: skipped -- {p2_path} already exists")
    else:
        print(f"\n  Part2 -- top-{args.n_top_mc} 06G FD Monte-Carlo "
              f"(x {args.n_mc_per_top} variations each)")
        run_part2_top10_mc(top_rows, space, thresholds, var_spec,
                           n_top=args.n_top_mc,
                           n_var=args.n_mc_per_top,
                           out_path=p2_path,
                           seed=args.seed)

    if args.skip_part3:
        print("  Part3: skipped (--skip_part3)")
    elif p3_path.exists():
        print(f"  Part3: skipped -- {p3_path} already exists")
    elif not rep_rows:
        print("  Part3: skipped -- 0 representative recipes supplied")
    else:
        print(f"\n  Part3 -- representative recipes FD MC "
              f"({len(rep_rows)} reps x {args.n_mc_per_rep} variations each)")
        run_part3_representative_mc(rep_rows, space, thresholds, var_spec,
                                      n_var=args.n_mc_per_rep,
                                      out_path=p3_path,
                                      seed=args.seed + 1)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
