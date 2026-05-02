"""Stage 06P-B -- targeted AL FD around G_4867 extreme time offsets.

Generates ~700 FD runs concentrated on the offsets where the 06R
surrogate still over-predicts G_4867 strict_pass_prob (06Q blindspot).
Three sub-phases:

  A. time_densification     (350 FD runs)
     7 deterministic time offsets x 50 chemistry jitter each.
     Standard process_variation widths from
     configs/yield_optimization.yaml.

  B. residual_max_jitter    (200 FD runs)
     4 offsets with the largest 06R residual (defaults to -4, -3, +4,
     +5) x 50 wider chemistry jitter (2x relative widths, 1.5x
     absolute widths) to densify the failure-mode boundary.

  C. boundary_candidates    (~150 FD runs)
     4 intermediate offsets (-2, -1, +1, +2) x ~38 standard chemistry
     jitter each. These cover the strict_pass_prob in [0.3, 0.7]
     boundary band where 06R's surrogate proxy is most ambiguous.

All FD rows are tagged with:
    source            = stage06PB_G4867_targeted_AL
    source_recipe_id  = G_4867
    sub_phase         = time_densification | residual_max_jitter
                          | boundary_candidates
    scenario          = det_offset
    time_offset_s     = the deterministic offset (s)

Closed Stage 04C / 04D / 06C / 06L / 06P / 06R artefacts are not
mutated. New FD rows live under outputs/labels/stage06PB_targeted_fd_rows.csv.

Policy
    v2_OP_frozen           = true
    published_data_loaded  = false
    No external calibration.
"""
from __future__ import annotations

import argparse
import copy
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
)
from reaction_diffusion_peb_v3_screening.src.process_variation import (
    VariationSpec,
    sample_variations,
)


V3_DIR = ROOT / "reaction_diffusion_peb_v3_screening"


EXTRA_COLS = [
    "source",
    "source_recipe_id",
    "sub_phase",       # time_densification | residual_max_jitter | boundary_candidates
    "scenario",        # always det_offset for 06PB
    "time_offset_s",
    "variation_idx",
]
WRITE_COLS = LABEL_CSV_COLUMNS + EXTRA_COLS


def _safe_float(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def _open_csv(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    f = path.open("w", newline="")
    w = csv.DictWriter(f, fieldnames=WRITE_COLS, extrasaction="ignore")
    w.writeheader()
    return f, w


def _write_fd_row(writer, row: dict, *,
                    source: str,
                    source_recipe_id: str,
                    sub_phase: str,
                    time_offset_s: float,
                    variation_idx: int) -> None:
    out = dict(row)
    out["source"]          = source
    out["source_recipe_id"] = source_recipe_id
    out["sub_phase"]       = sub_phase
    out["scenario"]        = "det_offset"
    out["time_offset_s"]   = float(time_offset_s)
    out["variation_idx"]   = int(variation_idx)
    writer.writerow(out)


def _base_candidate(base_params: dict, space: CandidateSpace,
                      recipe_id: str) -> dict:
    out = {k: float(v) for k, v in base_params.items()}
    out["pitch_nm"]    = float(out["pitch_nm"])
    out["line_cd_nm"]  = out["pitch_nm"] * out["line_cd_ratio"]
    out["domain_x_nm"] = out["pitch_nm"] * 5.0
    out["dose_norm"]   = (out["dose_mJ_cm2"]
                            / float(space.fixed["reference_dose_mJ_cm2"]))
    for fk, fv in space.fixed.items():
        out.setdefault(fk, fv)
    out["_id"] = recipe_id
    return out


def _override_time_and_clip(v: dict, new_time: float,
                              space: CandidateSpace) -> None:
    bounds = {p["name"]: (float(p["low"]), float(p["high"]))
                for p in space.parameters if p["type"] == "uniform"}
    lo, hi = bounds.get("time_s", (-np.inf, np.inf))
    v["time_s"] = float(np.clip(new_time, lo, hi))
    v["domain_x_nm"] = float(v["pitch_nm"]) * 5.0
    v["dose_norm"]   = (float(v["dose_mJ_cm2"])
                          / float(space.fixed["reference_dose_mJ_cm2"]))


def _wider_var_spec(var_spec_yaml: dict,
                      relative_scale: float = 2.0,
                      absolute_scale: float = 1.5) -> VariationSpec:
    """Build a VariationSpec where each knob's width is scaled. Used
    only for the residual_max_jitter sub-phase to densify the failure
    boundary."""
    wide = copy.deepcopy(var_spec_yaml)
    for k in wide.get("knobs", []):
        if "width" in k:
            scale = absolute_scale if k.get("absolute") else relative_scale
            k["width"] = float(k["width"]) * float(scale)
    if "line_cd_abs_nm" in wide:
        wide["line_cd_abs_nm"] = float(wide["line_cd_abs_nm"]) * absolute_scale
    return VariationSpec.from_yaml_dict(wide)


# --------------------------------------------------------------------------
# Sub-phase runners
# --------------------------------------------------------------------------
def run_time_cell_densification(base_params: dict,
                                     offsets: list[float],
                                     n_var_per_cell: int,
                                     space: CandidateSpace,
                                     thresholds: LabelThresholds,
                                     var_spec: VariationSpec,
                                     seed: int,
                                     recipe_id: str,
                                     writer,
                                     source_tag: str,
                                     progress_every: int = 50) -> int:
    base_cand = _base_candidate(base_params, space, recipe_id)
    base_time = float(base_cand["time_s"])
    base_rng = np.random.default_rng(seed)
    total = 0; t0 = time.time()
    for off in offsets:
        sub_rng = np.random.default_rng(int(base_rng.integers(0, 2**31 - 1)))
        variations = sample_variations(base_cand, var_spec, n_var_per_cell,
                                          space, rng=sub_rng)
        target_time = base_time + float(off)
        for j, v in enumerate(variations, start=1):
            _override_time_and_clip(v, target_time, space)
            v["_id"] = f"{recipe_id}__pb_t{int(round(off)):+02d}_mc{j:03d}"
            for fk, fv in space.fixed.items():
                v.setdefault(fk, fv)
            fd_row = run_one_candidate(v, thresholds)
            _write_fd_row(writer, fd_row,
                            source=source_tag,
                            source_recipe_id=recipe_id,
                            sub_phase="time_densification",
                            time_offset_s=float(off),
                            variation_idx=j)
            total += 1
            if total % progress_every == 0:
                rate = total / max(time.time() - t0, 1e-9)
                eta = (len(offsets) * n_var_per_cell - total) / max(rate, 1e-9)
                print(f"  time_densification: {total}/"
                      f"{len(offsets) * n_var_per_cell} done  "
                      f"({rate:.2f} runs/s, ETA {eta:.0f}s)")
    print(f"  -> time_densification: {total} FD runs in {time.time() - t0:.1f}s")
    return total


def run_residual_max_jitter(base_params: dict,
                                  offsets: list[float],
                                  n_var_per_cell: int,
                                  space: CandidateSpace,
                                  thresholds: LabelThresholds,
                                  var_spec_wide: VariationSpec,
                                  seed: int,
                                  recipe_id: str,
                                  writer,
                                  source_tag: str) -> int:
    base_cand = _base_candidate(base_params, space, recipe_id)
    base_time = float(base_cand["time_s"])
    base_rng = np.random.default_rng(seed)
    total = 0; t0 = time.time()
    for off in offsets:
        sub_rng = np.random.default_rng(int(base_rng.integers(0, 2**31 - 1)))
        variations = sample_variations(base_cand, var_spec_wide,
                                          n_var_per_cell, space, rng=sub_rng)
        target_time = base_time + float(off)
        for j, v in enumerate(variations, start=1):
            _override_time_and_clip(v, target_time, space)
            v["_id"] = f"{recipe_id}__pb_wide_t{int(round(off)):+02d}_mc{j:03d}"
            for fk, fv in space.fixed.items():
                v.setdefault(fk, fv)
            fd_row = run_one_candidate(v, thresholds)
            _write_fd_row(writer, fd_row,
                            source=source_tag,
                            source_recipe_id=recipe_id,
                            sub_phase="residual_max_jitter",
                            time_offset_s=float(off),
                            variation_idx=j)
            total += 1
    print(f"  -> residual_max_jitter: {total} FD runs in "
          f"{time.time() - t0:.1f}s")
    return total


def run_boundary_candidates(base_params: dict,
                                 offsets: list[float],
                                 n_var_per_cell: int,
                                 space: CandidateSpace,
                                 thresholds: LabelThresholds,
                                 var_spec: VariationSpec,
                                 seed: int,
                                 recipe_id: str,
                                 writer,
                                 source_tag: str) -> int:
    base_cand = _base_candidate(base_params, space, recipe_id)
    base_time = float(base_cand["time_s"])
    base_rng = np.random.default_rng(seed)
    total = 0; t0 = time.time()
    for off in offsets:
        sub_rng = np.random.default_rng(int(base_rng.integers(0, 2**31 - 1)))
        variations = sample_variations(base_cand, var_spec, n_var_per_cell,
                                          space, rng=sub_rng)
        target_time = base_time + float(off)
        for j, v in enumerate(variations, start=1):
            _override_time_and_clip(v, target_time, space)
            v["_id"] = f"{recipe_id}__pb_bnd_t{int(round(off)):+02d}_mc{j:03d}"
            for fk, fv in space.fixed.items():
                v.setdefault(fk, fv)
            fd_row = run_one_candidate(v, thresholds)
            _write_fd_row(writer, fd_row,
                            source=source_tag,
                            source_recipe_id=recipe_id,
                            sub_phase="boundary_candidates",
                            time_offset_s=float(off),
                            variation_idx=j)
            total += 1
    print(f"  -> boundary_candidates: {total} FD runs in "
          f"{time.time() - t0:.1f}s")
    return total


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str,
                   default=str(V3_DIR / "configs" / "yield_optimization.yaml"))
    p.add_argument("--manifest_yaml", type=str,
                   default=str(V3_DIR / "outputs" / "yield_optimization"
                               / "stage06I_mode_a_final_recipes.yaml"))
    p.add_argument("--recipe_id", type=str, default="G_4867")
    p.add_argument("--time_densification_offsets", type=str,
                   default="-5,-4,-3,2,3,4,5",
                   help="Time offsets (s) for sub-phase A.")
    p.add_argument("--n_var_time_cell", type=int, default=50)
    p.add_argument("--residual_max_offsets", type=str,
                   default="-4,-3,4,5",
                   help="Top-residual offsets for sub-phase B.")
    p.add_argument("--n_var_residual_max", type=int, default=50)
    p.add_argument("--boundary_offsets", type=str,
                   default="-2,-1,1,2",
                   help="Intermediate offsets for sub-phase C.")
    p.add_argument("--n_var_boundary", type=int, default=38)
    p.add_argument("--seed_time",       type=int, default=11111)
    p.add_argument("--seed_residual",   type=int, default=22222)
    p.add_argument("--seed_boundary",   type=int, default=33333)
    p.add_argument("--label_schema",
                   default=str(V3_DIR / "configs" / "label_schema.yaml"))
    p.add_argument("--out_csv", type=str,
                   default=str(V3_DIR / "outputs" / "labels"
                               / "stage06PB_targeted_fd_rows.csv"))
    p.add_argument("--source_tag", type=str,
                   default="stage06PB_G4867_targeted_AL")
    p.add_argument("--skip_a", action="store_true")
    p.add_argument("--skip_b", action="store_true")
    p.add_argument("--skip_c", action="store_true")
    args = p.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    space = CandidateSpace.from_yaml(cfg["candidate_space_yaml"])
    var_spec = VariationSpec.from_yaml_dict(cfg["process_variation"])
    var_spec_wide = _wider_var_spec(cfg["process_variation"])
    thresholds = LabelThresholds.from_yaml(args.label_schema)

    manifest = yaml.safe_load(Path(args.manifest_yaml).read_text())
    base = next((r for r in manifest["representatives"]
                   if r["recipe_id"] == args.recipe_id), None)
    if base is None:
        raise SystemExit(f"recipe {args.recipe_id} not found in manifest "
                          f"{args.manifest_yaml}")
    base_params = {k: float(v) for k, v in base["parameters"].items()}

    off_a = [float(s) for s in args.time_densification_offsets.split(",") if s.strip()]
    off_b = [float(s) for s in args.residual_max_offsets.split(",")        if s.strip()]
    off_c = [float(s) for s in args.boundary_offsets.split(",")            if s.strip()]
    print(f"Stage 06P-B -- targeted AL FD for {args.recipe_id}")
    print(f"  policy: v2_OP_frozen={cfg['policy']['v2_OP_frozen']}, "
          f"published_data_loaded={cfg['policy']['published_data_loaded']}")
    print(f"  base parameters:")
    for k, v in base_params.items():
        print(f"    {k:<14} = {v}")
    print(f"  sub-phase A (time_densification):  "
          f"{len(off_a)} offsets x {args.n_var_time_cell}")
    print(f"  sub-phase B (residual_max_jitter): "
          f"{len(off_b)} offsets x {args.n_var_residual_max}  (wider widths)")
    print(f"  sub-phase C (boundary_candidates): "
          f"{len(off_c)} offsets x {args.n_var_boundary}")
    total_planned = (
        (len(off_a) * args.n_var_time_cell    if not args.skip_a else 0)
        + (len(off_b) * args.n_var_residual_max if not args.skip_b else 0)
        + (len(off_c) * args.n_var_boundary     if not args.skip_c else 0)
    )
    print(f"  total planned FD runs: {total_planned}")

    out_path = Path(args.out_csv)
    f, writer = _open_csv(out_path)
    grand_total = 0
    try:
        if not args.skip_a:
            grand_total += run_time_cell_densification(
                base_params, off_a, args.n_var_time_cell, space, thresholds,
                var_spec, seed=args.seed_time, recipe_id=args.recipe_id,
                writer=writer, source_tag=args.source_tag,
            )
        if not args.skip_b:
            grand_total += run_residual_max_jitter(
                base_params, off_b, args.n_var_residual_max, space, thresholds,
                var_spec_wide, seed=args.seed_residual,
                recipe_id=args.recipe_id, writer=writer,
                source_tag=args.source_tag,
            )
        if not args.skip_c:
            grand_total += run_boundary_candidates(
                base_params, off_c, args.n_var_boundary, space, thresholds,
                var_spec, seed=args.seed_boundary, recipe_id=args.recipe_id,
                writer=writer, source_tag=args.source_tag,
            )
    finally:
        f.close()
    print(f"  total FD runs written: {grand_total}")
    print(f"  -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
