"""Stage 06M -- targeted time_s deep Monte Carlo for G_4867.

Stage 06K identified time_s as the dominant fragile knob around the
Mode A default recipe G_4867 -- a +/-5 s OAT sweep dropped FD MC
strict_pass_prob from 0.72 to 0.12. Stage 06M quantifies that more
densely with FD MC at every integer-second offset and a complementary
Gaussian time-smearing scenario.

Two scenarios:
    1. Deterministic time offsets:
       time_s = nominal + [-5, -4, -3, -2, -1, 0, +1, +2, +3, +4, +5]
       For each cell, 100 FD variations on the OTHER knobs at the
       fixed time offset. 11 cells x 100 FD runs = 1,100 FD runs.

    2. Gaussian time-smearing scenario:
       time_s ~ Normal(nominal, sigma_time = 2 s) and the standard
       process_variation YAML widths on the other knobs. 300 FD runs.
       (Surrogate-only sigma_time = 1 / 3 results live in
        analyze_stage06m.py since they are cheap.)

Variation widths for the OTHER knobs come straight from the existing
configs/yield_optimization.yaml `process_variation` block: dose +/-5%,
sigma +/-0.2 nm, DH +/-10%, Hmax +/-5%, Q0 +/-10%, line_cd +/-0.5 nm
via line_cd_ratio. time_s is NOT applied as random variation in
scenario 1 -- we explicitly overwrite time_s to (nominal + offset)
after sample_variations() returns, so there is no double-counting.

Policy
    v2_OP_frozen           = true
    published_data_loaded  = false
    No external calibration. Closed Stage 04C / 04D / 06C / 06H joblibs
    are read-only. New FD rows are written to a Stage 06M-only CSV.
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
)
from reaction_diffusion_peb_v3_screening.src.process_variation import (
    VariationSpec,
    sample_variations,
)


V3_DIR = ROOT / "reaction_diffusion_peb_v3_screening"


EXTRA_COLS = [
    "source_recipe_id",
    "scenario",        # "det_offset" | "gaussian_time"
    "time_offset_s",   # for det_offset: deterministic offset (s);
                        # for gaussian: realised time_s - nominal time_s
    "sigma_time_s",    # for gaussian only; 0 for deterministic
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


def _write_fd_row(writer, row: dict, *, source_recipe_id: str,
                    scenario: str, time_offset_s: float,
                    sigma_time_s: float, variation_idx: int) -> None:
    out = dict(row)
    out["source_recipe_id"] = source_recipe_id
    out["scenario"]         = scenario
    out["time_offset_s"]    = float(time_offset_s)
    out["sigma_time_s"]     = float(sigma_time_s)
    out["variation_idx"]    = int(variation_idx)
    writer.writerow(out)


def _base_candidate(base_params: dict, space: CandidateSpace,
                      recipe_id: str) -> dict:
    out = {k: float(v) for k, v in base_params.items()}
    out["pitch_nm"]    = float(out["pitch_nm"])
    out["line_cd_nm"]  = out["pitch_nm"] * out["line_cd_ratio"]
    out["domain_x_nm"] = out["pitch_nm"] * 5.0
    out["dose_norm"]   = out["dose_mJ_cm2"] / float(space.fixed["reference_dose_mJ_cm2"])
    for fk, fv in space.fixed.items():
        out.setdefault(fk, fv)
    out["_id"] = recipe_id
    return out


def _override_time_and_clip(v: dict, new_time: float,
                              space: CandidateSpace) -> None:
    """Clamp `new_time` into the candidate-space time bound and write
    it onto the variation; refresh derived fields (dose_norm,
    domain_x_nm, line_cd_nm)."""
    bounds = {p["name"]: (float(p["low"]), float(p["high"]))
                for p in space.parameters if p["type"] == "uniform"}
    lo, hi = bounds.get("time_s", (-np.inf, np.inf))
    v["time_s"] = float(np.clip(new_time, lo, hi))
    # Other derived fields recomputed for safety; sample_variations
    # already maintains them but we just rewrote a knob, so be defensive.
    v["domain_x_nm"] = float(v["pitch_nm"]) * 5.0
    v["dose_norm"]   = float(v["dose_mJ_cm2"]) / float(space.fixed["reference_dose_mJ_cm2"])


# --------------------------------------------------------------------------
# Scenario 1 -- deterministic time offsets.
# --------------------------------------------------------------------------
def run_deterministic_offsets(base_params: dict,
                                  offsets: list[float],
                                  space: CandidateSpace,
                                  thresholds: LabelThresholds,
                                  var_spec: VariationSpec,
                                  n_var_per_cell: int,
                                  seed: int,
                                  recipe_id: str,
                                  writer,
                                  progress_every: int = 50) -> int:
    base_cand = _base_candidate(base_params, space, recipe_id)
    base_time = float(base_cand["time_s"])
    base_rng = np.random.default_rng(seed)
    total = 0; t0 = time.time()
    for cell_idx, off in enumerate(offsets, start=1):
        sub_rng = np.random.default_rng(int(base_rng.integers(0, 2**31 - 1)))
        variations = sample_variations(base_cand, var_spec, n_var_per_cell,
                                          space, rng=sub_rng)
        target_time = base_time + float(off)
        for j, v in enumerate(variations, start=1):
            _override_time_and_clip(v, target_time, space)
            v["_id"] = f"{recipe_id}__det_t{int(off):+02d}_mc{j:03d}"
            for fk, fv in space.fixed.items():
                v.setdefault(fk, fv)
            fd_row = run_one_candidate(v, thresholds)
            _write_fd_row(
                writer, fd_row,
                source_recipe_id=recipe_id,
                scenario="det_offset",
                time_offset_s=float(off),
                sigma_time_s=0.0,
                variation_idx=j,
            )
            total += 1
            if total % progress_every == 0:
                rate = total / max(time.time() - t0, 1e-9)
                eta = (len(offsets) * n_var_per_cell - total) / max(rate, 1e-9)
                print(f"  det_offset: {total}/{len(offsets) * n_var_per_cell} done  "
                      f"({rate:.2f} runs/s, ETA {eta:.0f}s)")
        print(f"  det_offset: cell {cell_idx}/{len(offsets)} (offset = {off:+.0f} s, "
              f"target time = {target_time:.2f} s) done")
    print(f"  scenario 1 (deterministic) -> {total} FD runs in {time.time() - t0:.1f}s")
    return total


# --------------------------------------------------------------------------
# Scenario 2 -- Gaussian time-smearing FD.
# --------------------------------------------------------------------------
def run_gaussian_time_fd(base_params: dict,
                            sigma_time_s: float,
                            n_var: int,
                            space: CandidateSpace,
                            thresholds: LabelThresholds,
                            var_spec: VariationSpec,
                            seed: int,
                            recipe_id: str,
                            writer,
                            progress_every: int = 50) -> int:
    base_cand = _base_candidate(base_params, space, recipe_id)
    base_time = float(base_cand["time_s"])
    rng = np.random.default_rng(seed)
    variations = sample_variations(base_cand, var_spec, n_var, space, rng=rng)
    total = 0; t0 = time.time()
    for j, v in enumerate(variations, start=1):
        # Replace whatever time was sampled by the YAML ±2s rule with a
        # Gaussian draw centred on base_time and clipped to bounds.
        new_time = float(base_time + rng.normal(0.0, float(sigma_time_s)))
        _override_time_and_clip(v, new_time, space)
        v["_id"] = f"{recipe_id}__gauss_s{int(sigma_time_s)}_mc{j:03d}"
        for fk, fv in space.fixed.items():
            v.setdefault(fk, fv)
        fd_row = run_one_candidate(v, thresholds)
        _write_fd_row(
            writer, fd_row,
            source_recipe_id=recipe_id,
            scenario="gaussian_time",
            time_offset_s=float(v["time_s"] - base_time),
            sigma_time_s=float(sigma_time_s),
            variation_idx=j,
        )
        total += 1
        if total % progress_every == 0:
            rate = total / max(time.time() - t0, 1e-9)
            eta = (n_var - total) / max(rate, 1e-9)
            print(f"  gaussian: {total}/{n_var} done  "
                  f"({rate:.2f} runs/s, ETA {eta:.0f}s)")
    print(f"  scenario 2 (gaussian sigma_t = {sigma_time_s}s) -> "
          f"{total} FD runs in {time.time() - t0:.1f}s")
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
    p.add_argument("--recipe_id", type=str, default="G_4867",
                   help="Read parameters from this manifest entry.")
    p.add_argument("--offsets", type=str,
                   default="-5,-4,-3,-2,-1,0,1,2,3,4,5",
                   help="Comma-separated deterministic time_s offsets (s).")
    p.add_argument("--n_var_per_cell", type=int, default=100)
    p.add_argument("--gauss_sigma_time_s", type=float, default=2.0)
    p.add_argument("--gauss_n_var", type=int, default=300,
                   help="0 to skip the Gaussian FD scenario.")
    p.add_argument("--seed_det",   type=int, default=9191)
    p.add_argument("--seed_gauss", type=int, default=9292)
    p.add_argument("--label_schema",
                   default=str(V3_DIR / "configs" / "label_schema.yaml"))
    p.add_argument("--out_csv", type=str,
                   default=str(V3_DIR / "outputs" / "yield_optimization"
                               / "stage06M_time_deep_mc.csv"))
    p.add_argument("--skip_det",   action="store_true")
    p.add_argument("--skip_gauss", action="store_true")
    args = p.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    space = CandidateSpace.from_yaml(cfg["candidate_space_yaml"])
    var_spec = VariationSpec.from_yaml_dict(cfg["process_variation"])
    thresholds = LabelThresholds.from_yaml(args.label_schema)

    manifest = yaml.safe_load(Path(args.manifest_yaml).read_text())
    base = next((r for r in manifest["representatives"]
                   if r["recipe_id"] == args.recipe_id), None)
    if base is None:
        raise SystemExit(f"recipe {args.recipe_id} not found in manifest "
                         f"{args.manifest_yaml}")
    base_params = {k: float(v) for k, v in base["parameters"].items()}

    offsets = [float(s) for s in args.offsets.split(",") if s.strip()]
    print(f"Stage 06M -- targeted time_s deep MC for {args.recipe_id}")
    print(f"  policy: v2_OP_frozen={cfg['policy']['v2_OP_frozen']}, "
          f"published_data_loaded={cfg['policy']['published_data_loaded']}")
    print(f"  base parameters:")
    for k, v in base_params.items():
        print(f"    {k:<14} = {v}")
    print(f"  scenario 1: {len(offsets)} deterministic offsets x "
          f"{args.n_var_per_cell} FD = "
          f"{len(offsets) * args.n_var_per_cell} runs")
    if not args.skip_gauss and args.gauss_n_var > 0:
        print(f"  scenario 2: gaussian sigma_t = {args.gauss_sigma_time_s} s "
              f"x {args.gauss_n_var} FD")

    out_path = Path(args.out_csv)
    f, writer = _open_csv(out_path)
    try:
        total = 0
        if not args.skip_det:
            total += run_deterministic_offsets(
                base_params, offsets, space, thresholds, var_spec,
                n_var_per_cell=args.n_var_per_cell,
                seed=args.seed_det,
                recipe_id=args.recipe_id,
                writer=writer,
            )
        if not args.skip_gauss and args.gauss_n_var > 0:
            total += run_gaussian_time_fd(
                base_params, args.gauss_sigma_time_s, args.gauss_n_var,
                space, thresholds, var_spec,
                seed=args.seed_gauss,
                recipe_id=args.recipe_id,
                writer=writer,
            )
        print(f"\n  total FD runs: {total} -> {out_path}")
    finally:
        f.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
