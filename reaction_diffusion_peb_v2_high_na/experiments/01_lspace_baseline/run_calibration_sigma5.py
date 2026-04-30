"""σ=5 calibration: find an exposure/PEB budget where the High-NA-realistic
electron blur (σ=5 nm) produces a line-space pattern that satisfies the
interior gate at pitch=24 nm / CD=12.5 nm.

Search order (cheapest → most invasive):
    Stage A:  time_s × DH_nm2_s grid at kdep=0.5, Hmax=0.2
              time_s    ∈ {10, 15, 20, 30}
              DH_nm2_s  ∈ {0.3, 0.8}
    Stage B (only if A finds none):  Hmax sweep at the highest-contrast (t, DH)
              Hmax_mol_dm3 ∈ {0.1, 0.15, 0.2}

Gate (same as run_sigma_sweep.py):
    P_space_center_mean < 0.50
    P_line_center_mean  > 0.65
    contrast            > 0.15
    area_frac           < 0.90
    CD_final / pitch    < 0.85
    CD_final, LER_final finite
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reaction_diffusion_peb_v2_high_na.experiments.run_sigma_sweep_helpers import (  # noqa: E402
    print_table,
    run_one_with_overrides,
    save_outputs,
)

V2_DIR = Path(__file__).resolve().parents[2]
DEFAULT_OUT = V2_DIR / "outputs"


SIGMA = 5.0


def _passing(rows):
    return [r for r in rows if r["passed"]]


def _summary(r):
    return {k: r[k] for k in [
        "sigma_nm", "time_s", "DH_nm2_s", "Hmax_mol_dm3",
        "P_space_center_mean", "P_line_center_mean",
        "contrast", "area_frac",
        "CD_initial_nm", "CD_final_nm", "CD_shift_nm", "CD_pitch_frac",
        "LER_initial_nm", "LER_final_nm",
    ]}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--tag", type=str, default="01_calibration_sigma5")
    args = p.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())

    fig_dir_a = DEFAULT_OUT / "figures" / f"{args.tag}_stageA"
    fig_dir_b = DEFAULT_OUT / "figures" / f"{args.tag}_stageB"
    logs_dir = DEFAULT_OUT / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    # ---- Stage A: time × DH grid ----
    times = [10.0, 15.0, 20.0, 30.0]
    DHs = [0.3, 0.8]
    print(f"=== Stage A: σ=5 budget grid (time × DH) at kdep=0.5, Hmax={cfg['exposure']['Hmax_mol_dm3']} ===")
    rows_a = []
    for DH in DHs:
        for t in times:
            r = run_one_with_overrides(cfg, sigma_nm=SIGMA, time_s=t,
                                       DH_nm2_s=DH, kdep_s_inv=0.5,
                                       Hmax_mol_dm3=cfg["exposure"]["Hmax_mol_dm3"])
            rows_a.append(r)
    print_table(rows_a, extra_keys=("DH_nm2_s",))
    save_outputs(rows_a, cfg, fig_dir_a, logs_dir / f"{args.tag}_stageA.csv")

    passed_a = _passing(rows_a)
    if passed_a:
        # Pick: smallest area_frac among passers ≈ best margin.
        chosen = min(passed_a, key=lambda r: r["area_frac"])
        print(f"\nRECOMMENDED (Stage A; best margin): σ=5, time={chosen['time_s']:.0f} s, DH={chosen['DH_nm2_s']}")
        print(json.dumps(_summary(chosen), indent=2))
        return 0

    # ---- Stage B: Hmax sweep at best Stage-A point (highest contrast) ----
    base = max(rows_a, key=lambda r: r["contrast"])
    print(f"\nStage A: no pass. Falling back to Hmax sweep at "
          f"time={base['time_s']:.0f}, DH={base['DH_nm2_s']} (highest contrast in A).")
    Hmaxes = [0.1, 0.15, 0.2]
    rows_b = []
    for H in Hmaxes:
        r = run_one_with_overrides(cfg, sigma_nm=SIGMA, time_s=base["time_s"],
                                   DH_nm2_s=base["DH_nm2_s"], kdep_s_inv=0.5,
                                   Hmax_mol_dm3=H)
        rows_b.append(r)
    print(f"=== Stage B: Hmax sweep (σ=5, time={base['time_s']:.0f}, DH={base['DH_nm2_s']}, kdep=0.5) ===")
    print_table(rows_b, extra_keys=("DH_nm2_s", "Hmax_mol_dm3"))
    save_outputs(rows_b, cfg, fig_dir_b, logs_dir / f"{args.tag}_stageB.csv")

    passed_b = _passing(rows_b)
    if passed_b:
        chosen = max(passed_b, key=lambda r: r["Hmax_mol_dm3"])  # highest viable Hmax
        print(f"\nRECOMMENDED (Stage B): σ=5, time={chosen['time_s']:.0f} s, "
              f"DH={chosen['DH_nm2_s']}, Hmax={chosen['Hmax_mol_dm3']}")
        print(json.dumps(_summary(chosen), indent=2))
        return 0

    print("\nNo σ=5 budget combination passed. Consider widening DH or relaxing kdep.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
