"""Phase 2B Part C — Stage 5C small-pitch hypothesis study.

NOT a calibration. Tests whether weakening / removing the quencher and/or
lowering sigma re-opens the pitch ∈ {18, 20} process window.

Sweep: pitch × dose × sigma × DH × time × quencher_mode.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reaction_diffusion_peb_v2_high_na.calibration._common import (  # noqa: E402
    apply_xy_overrides, classify,
)
from reaction_diffusion_peb_v2_high_na.experiments.run_sigma_sweep_helpers import (  # noqa: E402
    run_one_with_overrides,
)

V2_DIR = Path(__file__).resolve().parents[3]
DEFAULT_OUT = V2_DIR / "outputs"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--tag", type=str, default="cal05_smallpitch")
    args = p.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    fig_dir = DEFAULT_OUT / "figures" / args.tag
    fig_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = DEFAULT_OUT / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    pitches = cfg["sweeps"]["pitch_nm"]
    doses = cfg["sweeps"]["dose_mJ_cm2"]
    sigmas = cfg["sweeps"]["electron_blur_sigma_nm"]
    DHs = cfg["sweeps"]["DH_nm2_s"]
    times = cfg["sweeps"]["time_s"]
    qmodes = cfg["sweeps"]["quencher_modes"]

    rows: list[dict] = []
    print(f"=== Phase 2B Part C — Stage 5C small-pitch hypothesis study ===")
    n_total = len(pitches) * len(doses) * len(sigmas) * len(DHs) * len(times) * len(qmodes)
    print(f"  pitch × dose × σ × DH × t × q_mode = {n_total} runs")

    for pitch in pitches:
        for dose in doses:
            for sigma in sigmas:
                for DH in DHs:
                    for t in times:
                        for q_name, q_dict in qmodes.items():
                            cfg_run = apply_xy_overrides(cfg, {
                                "pitch_nm": pitch, "dose_mJ_cm2": dose,
                                "electron_blur_sigma_nm": sigma,
                                "DH_nm2_s": DH, "time_s": t,
                            })
                            cfg_run["quencher"]["enabled"] = bool(q_dict["enabled"])
                            cfg_run["quencher"]["Q0_mol_dm3"] = float(q_dict["Q0_mol_dm3"])
                            cfg_run["quencher"]["kq_s_inv"] = float(q_dict["kq_s_inv"])

                            r = run_one_with_overrides(
                                cfg_run,
                                sigma_nm=cfg_run["exposure"]["electron_blur_sigma_nm"],
                                time_s=cfg_run["peb"]["time_s"],
                                DH_nm2_s=cfg_run["peb"]["DH_nm2_s"],
                                kdep_s_inv=cfg_run["peb"]["kdep_s_inv"],
                                Hmax_mol_dm3=cfg_run["exposure"]["Hmax_mol_dm3"],
                                quencher_enabled=cfg_run["quencher"]["enabled"],
                                Q0_mol_dm3=cfg_run["quencher"]["Q0_mol_dm3"],
                                DQ_nm2_s=cfg_run["quencher"]["DQ_nm2_s"],
                                kq_s_inv=cfg_run["quencher"]["kq_s_inv"],
                            )
                            r["status"] = classify(r)
                            r["pitch_nm"] = float(pitch)
                            r["dose_mJ_cm2"] = float(dose)
                            r["sigma_nm"] = float(sigma)
                            r["DH"] = float(DH)
                            r["time_s_set"] = float(t)
                            r["quencher_mode"] = q_name
                            rows.append(r)

    # Status counts grouped by (pitch, sigma, q_mode).
    print("\nStatus counts per (pitch, σ, quencher):")
    print("  pitch  σ   q_mode    unstable  merged  under  low_c  valid  robust")
    counts_summary = {}
    for pitch in pitches:
        for sigma in sigmas:
            for q_name in qmodes:
                subset = [r for r in rows
                          if r["pitch_nm"] == pitch
                          and r["sigma_nm"] == sigma
                          and r["quencher_mode"] == q_name]
                c = {s: 0 for s in
                     ["unstable", "merged", "under_exposed", "low_contrast", "valid", "robust_valid"]}
                for r in subset:
                    c[r["status"]] = c.get(r["status"], 0) + 1
                counts_summary[(pitch, sigma, q_name)] = (c, len(subset))
                print(f"  {pitch:>4}  {sigma:>2}   {q_name:<5}     "
                      f"{c['unstable']:>5}    {c['merged']:>5}   {c['under_exposed']:>5}   "
                      f"{c['low_contrast']:>5}  {c['valid']:>5}   {c['robust_valid']:>5}    "
                      f"  (of {len(subset)})")

    # Best robust_valid (or valid) candidate per pitch.
    print("\nBest cell per pitch (robust_valid > valid; min |CD_shift|; max margin):")
    best_per_pitch = {}
    for pitch in pitches:
        cands = [r for r in rows if r["pitch_nm"] == pitch
                 and r["status"] in ("robust_valid", "valid")]
        if not cands:
            best_per_pitch[pitch] = None
            print(f"  pitch={pitch}: no robust_valid or valid cell.")
            continue
        cands.sort(key=lambda r: (
            0 if r["status"] == "robust_valid" else 1,
            abs(r["CD_shift_nm"]) if np.isfinite(r["CD_shift_nm"]) else 1e9,
            -r["P_line_margin"],
        ))
        b = cands[0]
        best_per_pitch[pitch] = b
        print(f"  pitch={pitch}: dose={b['dose_mJ_cm2']:.1f}, σ={b['sigma_nm']}, "
              f"DH={b['DH']}, t={b['time_s_set']}, q={b['quencher_mode']} "
              f"→ status={b['status']}, CD_shift={b['CD_shift_nm']:+.2f}, "
              f"LER_lock={b['LER_CD_locked_nm']:.3f}, margin={b['P_line_margin']:+.3f}")

    # Save CSV.
    keys = ["pitch_nm", "dose_mJ_cm2", "sigma_nm", "DH", "time_s_set", "quencher_mode",
            "status",
            "CD_initial_nm", "CD_final_nm", "CD_shift_nm",
            "CD_locked_nm", "LER_CD_locked_nm", "P_threshold_locked",
            "P_line_center_mean", "P_space_center_mean", "P_line_margin",
            "contrast", "area_frac", "CD_pitch_frac",
            "psd_locked_mid"]
    with (logs_dir / f"{args.tag}_summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in keys})

    # Save count summary JSON.
    serialisable = {f"{p}_{s}_{q}": {"counts": c, "total": n}
                     for (p, s, q), (c, n) in counts_summary.items()}
    (logs_dir / f"{args.tag}_status_counts.json").write_text(json.dumps(serialisable, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
