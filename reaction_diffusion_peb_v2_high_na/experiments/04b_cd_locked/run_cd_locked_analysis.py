"""Stage 4B — CD-locked LER re-analysis (and pitch-dependent quencher mini-sweep).

Block A: CD-lock comparison
    primary OP : sigma=2, Q0=0.02, kq=1.0, DH=0.5, t=30
    control OP : sigma=0, quencher off,    DH=0.5, t=30
    pitch ∈ {18, 20, 24}, dose ∈ {28.4, 40}  → 12 runs.

For each run we extract the P field once and then evaluate two metric sets:
    fixed-threshold  (P=0.5)
    CD-locked        (P_threshold bisected so CD_overall ≈ initial CD,
                      tol = 0.25 nm, P_threshold ∈ [0.2, 0.8]).

Block B: Pitch-dependent weak-quencher mini-sweep (Stage-5 follow-up)
    pitch ∈ {18, 20}, dose = Stage-5 recommendation per pitch
        (pitch=18 → 28.4, pitch=20 → 40)
    sigma=2, DH=0.5, t=30, kdep=0.5, Hmax=0.2
    Q0 ∈ {0 (baseline), 0.005, 0.01, 0.02}, kq ∈ {0.5, 1.0, 2.0}  → 20 runs
    Each run reports CD-locked LER alongside fixed-threshold LER.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reaction_diffusion_peb_v2_high_na.experiments.run_sigma_sweep_helpers import (  # noqa: E402
    run_one_with_overrides,
)
from reaction_diffusion_peb_v2_high_na.src.metrics_edge import (  # noqa: E402
    CD_LOCK_OK,
    compute_edge_band_powers,
    extract_edges,
    find_cd_lock_threshold,
    stack_lr_edges,
)

V2_DIR = Path(__file__).resolve().parents[2]
DEFAULT_OUT = V2_DIR / "outputs"

PITCHES_BLOCK_A = [18, 20, 24]
DOSES_BLOCK_A = [28.4, 40.0]

PITCH_DOSE_BLOCK_B = [(18, 28.4), (20, 40.0)]
Q0S_B = [0.005, 0.01, 0.02]
KQS_B = [0.5, 1.0, 2.0]


def cd_locked_metrics(r: dict, cd_target_nm: float) -> dict:
    """Re-evaluate edges and metrics on the same P field at a CD-locked threshold."""
    grid = r["_grid"]
    P = r["_P_final"]
    P_locked, cd_locked, status = find_cd_lock_threshold(
        P, x_nm=grid.x_nm, line_centers_nm=grid.line_centers_nm,
        pitch_nm=grid.pitch_nm, cd_target_nm=cd_target_nm,
    )
    if P_locked is None or status != CD_LOCK_OK:
        return {
            "P_threshold_locked": P_locked,
            "CD_locked_nm": cd_locked,
            "LER_CD_locked_nm": float("nan"),
            "psd_locked_low": float("nan"),
            "psd_locked_mid": float("nan"),
            "psd_locked_high": float("nan"),
            "cd_lock_status": status,
        }
    edges_locked = extract_edges(P, grid.x_nm, grid.line_centers_nm, grid.pitch_nm, P_locked)
    bp = compute_edge_band_powers(stack_lr_edges(edges_locked), dy_nm=grid.dx_nm)
    return {
        "P_threshold_locked": float(P_locked),
        "CD_locked_nm": float(cd_locked),
        "LER_CD_locked_nm": float(edges_locked.ler_mean_nm),
        "psd_locked_low": float(bp[0]),
        "psd_locked_mid": float(bp[1]),
        "psd_locked_high": float(bp[2]),
        "cd_lock_status": status,
    }


def status_at_threshold(r: dict, threshold: float, cd_value: float, area_value: float) -> str:
    """Reuse the Stage-5 classifier semantics with overridden CD/area inputs."""
    P_space = r["P_space_center_mean"]
    P_line = r["P_line_center_mean"]
    contrast = r["contrast"]
    pitch = r["_grid"].pitch_nm
    if not np.isfinite(cd_value):
        return "unstable"
    if (P_space >= 0.50 or area_value >= 0.90 or cd_value / pitch >= 0.85):
        return "merged"
    if P_line < 0.65:
        return "under_exposed"
    if contrast <= 0.15:
        return "low_contrast"
    if r["P_line_margin"] >= 0.05:
        return "robust_valid"
    return "valid"


def run_one(cfg_base: dict, sigma_nm: float, quencher_enabled: bool,
            Q0_mol_dm3: float, kq_s_inv: float, pitch_nm: float, dose_mJ_cm2: float,
            n_periods_x: int = 5) -> dict:
    cfg = copy.deepcopy(cfg_base)
    cfg["geometry"]["pitch_nm"] = float(pitch_nm)
    cfg["geometry"]["half_pitch_nm"] = 0.5 * float(pitch_nm)
    cfg["geometry"]["domain_x_nm"] = float(pitch_nm * n_periods_x)
    cfg["exposure"]["dose_mJ_cm2"] = float(dose_mJ_cm2)
    cfg["exposure"]["dose_norm"] = float(dose_mJ_cm2) / float(cfg["exposure"]["reference_dose_mJ_cm2"])
    cfg["exposure"]["electron_blur_sigma_nm"] = float(sigma_nm)
    cfg["exposure"]["electron_blur_enabled"] = sigma_nm > 0.0
    cfg["quencher"]["enabled"] = bool(quencher_enabled)
    cfg["quencher"]["Q0_mol_dm3"] = float(Q0_mol_dm3)
    cfg["quencher"]["kq_s_inv"] = float(kq_s_inv)

    r = run_one_with_overrides(
        cfg,
        sigma_nm=sigma_nm,
        time_s=cfg["peb"]["time_s"],
        DH_nm2_s=cfg["peb"]["DH_nm2_s"],
        kdep_s_inv=cfg["peb"]["kdep_s_inv"],
        Hmax_mol_dm3=cfg["exposure"]["Hmax_mol_dm3"],
        quencher_enabled=quencher_enabled,
        Q0_mol_dm3=Q0_mol_dm3,
        DQ_nm2_s=0.0,
        kq_s_inv=kq_s_inv,
    )
    r["pitch_nm"] = float(pitch_nm)
    r["dose_mJ_cm2"] = float(dose_mJ_cm2)
    return r


def annotate_with_cd_lock(r: dict) -> dict:
    """Add fixed-vs-locked metrics to a row in place; returns row."""
    cd_target = r["CD_initial_nm"]
    locked = cd_locked_metrics(r, cd_target_nm=cd_target)
    r.update(locked)
    r["LER_fixed_threshold_nm"] = r["LER_after_PEB_P_nm"]
    r["CD_fixed_nm"] = r["CD_final_nm"]
    if np.isfinite(locked["LER_CD_locked_nm"]) and np.isfinite(r["LER_fixed_threshold_nm"]):
        r["LER_delta_nm"] = float(r["LER_fixed_threshold_nm"] - locked["LER_CD_locked_nm"])
    else:
        r["LER_delta_nm"] = float("nan")

    # area at locked threshold
    P = r["_P_final"]
    grid = r["_grid"]
    if np.isfinite(locked["P_threshold_locked"]) if locked["P_threshold_locked"] is not None else False:
        area_locked = float((P >= locked["P_threshold_locked"]).sum() * grid.dx_nm ** 2)
        domain_area = float(P.shape[0] * P.shape[1]) * (grid.dx_nm ** 2)
        r["area_frac_locked"] = area_locked / domain_area
    else:
        r["area_frac_locked"] = float("nan")

    r["status_fixed"] = status_at_threshold(r, threshold=0.5,
                                             cd_value=r["CD_fixed_nm"],
                                             area_value=r["area_frac"])
    r["status_CD_locked"] = status_at_threshold(r, threshold=locked["P_threshold_locked"] or float("nan"),
                                                  cd_value=locked["CD_locked_nm"],
                                                  area_value=r["area_frac_locked"]
                                                            if np.isfinite(r["area_frac_locked"]) else 1.0)
    return r


def block_a(cfg_base: dict, fig_dir: Path) -> list[dict]:
    rows = []
    print("=== Stage 4B Block A — fixed vs CD-locked at primary and control OPs ===")
    for ops in [("primary", 2.0, True, 0.02, 1.0),
                ("control_sigma0_no_q", 0.0, False, 0.0, 0.0)]:
        op_label, sigma, qen, Q0, kq = ops
        for pitch in PITCHES_BLOCK_A:
            for dose in DOSES_BLOCK_A:
                r = run_one(cfg_base, sigma_nm=sigma, quencher_enabled=qen,
                            Q0_mol_dm3=Q0, kq_s_inv=kq, pitch_nm=pitch, dose_mJ_cm2=dose)
                r["op_label"] = op_label
                annotate_with_cd_lock(r)
                rows.append(r)
                print(
                    f"  [{op_label:<22}] pitch={pitch} dose={dose:>5.1f}  "
                    f"P_th_lock={r['P_threshold_locked'] if r['P_threshold_locked'] is not None else float('nan'):>5.3f}  "
                    f"CD: fixed={r['CD_fixed_nm']:>6.2f}  locked={r['CD_locked_nm']:>6.2f}  "
                    f"LER: fixed={r['LER_fixed_threshold_nm']:>5.2f}  locked={r['LER_CD_locked_nm']:>5.2f}  "
                    f"Δ={r['LER_delta_nm']:>+5.2f}  PSDmid: f={r['psd_eblur_mid']:.2f}→{r['psd_PEB_mid']:.2f}→L={r['psd_locked_mid']:.2f}  "
                    f"status: {r['status_fixed']:<13}|{r['status_CD_locked']:<13}  cd_lock={r['cd_lock_status']}"
                )

                # Figure: P field with both contours.
                P_field = r["_P_final"]
                grid = r["_grid"]
                fig, ax = plt.subplots(figsize=(5.5, 4.5))
                im = ax.imshow(P_field, origin="lower",
                               extent=[grid.x_nm[0], grid.x_nm[-1], grid.y_nm[0], grid.y_nm[-1]],
                               cmap="viridis", vmin=0, vmax=1, aspect="equal")
                ax.contour(grid.x_nm, grid.y_nm, P_field, levels=[0.5],
                           colors="red", linewidths=1.0, linestyles="-")
                if r["P_threshold_locked"] is not None and np.isfinite(r["P_threshold_locked"]):
                    ax.contour(grid.x_nm, grid.y_nm, P_field, levels=[r["P_threshold_locked"]],
                               colors="white", linewidths=1.0, linestyles="--")
                ax.set_xlabel("x [nm]")
                ax.set_ylabel("y [nm]")
                ax.set_title(
                    f"{op_label}  pitch={pitch}  dose={dose:.1f}\n"
                    f"red=fixed P=0.5  white=locked P={r['P_threshold_locked'] if r['P_threshold_locked'] is not None else float('nan'):.3f}  ΔLER={r['LER_delta_nm']:+.2f} nm"
                )
                fig.colorbar(im, ax=ax)
                fig.tight_layout()
                fig.savefig(fig_dir / f"P_{op_label}_pitch_{int(pitch)}_dose_{dose:.1f}.png", dpi=150)
                plt.close(fig)
    return rows


def block_b(cfg_base: dict, fig_dir: Path) -> list[dict]:
    rows = []
    print("\n=== Stage 5B (in 4B) — pitch-dependent weak-quencher mini-sweep ===")
    sigma = 2.0
    for pitch, dose in PITCH_DOSE_BLOCK_B:
        # Baseline (no quencher)
        r0 = run_one(cfg_base, sigma_nm=sigma, quencher_enabled=False,
                     Q0_mol_dm3=0.0, kq_s_inv=0.0, pitch_nm=pitch, dose_mJ_cm2=dose)
        r0["row_kind"] = "baseline"
        r0["Q0_mol_dm3"] = 0.0
        r0["kq_s_inv"] = 0.0
        annotate_with_cd_lock(r0)
        rows.append(r0)

        for Q0 in Q0S_B:
            for kq in KQS_B:
                r = run_one(cfg_base, sigma_nm=sigma, quencher_enabled=True,
                            Q0_mol_dm3=Q0, kq_s_inv=kq, pitch_nm=pitch, dose_mJ_cm2=dose)
                r["row_kind"] = "quencher"
                annotate_with_cd_lock(r)
                rows.append(r)

    # Compute per-pitch baseline LER for delta
    base_by_pd = {(r["pitch_nm"], r["dose_mJ_cm2"]): r for r in rows if r["row_kind"] == "baseline"}
    for r in rows:
        b = base_by_pd[(r["pitch_nm"], r["dose_mJ_cm2"])]
        r["dCD_shift_nm"] = float(r["CD_shift_nm"] - b["CD_shift_nm"])
        r["dLER_fixed_nm"] = float(r["LER_fixed_threshold_nm"] - b["LER_fixed_threshold_nm"])
        if np.isfinite(r["LER_CD_locked_nm"]) and np.isfinite(b["LER_CD_locked_nm"]):
            r["dLER_locked_nm"] = float(r["LER_CD_locked_nm"] - b["LER_CD_locked_nm"])
        else:
            r["dLER_locked_nm"] = float("nan")

    print()
    print("  pitch  dose  Q0       kq     CD_fix  CD_lock  LER_fix  LER_lock  ΔLER_lock  Pmin_marg  status_fix  status_lock  cd_lock")
    for r in rows:
        print(
            f"  {int(r['pitch_nm']):>5}  {r['dose_mJ_cm2']:>4.1f}  "
            f"{r['Q0_mol_dm3']:>6.4f}   {r['kq_s_inv']:>4.2f}   "
            f"{r['CD_fixed_nm']:>5.2f}   {r['CD_locked_nm']:>5.2f}    "
            f"{r['LER_fixed_threshold_nm']:>5.2f}    {r['LER_CD_locked_nm']:>5.2f}    "
            f"{r['dLER_locked_nm']:>+5.2f}      {r['P_line_margin']:>+5.3f}    "
            f"{r['status_fixed']:<13}  {r['status_CD_locked']:<13}  {r['cd_lock_status']}"
        )

    # Heatmaps per pitch.
    Q0_labels = [0.005, 0.01, 0.02]
    kq_labels = [0.5, 1.0, 2.0]
    for pitch, dose in PITCH_DOSE_BLOCK_B:
        for metric, fname, cmap, vmin, vmax in [
            ("LER_CD_locked_nm", f"pitch_{int(pitch)}_LER_locked.png", "viridis", None, None),
            ("dLER_locked_nm", f"pitch_{int(pitch)}_dLER_locked.png", "RdBu_r", None, None),
            ("dCD_shift_nm", f"pitch_{int(pitch)}_dCD_shift.png", "RdBu_r", None, None),
            ("P_line_margin", f"pitch_{int(pitch)}_P_line_margin.png", "viridis", None, None),
        ]:
            grid_arr = np.full((len(Q0_labels), len(kq_labels)), np.nan)
            for i, Q0 in enumerate(Q0_labels):
                for j, kq in enumerate(kq_labels):
                    rs = [r for r in rows
                          if r["pitch_nm"] == pitch and r["dose_mJ_cm2"] == dose
                          and r["row_kind"] == "quencher"
                          and abs(r["Q0_mol_dm3"] - Q0) < 1e-9
                          and abs(r["kq_s_inv"] - kq) < 1e-9]
                    if rs:
                        v = rs[0][metric]
                        if v is not None and (not isinstance(v, float) or np.isfinite(v)):
                            grid_arr[i, j] = float(v)
            fig, ax = plt.subplots(figsize=(5.0, 4.0))
            im = ax.imshow(grid_arr, origin="lower", aspect="auto", cmap=cmap,
                           vmin=vmin, vmax=vmax,
                           extent=[kq_labels[0] - 0.25, kq_labels[-1] + 0.25,
                                   Q0_labels[0] - 0.0025, Q0_labels[-1] + 0.0025])
            ax.set_xticks(kq_labels)
            ax.set_yticks(Q0_labels)
            ax.set_xlabel("kq [s⁻¹]")
            ax.set_ylabel("Q0 [mol/dm³]")
            ax.set_title(f"pitch={int(pitch)} dose={dose:.1f}  {metric}")
            for i, Q0 in enumerate(Q0_labels):
                for j, kq in enumerate(kq_labels):
                    v = grid_arr[i, j]
                    if np.isfinite(v):
                        ax.text(kq, Q0, f"{v:+.2f}", ha="center", va="center",
                                color="white", fontsize=8)
            fig.colorbar(im, ax=ax)
            fig.tight_layout()
            fig.savefig(fig_dir / fname, dpi=150)
            plt.close(fig)
    return rows


def write_csv(rows: list[dict], path: Path, extra_keys=()):
    keys = [
        "op_label", "row_kind", "pitch_nm", "dose_mJ_cm2",
        "sigma_nm", "DH_nm2_s", "time_s", "Q0_mol_dm3", "kq_s_inv",
        "P_space_center_mean", "P_line_center_mean", "P_line_margin",
        "contrast", "area_frac", "area_frac_locked", "CD_pitch_frac",
        "CD_initial_nm", "CD_fixed_nm", "CD_locked_nm",
        "P_threshold_locked", "cd_lock_status",
        "LER_design_initial_nm", "LER_after_eblur_H0_nm",
        "LER_fixed_threshold_nm", "LER_CD_locked_nm", "LER_delta_nm",
        "psd_eblur_mid", "psd_PEB_mid", "psd_locked_mid",
        "status_fixed", "status_CD_locked",
        *list(extra_keys),
    ]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            row = {k: r.get(k) for k in keys}
            w.writerow(row)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--tag", type=str, default="04b_cd_locked")
    args = p.parse_args()

    cfg_base = yaml.safe_load(Path(args.config).read_text())

    fig_dir_a = DEFAULT_OUT / "figures" / f"{args.tag}_block_a"
    fig_dir_b = DEFAULT_OUT / "figures" / f"{args.tag}_block_b"
    fig_dir_a.mkdir(parents=True, exist_ok=True)
    fig_dir_b.mkdir(parents=True, exist_ok=True)
    logs_dir = DEFAULT_OUT / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    rows_a = block_a(cfg_base, fig_dir_a)
    write_csv(rows_a, logs_dir / f"{args.tag}_block_a.csv")

    rows_b = block_b(cfg_base, fig_dir_b)
    write_csv(rows_b, logs_dir / f"{args.tag}_block_b.csv",
              extra_keys=("dCD_shift_nm", "dLER_fixed_nm", "dLER_locked_nm"))

    # Per-row decision label for Block A.
    print("\n=== Stage 4B decisions (Block A) ===")
    DECISION_TOL = 0.20
    for r in rows_a:
        ler_design = r["LER_design_initial_nm"]
        ler_fixed = r["LER_fixed_threshold_nm"]
        ler_locked = r["LER_CD_locked_nm"]
        if not (np.isfinite(ler_fixed) and np.isfinite(ler_locked)):
            label = "unstable (CD-lock failed)"
        else:
            fixed_bad = ler_fixed >= ler_design + DECISION_TOL
            locked_bad = ler_locked >= ler_design + DECISION_TOL
            fixed_underestimate = (not fixed_bad) and locked_bad and (ler_locked > ler_fixed + DECISION_TOL)
            displacement_artifact = fixed_bad and (not locked_bad) and (ler_locked + DECISION_TOL < ler_fixed)
            both_ok = (not fixed_bad) and (not locked_bad)
            if fixed_underestimate:
                label = "fixed underestimates (merged-line / collapsed-contour artifact)"
            elif displacement_artifact:
                label = "fixed overestimates (contour-displacement artifact); locked recovers"
            elif fixed_bad and locked_bad:
                label = "real roughness degradation (both fixed & locked > design)"
            elif both_ok:
                label = "OK (both fixed & locked near design)"
            else:
                label = "ambiguous"
        r["decision_label"] = label
        print(f"  [{r['op_label']:<22}] pitch={int(r['pitch_nm'])} dose={r['dose_mJ_cm2']:.1f}: "
              f"LER design={ler_design:.2f}  fixed={ler_fixed:.2f}  locked={ler_locked:.2f} → {label}")
    # Add label to CSV.
    with (logs_dir / f"{args.tag}_block_a_decisions.json").open("w") as f:
        json.dump([{
            "op_label": r["op_label"], "pitch_nm": r["pitch_nm"], "dose_mJ_cm2": r["dose_mJ_cm2"],
            "LER_design": r["LER_design_initial_nm"], "LER_fixed": r["LER_fixed_threshold_nm"],
            "LER_locked": r["LER_CD_locked_nm"], "status_fixed": r["status_fixed"],
            "status_CD_locked": r["status_CD_locked"], "cd_lock_status": r["cd_lock_status"],
            "decision_label": r["decision_label"],
        } for r in rows_a], f, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
