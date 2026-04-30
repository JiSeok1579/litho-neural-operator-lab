"""Stage 5 — pitch x dose process window at the Stage-4 balanced OP.

Primary OP:
    sigma=2 nm, DH=0.5 nm^2/s, time=30 s, kdep=0.5 s^-1, kloss=0.005 s^-1,
    Hmax=0.2 mol/dm^3, quencher enabled (Q0=0.02 mol/dm^3, kq=1.0 s^-1, DQ=0).

Sweep:
    pitch_nm     ∈ {16, 18, 20, 24, 28, 32}
    dose_mJ_cm2  ∈ {21, 28.4, 40, 44.2, 59, 60}
    domain_x_nm  = pitch * 5  (n_periods_x = 5)
    line_cd_nm   = 12.5  (fixed across pitches; duty changes)

Optional controls:
    control_sigma0_no_q : sigma=0, quencher disabled
    control_sigma2_no_q : sigma=2, quencher disabled

Per-run classification (precedence order):
    unstable      : NaN / Inf / bounds violation / no extractable contour
    merged        : P_space_mean >= 0.5  OR  area_frac >= 0.90  OR  CD/pitch >= 0.85
    under_exposed : P_line_mean  < 0.65
    low_contrast  : contrast      <= 0.15
    valid         : interior gate pass without margin
    robust_valid  : interior gate pass AND P_line_margin >= 0.05

Recommendation per pitch (primary OP):
    robust_valid first → minimize |CD_shift_nm| → maximize total_LER_reduction_pct
    → prefer larger P_line_margin on tie.
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
from reaction_diffusion_peb_v2_high_na.src.visualization import plot_contour_overlay  # noqa: E402

V2_DIR = Path(__file__).resolve().parents[2]
DEFAULT_OUT = V2_DIR / "outputs"

PITCHES = [16, 18, 20, 24, 28, 32]
DOSES = [21.0, 28.4, 40.0, 44.2, 59.0, 60.0]
N_PERIODS_X = 5

# Status enum.
STATUS_ORDER = ["unstable", "merged", "under_exposed", "low_contrast", "valid", "robust_valid"]
STATUS_NUM = {s: i for i, s in enumerate(STATUS_ORDER)}


def classify(r: dict) -> str:
    """Assign one mutually-exclusive status per the precedence above."""
    # unstable
    if not (r["cond_metrics"] and np.isfinite(r["P_max"]) and np.isfinite(r["P_min"])
            and np.isfinite(r["LER_after_PEB_P_nm"])):
        return "unstable"
    # bounds: H/P should be in expected ranges
    if r["H_min"] < -1e-6 or r["P_min"] < -1e-6 or r["P_max"] > 1.0 + 1e-6:
        return "unstable"
    # merged
    if (r["P_space_center_mean"] >= 0.50
            or r["area_frac"] >= 0.90
            or (np.isfinite(r["CD_pitch_frac"]) and r["CD_pitch_frac"] >= 0.85)):
        return "merged"
    # under_exposed
    if r["P_line_center_mean"] < 0.65:
        return "under_exposed"
    # low_contrast (rare; falls between cracks)
    if r["contrast"] <= 0.15:
        return "low_contrast"
    # robust_valid vs valid
    if r["P_line_margin"] >= 0.05:
        return "robust_valid"
    return "valid"


def build_cfg_for_run(base_cfg: dict, pitch_nm: float, dose_mJ_cm2: float,
                      sigma_nm: float | None, quencher_enabled: bool | None) -> dict:
    cfg = copy.deepcopy(base_cfg)
    cfg["geometry"]["pitch_nm"] = float(pitch_nm)
    cfg["geometry"]["half_pitch_nm"] = 0.5 * float(pitch_nm)
    cfg["geometry"]["domain_x_nm"] = float(pitch_nm * N_PERIODS_X)
    cfg["exposure"]["dose_mJ_cm2"] = float(dose_mJ_cm2)
    cfg["exposure"]["dose_norm"] = float(dose_mJ_cm2) / float(cfg["exposure"]["reference_dose_mJ_cm2"])
    if sigma_nm is not None:
        cfg["exposure"]["electron_blur_sigma_nm"] = float(sigma_nm)
        cfg["exposure"]["electron_blur_enabled"] = sigma_nm > 0.0
    if quencher_enabled is not None:
        cfg["quencher"]["enabled"] = bool(quencher_enabled)
    return cfg


def run_block(label: str, base_cfg: dict, sigma_nm: float | None,
              quencher_enabled: bool | None) -> list[dict]:
    rows = []
    print(f"\n=== Stage 5 [{label}] ===")
    for pitch in PITCHES:
        for dose in DOSES:
            cfg = build_cfg_for_run(base_cfg, pitch_nm=pitch, dose_mJ_cm2=dose,
                                    sigma_nm=sigma_nm, quencher_enabled=quencher_enabled)
            sigma_use = cfg["exposure"]["electron_blur_sigma_nm"] if cfg["exposure"]["electron_blur_enabled"] else 0.0
            quencher_use = cfg["quencher"]["enabled"]
            r = run_one_with_overrides(
                cfg,
                sigma_nm=sigma_use,
                time_s=cfg["peb"]["time_s"],
                DH_nm2_s=cfg["peb"]["DH_nm2_s"],
                kdep_s_inv=cfg["peb"]["kdep_s_inv"],
                Hmax_mol_dm3=cfg["exposure"]["Hmax_mol_dm3"],
                quencher_enabled=quencher_use,
                Q0_mol_dm3=cfg["quencher"]["Q0_mol_dm3"],
                DQ_nm2_s=cfg["quencher"]["DQ_nm2_s"],
                kq_s_inv=cfg["quencher"]["kq_s_inv"],
            )
            r["block_label"] = label
            r["pitch_nm"] = float(pitch)
            r["dose_mJ_cm2"] = float(dose)
            r["status"] = classify(r)
            r["status_num"] = STATUS_NUM[r["status"]]
            rows.append(r)
            print(f"  pitch={pitch:>2.0f}  dose={dose:>5.1f}  → {r['status']:<14}  "
                  f"P_sp={r['P_space_center_mean']:.3f}  P_ln={r['P_line_center_mean']:.3f}  "
                  f"margin={r['P_line_margin']:+.3f}  CD/p={r['CD_pitch_frac']:.3f}  "
                  f"CDshift={r['CD_shift_nm']:+.2f}  LER%={r['total_LER_reduction_pct']:+.2f}")
    return rows


def heatmap(rows: list[dict], metric: str, title: str, out_path: Path,
            cmap: str = "viridis", vmin=None, vmax=None, fmt: str = "{:+.2f}"):
    M = np.full((len(PITCHES), len(DOSES)), np.nan)
    for r in rows:
        i = PITCHES.index(int(r["pitch_nm"]))
        j = DOSES.index(float(r["dose_mJ_cm2"]))
        v = r[metric]
        if v is None or (isinstance(v, float) and not np.isfinite(v)):
            M[i, j] = np.nan
        else:
            M[i, j] = float(v)
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    im = ax.imshow(M, origin="lower", aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax,
                   extent=[-0.5, len(DOSES) - 0.5, -0.5, len(PITCHES) - 0.5])
    ax.set_xticks(range(len(DOSES)))
    ax.set_xticklabels([f"{d:g}" for d in DOSES])
    ax.set_yticks(range(len(PITCHES)))
    ax.set_yticklabels([f"{p}" for p in PITCHES])
    ax.set_xlabel("dose [mJ/cm²]")
    ax.set_ylabel("pitch [nm]")
    ax.set_title(title)
    for i in range(len(PITCHES)):
        for j in range(len(DOSES)):
            v = M[i, j]
            if np.isfinite(v):
                ax.text(j, i, fmt.format(v), ha="center", va="center",
                        color="white", fontsize=8)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)


def status_heatmap(rows: list[dict], title: str, out_path: Path):
    M = np.full((len(PITCHES), len(DOSES)), -1, dtype=int)
    for r in rows:
        i = PITCHES.index(int(r["pitch_nm"]))
        j = DOSES.index(float(r["dose_mJ_cm2"]))
        M[i, j] = r["status_num"]

    palette = {
        "unstable":      "#222222",
        "merged":        "#b30000",
        "under_exposed": "#5b8def",
        "low_contrast":  "#9b59b6",
        "valid":         "#f0c419",
        "robust_valid":  "#27ae60",
    }
    from matplotlib.colors import ListedColormap, BoundaryNorm
    cmap = ListedColormap([palette[s] for s in STATUS_ORDER])
    norm = BoundaryNorm(np.arange(len(STATUS_ORDER) + 1) - 0.5, cmap.N)

    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    im = ax.imshow(M, origin="lower", aspect="auto", cmap=cmap, norm=norm,
                   extent=[-0.5, len(DOSES) - 0.5, -0.5, len(PITCHES) - 0.5])
    ax.set_xticks(range(len(DOSES)))
    ax.set_xticklabels([f"{d:g}" for d in DOSES])
    ax.set_yticks(range(len(PITCHES)))
    ax.set_yticklabels([f"{p}" for p in PITCHES])
    ax.set_xlabel("dose [mJ/cm²]")
    ax.set_ylabel("pitch [nm]")
    ax.set_title(title)
    for i in range(len(PITCHES)):
        for j in range(len(DOSES)):
            label = STATUS_ORDER[M[i, j]]
            ax.text(j, i, label[:4], ha="center", va="center",
                    color="white", fontsize=7)
    cb = fig.colorbar(im, ax=ax, ticks=range(len(STATUS_ORDER)))
    cb.set_ticklabels(STATUS_ORDER)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)


def select_recommendation(rows: list[dict]) -> dict[int, dict | None]:
    """Per pitch: robust_valid first; among those minimise |CD_shift|, then
    maximise total_LER_reduction_pct, then maximise P_line_margin."""
    by_pitch: dict[int, list[dict]] = {p: [] for p in PITCHES}
    for r in rows:
        by_pitch[int(r["pitch_nm"])].append(r)

    rec: dict[int, dict | None] = {}
    for pitch, rs in by_pitch.items():
        # Filter: robust_valid > valid > nothing.
        candidates_rv = [r for r in rs if r["status"] == "robust_valid"]
        candidates_v = [r for r in rs if r["status"] == "valid"]
        pool = candidates_rv if candidates_rv else candidates_v
        if not pool:
            rec[pitch] = None
            continue
        pool = sorted(
            pool,
            key=lambda r: (
                abs(r["CD_shift_nm"]) if np.isfinite(r["CD_shift_nm"]) else 1e9,
                -r["total_LER_reduction_pct"] if np.isfinite(r["total_LER_reduction_pct"]) else 1e9,
                -r["P_line_margin"],
            ),
        )
        rec[pitch] = pool[0]
    return rec


def write_csv(rows: list[dict], path: Path):
    keys = [
        "block_label", "pitch_nm", "dose_mJ_cm2",
        "sigma_nm", "DH_nm2_s", "time_s", "Q0_mol_dm3", "kq_s_inv",
        "P_space_center_mean", "P_line_center_mean", "P_line_margin",
        "contrast", "area_frac", "CD_pitch_frac",
        "CD_initial_nm", "CD_final_nm", "CD_shift_nm",
        "LER_design_initial_nm", "LER_after_eblur_H0_nm", "LER_after_PEB_P_nm",
        "electron_blur_LER_reduction_pct", "PEB_LER_reduction_pct", "total_LER_reduction_pct",
        "psd_PEB_high", "psd_high_band_reduction_pct",
        "status",
    ]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in keys})


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--tag", type=str, default="05_pitch_dose")
    p.add_argument("--skip_controls", action="store_true",
                   help="Skip the two no-quencher control blocks.")
    args = p.parse_args()

    base_cfg = yaml.safe_load(Path(args.config).read_text())

    fig_dir = DEFAULT_OUT / "figures" / args.tag
    fig_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = DEFAULT_OUT / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Primary block.
    primary = run_block("primary",
                         base_cfg,
                         sigma_nm=base_cfg["exposure"]["electron_blur_sigma_nm"],
                         quencher_enabled=base_cfg["quencher"]["enabled"])

    # Optional controls.
    controls = {}
    if not args.skip_controls:
        controls["control_sigma0_no_q"] = run_block(
            "control_sigma0_no_q", base_cfg,
            sigma_nm=0.0, quencher_enabled=False,
        )
        controls["control_sigma2_no_q"] = run_block(
            "control_sigma2_no_q", base_cfg,
            sigma_nm=2.0, quencher_enabled=False,
        )

    all_rows = primary[:]
    for rs in controls.values():
        all_rows.extend(rs)

    # Heatmaps for each block.
    blocks = {"primary": primary, **controls}
    for label, rs in blocks.items():
        block_dir = fig_dir / label
        block_dir.mkdir(parents=True, exist_ok=True)
        status_heatmap(rs, f"status — {label}", block_dir / "heatmap_status.png")
        heatmap(rs, "CD_shift_nm", f"CD_shift_nm — {label}",
                block_dir / "heatmap_CD_shift.png", cmap="RdBu_r", vmin=-6, vmax=12, fmt="{:+.2f}")
        heatmap(rs, "total_LER_reduction_pct", f"total_LER_reduction_pct — {label}",
                block_dir / "heatmap_total_LER.png", cmap="RdBu_r", vmin=-30, vmax=15, fmt="{:+.1f}")
        heatmap(rs, "P_line_margin", f"P_line_margin — {label}",
                block_dir / "heatmap_P_line_margin.png", cmap="viridis", fmt="{:+.3f}")
        heatmap(rs, "area_frac", f"area_frac — {label}",
                block_dir / "heatmap_area_frac.png", cmap="viridis", vmin=0, vmax=1, fmt="{:.2f}")
        heatmap(rs, "contrast", f"contrast — {label}",
                block_dir / "heatmap_contrast.png", cmap="viridis", vmin=0, vmax=0.7, fmt="{:.2f}")

    # Primary contour overlays (1 per cell).
    primary_contour_dir = fig_dir / "primary_contours"
    primary_contour_dir.mkdir(parents=True, exist_ok=True)
    for r in primary:
        grid = r["_grid"]
        tag = f"pitch_{int(r['pitch_nm'])}_dose_{r['dose_mJ_cm2']:.1f}"
        plot_contour_overlay(
            P_field=r["_P_final"],
            x_nm=grid.x_nm,
            y_nm=grid.y_nm,
            threshold=base_cfg["development"]["P_threshold"],
            initial_edges=(grid.line_centers_nm,
                           r["_initial_edges"].left_edges_nm,
                           r["_initial_edges"].right_edges_nm),
            final_edges=(grid.line_centers_nm,
                         r["_final_edges"].left_edges_nm,
                         r["_final_edges"].right_edges_nm),
            title=f"pitch={int(r['pitch_nm'])} dose={r['dose_mJ_cm2']:.1f} → {r['status']}",
            out_path=str(primary_contour_dir / f"contour_{tag}.png"),
        )

    # CSV.
    write_csv(all_rows, logs_dir / f"{args.tag}_summary.csv")
    (logs_dir / f"{args.tag}_summary.json").write_text(json.dumps(
        [{k: r[k] for k in [
            "block_label", "pitch_nm", "dose_mJ_cm2", "sigma_nm", "DH_nm2_s",
            "Q0_mol_dm3", "kq_s_inv",
            "P_space_center_mean", "P_line_center_mean", "P_line_margin",
            "contrast", "area_frac", "CD_pitch_frac", "CD_shift_nm",
            "total_LER_reduction_pct", "status",
        ]} for r in all_rows],
        indent=2,
    ))

    # Recommendation per pitch (primary block only).
    rec = select_recommendation(primary)
    print("\n=== Stage 5 recommended dose per pitch (primary OP) ===")
    rec_rows = []
    for pitch in PITCHES:
        r = rec[pitch]
        if r is None:
            print(f"  pitch={pitch}: no robust_valid or valid run in dose sweep.")
            rec_rows.append({"pitch_nm": pitch, "rec_dose_mJ_cm2": None, "status": "none"})
            continue
        print(f"  pitch={pitch:>2}: dose={r['dose_mJ_cm2']:>5.1f}  "
              f"({r['status']:<13}) | CD_shift={r['CD_shift_nm']:+.2f} | "
              f"LER%={r['total_LER_reduction_pct']:+.2f} | margin={r['P_line_margin']:+.3f}")
        rec_rows.append({
            "pitch_nm": pitch,
            "rec_dose_mJ_cm2": r["dose_mJ_cm2"],
            "status": r["status"],
            "CD_shift_nm": r["CD_shift_nm"],
            "total_LER_reduction_pct": r["total_LER_reduction_pct"],
            "P_line_margin": r["P_line_margin"],
            "area_frac": r["area_frac"],
            "CD_pitch_frac": r["CD_pitch_frac"],
        })
    (logs_dir / f"{args.tag}_recommendation.json").write_text(json.dumps(rec_rows, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
