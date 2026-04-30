"""Phase 2B Part A — x-y sensitivity atlas at the frozen v2 OP.

NOT a calibration. Sensitivity / controllability study only.

OAT sweeps + selected pair sweeps as defined in the config.
Per cell reports: CD_fixed, CD_locked, LER_locked, P_line_margin,
area_frac, contrast, psd_mid (locked), status.
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
    apply_xy_overrides,
    classify,
)
from reaction_diffusion_peb_v2_high_na.experiments.run_sigma_sweep_helpers import (  # noqa: E402
    run_one_with_overrides,
)

V2_DIR = Path(__file__).resolve().parents[3]
DEFAULT_OUT = V2_DIR / "outputs"

REPORT_KEYS = [
    "CD_final_nm",            # CD_fixed
    "CD_locked_nm",
    "LER_CD_locked_nm",
    "P_line_margin",
    "area_frac",
    "contrast",
    "psd_locked_mid",
    "status",
]


def run_one(base_cfg: dict, overrides: dict) -> dict:
    cfg = apply_xy_overrides(base_cfg, overrides)
    r = run_one_with_overrides(
        cfg,
        sigma_nm=cfg["exposure"]["electron_blur_sigma_nm"],
        time_s=cfg["peb"]["time_s"],
        DH_nm2_s=cfg["peb"]["DH_nm2_s"],
        kdep_s_inv=cfg["peb"]["kdep_s_inv"],
        Hmax_mol_dm3=cfg["exposure"]["Hmax_mol_dm3"],
        quencher_enabled=cfg["quencher"]["enabled"],
        Q0_mol_dm3=cfg["quencher"]["Q0_mol_dm3"],
        DQ_nm2_s=cfg["quencher"]["DQ_nm2_s"],
        kq_s_inv=cfg["quencher"]["kq_s_inv"],
    )
    r["status"] = classify(r)
    r["overrides"] = overrides
    return r


def line_plot(xs, ys_dict, xlabel, title, out_path):
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    for name, ys in ys_dict.items():
        clean = [v if v is not None and (not isinstance(v, float) or np.isfinite(v)) else np.nan for v in ys]
        ax.plot(xs, clean, marker="o", label=name)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def heatmap_2d(M, x_vals, y_vals, xlabel, ylabel, title, out_path,
                cmap="viridis", vmin=None, vmax=None, fmt="{:+.2f}",
                text_color="white"):
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    im = ax.imshow(M, origin="lower", aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax,
                   extent=[-0.5, len(x_vals) - 0.5, -0.5, len(y_vals) - 0.5])
    ax.set_xticks(range(len(x_vals)))
    ax.set_xticklabels([str(v) for v in x_vals])
    ax.set_yticks(range(len(y_vals)))
    ax.set_yticklabels([str(v) for v in y_vals])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    for i in range(len(y_vals)):
        for j in range(len(x_vals)):
            v = M[i, j]
            if isinstance(v, str):
                ax.text(j, i, v[:4], ha="center", va="center", color=text_color, fontsize=7)
            elif np.isfinite(v):
                ax.text(j, i, fmt.format(v), ha="center", va="center", color=text_color, fontsize=7)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def status_heatmap(M_status, x_vals, y_vals, xlabel, ylabel, title, out_path):
    palette_order = ["unstable", "merged", "under_exposed", "low_contrast", "valid", "robust_valid"]
    palette_color = {
        "unstable": "#222222", "merged": "#b30000", "under_exposed": "#5b8def",
        "low_contrast": "#9b59b6", "valid": "#f0c419", "robust_valid": "#27ae60",
    }
    from matplotlib.colors import ListedColormap, BoundaryNorm
    M_num = np.full(M_status.shape, -1, dtype=int)
    for i in range(M_status.shape[0]):
        for j in range(M_status.shape[1]):
            s = M_status[i, j]
            M_num[i, j] = palette_order.index(s) if s in palette_order else -1
    cmap = ListedColormap([palette_color[s] for s in palette_order])
    norm = BoundaryNorm(np.arange(len(palette_order) + 1) - 0.5, cmap.N)

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    im = ax.imshow(M_num, origin="lower", aspect="auto", cmap=cmap, norm=norm,
                   extent=[-0.5, len(x_vals) - 0.5, -0.5, len(y_vals) - 0.5])
    ax.set_xticks(range(len(x_vals))); ax.set_xticklabels([str(v) for v in x_vals])
    ax.set_yticks(range(len(y_vals))); ax.set_yticklabels([str(v) for v in y_vals])
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title)
    for i in range(M_status.shape[0]):
        for j in range(M_status.shape[1]):
            label = M_status[i, j]
            ax.text(j, i, label[:4], ha="center", va="center", color="white", fontsize=7)
    cb = fig.colorbar(im, ax=ax, ticks=range(len(palette_order)))
    cb.set_ticklabels(palette_order)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--tag", type=str, default="cal03_atlas_xy")
    args = p.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    fig_dir = DEFAULT_OUT / "figures" / args.tag
    fig_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = DEFAULT_OUT / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []

    # ---- Anchor ----
    anchor = run_one(cfg, overrides={})
    all_rows.append({"sweep": "anchor", "x_var": None, "y_var": None,
                     "x_val": None, "y_val": None,
                     **{k: anchor[k] for k in REPORT_KEYS}})
    print("\n=== Phase 2B Part A — x-y sensitivity atlas ===")
    print(
        "anchor: "
        f"CD_fix={anchor['CD_final_nm']:.3f}  "
        f"CD_lock={anchor['CD_locked_nm']:.3f}  "
        f"LER_lock={anchor['LER_CD_locked_nm']:.3f}  "
        f"margin={anchor['P_line_margin']:+.3f}  "
        f"area={anchor['area_frac']:.3f}  "
        f"status={anchor['status']}"
    )

    # ---- OAT sweeps ----
    print("\n[OAT sweeps]")
    oat_results: dict[str, list[dict]] = {}
    for var, values in cfg["oat"].items():
        rows = []
        print(f"  {var} ∈ {values}")
        for v in values:
            r = run_one(cfg, overrides={var: v})
            rows.append({"value": v, **{k: r[k] for k in REPORT_KEYS}})
            all_rows.append({"sweep": f"oat:{var}", "x_var": var, "y_var": None,
                              "x_val": v, "y_val": None,
                              **{k: r[k] for k in REPORT_KEYS}})
        oat_results[var] = rows

        # Line plot of metrics vs variable.
        xs = [row["value"] for row in rows]
        ys_dict = {m: [row[m] for row in rows] for m in
                    ["CD_final_nm", "CD_locked_nm", "LER_CD_locked_nm",
                     "P_line_margin", "area_frac", "contrast", "psd_locked_mid"]}
        line_plot(xs, ys_dict, var, f"x-y OAT vs {var}",
                   fig_dir / f"oat_{var}.png")

    # ---- Pair sweeps ----
    pair_axis_values = cfg.get("pair_axis_values", {})
    print("\n[Pair sweeps]")
    for pair in cfg["pairs"]:
        x_var, y_var = pair["x"], pair["y"]
        x_vals = pair_axis_values.get(x_var) or cfg["oat"].get(x_var)
        y_vals = pair_axis_values.get(y_var) or cfg["oat"].get(y_var)
        if x_vals is None or y_vals is None:
            print(f"  SKIP pair {x_var}×{y_var}: axis values missing")
            continue
        print(f"  {x_var}×{y_var}: {len(x_vals)}×{len(y_vals)}={len(x_vals)*len(y_vals)} runs")

        # Per-cell run.
        cells = np.empty((len(y_vals), len(x_vals)), dtype=object)
        for i, y in enumerate(y_vals):
            for j, x in enumerate(x_vals):
                r = run_one(cfg, overrides={x_var: x, y_var: y})
                cells[i, j] = r
                all_rows.append({"sweep": f"pair:{x_var}×{y_var}",
                                   "x_var": x_var, "y_var": y_var,
                                   "x_val": x, "y_val": y,
                                   **{k: r[k] for k in REPORT_KEYS}})

        # Heatmaps for each metric.
        def _grid(metric_key):
            M = np.full((len(y_vals), len(x_vals)), np.nan)
            for i in range(len(y_vals)):
                for j in range(len(x_vals)):
                    v = cells[i, j][metric_key]
                    if v is not None and (not isinstance(v, float) or np.isfinite(v)):
                        M[i, j] = float(v)
            return M

        pair_tag = f"{x_var}__{y_var}"
        heatmap_2d(_grid("CD_final_nm"), x_vals, y_vals, x_var, y_var,
                    f"CD_fixed (nm)  {x_var}×{y_var}",
                    fig_dir / f"pair_{pair_tag}_CD_fixed.png",
                    cmap="viridis", fmt="{:.2f}")
        heatmap_2d(_grid("CD_locked_nm"), x_vals, y_vals, x_var, y_var,
                    f"CD_locked (nm)  {x_var}×{y_var}",
                    fig_dir / f"pair_{pair_tag}_CD_locked.png",
                    cmap="viridis", fmt="{:.2f}")
        heatmap_2d(_grid("LER_CD_locked_nm"), x_vals, y_vals, x_var, y_var,
                    f"LER_locked (nm)  {x_var}×{y_var}",
                    fig_dir / f"pair_{pair_tag}_LER_locked.png",
                    cmap="viridis", fmt="{:.2f}")
        heatmap_2d(_grid("P_line_margin"), x_vals, y_vals, x_var, y_var,
                    f"P_line_margin  {x_var}×{y_var}",
                    fig_dir / f"pair_{pair_tag}_margin.png",
                    cmap="viridis", fmt="{:+.3f}")
        heatmap_2d(_grid("area_frac"), x_vals, y_vals, x_var, y_var,
                    f"area_frac  {x_var}×{y_var}",
                    fig_dir / f"pair_{pair_tag}_area.png",
                    cmap="viridis", vmin=0.0, vmax=1.0, fmt="{:.2f}")
        heatmap_2d(_grid("psd_locked_mid"), x_vals, y_vals, x_var, y_var,
                    f"PSD_mid_locked  {x_var}×{y_var}",
                    fig_dir / f"pair_{pair_tag}_psd_mid.png",
                    cmap="viridis", fmt="{:.2f}")

        # Status heatmap.
        S = np.empty((len(y_vals), len(x_vals)), dtype=object)
        for i in range(len(y_vals)):
            for j in range(len(x_vals)):
                S[i, j] = cells[i, j]["status"]
        status_heatmap(S, x_vals, y_vals, x_var, y_var,
                        f"status  {x_var}×{y_var}",
                        fig_dir / f"pair_{pair_tag}_status.png")

    # ---- Write CSV ----
    keys = ["sweep", "x_var", "y_var", "x_val", "y_val",
            "CD_final_nm", "CD_locked_nm", "LER_CD_locked_nm",
            "P_line_margin", "area_frac", "contrast", "psd_locked_mid", "status"]
    csv_path = logs_dir / f"{args.tag}_summary.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for row in all_rows:
            w.writerow({k: row.get(k) for k in keys})
    print(f"\nWrote {len(all_rows)} rows to {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
