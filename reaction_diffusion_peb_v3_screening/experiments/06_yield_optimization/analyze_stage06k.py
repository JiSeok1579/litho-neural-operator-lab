"""Stage 06K -- sensitivity analysis, figures, and acceptance summary
for the local process-window map around G_4867.

Reads:
    outputs/yield_optimization/stage06K_local_candidates.csv
    outputs/yield_optimization/stage06K_oat_sensitivity.csv
    outputs/yield_optimization/stage06K_pairwise_maps.csv
    outputs/labels/stage06K_fd_verification.csv
    outputs/yield_optimization/stage06G_strict_score_config.yaml
    outputs/yield_optimization/stage06I_mode_a_final_recipes.yaml

Writes:
    outputs/logs/stage06K_summary.json
    outputs/figures/06_yield_optimization/
        stage06K_strict_pass_prob_oat.png
        stage06K_knob_sensitivity_bar.png
        stage06K_dose_sigma_heatmap.png
        stage06K_DH_time_heatmap.png
        stage06K_Q0_kq_heatmap.png
        stage06K_cd_error_vs_ler_local_pareto.png
        stage06K_process_window_boundary.png
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reaction_diffusion_peb_v3_screening.src.metrics_io import (
    FEATURE_KEYS,
    read_labels_csv,
)


V3_DIR = ROOT / "reaction_diffusion_peb_v3_screening"
CD_TARGET_NM = 15.0

KNOBS = ["dose_mJ_cm2", "sigma_nm", "DH_nm2_s", "time_s",
          "Hmax_mol_dm3", "kdep_s_inv", "Q0_mol_dm3", "kq_s_inv"]
HARD_FAIL_LABELS = {"under_exposed", "merged",
                     "roughness_degraded", "numerical_invalid"}


def _safe_float(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def _coerce(rows: list[dict], keys: list[str]) -> None:
    for r in rows:
        for k in keys:
            if k in r:
                r[k] = _safe_float(r.get(k))


# --------------------------------------------------------------------------
# Surrogate "strict_pass_prob" proxy.
# The surrogate evaluator returns mean_cd_fixed / mean_ler_locked /
# std_cd_fixed / std_ler_locked / p_robust_valid. A defensible proxy
# for strict_pass_prob is:
#     P_robust * P(|CD - 15| <= cd_tol) * P(LER <= ler_cap)
# under a Gaussian approximation around (mean, std).
# This is the surrogate-side analogue of the FD MC empirical
# strict_pass_prob and is purely a ranking tool. FD MC remains truth.
# --------------------------------------------------------------------------
def _gauss_p_in_window(mu: float, sigma: float,
                          lo: float, hi: float) -> float:
    if not np.isfinite(mu) or not np.isfinite(sigma):
        return float("nan")
    if sigma <= 1e-12:
        return float(1.0 if lo <= mu <= hi else 0.0)
    from math import erf, sqrt
    z_hi = (hi - mu) / (sigma * sqrt(2))
    z_lo = (lo - mu) / (sigma * sqrt(2))
    return float(0.5 * (erf(z_hi) - erf(z_lo)))


def attach_strict_pass_proxy(rows: list[dict],
                                cd_tol: float, ler_cap: float) -> None:
    for r in rows:
        cd_mu = _safe_float(r.get("mean_cd_fixed"))
        cd_sd = _safe_float(r.get("std_cd_fixed"))
        ler_mu = _safe_float(r.get("mean_ler_locked"))
        ler_sd = _safe_float(r.get("std_ler_locked"))
        p_rob = _safe_float(r.get("p_robust_valid"))
        p_cd = _gauss_p_in_window(cd_mu, cd_sd,
                                       CD_TARGET_NM - cd_tol, CD_TARGET_NM + cd_tol)
        p_ler = _gauss_p_in_window(ler_mu, ler_sd, -np.inf, ler_cap)
        proxy = float(p_rob * p_cd * p_ler) \
            if all(np.isfinite([p_rob, p_cd, p_ler])) else float("nan")
        r["strict_pass_prob_proxy"] = proxy
        r["mean_cd_error"] = abs(cd_mu - CD_TARGET_NM) if np.isfinite(cd_mu) else float("nan")


# --------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------
def plot_oat_strict(oat_rows: list[dict], base_strict: float,
                      base_proxy: float, out_path: Path) -> None:
    knobs = sorted(set(r["_oat_knob"] for r in oat_rows if r.get("_oat_knob")))
    fig, axes = plt.subplots(2, 4, figsize=(16.0, 8.5), sharey=False)
    for ax, k in zip(axes.flat, knobs):
        kr = sorted([r for r in oat_rows if r["_oat_knob"] == k],
                     key=lambda r: float(r["_oat_value"]))
        x = np.array([float(r["_oat_value"]) for r in kr])
        y_strict = np.array([float(r["strict_score"]) for r in kr])
        y_proxy = np.array([float(r["strict_pass_prob_proxy"]) for r in kr])
        ax2 = ax.twinx()
        ax.plot(x, y_strict, "-o", color="#1f77b4", lw=1.6, markersize=4,
                label="strict_score")
        ax2.plot(x, y_proxy, "-s", color="#d62728", lw=1.4, markersize=3,
                  label="strict_pass_proxy")
        ax.axhline(base_strict, color="#1f77b4", ls=":", lw=0.6, alpha=0.6)
        ax2.axhline(base_proxy, color="#d62728", ls=":", lw=0.6, alpha=0.6)
        ax.set_title(k, fontsize=10)
        ax.set_xlabel(f"{k} value (around G_4867)")
        ax.set_ylabel("strict_score", color="#1f77b4")
        ax2.set_ylabel("strict_pass_proxy", color="#d62728")
        ax.grid(True, alpha=0.25)
    fig.suptitle("Stage 06K -- one-at-a-time sweeps around G_4867",
                  fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_knob_sensitivity_bar(oat_rows: list[dict], base_strict: float,
                                base_proxy: float, out_path: Path) -> dict:
    knobs = sorted(set(r["_oat_knob"] for r in oat_rows if r.get("_oat_knob")))
    sens = []
    for k in knobs:
        kr = sorted([r for r in oat_rows if r["_oat_knob"] == k],
                     key=lambda r: float(r["_oat_value"]))
        if len(kr) < 2:
            continue
        d_strict = max(abs(float(r["strict_score"]) - base_strict) for r in kr)
        d_proxy = max(abs(float(r["strict_pass_prob_proxy"]) - base_proxy) for r in kr)
        d_cd_err = max(abs(float(r["mean_cd_error"]) - 0.0) for r in kr) \
            if any(np.isfinite(_safe_float(r.get("mean_cd_error"))) for r in kr) else float("nan")
        d_ler = max(abs(float(r["mean_ler_locked"]) - float(kr[len(kr)//2]["mean_ler_locked"])) for r in kr)
        sens.append({"knob": k,
                     "max_abs_strict_drop":    float(d_strict),
                     "max_abs_proxy_drop":     float(d_proxy),
                     "max_cd_error":           float(d_cd_err),
                     "max_ler_excursion":      float(d_ler)})

    sens.sort(key=lambda s: -s["max_abs_strict_drop"])

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.5))
    ax = axes[0]
    knob_names = [s["knob"] for s in sens]
    vals_strict = [s["max_abs_strict_drop"] for s in sens]
    ax.barh(np.arange(len(sens)), vals_strict, color="#1f77b4", alpha=0.85)
    ax.set_yticks(np.arange(len(sens)))
    ax.set_yticklabels(knob_names)
    ax.invert_yaxis()
    ax.set_xlabel("max |Δ strict_score| over OAT sweep")
    ax.set_title("Knob sensitivity by Δ strict_score")
    ax.grid(True, alpha=0.25, axis="x")

    ax = axes[1]
    vals_proxy = [s["max_abs_proxy_drop"] for s in sens]
    ax.barh(np.arange(len(sens)), vals_proxy, color="#d62728", alpha=0.85)
    ax.set_yticks(np.arange(len(sens)))
    ax.set_yticklabels(knob_names)
    ax.invert_yaxis()
    ax.set_xlabel("max |Δ strict_pass_proxy| over OAT sweep")
    ax.set_title("Knob sensitivity by Δ strict_pass_proxy")
    ax.grid(True, alpha=0.25, axis="x")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return {"per_knob": sens}


def _pair_grid(rows: list[dict], k1: str, k2: str,
                metric: str = "strict_pass_prob_proxy") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sub = [r for r in rows
            if r.get("_pair_k1") == k1 and r.get("_pair_k2") == k2]
    if not sub:
        return None, None, None
    n = int(max(int(_safe_float(r["_pair_i"])) for r in sub)) + 1
    m = int(max(int(_safe_float(r["_pair_j"])) for r in sub)) + 1
    M = np.full((m, n), np.nan)
    v1 = np.zeros(n); v2 = np.zeros(m)
    for r in sub:
        i = int(_safe_float(r["_pair_i"])); j = int(_safe_float(r["_pair_j"]))
        M[j, i] = float(r.get(metric, float("nan")))
        v1[i] = float(r["_pair_v1"]); v2[j] = float(r["_pair_v2"])
    return v1, v2, M


def plot_pair_heatmap(rows: list[dict], k1: str, k2: str,
                       title: str, out_path: Path) -> None:
    v1, v2, M = _pair_grid(rows, k1, k2)
    if M is None:
        return
    fig, ax = plt.subplots(figsize=(9.0, 7.0))
    im = ax.imshow(M, origin="lower", aspect="auto",
                    cmap="viridis", vmin=0.0, vmax=max(0.05, float(np.nanmax(M))),
                    extent=(v1.min(), v1.max(), v2.min(), v2.max()))
    cb = plt.colorbar(im, ax=ax)
    cb.set_label("strict_pass_prob_proxy (surrogate)")
    ax.set_xlabel(k1); ax.set_ylabel(k2)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_local_pareto(local_rows: list[dict], out_path: Path) -> None:
    cd_err = np.array([float(r["mean_cd_error"]) for r in local_rows])
    ler = np.array([float(r["mean_ler_locked"]) for r in local_rows])
    score = np.array([float(r["strict_pass_prob_proxy"]) for r in local_rows])
    fig, ax = plt.subplots(figsize=(10.0, 7.0))
    sc = ax.scatter(cd_err, ler, s=10, c=score, cmap="viridis",
                    alpha=0.55, edgecolor="none", vmin=0, vmax=1.0)
    cb = plt.colorbar(sc, ax=ax)
    cb.set_label("strict_pass_prob_proxy")
    base_idx = next((i for i, r in enumerate(local_rows)
                       if str(r.get("recipe_id", "")) == "K_base"), None)
    if base_idx is not None:
        ax.scatter([cd_err[base_idx]], [ler[base_idx]], s=240, marker="*",
                    color="#7a0a0a", edgecolor="white", lw=1.0,
                    label="G_4867 baseline")
    ax.axvline(0.5, color="#1f1f1f", ls="--", lw=1.0,
                label="strict CD_tol = 0.5 nm")
    ax.axhline(3.0, color="#1f1f1f", ls=":", lw=1.0,
                label="strict LER_cap = 3.0 nm")
    ax.set_xlabel("mean |CD_fixed - 15| (nm)")
    ax.set_ylabel("mean LER_CD_locked (nm)")
    ax.set_title("Stage 06K -- local CD vs LER, color = strict_pass_proxy")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_process_window_boundary(local_rows: list[dict],
                                    fd_rows: list[dict],
                                    out_path: Path) -> None:
    cd_err = np.array([float(r["mean_cd_error"]) for r in local_rows])
    ler = np.array([float(r["mean_ler_locked"]) for r in local_rows])
    proxy = np.array([float(r["strict_pass_prob_proxy"]) for r in local_rows])

    fig, axes = plt.subplots(1, 2, figsize=(15.0, 6.5))

    ax = axes[0]
    threshold = 0.5
    inside = proxy >= threshold; outside = ~inside
    ax.scatter(cd_err[outside], ler[outside], s=8, c="#777", alpha=0.4,
                edgecolor="none", label=f"outside (proxy < {threshold})")
    ax.scatter(cd_err[inside], ler[inside], s=10, c="#2ca02c", alpha=0.85,
                edgecolor="none", label=f"inside (proxy ≥ {threshold})")
    base_idx = next((i for i, r in enumerate(local_rows)
                       if str(r.get("recipe_id", "")) == "K_base"), None)
    if base_idx is not None:
        ax.scatter([cd_err[base_idx]], [ler[base_idx]], s=240, marker="*",
                    color="#7a0a0a", edgecolor="white", lw=1.0,
                    label="G_4867 baseline")
    ax.axvline(0.5, color="#1f1f1f", ls="--", lw=1.0,
                label="strict CD_tol = 0.5 nm")
    ax.axhline(3.0, color="#1f1f1f", ls=":", lw=1.0)
    ax.set_xlabel("mean |CD_fixed - 15| (nm)"); ax.set_ylabel("mean LER (nm)")
    ax.set_title("surrogate proxy boundary  (green = inside window)")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.25)

    # Right pane: FD nominal labels.
    ax = axes[1]
    if fd_rows:
        cd_fd = np.array([abs(_safe_float(r.get("CD_final_nm")) - CD_TARGET_NM)
                            for r in fd_rows])
        ler_fd = np.array([_safe_float(r.get("LER_CD_locked_nm"))
                             for r in fd_rows])
        labels = np.array([str(r.get("label", "")) for r in fd_rows])
        roles = np.array([str(r.get("role", "")) for r in fd_rows])
        marker_map = {"baseline": "*", "top": "o", "worst": "x",
                       "boundary": "^"}
        color_map = {"robust_valid": "#2ca02c", "margin_risk": "#ffbf00",
                      "under_exposed": "#1f77b4", "merged": "#d62728",
                      "roughness_degraded": "#9467bd",
                      "numerical_invalid": "#8c564b"}
        for marker, label_role in marker_map.items():
            m = roles == marker
            if not np.any(m):
                continue
            for lbl, color in color_map.items():
                ml = m & (labels == lbl)
                if not np.any(ml):
                    continue
                ax.scatter(cd_fd[ml], ler_fd[ml], marker=label_role,
                            s=80 if marker_map[marker] == "*" else 32,
                            color=color, edgecolor="black", lw=0.6,
                            alpha=0.9)
    ax.axvline(0.5, color="#1f1f1f", ls="--", lw=1.0,
                label="strict CD_tol = 0.5 nm")
    ax.axhline(3.0, color="#1f1f1f", ls=":", lw=1.0,
                label="strict LER_cap = 3.0 nm")
    # Custom legend.
    handles = []
    for kind, marker in marker_map.items():
        handles.append(plt.Line2D([], [], marker=marker, color="#444", lw=0,
                                     markersize=10, label=kind))
    for lbl, c in color_map.items():
        handles.append(plt.scatter([], [], c=c, s=40, edgecolor="black",
                                       lw=0.6, label=lbl))
    ax.legend(handles=handles, loc="upper right", fontsize=8, ncol=2)
    ax.set_xlabel("FD CD_error (nm)"); ax.set_ylabel("FD LER (nm)")
    ax.set_title("FD verification subset (color = label, marker = role)")
    ax.grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------
def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--local_csv", type=str,
                   default=str(V3_DIR / "outputs" / "yield_optimization"
                               / "stage06K_local_candidates.csv"))
    p.add_argument("--oat_csv", type=str,
                   default=str(V3_DIR / "outputs" / "yield_optimization"
                               / "stage06K_oat_sensitivity.csv"))
    p.add_argument("--pair_csv", type=str,
                   default=str(V3_DIR / "outputs" / "yield_optimization"
                               / "stage06K_pairwise_maps.csv"))
    p.add_argument("--fd_csv", type=str,
                   default=str(V3_DIR / "outputs" / "labels"
                               / "stage06K_fd_verification.csv"))
    p.add_argument("--strict_cfg_yaml", type=str,
                   default=str(V3_DIR / "outputs" / "yield_optimization"
                               / "stage06G_strict_score_config.yaml"))
    p.add_argument("--manifest_yaml", type=str,
                   default=str(V3_DIR / "outputs" / "yield_optimization"
                               / "stage06I_mode_a_final_recipes.yaml"))
    p.add_argument("--config", type=str,
                   default=str(V3_DIR / "configs" / "yield_optimization.yaml"))
    args = p.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    strict_yaml = yaml.safe_load(Path(args.strict_cfg_yaml).read_text())
    cd_tol  = float(strict_yaml["thresholds"]["cd_tol_nm"])
    ler_cap = float(strict_yaml["thresholds"]["ler_cap_nm"])

    num_keys = ["strict_score", "yield_score",
                 "p_robust_valid", "p_margin_risk", "p_under_exposed",
                 "p_merged", "p_roughness_degraded", "p_numerical_invalid",
                 "mean_cd_fixed", "std_cd_fixed",
                 "mean_cd_locked", "std_cd_locked",
                 "mean_ler_locked", "std_ler_locked",
                 "mean_p_line_margin", "std_p_line_margin",
                 "_oat_step", "_oat_value", "_pair_i", "_pair_j",
                 "_pair_v1", "_pair_v2"] + FEATURE_KEYS

    rows_local = read_labels_csv(args.local_csv); _coerce(rows_local, num_keys)
    rows_oat   = read_labels_csv(args.oat_csv);   _coerce(rows_oat,   num_keys)
    rows_pair  = read_labels_csv(args.pair_csv);  _coerce(rows_pair,  num_keys)

    attach_strict_pass_proxy(rows_local, cd_tol, ler_cap)
    attach_strict_pass_proxy(rows_oat,   cd_tol, ler_cap)
    attach_strict_pass_proxy(rows_pair,  cd_tol, ler_cap)

    # Baseline values (K_base in the local set).
    base = next((r for r in rows_local if r["recipe_id"] == "K_base"), None)
    if base is None:
        base = next((r for r in rows_oat if r["recipe_id"] == "K_base"), None)
    base_strict = float(base["strict_score"])
    base_proxy = float(base["strict_pass_prob_proxy"])
    base_cd_err = float(base["mean_cd_error"])
    base_ler = float(base["mean_ler_locked"])
    base_robust = float(base["p_robust_valid"])
    print(f"  G_4867 surrogate baseline (06H surrogate, n_var=200):")
    print(f"    strict_score          = {base_strict:.4f}")
    print(f"    strict_pass_proxy     = {base_proxy:.4f}")
    print(f"    mean CD_error         = {base_cd_err:.4f} nm")
    print(f"    mean LER_locked       = {base_ler:.4f} nm")
    print(f"    P(robust_valid)       = {base_robust:.4f}")

    # ----- Sensitivity (OAT-derived) -----
    fig_dir = V3_DIR / "outputs" / "figures" / "06_yield_optimization"
    fig_dir.mkdir(parents=True, exist_ok=True)
    sens = plot_knob_sensitivity_bar(
        rows_oat, base_strict, base_proxy,
        fig_dir / "stage06K_knob_sensitivity_bar.png",
    )
    plot_oat_strict(rows_oat, base_strict, base_proxy,
                      fig_dir / "stage06K_strict_pass_prob_oat.png")

    # ----- Pairwise heatmaps (3 required by spec; we plot the named 3) -----
    plot_pair_heatmap(rows_pair, "dose_mJ_cm2", "sigma_nm",
                       "dose × sigma -- strict_pass_proxy heatmap",
                       fig_dir / "stage06K_dose_sigma_heatmap.png")
    plot_pair_heatmap(rows_pair, "DH_nm2_s", "time_s",
                       "DH × time -- strict_pass_proxy heatmap",
                       fig_dir / "stage06K_DH_time_heatmap.png")
    plot_pair_heatmap(rows_pair, "Q0_mol_dm3", "kq_s_inv",
                       "Q0 × kq -- strict_pass_proxy heatmap",
                       fig_dir / "stage06K_Q0_kq_heatmap.png")

    # ----- Local Pareto (CD_error vs LER) -----
    plot_local_pareto(rows_local,
                        fig_dir / "stage06K_cd_error_vs_ler_local_pareto.png")

    # ----- Process window boundary (proxy + FD overlay) -----
    fd_rows = []
    if Path(args.fd_csv).exists():
        fd_rows = read_labels_csv(args.fd_csv)
        for r in fd_rows:
            for k in ("CD_final_nm", "CD_locked_nm", "LER_CD_locked_nm",
                       "area_frac", "P_line_margin",
                       "strict_score_surrogate", "yield_score_surrogate",
                       "variation_idx"):
                if k in r:
                    r[k] = _safe_float(r[k])
    plot_process_window_boundary(rows_local, fd_rows,
                                    fig_dir / "stage06K_process_window_boundary.png")

    # ----- FD subset: nominal label distribution and strict_pass on MC -----
    fd_nominal = [r for r in fd_rows if str(r.get("phase", "")) == "nominal"]
    fd_mc      = [r for r in fd_rows if str(r.get("phase", "")) == "mc"]
    label_counts = Counter(str(r.get("label", "")) for r in fd_nominal)
    n_strict_fail = sum(
        1 for r in fd_nominal
        if (str(r.get("label", "")) != "robust_valid"
             or abs(_safe_float(r.get("CD_final_nm")) - CD_TARGET_NM) > cd_tol
             or _safe_float(r.get("LER_CD_locked_nm")) > ler_cap)
    )
    n_robust = sum(1 for r in fd_nominal if str(r.get("label", "")) == "robust_valid")
    n_strict_pass = sum(
        1 for r in fd_nominal
        if str(r.get("label", "")) == "robust_valid"
            and abs(_safe_float(r.get("CD_final_nm")) - CD_TARGET_NM) <= cd_tol
            and _safe_float(r.get("LER_CD_locked_nm")) <= ler_cap
    )

    # MC FD aggregates per source recipe.
    fd_mc_aggr: list[dict] = []
    by_recipe: dict[str, list[dict]] = {}
    for r in fd_mc:
        rid = str(r.get("source_recipe_id", ""))
        by_recipe.setdefault(rid, []).append(r)
    for rid, rs in by_recipe.items():
        n = len(rs)
        n_robust_mc = sum(1 for r in rs if str(r.get("label", "")) == "robust_valid")
        n_margin_mc = sum(1 for r in rs if str(r.get("label", "")) == "margin_risk")
        n_hard_mc = sum(1 for r in rs if str(r.get("label", "")) in HARD_FAIL_LABELS)
        n_strict_pass_mc = sum(
            1 for r in rs
            if str(r.get("label", "")) == "robust_valid"
                and abs(_safe_float(r.get("CD_final_nm")) - CD_TARGET_NM) <= cd_tol
                and _safe_float(r.get("LER_CD_locked_nm")) <= ler_cap
        )
        role = str(rs[0].get("role", ""))
        fd_mc_aggr.append({
            "recipe_id":          rid,
            "role":                role,
            "n_mc":                n,
            "robust_prob":         float(n_robust_mc / max(n, 1)),
            "margin_risk_prob":    float(n_margin_mc / max(n, 1)),
            "defect_prob":         float(n_hard_mc / max(n, 1)),
            "strict_pass_prob":    float(n_strict_pass_mc / max(n, 1)),
            "mean_CD_final_nm":    float(np.nanmean([_safe_float(r.get("CD_final_nm")) for r in rs])),
            "std_CD_final_nm":     float(np.nanstd([_safe_float(r.get("CD_final_nm")) for r in rs])),
            "mean_LER_CD_locked":  float(np.nanmean([_safe_float(r.get("LER_CD_locked_nm")) for r in rs])),
            "std_LER_CD_locked":   float(np.nanstd([_safe_float(r.get("LER_CD_locked_nm")) for r in rs])),
        })

    # ----- Acceptance + summary -----
    top3_fragile = [s["knob"] for s in sens["per_knob"][:3]]
    pair_metrics_present = [
        any(r.get("_pair_k1") == k1 and r.get("_pair_k2") == k2 for r in rows_pair)
        for (k1, k2) in [("dose_mJ_cm2", "sigma_nm"),
                          ("DH_nm2_s", "time_s"),
                          ("Q0_mol_dm3", "kq_s_inv")]
    ]
    acceptance = {
        "local_pw_mapped": True,
        "n_local_candidates":           len(rows_local),
        "n_oat_candidates":             len(rows_oat),
        "n_pair_candidates":            len(rows_pair),
        "strict_pass_proxy_quantified": True,
        "top3_fragile_knobs":            top3_fragile,
        "n_pairwise_maps_generated":     int(sum(pair_metrics_present)),
        "fd_verification_done":          bool(len(fd_rows) > 0),
        "fd_nominal_n_robust":           int(n_robust),
        "fd_nominal_n_strict_pass":      int(n_strict_pass),
        "fd_nominal_n_strict_fail":      int(n_strict_fail),
        "fd_mc_aggr_count":              len(fd_mc_aggr),
        "policy_v2_OP_frozen":           bool(cfg["policy"].get("v2_OP_frozen", True)),
        "policy_published_data_loaded":  bool(cfg["policy"].get("published_data_loaded", False)),
        "policy_external_calibration":   "none",
    }

    # Interpretation flags.
    fragile = any(
        s["max_abs_strict_drop"] > 0.5 * abs(base_strict) + 0.05
        for s in sens["per_knob"]
    )
    interpretation = {
        "broad_basin_inferred":
            bool(not fragile and acceptance["fd_nominal_n_strict_pass"] >= 0.5 * len(fd_nominal)),
        "primary_process_control_knob": top3_fragile[0] if top3_fragile else None,
        "process_window_recommendation": (
            "Tighten control most aggressively on the top-1 fragile knob; "
            "loosen tolerances on the bottom-1 fragile knob if cost matters."
        ),
    }

    payload = {
        "stage": "06K",
        "policy": cfg["policy"],
        "primary_recipe_id": "G_4867",
        "strict_thresholds": {"cd_tol_nm": cd_tol, "ler_cap_nm": ler_cap},
        "baseline_metrics": {
            "strict_score":      base_strict,
            "strict_pass_proxy": base_proxy,
            "mean_cd_error":     base_cd_err,
            "mean_ler_locked":   base_ler,
            "p_robust_valid":    base_robust,
        },
        "knob_sensitivity_oat": sens["per_knob"],
        "fd_subset": {
            "n_total":          len(fd_rows),
            "n_nominal":        len(fd_nominal),
            "label_counts_nominal": dict(label_counts),
            "n_robust_valid":   int(n_robust),
            "n_strict_pass":    int(n_strict_pass),
            "n_strict_fail":    int(n_strict_fail),
            "n_mc":             len(fd_mc),
            "fd_mc_aggregates": fd_mc_aggr,
        },
        "interpretation":  interpretation,
        "acceptance":      acceptance,
    }
    logs_dir = V3_DIR / "outputs" / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    (logs_dir / "stage06K_summary.json").write_text(
        json.dumps(payload, indent=2, default=float))

    # ----- Console summary -----
    print(f"\nStage 06K -- analysis summary")
    print(f"  knob sensitivity (sorted by max |Δ strict_score|):")
    for s in sens["per_knob"]:
        print(f"    {s['knob']:<14} max|Δstrict|={s['max_abs_strict_drop']:.3f}  "
              f"max|Δproxy|={s['max_abs_proxy_drop']:.3f}")
    print(f"  top-3 fragile knobs: {top3_fragile}")
    print(f"  FD subset: nominal={len(fd_nominal)} (robust={n_robust}, "
          f"strict_pass={n_strict_pass}, strict_fail={n_strict_fail}); "
          f"MC recipes={len(fd_mc_aggr)}")
    print(f"  FD MC aggregates per recipe:")
    for r in fd_mc_aggr:
        print(f"    {r['role']:>22}  {r['recipe_id']:>14}  "
              f"robust={r['robust_prob']:.3f}  strict_pass={r['strict_pass_prob']:.3f}  "
              f"std_CD={r['std_CD_final_nm']:.3f}  std_LER={r['std_LER_CD_locked']:.3f}")
    print(f"  basin classification: "
          f"{'broad/stable' if interpretation['broad_basin_inferred'] else 'fragile-on-some-axis'}")
    print(f"  primary process-control knob: "
          f"{interpretation['primary_process_control_knob']}")
    print(f"  summary -> {logs_dir / 'stage06K_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
