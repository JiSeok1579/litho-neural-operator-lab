"""Stage 06E -- analysis, figures, AL additions for the FD verification
of Stage 06D recipes.

Reads
    outputs/yield_optimization/stage06D_top_recipes.csv
    outputs/yield_optimization/stage06D_disagreement_candidates.csv
    outputs/yield_optimization/stage06D_recipe_summary.csv
    outputs/labels/stage06E_fd_baseline_v2_op.csv          (Stage 06E Part 0)
    outputs/labels/stage06E_fd_top100_nominal.csv          (Stage 06E Part 1)
    outputs/labels/stage06E_fd_top10_mc.csv                (Stage 06E Part 2)
    outputs/labels/stage06E_fd_disagreement.csv            (Stage 06E Part 3)
    outputs/labels/fd_top10_mc_verification.csv            (Stage 06B, optional)
    outputs/labels/06_top_recipes_fixed_design_surrogate.csv (Stage 06A)
    outputs/logs/stage06D_summary.json
    outputs/logs/06_yield_optimization_summary.json

Writes
    outputs/labels/stage06E_surrogate_vs_fd_metrics.csv
    outputs/labels/stage06E_false_pass_cases.csv
    outputs/labels/stage06E_disagreement_fd_breakdown.csv
    outputs/labels/stage06E_al_additions.csv
    outputs/logs/stage06E_summary.json
    outputs/logs/stage06E_false_pass_summary.json
    outputs/figures/06_yield_optimization/
        stage06E_surrogate_vs_fd_yield_score_scatter.png
        stage06E_surrogate_vs_fd_cd_ler_scatter.png
        stage06E_top10_fd_yield_barplot.png
        stage06E_defect_breakdown_top10.png
        stage06E_false_pass_parameter_parallel_coordinates.png
        stage06E_disagreement_fd_breakdown.png
        stage06E_06D_vs_06A_fd_top10.png

Policy
    v2_OP_frozen           = true
    published_data_loaded  = false
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

from reaction_diffusion_peb_v3_screening.src.fd_yield_score import (
    fd_yield_score_per_recipe,
    nominal_yield_score,
    spearman,
    topk_overlap,
)
from reaction_diffusion_peb_v3_screening.src.metrics_io import read_labels_csv
from reaction_diffusion_peb_v3_screening.src.yield_optimizer import (
    YieldScoreConfig,
)


V3_DIR = ROOT / "reaction_diffusion_peb_v3_screening"

PARAM_AXES = [
    "dose_mJ_cm2", "sigma_nm", "DH_nm2_s", "time_s",
    "Hmax_mol_dm3", "kdep_s_inv", "Q0_mol_dm3", "kq_s_inv",
    "line_cd_ratio",
]

HARD_FAIL_LABELS = ("under_exposed", "merged",
                    "roughness_degraded", "numerical_invalid")
SOFT_FAIL_LABEL  = "margin_risk"


def _safe_float(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def _coerce_floats(rows: list[dict], keys: list[str]) -> None:
    for r in rows:
        for k in keys:
            if k in r:
                r[k] = _safe_float(r.get(k))


# --------------------------------------------------------------------------
# Surrogate vs FD per-recipe metrics on top-100 nominal.
# --------------------------------------------------------------------------
def build_surrogate_vs_fd_metrics(top_06d: list[dict],
                                   fd_nominal: list[dict],
                                   score_cfg: YieldScoreConfig) -> list[dict]:
    fd_by_id = {r["source_recipe_id"]: r for r in fd_nominal}
    out = []
    for s in top_06d:
        rid = s["recipe_id"]
        fd_row = fd_by_id.get(rid)
        merged = {
            "recipe_id": rid,
            "rank_06c":           int(_safe_float(s.get("rank_06c", 0))),
            "yield_score_06c":    _safe_float(s.get("yield_score_06c")),
            "yield_score_06a":    _safe_float(s.get("yield_score_06a")),
            "p_robust_valid_06c": _safe_float(s.get("p_robust_valid_06c")),
            "p_robust_valid_06a": _safe_float(s.get("p_robust_valid_06a")),
            "mean_cd_fixed_06c":  _safe_float(s.get("mean_cd_fixed_06c")),
            "mean_ler_locked_06c": _safe_float(s.get("mean_ler_locked_06c")),
        }
        if fd_row is None:
            merged.update({
                "fd_label":              "",
                "FD_yield_score_nominal": float("nan"),
                "CD_final_fd":           float("nan"),
                "CD_locked_fd":          float("nan"),
                "LER_CD_locked_fd":      float("nan"),
                "area_frac_fd":          float("nan"),
                "P_line_margin_fd":      float("nan"),
            })
        else:
            nom = nominal_yield_score(fd_row, score_cfg)
            merged.update({
                "fd_label":               str(fd_row.get("label", "")),
                "FD_yield_score_nominal": float(nom["FD_yield_score"]),
                "CD_final_fd":      _safe_float(fd_row.get("CD_final_nm")),
                "CD_locked_fd":     _safe_float(fd_row.get("CD_locked_nm")),
                "LER_CD_locked_fd": _safe_float(fd_row.get("LER_CD_locked_nm")),
                "area_frac_fd":     _safe_float(fd_row.get("area_frac")),
                "P_line_margin_fd": _safe_float(fd_row.get("P_line_margin")),
            })
        out.append(merged)
    return out


def regression_target_errors(rows: list[dict]) -> dict:
    pairs = {
        "CD_fixed":   ("mean_cd_fixed_06c",  "CD_final_fd"),
        "LER_locked": ("mean_ler_locked_06c", "LER_CD_locked_fd"),
    }
    out = {}
    for name, (a_key, b_key) in pairs.items():
        a = np.array([r[a_key] for r in rows], dtype=np.float64)
        b = np.array([r[b_key] for r in rows], dtype=np.float64)
        finite = np.isfinite(a) & np.isfinite(b)
        if finite.sum() == 0:
            out[name] = {"mae": None, "rmse": None, "n": 0}
            continue
        d = a[finite] - b[finite]
        out[name] = {
            "mae":  float(np.mean(np.abs(d))),
            "rmse": float(np.sqrt(np.mean(d ** 2))),
            "n":    int(finite.sum()),
        }
    return out


# --------------------------------------------------------------------------
# Top-10 MC FD scoring + ranking.
# --------------------------------------------------------------------------
def score_top10_mc(fd_mc_rows: list[dict],
                   score_cfg: YieldScoreConfig) -> list[dict]:
    by_recipe = fd_yield_score_per_recipe(fd_mc_rows, score_cfg)
    rank_lookup = {}
    for fd in fd_mc_rows:
        rid = fd.get("source_recipe_id")
        if rid not in rank_lookup:
            rank_lookup[rid] = int(_safe_float(fd.get("rank_surrogate", 0)))
    rows = []
    for rid, payload in by_recipe.items():
        out = dict(payload)
        out["recipe_id"] = rid
        out["rank_surrogate"] = rank_lookup.get(rid, 0)
        rows.append(out)
    rows.sort(key=lambda r: r["rank_surrogate"])
    return rows


# --------------------------------------------------------------------------
# False-PASS detection.
# --------------------------------------------------------------------------
def detect_false_pass(nominal_rows: list[dict],
                       top_06d: list[dict]) -> tuple[list[dict], dict]:
    sur_lookup = {s["recipe_id"]: s for s in top_06d}
    rows: list[dict] = []
    n_total = len(nominal_rows)
    counts = Counter(r.get("label", "") for r in nominal_rows)

    for fd in nominal_rows:
        label = str(fd.get("label", ""))
        if label == "robust_valid":
            continue
        rid = fd.get("source_recipe_id")
        s = sur_lookup.get(rid, {})
        kind = (
            "hard_false_pass" if label in HARD_FAIL_LABELS
            else "soft_false_pass" if label == SOFT_FAIL_LABEL
            else "other_non_robust"
        )
        rows.append({
            "recipe_id":              rid,
            "rank_06c":               int(_safe_float(s.get("rank_06c", 0))),
            "fd_label":               label,
            "false_pass_kind":        kind,
            "yield_score_06c":        _safe_float(s.get("yield_score_06c")),
            "yield_score_06a":        _safe_float(s.get("yield_score_06a")),
            "p_robust_valid_06c":     _safe_float(s.get("p_robust_valid_06c")),
            "CD_final_fd":            _safe_float(fd.get("CD_final_nm")),
            "CD_locked_fd":           _safe_float(fd.get("CD_locked_nm")),
            "LER_CD_locked_fd":       _safe_float(fd.get("LER_CD_locked_nm")),
            "area_frac_fd":           _safe_float(fd.get("area_frac")),
            "P_line_margin_fd":       _safe_float(fd.get("P_line_margin")),
            **{k: _safe_float(s.get(k)) for k in PARAM_AXES if k in s},
        })

    summary = {
        "n_top": n_total,
        "label_breakdown": dict(counts),
        "n_hard_false_pass": int(sum(1 for r in rows if r["false_pass_kind"] == "hard_false_pass")),
        "n_soft_false_pass": int(sum(1 for r in rows if r["false_pass_kind"] == "soft_false_pass")),
        "hard_false_pass_rate": float(sum(1 for r in rows if r["false_pass_kind"] == "hard_false_pass") / max(n_total, 1)),
        "soft_false_pass_rate": float(sum(1 for r in rows if r["false_pass_kind"] == "soft_false_pass") / max(n_total, 1)),
    }
    return rows, summary


# --------------------------------------------------------------------------
# Disagreement-candidate FD breakdown.
# --------------------------------------------------------------------------
def build_disagreement_breakdown(disagree_rows: list[dict],
                                  fd_disagree: list[dict],
                                  score_cfg: YieldScoreConfig) -> tuple[list[dict], dict]:
    fd_by_id = {r["source_recipe_id"]: r for r in fd_disagree}
    out = []
    label_counter = Counter()
    for s in disagree_rows:
        rid = s["recipe_id"]
        fd_row = fd_by_id.get(rid)
        if fd_row is None:
            continue
        nom = nominal_yield_score(fd_row, score_cfg)
        label = str(fd_row.get("label", ""))
        label_counter[label] += 1
        # Which surrogate "wins" the label? Compare against FD label.
        agree_06c = (label == "robust_valid"
                     and _safe_float(s.get("yield_score_06c")) > 0.5)
        agree_06a = (label == "robust_valid"
                     and _safe_float(s.get("yield_score_06a")) > 0.5)
        out.append({
            "recipe_id":              rid,
            "rank_06c":               int(_safe_float(s.get("rank_06c", 0))),
            "yield_score_06c":        _safe_float(s.get("yield_score_06c")),
            "yield_score_06a":        _safe_float(s.get("yield_score_06a")),
            "score_gap":              _safe_float(s.get("score_gap")),
            "fd_label":               label,
            "FD_yield_score_nominal": float(nom["FD_yield_score"]),
            "06c_agrees_with_fd":     bool(agree_06c),
            "06a_agrees_with_fd":     bool(agree_06a),
            "CD_final_fd":            _safe_float(fd_row.get("CD_final_nm")),
            "LER_CD_locked_fd":       _safe_float(fd_row.get("LER_CD_locked_nm")),
            "P_line_margin_fd":       _safe_float(fd_row.get("P_line_margin")),
        })
    summary = {
        "n_disagreement":     len(out),
        "label_breakdown":    dict(label_counter),
        "n_robust_valid":     int(label_counter.get("robust_valid", 0)),
        "n_margin_risk":      int(label_counter.get("margin_risk", 0)),
        "n_hard_fail":        int(sum(label_counter.get(k, 0) for k in HARD_FAIL_LABELS)),
        "06c_correct":        int(sum(1 for r in out if r["06c_agrees_with_fd"])),
        "06a_correct":        int(sum(1 for r in out if r["06a_agrees_with_fd"])),
        "06c_correct_rate":   float(sum(1 for r in out if r["06c_agrees_with_fd"]) / max(len(out), 1)),
    }
    return out, summary


# --------------------------------------------------------------------------
# AL additions writer.
# --------------------------------------------------------------------------
def write_al_additions(parts: list[list[dict]], out_path: Path) -> int:
    rows = [r for src in parts for r in src]
    if not rows:
        return 0
    cols = list(rows[0].keys())
    extras = []
    for src in parts:
        for r in src:
            for k in r.keys():
                if k not in cols and k not in extras:
                    extras.append(k)
    cols = cols + extras
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return len(rows)


# --------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------
def plot_yield_score_scatter(pair_rows, baseline, out_path):
    sx = np.array([r["yield_score_06c"]      for r in pair_rows])
    sy = np.array([r["FD_yield_score_nominal"] for r in pair_rows])
    fig, ax = plt.subplots(figsize=(8.5, 7.0))
    ax.scatter(sx, sy, s=30, c="#d62728", alpha=0.75,
               edgecolor="white", lw=0.5, label="06D top-100")
    lo = float(np.nanmin([np.nanmin(sx), np.nanmin(sy), -0.5]))
    hi = float(np.nanmax([np.nanmax(sx), np.nanmax(sy), 1.05]))
    ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, alpha=0.5, label="y = x")
    if baseline is not None:
        ax.axhline(baseline, color="#1f77b4", ls=":", lw=1.0,
                   label=f"v2 frozen OP FD yield_score = {baseline:.3f}")
    rho = spearman(sx, sy)
    ax.set_xlabel("06C surrogate yield_score (mean over 200 MC variations)")
    ax.set_ylabel("Stage 06E FD yield_score (single nominal FD)")
    ax.set_title(f"Stage 06E -- 06C surrogate vs FD on 06D top-100  "
                 f"(Spearman rho = {rho:.3f})")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_cd_ler_scatter(pair_rows, out_path):
    cd_s = np.array([r["mean_cd_fixed_06c"] for r in pair_rows])
    cd_f = np.array([r["CD_final_fd"]        for r in pair_rows])
    ler_s = np.array([r["mean_ler_locked_06c"] for r in pair_rows])
    ler_f = np.array([r["LER_CD_locked_fd"]     for r in pair_rows])
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 6.0))

    ax = axes[0]
    ax.scatter(cd_s, cd_f, s=22, c="#2ca02c", alpha=0.7, edgecolor="white", lw=0.5)
    lo = float(np.nanmin([np.nanmin(cd_s), np.nanmin(cd_f)]))
    hi = float(np.nanmax([np.nanmax(cd_s), np.nanmax(cd_f)]))
    pad = 0.05 * (hi - lo + 1e-9)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad],
            "k--", lw=0.8, alpha=0.5, label="y = x")
    rho = spearman(cd_s, cd_f)
    ax.axhspan(14.0, 16.0, color="#aaa", alpha=0.15, label="CD target +/-1 nm")
    ax.set_xlabel("06C surrogate mean CD_fixed (nm)")
    ax.set_ylabel("FD nominal CD_final (nm)")
    ax.set_title(f"CD_fixed: 06C vs FD  (Spearman rho = {rho:.3f})")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.25)

    ax = axes[1]
    ax.scatter(ler_s, ler_f, s=22, c="#d62728", alpha=0.7, edgecolor="white", lw=0.5)
    lo = float(np.nanmin([np.nanmin(ler_s), np.nanmin(ler_f)]))
    hi = float(np.nanmax([np.nanmax(ler_s), np.nanmax(ler_f)]))
    pad = 0.05 * (hi - lo + 1e-9)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad],
            "k--", lw=0.8, alpha=0.5, label="y = x")
    rho = spearman(ler_s, ler_f)
    ax.axhspan(0, 3.0, color="#aaa", alpha=0.15, label="LER <= 3.0 nm")
    ax.set_xlabel("06C surrogate mean LER_CD_locked (nm)")
    ax.set_ylabel("FD nominal LER_CD_locked (nm)")
    ax.set_title(f"LER_locked: 06C vs FD  (Spearman rho = {rho:.3f})")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_top10_fd_yield_bar(top10_rows, baseline, out_path):
    rows = sorted(top10_rows, key=lambda r: r["rank_surrogate"])
    fig, ax = plt.subplots(figsize=(11.0, 5.5))
    x = np.arange(len(rows))
    sur_score = [r.get("FD_yield_score", 0) for r in rows]
    ax.bar(x, sur_score, color="#d62728", alpha=0.85,
           edgecolor="#1f1f1f", label="FD MC yield_score (100 variations)")
    if baseline is not None:
        ax.axhline(baseline, color="#1f77b4", ls="--", lw=1.2,
                   label=f"v2 frozen OP FD yield_score = {baseline:.3f}")
    for i, r in enumerate(rows):
        ax.text(i, r["FD_yield_score"] + 0.01,
                f"{r['FD_yield_score']:.3f}",
                ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"#{r['rank_surrogate']}\n{r['recipe_id']}" for r in rows],
                       fontsize=8)
    ax.set_xlabel("06D top-10 surrogate-rank recipe (Mode A)")
    ax.set_ylabel("FD MC yield_score")
    ax.set_title("Stage 06E -- top-10 06D FD Monte-Carlo yield_score")
    ymin = min(0.0, (baseline or 0.0) - 0.05)
    ax.set_ylim(ymin, 1.05)
    ax.grid(True, alpha=0.25, axis="y")
    ax.legend(loc="lower right", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_defect_breakdown_top10(top10_rows, out_path):
    rows = sorted(top10_rows, key=lambda r: r["rank_surrogate"])
    classes = [
        ("P_FD_robust_valid", "robust_valid"),
        ("P_FD_margin_risk", "margin_risk"),
        ("P_FD_under_exposed", "under_exposed"),
        ("P_FD_merged", "merged"),
        ("P_FD_roughness_degraded", "roughness"),
        ("P_FD_numerical_invalid", "numerical"),
    ]
    M = np.array([[r.get(k, 0.0) for k, _ in classes] for r in rows])
    fig, ax = plt.subplots(figsize=(11.0, 5.5))
    bottoms = np.zeros(len(rows))
    palette = ["#2ca02c", "#ffbf00", "#1f77b4", "#d62728", "#9467bd", "#8c564b"]
    for j, (col, label) in enumerate(classes):
        ax.bar(np.arange(len(rows)), M[:, j], bottom=bottoms,
               color=palette[j], alpha=0.85, label=label,
               edgecolor="white", lw=0.4)
        bottoms = bottoms + M[:, j]
    ax.set_xticks(np.arange(len(rows)))
    ax.set_xticklabels([f"#{r['rank_surrogate']}\n{r['recipe_id']}" for r in rows],
                       fontsize=8)
    ax.set_xlabel("06D top-10 surrogate-rank recipe (Mode A)")
    ax.set_ylabel("FD MC class probability")
    ax.set_title("Stage 06E top-10 06D FD MC defect breakdown (100 variations)")
    ax.legend(loc="lower right", fontsize=9, framealpha=0.95)
    ax.set_ylim(0, 1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_false_pass_parallel(false_rows, out_path):
    if not false_rows:
        fig, ax = plt.subplots(figsize=(11.0, 5.0))
        ax.text(0.5, 0.55, "Stage 06E -- false-PASS parallel coordinates",
                ha="center", va="center", fontsize=14, fontweight="bold")
        ax.text(0.5, 0.30,
                "Zero false-PASS cases in the top-100 nominal FD verification.\n"
                "All top-100 06D recipes were FD-labelled robust_valid.",
                ha="center", va="center", fontsize=11)
        ax.axis("off")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        return
    axes_keys = [k for k in PARAM_AXES if any(k in r for r in false_rows)]
    if not axes_keys:
        return
    fig, ax = plt.subplots(figsize=(13.0, 6.0))
    M = np.array([[r.get(k, np.nan) for k in axes_keys] for r in false_rows], dtype=float)
    lo = np.nanmin(M, axis=0); hi = np.nanmax(M, axis=0)
    span = np.where(hi - lo > 1e-12, hi - lo, 1.0)
    Mn = (M - lo) / span
    color_by_kind = {
        "hard_false_pass": "#7a0a0a",
        "soft_false_pass": "#ffbf00",
        "other_non_robust": "#777",
    }
    for i, r in enumerate(false_rows):
        ax.plot(range(len(axes_keys)), Mn[i], "-",
                color=color_by_kind.get(r["false_pass_kind"], "#1f77b4"),
                alpha=0.7, lw=1.4)
    ax.set_xticks(range(len(axes_keys)))
    ax.set_xticklabels(axes_keys, rotation=15)
    for i, k in enumerate(axes_keys):
        ax.text(i, -0.07, f"[{lo[i]:.2g}, {hi[i]:.2g}]",
                ha="center", fontsize=8, color="#555",
                transform=ax.get_xaxis_transform())
    ax.set_ylabel("normalised value")
    ax.set_title(f"Stage 06E false-PASS recipe parameter pattern  "
                 f"(N={len(false_rows)})")
    seen = set(); handles = []
    for r in false_rows:
        if r["false_pass_kind"] not in seen:
            seen.add(r["false_pass_kind"])
            handles.append(plt.Line2D([0], [0], color=color_by_kind[r["false_pass_kind"]],
                                       lw=1.5, label=r["false_pass_kind"]))
    if handles:
        ax.legend(handles=handles, loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_disagreement_breakdown(disagree_rows, out_path):
    if not disagree_rows:
        return
    rows = sorted(disagree_rows, key=lambda r: -r["yield_score_06c"])
    fig, ax = plt.subplots(figsize=(12.0, 5.5))
    x = np.arange(len(rows))
    score_06c = np.array([r["yield_score_06c"] for r in rows])
    score_06a = np.array([r["yield_score_06a"] for r in rows])
    fd_score = np.array([r["FD_yield_score_nominal"] for r in rows])
    fd_label = [r["fd_label"] for r in rows]

    label_color = {
        "robust_valid":       "#2ca02c",
        "margin_risk":        "#ffbf00",
        "under_exposed":      "#1f77b4",
        "merged":             "#d62728",
        "roughness_degraded": "#9467bd",
        "numerical_invalid":  "#8c564b",
    }
    bar_colors = [label_color.get(lbl, "#777") for lbl in fd_label]

    width = 0.27
    ax.bar(x - width, score_06c, width=width, color="#d62728", alpha=0.85, label="06C yield_score")
    ax.bar(x,         score_06a, width=width, color="#1f77b4", alpha=0.85, label="06A yield_score")
    ax.bar(x + width, fd_score,  width=width, color=bar_colors, alpha=0.95,
           edgecolor="#1f1f1f", lw=0.5, label="FD nominal yield_score (colored by FD label)")

    ax.axhline(0.0, color="#1f1f1f", lw=0.6, alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([r["recipe_id"] for r in rows], rotation=45, ha="right", fontsize=8)
    ax.set_xlabel("disagreement candidate (sorted by 06C yield_score)")
    ax.set_ylabel("yield_score")
    ax.set_title("Stage 06E -- 17 disagreement candidates: 06C vs 06A vs FD nominal")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c, alpha=0.95, label=lbl)
                for lbl, c in label_color.items()
                if lbl in {r["fd_label"] for r in rows}]
    leg1 = ax.legend(loc="upper right", fontsize=9)
    ax.add_artist(leg1)
    if handles:
        ax.legend(handles=handles, loc="lower right", fontsize=8, title="FD label", framealpha=0.95)
    ax.grid(True, alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_06d_vs_06a_top10(top10_06d_rows, top10_06a_rows, baseline, out_path):
    rows_d = sorted(top10_06d_rows, key=lambda r: r["rank_surrogate"])
    rows_a = sorted(top10_06a_rows, key=lambda r: r["rank_surrogate"]) if top10_06a_rows else []

    fig, ax = plt.subplots(figsize=(12.0, 5.8))
    n_d = len(rows_d); n_a = len(rows_a)
    width = 0.4
    if rows_a:
        x_d = np.arange(n_d) - width / 2
        x_a = np.arange(n_a) + width / 2
        ax.bar(x_a, [r["FD_yield_score"] for r in rows_a], width=width,
               color="#1f77b4", alpha=0.85, label="06A top-10 (Stage 06B FD MC)")
    else:
        x_d = np.arange(n_d)
    ax.bar(x_d, [r["FD_yield_score"] for r in rows_d], width=width,
           color="#d62728", alpha=0.85, label="06D top-10 (Stage 06E FD MC)")
    if baseline is not None:
        ax.axhline(baseline, color="#1f1f1f", ls="--", lw=1.0,
                   label=f"v2 frozen OP FD = {baseline:.3f}")
    n_max = max(n_d, n_a)
    ax.set_xticks(np.arange(n_max))
    ax.set_xticklabels([f"#{i+1}" for i in range(n_max)])
    ax.set_xlabel("Surrogate rank (top-10)")
    ax.set_ylabel("FD MC yield_score")
    ax.set_title("Stage 06E -- 06D vs 06A FD MC top-10 yield_score")
    ax.set_ylim(min(0.0, (baseline or 0.0) - 0.05), 1.05)
    ax.grid(True, alpha=0.25, axis="y")
    ax.legend(loc="lower right", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------
def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str,
                   default=str(V3_DIR / "configs" / "yield_optimization.yaml"))
    p.add_argument("--top_06d_csv", type=str,
                   default=str(V3_DIR / "outputs" / "yield_optimization"
                               / "stage06D_top_recipes.csv"))
    p.add_argument("--disagree_06d_csv", type=str,
                   default=str(V3_DIR / "outputs" / "yield_optimization"
                               / "stage06D_disagreement_candidates.csv"))
    p.add_argument("--top_06a_csv", type=str,
                   default=str(V3_DIR / "outputs" / "labels"
                               / "06_top_recipes_fixed_design_surrogate.csv"))
    p.add_argument("--fd_baseline_csv", type=str,
                   default=str(V3_DIR / "outputs" / "labels"
                               / "stage06E_fd_baseline_v2_op.csv"))
    p.add_argument("--fd_baseline_mc_csv", type=str,
                   default=str(V3_DIR / "outputs" / "labels"
                               / "stage06E_fd_baseline_v2_op_mc.csv"))
    p.add_argument("--fd_nominal_csv", type=str,
                   default=str(V3_DIR / "outputs" / "labels"
                               / "stage06E_fd_top100_nominal.csv"))
    p.add_argument("--fd_mc_csv", type=str,
                   default=str(V3_DIR / "outputs" / "labels"
                               / "stage06E_fd_top10_mc.csv"))
    p.add_argument("--fd_disagree_csv", type=str,
                   default=str(V3_DIR / "outputs" / "labels"
                               / "stage06E_fd_disagreement.csv"))
    p.add_argument("--fd_06a_mc_csv", type=str,
                   default=str(V3_DIR / "outputs" / "labels"
                               / "fd_top10_mc_verification.csv"))
    p.add_argument("--summary_06d_json", type=str,
                   default=str(V3_DIR / "outputs" / "logs" / "stage06D_summary.json"))
    p.add_argument("--summary_06a_json", type=str,
                   default=str(V3_DIR / "outputs" / "logs"
                               / "06_yield_optimization_summary.json"))
    p.add_argument("--false_pass_threshold", type=float, default=0.20)
    args = p.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    score_cfg = YieldScoreConfig.from_yaml_dict(cfg["yield_score"])

    # ----- Read inputs -----
    top_06d = read_labels_csv(args.top_06d_csv)
    num_keys_06d = ["rank_06c", "yield_score_06c", "yield_score_06a",
                    "p_robust_valid_06c", "p_robust_valid_06a",
                    "mean_cd_fixed_06c", "mean_ler_locked_06c"] + PARAM_AXES
    _coerce_floats(top_06d, num_keys_06d)
    top_06d.sort(key=lambda r: r.get("rank_06c", 1e9))

    disagree_06d = read_labels_csv(args.disagree_06d_csv) if Path(args.disagree_06d_csv).exists() else []
    _coerce_floats(disagree_06d, num_keys_06d + ["score_gap"])

    fd_baseline = read_labels_csv(args.fd_baseline_csv)
    _coerce_floats(fd_baseline, ["CD_final_nm", "CD_locked_nm",
                                  "LER_CD_locked_nm", "area_frac",
                                  "P_line_margin"])
    fd_baseline_mc = read_labels_csv(args.fd_baseline_mc_csv) if Path(args.fd_baseline_mc_csv).exists() else []
    _coerce_floats(fd_baseline_mc, ["CD_final_nm", "CD_locked_nm",
                                     "LER_CD_locked_nm", "area_frac",
                                     "P_line_margin"])
    fd_nominal = read_labels_csv(args.fd_nominal_csv)
    _coerce_floats(fd_nominal, ["CD_final_nm", "CD_locked_nm",
                                 "LER_CD_locked_nm", "area_frac",
                                 "P_line_margin", "rank_surrogate"])
    fd_mc = read_labels_csv(args.fd_mc_csv)
    _coerce_floats(fd_mc, ["CD_final_nm", "CD_locked_nm",
                            "LER_CD_locked_nm", "area_frac",
                            "P_line_margin", "rank_surrogate"])
    fd_disagree = read_labels_csv(args.fd_disagree_csv) if Path(args.fd_disagree_csv).exists() else []
    _coerce_floats(fd_disagree, ["CD_final_nm", "CD_locked_nm",
                                  "LER_CD_locked_nm", "area_frac",
                                  "P_line_margin"])

    # ----- FD baseline (Part 0 = single nominal, Part 4 = 100 MC) -----
    if fd_baseline:
        baseline_payload = nominal_yield_score(fd_baseline[0], score_cfg)
        fd_baseline_score = float(baseline_payload["FD_yield_score"])
        fd_baseline_label = str(fd_baseline[0].get("label", ""))
    else:
        fd_baseline_score = float("nan")
        fd_baseline_label = ""

    if fd_baseline_mc:
        from reaction_diffusion_peb_v3_screening.src.fd_yield_score import (
            fd_yield_score_from_rows,
        )
        bl_mc = fd_yield_score_from_rows(fd_baseline_mc, score_cfg)
        fd_baseline_mc_score = float(bl_mc["FD_yield_score"])
        fd_baseline_mc_p_robust = float(bl_mc["P_FD_robust_valid"])
        fd_baseline_mc_n = int(bl_mc["n_fd_rows"])
    else:
        fd_baseline_mc_score = float("nan")
        fd_baseline_mc_p_robust = float("nan")
        fd_baseline_mc_n = 0

    # ----- Surrogate vs FD pair table -----
    pair_rows = build_surrogate_vs_fd_metrics(top_06d, fd_nominal, score_cfg)
    errs = regression_target_errors(pair_rows)

    # Spearman top-100 nominal.
    sx = np.array([r["yield_score_06c"]      for r in pair_rows])
    sy = np.array([r["FD_yield_score_nominal"] for r in pair_rows])
    rho100 = spearman(sx, sy)

    # Top-10 MC FD.
    top10 = score_top10_mc(fd_mc, score_cfg)
    sur_lookup = {r["recipe_id"]: r for r in top_06d}
    rec_ids_top10 = [r["recipe_id"] for r in sorted(top10, key=lambda r: r["rank_surrogate"])]
    sur_yields = np.array([sur_lookup[rid]["yield_score_06c"] for rid in rec_ids_top10])
    fd_yields = np.array([r["FD_yield_score"] for r in
                          sorted(top10, key=lambda r: r["rank_surrogate"])])
    rho10 = spearman(sur_yields, fd_yields)

    # Top-1 / 3 / 5 overlap on top-100.
    sur_rank_100 = [r["recipe_id"] for r in sorted(pair_rows,
                       key=lambda r: -r["yield_score_06c"])]
    fd_rank_100 = [r["recipe_id"] for r in sorted(pair_rows,
                      key=lambda r: -r["FD_yield_score_nominal"])]
    top_1 = topk_overlap(sur_rank_100, fd_rank_100, 1)
    top_3 = topk_overlap(sur_rank_100, fd_rank_100, 3)
    top_5 = topk_overlap(sur_rank_100, fd_rank_100, 5)

    # Surrogate top-10 vs FD top-10.
    sur_top10_ids = [r["recipe_id"] for r in top_06d[:10]]
    fd_top10_by_score = sorted(top10, key=lambda r: -r["FD_yield_score"])
    fd_top10_ids = [r["recipe_id"] for r in fd_top10_by_score]
    sur_in_fd_top10 = len(set(sur_top10_ids) & set(fd_top10_ids))
    sur_top1_in_fd_top10 = sur_top10_ids[0] in fd_top10_ids if sur_top10_ids else False

    # Label agreement on top-100 (06C argmax).
    sur_pred_label = []
    for r in pair_rows:
        s = sur_lookup[r["recipe_id"]]
        # 06D top CSV only carries p_robust_valid_06c. Use a 2-class
        # heuristic: predict robust_valid if p >= 0.5, else "non_robust".
        p = _safe_float(s.get("p_robust_valid_06c"))
        sur_pred_label.append("robust_valid" if p >= 0.5 else "non_robust")
    fd_label_simplified = ["robust_valid" if r["fd_label"] == "robust_valid"
                            else "non_robust" for r in pair_rows]
    agree = sum(1 for a, b in zip(sur_pred_label, fd_label_simplified) if a == b)

    # False-PASS top-100.
    false_rows, false_summary = detect_false_pass(fd_nominal, top_06d)

    # Disagreement-candidate breakdown.
    disagree_breakdown_rows, disagree_summary = build_disagreement_breakdown(
        disagree_06d, fd_disagree, score_cfg)

    # ----- 06D vs 06A FD MC top-10 comparison -----
    fd_06a_mc_rows = []
    top10_06a = []
    if Path(args.fd_06a_mc_csv).exists():
        fd_06a_mc_rows = read_labels_csv(args.fd_06a_mc_csv)
        _coerce_floats(fd_06a_mc_rows, ["CD_final_nm", "CD_locked_nm",
                                          "LER_CD_locked_nm", "area_frac",
                                          "P_line_margin", "rank_surrogate"])
        top10_06a = score_top10_mc(fd_06a_mc_rows, score_cfg)

    # ----- Acceptance -----
    fd_beats_baseline_top100 = sum(
        1 for r in pair_rows
        if np.isfinite(r["FD_yield_score_nominal"])
           and np.isfinite(fd_baseline_score)
           and r["FD_yield_score_nominal"] > fd_baseline_score
    )
    # Compare top-10 MC FD vs MC baseline (apples-to-apples) when
    # available; fall back to the nominal baseline otherwise.
    if np.isfinite(fd_baseline_mc_score):
        mc_ref = fd_baseline_mc_score
        mc_ref_label = "v2 frozen OP MC"
    else:
        mc_ref = fd_baseline_score
        mc_ref_label = "v2 frozen OP nominal"
    fd_mc_beats_baseline = sum(
        1 for r in top10
        if np.isfinite(mc_ref) and r["FD_yield_score"] > mc_ref
    )
    fd_mc_ties_or_beats = sum(
        1 for r in top10
        if np.isfinite(mc_ref) and r["FD_yield_score"] >= mc_ref - 1e-9
    )
    false_pass_total = float(false_summary["hard_false_pass_rate"]
                              + false_summary["soft_false_pass_rate"])
    # Notes on "beats":
    #   * The v2 frozen-OP nominal FD lands on robust_valid -> score 1.0,
    #     which is the maximum a single nominal FD row can yield. Strict
    #     > 1.0 is therefore impossible for any nominal recipe; the more
    #     informative criterion is whether 06D top recipes TIE the
    #     nominal ceiling without false-PASS, AND whether they beat the
    #     v2 OP under the same MC variation distribution.
    nominal_top1_score = float(pair_rows[0]["FD_yield_score_nominal"]) if pair_rows else float("nan")
    n_top100_at_nominal_ceiling = sum(
        1 for r in pair_rows
        if np.isfinite(r["FD_yield_score_nominal"])
           and r["FD_yield_score_nominal"] >= fd_baseline_score - 1e-9
    )

    acceptance = {
        "fd_baseline_v2_op_nominal_yield_score": fd_baseline_score,
        "fd_baseline_v2_op_nominal_label":       fd_baseline_label,
        "fd_baseline_v2_op_mc_yield_score":      fd_baseline_mc_score,
        "fd_baseline_v2_op_mc_p_robust_valid":   fd_baseline_mc_p_robust,
        "fd_baseline_v2_op_mc_n":                fd_baseline_mc_n,
        "mc_baseline_used_for_beats_test":       mc_ref_label,
        "top100_FD_beats_nominal_baseline_count": int(fd_beats_baseline_top100),
        "top100_FD_ties_nominal_baseline_count":  int(n_top100_at_nominal_ceiling),
        "top100_FD_ties_nominal_baseline_pass":   bool(n_top100_at_nominal_ceiling >= 1),
        "fd_mc_beats_baseline_count":            int(fd_mc_beats_baseline),
        "fd_mc_beats_baseline_pass":             bool(fd_mc_beats_baseline >= 1),
        "fd_mc_ties_or_beats_baseline_count":    int(fd_mc_ties_or_beats),
        "spearman_top100":                       rho100,
        "spearman_top10":                        rho10,
        "topk_overlap":                          {"top1": top_1, "top3": top_3, "top5": top_5},
        "surrogate_top1_in_fd_top10":            bool(sur_top1_in_fd_top10),
        "surrogate_top10_overlap_fd_top10":      int(sur_in_fd_top10),
        "false_pass_rate_top100":                false_pass_total,
        "false_pass_within_threshold":           bool(false_pass_total <= args.false_pass_threshold),
        "false_pass_threshold":                  float(args.false_pass_threshold),
        "disagreement_n":                        int(disagree_summary["n_disagreement"]),
        "disagreement_n_robust_valid":           int(disagree_summary["n_robust_valid"]),
        "disagreement_n_hard_fail":              int(disagree_summary["n_hard_fail"]),
        "disagreement_06c_correct_rate":         float(disagree_summary["06c_correct_rate"]),
        "policy_v2_OP_frozen":                   bool(cfg["policy"].get("v2_OP_frozen", True)),
        "policy_published_data_loaded":          bool(cfg["policy"].get("published_data_loaded", False)),
        "policy_external_calibration":           "none",
    }

    # ----- Outputs (CSVs / JSON) -----
    labels_dir = V3_DIR / "outputs" / "labels"
    logs_dir   = V3_DIR / "outputs" / "logs"
    fig_dir    = V3_DIR / "outputs" / "figures" / "06_yield_optimization"
    fig_dir.mkdir(parents=True, exist_ok=True)

    pair_cols = list(pair_rows[0].keys()) if pair_rows else []
    with (labels_dir / "stage06E_surrogate_vs_fd_metrics.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=pair_cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(pair_rows)

    if false_rows:
        cols = list(false_rows[0].keys())
    else:
        cols = ["recipe_id", "rank_06c", "fd_label", "false_pass_kind",
                "yield_score_06c", "yield_score_06a", "p_robust_valid_06c",
                "CD_final_fd", "CD_locked_fd", "LER_CD_locked_fd",
                "area_frac_fd", "P_line_margin_fd"] + PARAM_AXES
    with (labels_dir / "stage06E_false_pass_cases.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in false_rows:
            w.writerow(r)

    if disagree_breakdown_rows:
        dis_cols = list(disagree_breakdown_rows[0].keys())
    else:
        dis_cols = ["recipe_id", "rank_06c", "yield_score_06c",
                    "yield_score_06a", "score_gap", "fd_label",
                    "FD_yield_score_nominal", "06c_agrees_with_fd",
                    "06a_agrees_with_fd", "CD_final_fd",
                    "LER_CD_locked_fd", "P_line_margin_fd"]
    with (labels_dir / "stage06E_disagreement_fd_breakdown.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=dis_cols, extrasaction="ignore")
        w.writeheader()
        for r in disagree_breakdown_rows:
            w.writerow(r)

    n_al = write_al_additions(
        [fd_baseline, fd_baseline_mc, fd_nominal, fd_mc, fd_disagree],
        labels_dir / "stage06E_al_additions.csv",
    )

    ranking_payload = {
        "stage": "06E",
        "policy": cfg["policy"],
        "n_top_nominal":            len(pair_rows),
        "n_top_mc_recipes":         len(top10),
        "n_mc_per_recipe":          int(len(fd_mc) / max(len(top10), 1)),
        "n_disagreement":           len(disagree_breakdown_rows),
        "fd_baseline_v2_op": {
            "nominal_yield_score":    fd_baseline_score,
            "nominal_label":          fd_baseline_label,
            "mc_yield_score":         fd_baseline_mc_score,
            "mc_p_robust_valid":      fd_baseline_mc_p_robust,
            "mc_n":                   fd_baseline_mc_n,
        },
        "regression_errors_nominal": errs,
        "spearman_top100_nominal_yield_score": rho100,
        "spearman_top10_mc_yield_score":       rho10,
        "topk_overlap_top100": {"top1": top_1, "top3": top_3, "top5": top_5},
        "surrogate_top10_overlap_with_fd_top10": sur_in_fd_top10,
        "surrogate_top1_in_fd_top10": bool(sur_top1_in_fd_top10),
        "label_agreement_top100_2class": {
            "n":     len(pair_rows),
            "agree": int(agree),
            "rate":  float(agree / max(len(pair_rows), 1)),
        },
        "fd_top10_recipes_by_score": [
            {
                "rank_surrogate":     int(r["rank_surrogate"]),
                "recipe_id":          r["recipe_id"],
                "FD_yield_score":     float(r["FD_yield_score"]),
                "P_FD_robust_valid":  float(r["P_FD_robust_valid"]),
                "mean_cd_fixed_fd":   float(r["mean_cd_fixed_fd"]),
                "mean_ler_locked_fd": float(r["mean_ler_locked_fd"]),
            } for r in fd_top10_by_score
        ],
        "fd_top10_recipes_06a_for_reference": [
            {
                "rank_surrogate":     int(r["rank_surrogate"]),
                "recipe_id":          r["recipe_id"],
                "FD_yield_score":     float(r["FD_yield_score"]),
                "P_FD_robust_valid":  float(r["P_FD_robust_valid"]),
            } for r in sorted(top10_06a, key=lambda r: -r["FD_yield_score"])[:10]
        ] if top10_06a else [],
        "disagreement_summary": disagree_summary,
        "n_al_additions": int(n_al),
        "acceptance": acceptance,
    }
    (logs_dir / "stage06E_summary.json").write_text(
        json.dumps(ranking_payload, indent=2, default=float))

    (logs_dir / "stage06E_false_pass_summary.json").write_text(
        json.dumps({**false_summary, "policy": cfg["policy"]}, indent=2, default=float))

    # ----- Figures -----
    plot_yield_score_scatter(pair_rows, fd_baseline_score,
                              fig_dir / "stage06E_surrogate_vs_fd_yield_score_scatter.png")
    plot_cd_ler_scatter(pair_rows,
                         fig_dir / "stage06E_surrogate_vs_fd_cd_ler_scatter.png")
    plot_top10_fd_yield_bar(top10, mc_ref,
                             fig_dir / "stage06E_top10_fd_yield_barplot.png")
    plot_defect_breakdown_top10(top10,
                                 fig_dir / "stage06E_defect_breakdown_top10.png")
    plot_false_pass_parallel(false_rows,
                              fig_dir / "stage06E_false_pass_parameter_parallel_coordinates.png")
    plot_disagreement_breakdown(disagree_breakdown_rows,
                                 fig_dir / "stage06E_disagreement_fd_breakdown.png")
    plot_06d_vs_06a_top10(top10, top10_06a, mc_ref,
                           fig_dir / "stage06E_06D_vs_06A_fd_top10.png")

    # ----- Console summary -----
    print(f"\nStage 06E -- analysis summary")
    print(f"  FD baseline (v2 frozen OP)")
    print(f"    nominal: yield_score = {fd_baseline_score:.4f}  "
          f"label = {fd_baseline_label!r}")
    print(f"    MC ({fd_baseline_mc_n} variations): yield_score = "
          f"{fd_baseline_mc_score:.4f}  P(robust) = {fd_baseline_mc_p_robust:.3f}")
    print(f"  Part 1 -- top-100 nominal FD")
    print(f"    Spearman rho (06C surrogate vs FD): {rho100:.4f}")
    print(f"    top-1/3/5 overlap: {top_1}/1, {top_3}/3, {top_5}/5")
    for tname, e in errs.items():
        if e["mae"] is None:
            continue
        print(f"    {tname:<14}  MAE={e['mae']:.3f}  RMSE={e['rmse']:.3f}  n={e['n']}")
    print(f"    FD recipes that beat FD baseline: {fd_beats_baseline_top100} / {len(pair_rows)}")
    print(f"  Part 2 -- top-10 FD MC")
    print(f"    Spearman rho (06C vs FD MC): {rho10:.4f}")
    print(f"    surrogate top-10 ∩ FD top-10: {sur_in_fd_top10}/10")
    print(f"    FD MC recipes vs MC baseline ({mc_ref_label} = {mc_ref:.4f}):")
    print(f"      beats: {fd_mc_beats_baseline} / {len(top10)}")
    print(f"      ties or beats: {fd_mc_ties_or_beats} / {len(top10)}")
    print(f"  Part 3 -- disagreement candidates")
    print(f"    n={disagree_summary['n_disagreement']}  "
          f"robust_valid={disagree_summary['n_robust_valid']}  "
          f"margin_risk={disagree_summary['n_margin_risk']}  "
          f"hard_fail={disagree_summary['n_hard_fail']}")
    print(f"    06C label-agreement rate: {disagree_summary['06c_correct_rate']*100:.1f}%")
    print(f"  False-PASS top-100")
    print(f"    hard: {false_summary['n_hard_false_pass']}  "
          f"soft: {false_summary['n_soft_false_pass']}  "
          f"total rate: {false_pass_total*100:.2f}% "
          f"(threshold {args.false_pass_threshold*100:.0f}%)")
    print(f"  AL additions: {n_al} new FD rows -> stage06E_al_additions.csv")
    print(f"  Acceptance: {acceptance}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
