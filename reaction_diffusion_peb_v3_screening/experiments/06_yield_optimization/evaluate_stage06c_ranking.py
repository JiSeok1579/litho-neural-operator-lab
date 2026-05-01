"""Stage 06C — re-score the original Stage 06A top-100 fixed-design
candidates with the refreshed surrogate, then compare the new ranking
to the Stage 06B FD ground truth.

Purpose: Stage 06B showed that the Stage 06A surrogate is great at
candidate proposal (top-1/3/5 overlap perfect, label agreement 100/100)
but weak at fine ranking inside the top tier (Spearman ≈ +0.07,
top-10 FD scores all tied at 1.000). Stage 06C tests whether folding
the FD-verified rows back into training improves the within-tier
ranking on the same 100 candidates.

Outputs (alongside Stage 06A / 06B figures):
    outputs/logs/stage06C_ranking_comparison.json
    outputs/logs/stage06C_false_pass_summary.json
    outputs/figures/06_yield_optimization/
        stage06C_surrogate_vs_fd_yield_scatter.png
        stage06C_cd_fixed_pred_vs_fd.png
        stage06C_ranking_before_after.png
        stage06C_feature_importance.png
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

from reaction_diffusion_peb_v3_screening.src.candidate_sampler import (
    CandidateSpace,
)
from reaction_diffusion_peb_v3_screening.src.fd_yield_score import (
    nominal_yield_score,
    spearman,
    topk_overlap,
)
from reaction_diffusion_peb_v3_screening.src.metrics_io import (
    FEATURE_KEYS,
    load_model,
    read_labels_csv,
)
from reaction_diffusion_peb_v3_screening.src.process_variation import (
    VariationSpec,
)
from reaction_diffusion_peb_v3_screening.src.yield_optimizer import (
    SUMMARY_COLUMNS,
    YieldScoreConfig,
    evaluate_recipes,
)


V3_DIR = ROOT / "reaction_diffusion_peb_v3_screening"


def _safe_float(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def _row_to_candidate(row: dict, space: CandidateSpace) -> dict:
    out = {}
    for k in FEATURE_KEYS:
        out[k] = _safe_float(row[k])
    out["pitch_nm"] = float(out["pitch_nm"])
    out["line_cd_nm"]  = out["pitch_nm"] * out["line_cd_ratio"]
    out["domain_x_nm"] = out["pitch_nm"] * 5.0
    out["dose_norm"]   = out["dose_mJ_cm2"] / float(space.fixed["reference_dose_mJ_cm2"])
    for fk, fv in space.fixed.items():
        out.setdefault(fk, fv)
    out["_id"] = row.get("recipe_id", "?")
    return out


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------
def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str,
                   default=str(V3_DIR / "configs" / "yield_optimization.yaml"))
    p.add_argument("--surrogate_top_csv", type=str,
                   default=str(V3_DIR / "outputs" / "labels"
                               / "06_top_recipes_fixed_design_surrogate.csv"))
    p.add_argument("--fd_nominal_csv", type=str,
                   default=str(V3_DIR / "outputs" / "labels"
                               / "fd_top100_nominal_verification.csv"))
    p.add_argument("--clf_06c", type=str,
                   default=str(V3_DIR / "outputs" / "models"
                               / "stage06C_classifier.joblib"))
    p.add_argument("--reg_06c", type=str,
                   default=str(V3_DIR / "outputs" / "models"
                               / "stage06C_regressor.joblib"))
    p.add_argument("--aux_06c", type=str,
                   default=str(V3_DIR / "outputs" / "models"
                               / "stage06C_aux_cd_fixed_regressor.joblib"))
    p.add_argument("--seed", type=int, default=2026)
    args = p.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    space = CandidateSpace.from_yaml(cfg["candidate_space_yaml"])
    var_spec = VariationSpec.from_yaml_dict(cfg["process_variation"])
    score_cfg = YieldScoreConfig.from_yaml_dict(cfg["yield_score"])

    classifier_06c, _ = load_model(args.clf_06c)
    regressor_06c,  _ = load_model(args.reg_06c)
    cd_fixed_06c,   _ = load_model(args.aux_06c)

    # Stage 06A top-100 surrogate-rank order.
    surrogate_top_rows = read_labels_csv(args.surrogate_top_csv)
    surrogate_top_rows.sort(key=lambda r: -float(r["yield_score"]))
    for i, r in enumerate(surrogate_top_rows, start=1):
        r["rank_06a_surrogate"] = i

    # Stage 06B FD nominal ground truth.
    fd_nominal = {r["source_recipe_id"]: r
                  for r in read_labels_csv(args.fd_nominal_csv)}

    # Re-score the same 100 recipes with the refreshed 06C surrogate
    # using the same MC pipeline (same variation widths, same n_var).
    n_var = int(cfg["sampling"]["variations_per_candidate"])
    candidates = [_row_to_candidate(r, space) for r in surrogate_top_rows]
    print(f"Stage 06C re-scoring on Stage 06A top-{len(candidates)} "
          f"(n_var={n_var})")
    scored_06c = evaluate_recipes(
        candidates, classifier_06c, regressor_06c, cd_fixed_06c,
        var_spec, n_var, space, score_cfg, seed=args.seed,
    )
    by_id_06c = {str(s["recipe_id"]): s for s in scored_06c}

    # Build paired-comparison rows.
    pairs: list[dict] = []
    for src in surrogate_top_rows:
        rid = src["recipe_id"]
        fd = fd_nominal.get(rid)
        s06c = by_id_06c.get(rid, {})
        pair = {
            "recipe_id":              rid,
            "rank_06a_surrogate":     int(src["rank_06a_surrogate"]),
            "yield_score_06a_surrogate":  float(src["yield_score"]),
            "yield_score_06c_surrogate":  float(s06c.get("yield_score", "nan")),
            "mean_cd_fixed_06a":      _safe_float(src.get("mean_cd_fixed")),
            "mean_cd_fixed_06c":      _safe_float(s06c.get("mean_cd_fixed")),
            "mean_ler_locked_06a":    _safe_float(src.get("mean_ler_locked")),
            "mean_ler_locked_06c":    _safe_float(s06c.get("mean_ler_locked")),
            "p_robust_valid_06a":     _safe_float(src.get("p_robust_valid")),
            "p_robust_valid_06c":     _safe_float(s06c.get("p_robust_valid")),
        }
        if fd is None:
            pair.update({
                "fd_label": "",
                "FD_yield_score_nominal": float("nan"),
                "CD_final_fd": float("nan"),
                "LER_CD_locked_fd": float("nan"),
                "P_line_margin_fd": float("nan"),
            })
        else:
            nom = nominal_yield_score(fd, score_cfg)
            pair.update({
                "fd_label": str(fd.get("label", "")),
                "FD_yield_score_nominal": float(nom["FD_yield_score"]),
                "CD_final_fd":      _safe_float(fd.get("CD_final_nm")),
                "LER_CD_locked_fd": _safe_float(fd.get("LER_CD_locked_nm")),
                "P_line_margin_fd": _safe_float(fd.get("P_line_margin")),
            })
        pairs.append(pair)

    # Re-rank by 06C yield_score.
    pairs_06c_sorted = sorted(
        pairs, key=lambda r: -r["yield_score_06c_surrogate"])
    for i, r in enumerate(pairs_06c_sorted, start=1):
        r["rank_06c_surrogate"] = i

    # ----- ranking comparison metrics -----
    arr_06a   = np.array([r["yield_score_06a_surrogate"]   for r in pairs])
    arr_06c   = np.array([r["yield_score_06c_surrogate"]   for r in pairs])
    arr_fd    = np.array([r["FD_yield_score_nominal"]      for r in pairs])
    cd_06a    = np.array([r["mean_cd_fixed_06a"]           for r in pairs])
    cd_06c    = np.array([r["mean_cd_fixed_06c"]           for r in pairs])
    cd_fd     = np.array([r["CD_final_fd"]                 for r in pairs])
    ler_06a   = np.array([r["mean_ler_locked_06a"]         for r in pairs])
    ler_06c   = np.array([r["mean_ler_locked_06c"]         for r in pairs])
    ler_fd    = np.array([r["LER_CD_locked_fd"]            for r in pairs])

    rho_yield_06a = spearman(arr_06a, arr_fd)
    rho_yield_06c = spearman(arr_06c, arr_fd)
    rho_cd_06a    = spearman(cd_06a, cd_fd)
    rho_cd_06c    = spearman(cd_06c, cd_fd)
    rho_ler_06a   = spearman(ler_06a, ler_fd)
    rho_ler_06c   = spearman(ler_06c, ler_fd)

    # MAE / RMSE.
    def _ae(a, b):
        m = np.isfinite(a) & np.isfinite(b)
        if m.sum() == 0:
            return None, None, 0
        d = a[m] - b[m]
        return (float(np.mean(np.abs(d))),
                float(np.sqrt(np.mean(d ** 2))),
                int(m.sum()))

    cd_mae_06a, cd_rmse_06a, cd_n = _ae(cd_06a, cd_fd)
    cd_mae_06c, cd_rmse_06c, _    = _ae(cd_06c, cd_fd)
    ler_mae_06a, ler_rmse_06a, ler_n = _ae(ler_06a, ler_fd)
    ler_mae_06c, ler_rmse_06c, _    = _ae(ler_06c, ler_fd)

    # Top-k overlap on the FD ranking vs surrogate ranking.
    fd_rank   = [r["recipe_id"] for r in sorted(pairs, key=lambda r: -r["FD_yield_score_nominal"])]
    rank_06a  = [r["recipe_id"] for r in sorted(pairs, key=lambda r: -r["yield_score_06a_surrogate"])]
    rank_06c  = [r["recipe_id"] for r in sorted(pairs, key=lambda r: -r["yield_score_06c_surrogate"])]
    overlap = {}
    for k in (1, 3, 5, 10):
        overlap[f"top{k}"] = {
            "06a_vs_fd": int(topk_overlap(rank_06a, fd_rank, k)),
            "06c_vs_fd": int(topk_overlap(rank_06c, fd_rank, k)),
            "06a_vs_06c": int(topk_overlap(rank_06a, rank_06c, k)),
        }

    # False-PASS — same definition as Stage 06B (FD label not robust_valid).
    HARD = ("under_exposed", "merged", "roughness_degraded", "numerical_invalid")
    SOFT = "margin_risk"
    false_pass_06c_rows = []
    for r in pairs:
        if r["fd_label"] == "robust_valid" or not r["fd_label"]:
            continue
        kind = ("hard_false_pass" if r["fd_label"] in HARD
                else "soft_false_pass" if r["fd_label"] == SOFT
                else "other_non_robust")
        false_pass_06c_rows.append({**r, "false_pass_kind": kind})
    fp_summary = {
        "n_top": len(pairs),
        "n_hard_false_pass":  int(sum(1 for r in false_pass_06c_rows
                                       if r["false_pass_kind"] == "hard_false_pass")),
        "n_soft_false_pass":  int(sum(1 for r in false_pass_06c_rows
                                       if r["false_pass_kind"] == "soft_false_pass")),
        "hard_false_pass_rate": float(sum(1 for r in false_pass_06c_rows
                                           if r["false_pass_kind"] == "hard_false_pass")
                                       / max(len(pairs), 1)),
        "soft_false_pass_rate": float(sum(1 for r in false_pass_06c_rows
                                           if r["false_pass_kind"] == "soft_false_pass")
                                       / max(len(pairs), 1)),
    }

    # ----- Acceptance vs user thresholds -----
    cd_mae_06b_baseline = 0.995          # from Stage 06B (06A aux on FD nominal)
    rho_yield_06b_baseline = 0.071       # from Stage 06B
    cd_mae_drop_pct = ((cd_mae_06b_baseline - (cd_mae_06c or float("inf")))
                       / cd_mae_06b_baseline) * 100.0 if cd_mae_06c else None

    acceptance = {
        "cd_fixed_mae_06b_baseline":  cd_mae_06b_baseline,
        "cd_fixed_mae_06c":           cd_mae_06c,
        "cd_fixed_mae_improvement_pct": cd_mae_drop_pct,
        "cd_fixed_mae_pass_strict":   bool(cd_mae_06c is not None
                                            and cd_mae_06c < 0.75),
        "cd_fixed_mae_pass_relative": bool(cd_mae_06c is not None
                                            and cd_mae_drop_pct is not None
                                            and cd_mae_drop_pct >= 20.0),
        "yield_spearman_06b_baseline": rho_yield_06b_baseline,
        "yield_spearman_06c":          rho_yield_06c,
        "yield_spearman_pass":         bool(np.isfinite(rho_yield_06c)
                                             and rho_yield_06c > 0.25),
        "false_pass_total_06c":       int(len(false_pass_06c_rows)),
        "false_pass_pass":            bool(len(false_pass_06c_rows) == 0),
    }

    payload = {
        "stage": "06C",
        "policy": cfg["policy"],
        "n_recipes_rescored": len(pairs),
        "n_var_per_recipe":   n_var,
        "ranking_metrics": {
            "spearman_yield_score": {"06a": rho_yield_06a, "06c": rho_yield_06c},
            "spearman_cd_fixed":     {"06a": rho_cd_06a,    "06c": rho_cd_06c},
            "spearman_ler_locked":   {"06a": rho_ler_06a,   "06c": rho_ler_06c},
            "topk_overlap": overlap,
        },
        "regression_vs_fd_nominal": {
            "cd_fixed":   {"mae_06a": cd_mae_06a, "rmse_06a": cd_rmse_06a,
                            "mae_06c": cd_mae_06c, "rmse_06c": cd_rmse_06c,
                            "n": cd_n},
            "ler_locked": {"mae_06a": ler_mae_06a, "rmse_06a": ler_rmse_06a,
                            "mae_06c": ler_mae_06c, "rmse_06c": ler_rmse_06c,
                            "n": ler_n},
        },
        "acceptance": acceptance,
    }

    logs_dir   = V3_DIR / "outputs" / "logs"
    fig_dir    = V3_DIR / "outputs" / "figures" / "06_yield_optimization"
    labels_dir = V3_DIR / "outputs" / "labels"
    logs_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    (logs_dir / "stage06C_ranking_comparison.json").write_text(
        json.dumps(payload, indent=2))
    (logs_dir / "stage06C_false_pass_summary.json").write_text(
        json.dumps({**fp_summary, "policy": cfg["policy"]}, indent=2))

    # Save the paired-comparison CSV.
    if pairs:
        cols = list(pairs[0].keys())
        with (labels_dir / "stage06C_ranking_comparison.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            for r in pairs:
                w.writerow(r)

    # ----- Figures -----
    plot_yield_scatter(pairs, rho_yield_06a, rho_yield_06c,
                        fig_dir / "stage06C_surrogate_vs_fd_yield_scatter.png")
    plot_cd_fixed_pred_vs_fd(pairs, rho_cd_06a, rho_cd_06c,
                              fig_dir / "stage06C_cd_fixed_pred_vs_fd.png")
    plot_ranking_before_after(pairs,
                               fig_dir / "stage06C_ranking_before_after.png")
    plot_feature_importance(classifier_06c, regressor_06c, cd_fixed_06c,
                              fig_dir / "stage06C_feature_importance.png")

    # ----- Console summary -----
    print(f"\nStage 06C ranking comparison (re-score Stage 06A top-100)")
    print(f"  Spearman ρ(yield_score, FD)         06A→FD: {rho_yield_06a:.3f}   06C→FD: {rho_yield_06c:.3f}")
    print(f"  Spearman ρ(CD_fixed,  FD)            06A→FD: {rho_cd_06a:.3f}   06C→FD: {rho_cd_06c:.3f}")
    print(f"  Spearman ρ(LER_locked, FD)           06A→FD: {rho_ler_06a:.3f}   06C→FD: {rho_ler_06c:.3f}")
    print(f"  CD_fixed MAE                         06A: {cd_mae_06a:.3f}     06C: {cd_mae_06c:.3f}     "
          f"improvement: {cd_mae_drop_pct:+.1f} %")
    print(f"  LER_locked MAE                       06A: {ler_mae_06a:.3f}    06C: {ler_mae_06c:.3f}")
    print(f"  Top-k overlap (06A vs FD / 06C vs FD)")
    for k in (1, 3, 5, 10):
        ov = overlap[f"top{k}"]
        print(f"    top{k:>2}:  06A∩FD={ov['06a_vs_fd']:>2}    06C∩FD={ov['06c_vs_fd']:>2}    "
              f"06A∩06C={ov['06a_vs_06c']:>2}")
    print(f"  false-PASS (06C, FD label != robust_valid): "
          f"{fp_summary['n_hard_false_pass']} hard + {fp_summary['n_soft_false_pass']} soft")
    print(f"  Acceptance:")
    for k, v in acceptance.items():
        print(f"    {k:<35} {v}")
    return 0


# --------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------
def plot_yield_scatter(pairs, rho_06a, rho_06c, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 6.0))
    for ax, key, rho, title in [
        (axes[0], "yield_score_06a_surrogate", rho_06a, "Stage 06A surrogate"),
        (axes[1], "yield_score_06c_surrogate", rho_06c, "Stage 06C surrogate (refreshed)"),
    ]:
        x = np.array([r[key] for r in pairs])
        y = np.array([r["FD_yield_score_nominal"] for r in pairs])
        ax.scatter(x, y, s=24, c="#1f77b4", alpha=0.75, edgecolor="white", lw=0.4)
        lo = float(np.nanmin([x.min(), y.min(), -0.5]))
        hi = float(np.nanmax([x.max(), y.max(), 1.05]))
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, alpha=0.5, label="y = x")
        ax.set_xlabel(f"{title} yield_score (mean over 200 MC variations)")
        ax.set_ylabel("Stage 06B FD yield_score (single nominal FD)")
        ax.set_title(f"{title} vs FD truth  (Spearman ρ = {rho:.3f})")
        ax.legend(loc="lower right", fontsize=10)
        ax.grid(True, alpha=0.25)
    fig.suptitle("Stage 06C — does the refresh improve yield_score ranking?",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_cd_fixed_pred_vs_fd(pairs, rho_06a, rho_06c, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 6.0))
    for ax, key, rho, title in [
        (axes[0], "mean_cd_fixed_06a", rho_06a, "Stage 06A surrogate (existing aux)"),
        (axes[1], "mean_cd_fixed_06c", rho_06c, "Stage 06C surrogate (refreshed aux)"),
    ]:
        x = np.array([r[key] for r in pairs])
        y = np.array([r["CD_final_fd"] for r in pairs])
        finite = np.isfinite(x) & np.isfinite(y)
        x = x[finite]; y = y[finite]
        ax.scatter(x, y, s=24, c="#2ca02c", alpha=0.75, edgecolor="white", lw=0.4)
        lo = float(min(x.min(), y.min()))
        hi = float(max(x.max(), y.max()))
        pad = 0.05 * (hi - lo + 1e-9)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k--", lw=0.8, alpha=0.5,
                label="y = x")
        ax.axhspan(14.0, 16.0, color="#aaa", alpha=0.15, label="CD target ±1 nm")
        d = x - y
        ax.set_xlabel(f"{title} mean CD_fixed prediction (nm)")
        ax.set_ylabel("Stage 06B FD nominal CD_final (nm)")
        ax.set_title(f"{title}  (Spearman ρ = {rho:.3f}, MAE = {np.mean(np.abs(d)):.3f})")
        ax.legend(loc="lower right", fontsize=9)
        ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_ranking_before_after(pairs, out_path):
    """Slope chart: each line is a recipe; left = 06A rank, right = 06C
    rank. Crossings show where the refresh moved a recipe."""
    fig, ax = plt.subplots(figsize=(11.0, 7.0))
    n = len(pairs)
    by_id = {r["recipe_id"]: r for r in pairs}

    # Re-rank by 06C surrogate yield_score.
    rank_06c = {r["recipe_id"]: i + 1 for i, r in
                enumerate(sorted(pairs, key=lambda r: -r["yield_score_06c_surrogate"]))}

    for r in pairs:
        rid = r["recipe_id"]
        a = r["rank_06a_surrogate"]
        c = rank_06c[rid]
        # Colour by FD outcome — green (FD ≥ baseline), red (FD < baseline).
        fd = r["FD_yield_score_nominal"]
        if not np.isfinite(fd):
            color = "#aaa"
        elif fd >= 0.95:
            color = "#2ca02c"
        elif fd >= 0.5:
            color = "#1f77b4"
        else:
            color = "#d62728"
        ax.plot([0, 1], [a, c], "-", color=color, alpha=0.55, lw=0.9)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Stage 06A surrogate rank", "Stage 06C surrogate rank"])
    ax.set_ylim(n + 0.5, 0.5)         # rank 1 at top
    ax.set_ylabel("rank within top-100")
    ax.set_title("Stage 06C ranking-before-after (top-100 fixed-design recipes; "
                 "colour = FD nominal yield_score band)")
    handles = [
        plt.Line2D([0], [0], color="#2ca02c", lw=2, label="FD ≥ 0.95"),
        plt.Line2D([0], [0], color="#1f77b4", lw=2, label="FD 0.5–0.95"),
        plt.Line2D([0], [0], color="#d62728", lw=2, label="FD < 0.5"),
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=10)
    ax.grid(True, alpha=0.2, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_feature_importance(classifier, regressor, aux_cd_fixed, out_path):
    """Stack the three models' feature importances side by side. They
    share the FEATURE_KEYS column order."""
    keys = FEATURE_KEYS
    n = len(keys)
    # sharex=False so each panel keeps its own descending-importance order
    # *and* its own axis labels — ranking changes between models.
    fig, axes = plt.subplots(3, 1, figsize=(11.0, 11.0), sharex=False)

    classifier_imp = classifier.feature_importances_
    reg_imp = regressor.feature_importances_
    aux_imp = aux_cd_fixed.feature_importances_

    for ax, imp, title, color in [
        (axes[0], classifier_imp, "Stage 06C classifier", "#1f77b4"),
        (axes[1], reg_imp,        "Stage 06C 4-target regressor", "#2ca02c"),
        (axes[2], aux_imp,        "Stage 06C auxiliary CD_fixed regressor", "#d62728"),
    ]:
        order = np.argsort(-imp)
        ranked_keys = [keys[i] for i in order]
        ranked_imp = imp[order]
        bars = ax.bar(np.arange(n), ranked_imp, color=color, alpha=0.85,
                      edgecolor="#1f1f1f")
        for b, v in zip(bars, ranked_imp):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.005,
                    f"{v:.2f}", ha="center", fontsize=8, color="#1f1f1f")
        ax.set_xticks(np.arange(n))
        ax.set_xticklabels(ranked_keys, rotation=18, ha="right", fontsize=9)
        ax.set_ylabel("feature importance")
        ax.set_title(title, fontsize=11)
        ax.grid(True, alpha=0.2, axis="y")
    fig.suptitle("Stage 06C feature importances (sorted per model)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
