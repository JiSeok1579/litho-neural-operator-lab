"""Stage 06L -- re-score 06G/06J candidates with the 06L surrogate
stack (classifier + 4-target + aux + strict_score head), do
false-PASS demotion analysis, write figures, write the AL-targeting
JSON.

Reads:
    outputs/yield_optimization/stage06G_top_recipes.csv
    outputs/yield_optimization/stage06J_mode_b_top_recipes.csv
    outputs/yield_optimization/stage06J_mode_b_recipe_summary.csv
    outputs/yield_optimization/stage06J_mode_b_vs_mode_a_comparison.csv
    outputs/yield_optimization/stage06I_mode_a_final_recipes.yaml
    outputs/yield_optimization/stage06G_strict_score_config.yaml
    outputs/labels/stage06J_mode_b_fd_sanity.csv
    outputs/labels/stage06J_mode_b_fd_mc_optional.csv
    outputs/logs/stage06H_fd_verification_summary.json
    outputs/logs/stage06L_model_metrics.json
    outputs/models/stage06L_*.joblib

Writes:
    outputs/yield_optimization/stage06L_mode_b_rescored_candidates.csv
    outputs/yield_optimization/stage06L_false_pass_demotions.csv
    outputs/yield_optimization/stage06L_al_targets.csv
    outputs/logs/stage06L_false_pass_reduction.json
    outputs/figures/06_yield_optimization/
        stage06L_strict_score_pred_vs_fd.png
        stage06L_strict_score_rank_before_after.png
        stage06L_mode_a_vs_mode_b_strict_score.png
        stage06L_false_pass_demotions.png
        stage06L_feature_importance_strict_score.png
        stage06L_J1453_vs_G4867_ranking.png
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

from reaction_diffusion_peb_v3_screening.src.candidate_sampler import (
    CandidateSpace,
)
from reaction_diffusion_peb_v3_screening.src.fd_yield_score import spearman, topk_overlap
from reaction_diffusion_peb_v3_screening.src.metrics_io import (
    FEATURE_KEYS, load_model, read_labels_csv,
)
from reaction_diffusion_peb_v3_screening.src.process_variation import (
    VariationSpec, sample_variations,
)
from reaction_diffusion_peb_v3_screening.src.yield_optimizer import (
    YieldScoreConfig, evaluate_recipes,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_stage06g_strict_optimization import (  # noqa: E402
    StrictScoreConfig, compute_strict_score,
)
from build_stage06l_dataset import per_row_strict_score  # noqa: E402


V3_DIR = ROOT / "reaction_diffusion_peb_v3_screening"
CD_TARGET_NM = 15.0
HARD_FAIL_LABELS = {"under_exposed", "merged",
                     "roughness_degraded", "numerical_invalid"}


def _safe_float(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def _coerce(rows, keys):
    for r in rows:
        for k in keys:
            if k in r:
                r[k] = _safe_float(r.get(k))


# --------------------------------------------------------------------------
# Re-score candidates: each candidate gets 200 process variations sampled,
# the 06L strict_score head predicts each variation's per-row strict_score,
# the mean is the candidate's "06L MC strict_score". Also evaluate 06L
# 4-target + classifier via evaluate_recipes for proxy / class-prob view.
# --------------------------------------------------------------------------
def rescore_candidates(rows: list[dict], clf, reg, aux, strict_reg,
                          var_spec: VariationSpec, n_var: int,
                          space: CandidateSpace, score_cfg: YieldScoreConfig,
                          strict_cfg: StrictScoreConfig, seed: int) -> list[dict]:
    cand_pool = []
    for r in rows:
        cand = {k: float(r[k]) for k in FEATURE_KEYS}
        cand["pitch_nm"]    = float(cand["pitch_nm"])
        cand["line_cd_nm"]  = cand["pitch_nm"] * cand["line_cd_ratio"]
        cand["domain_x_nm"] = cand["pitch_nm"] * 5.0
        cand["dose_norm"]   = cand["dose_mJ_cm2"] / float(space.fixed["reference_dose_mJ_cm2"])
        for fk, fv in space.fixed.items():
            cand.setdefault(fk, fv)
        cand["_id"] = f"L_rescore_{r.get('recipe_id', r.get('_id', '?'))}"
        cand_pool.append((r, cand))

    eval_rows = evaluate_recipes(
        [c for _, c in cand_pool], clf, reg, aux,
        var_spec, n_var, space, score_cfg, seed=seed,
    )
    for r in eval_rows:
        r.update(compute_strict_score(r, strict_cfg))

    # For the strict_score head, run sample_variations again with the same
    # seed structure as evaluate_recipes does internally is impractical;
    # instead, draw a fresh variation pool per candidate and predict per-row.
    base_rng = np.random.default_rng(seed + 1)
    out: list[dict] = []
    for (orig, cand), eval_row in zip(cand_pool, eval_rows):
        sub_rng = np.random.default_rng(int(base_rng.integers(0, 2**31 - 1)))
        variations = sample_variations(cand, var_spec, n_var, space, rng=sub_rng)
        Xv = np.array([[_safe_float(v.get(k)) for k in FEATURE_KEYS]
                         for v in variations])
        per_row_pred = strict_reg.predict(Xv)
        out.append({
            **{k: _safe_float(orig.get(k)) for k in FEATURE_KEYS},
            "recipe_id":         orig.get("recipe_id", orig.get("_id", "?")),
            "strict_score_06l_eval":         float(eval_row["strict_score"]),
            "strict_score_06l_direct_mean":  float(np.mean(per_row_pred)),
            "strict_score_06l_direct_std":   float(np.std(per_row_pred)),
            "yield_score_06l":               float(eval_row["yield_score"]),
            "p_robust_valid_06l":            float(eval_row["p_robust_valid"]),
            "mean_cd_fixed_06l":             float(eval_row["mean_cd_fixed"]),
            "std_cd_fixed_06l":              float(eval_row["std_cd_fixed"]),
            "mean_ler_locked_06l":           float(eval_row["mean_ler_locked"]),
            "std_ler_locked_06l":            float(eval_row["std_ler_locked"]),
            "mean_p_line_margin_06l":        float(eval_row["mean_p_line_margin"]),
        })
    return out


# --------------------------------------------------------------------------
# FD MC truth lookup from 06H + 06J FD CSVs.
# --------------------------------------------------------------------------
def _mc_aggregate_strict_pass(rows: list[dict], cd_tol: float, ler_cap: float,
                                 strict_yaml: dict) -> dict:
    n = len(rows)
    if n == 0:
        return {"n": 0, "strict_pass_prob": float("nan"),
                "robust_prob": float("nan"),
                "mean_strict_score_per_row": float("nan")}
    n_robust = sum(1 for r in rows if str(r.get("label", "")) == "robust_valid")
    n_sp = sum(
        1 for r in rows
        if str(r.get("label", "")) == "robust_valid"
            and abs(_safe_float(r.get("CD_final_nm")) - CD_TARGET_NM) <= cd_tol
            and _safe_float(r.get("LER_CD_locked_nm")) <= ler_cap
    )
    strict_per_row = np.array([per_row_strict_score(r, strict_yaml) for r in rows])
    return {
        "n": int(n),
        "strict_pass_prob": float(n_sp / n),
        "robust_prob":      float(n_robust / n),
        "mean_strict_score_per_row": float(np.mean(strict_per_row)),
    }


# --------------------------------------------------------------------------
# Plot helpers
# --------------------------------------------------------------------------
def plot_strict_score_pred_vs_fd(pairs: list[dict], out_path: Path) -> None:
    if not pairs:
        return
    rho = spearman(np.array([r["pred_strict_06l"] for r in pairs]),
                     np.array([r["fd_mc_strict_pass_prob"] for r in pairs]))
    fig, ax = plt.subplots(figsize=(9.0, 6.5))
    colors = {"mode_a": "#1f77b4", "mode_b": "#d62728"}
    for mode, color in colors.items():
        sub = [p for p in pairs if p["mode"] == mode]
        if not sub:
            continue
        ax.scatter([p["pred_strict_06l"] for p in sub],
                    [p["fd_mc_strict_pass_prob"] for p in sub],
                    c=color, s=80, alpha=0.85, edgecolor="white", lw=0.6,
                    label=f"{mode} (n={len(sub)})")
        for p in sub:
            ax.annotate(p["recipe_id"], (p["pred_strict_06l"], p["fd_mc_strict_pass_prob"]),
                          fontsize=8, xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("06L direct strict_score (mean over 200 MC variations)")
    ax.set_ylabel("FD MC strict_pass_prob")
    ax.set_title(f"Stage 06L -- direct strict_score head vs FD MC truth  "
                  f"(Spearman rho = {rho:.3f})")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_rank_before_after(rows_06g: list[dict], rows_06j: list[dict],
                              rescored_06g: list[dict], rescored_06j: list[dict],
                              highlight_ids: dict, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11.0, 6.0))

    def _ranks(rs_orig: list[dict], rs_new: list[dict],
                 sort_orig_key: str, sort_new_key: str,
                 mode_label: str, baseline_color: str):
        orig_sorted = sorted(rs_orig, key=lambda r: -float(r.get(sort_orig_key, 0)))
        orig_rank = {r.get("recipe_id", r.get("_id", "?")): i + 1
                     for i, r in enumerate(orig_sorted)}
        new_sorted = sorted(rs_new, key=lambda r: -float(r.get(sort_new_key, 0)))
        new_rank = {r["recipe_id"]: i + 1 for i, r in enumerate(new_sorted)}
        for r in rs_orig:
            rid = r.get("recipe_id", r.get("_id", "?"))
            if rid not in new_rank:
                continue
            old = orig_rank[rid]; new = new_rank[rid]
            is_hi = rid in highlight_ids
            ax.plot([0, 1], [old, new], "-",
                     color=highlight_ids.get(rid, baseline_color) if is_hi else baseline_color,
                     lw=1.7 if is_hi else 0.4,
                     alpha=0.95 if is_hi else 0.35,
                     zorder=5 if is_hi else 1)
            if is_hi:
                ax.scatter([0, 1], [old, new], s=44,
                            color=highlight_ids[rid], zorder=6)
                ax.annotate(rid, (1.02, new), fontsize=8,
                              color=highlight_ids[rid], verticalalignment="center")
        return mode_label

    _ranks(rows_06g, rescored_06g,
            sort_orig_key="strict_score", sort_new_key="strict_score_06l_direct_mean",
            mode_label="06G top-100", baseline_color="#9aaecf")
    _ranks(rows_06j, rescored_06j,
            sort_orig_key="strict_score", sort_new_key="strict_score_06l_direct_mean",
            mode_label="06J top-100", baseline_color="#cf9aaa")
    ax.invert_yaxis()
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["original surrogate (06G/06J) strict_score rank",
                          "06L direct strict_score head rank"])
    ax.set_ylabel("rank (lower = better)")
    ax.set_title("Stage 06L -- rank movement under direct strict_score head")
    ax.set_xlim(-0.05, 1.20)
    ax.grid(True, alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_mode_a_vs_mode_b_distribution(rescored_06g, rescored_06j, out_path):
    sa = np.array([_safe_float(r.get("strict_score_06l_direct_mean"))
                     for r in rescored_06g])
    sb = np.array([_safe_float(r.get("strict_score_06l_direct_mean"))
                     for r in rescored_06j])
    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    bins = np.linspace(min(sa.min(), sb.min()), max(sa.max(), sb.max()), 40)
    ax.hist(sa, bins=bins, alpha=0.55, color="#1f77b4",
            label=f"Mode A 06G top-100 (n={len(sa)})")
    ax.hist(sb, bins=bins, alpha=0.55, color="#d62728",
            label=f"Mode B 06J top-100 (n={len(sb)})")
    ax.set_xlabel("06L direct strict_score (mean over 200 MC)")
    ax.set_ylabel("count")
    ax.set_title("Stage 06L -- Mode A vs Mode B strict_score distribution under refreshed head")
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_false_pass_demotions(demos: list[dict], out_path: Path) -> None:
    if not demos:
        fig, ax = plt.subplots(figsize=(8.5, 4.0))
        ax.text(0.5, 0.5, "No 06J false-PASS candidates demoted by 06L "
                            "(or none had original ranks).",
                ha="center", va="center", fontsize=11)
        ax.axis("off")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        return
    rids = [d["recipe_id"] for d in demos]
    old = np.array([float(d["rank_06j"]) for d in demos])
    new = np.array([float(d["rank_06l"]) for d in demos])
    fig, ax = plt.subplots(figsize=(11.0, 5.5))
    x = np.arange(len(rids))
    ax.bar(x - 0.18, old, width=0.35, color="#9aaecf",
            label="06J surrogate rank")
    ax.bar(x + 0.18, new, width=0.35, color="#d62728",
            label="06L direct strict_score rank")
    for i, d in enumerate(demos):
        delta = float(d["rank_06l"] - d["rank_06j"])
        ax.text(i, max(old[i], new[i]) + 1, f"Δ={delta:+.0f}\n{d['fd_label']}",
                ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(rids, rotation=20, ha="right", fontsize=9)
    ax.set_xlabel("06J false-PASS recipe (FD label != robust_valid)")
    ax.set_ylabel("rank under each ranker (lower = higher up)")
    ax.set_title("Stage 06L -- ranking change for 06J false-PASS recipes")
    ax.grid(True, alpha=0.25, axis="y")
    ax.legend(loc="best", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_feature_importance(metrics: dict, out_path: Path) -> None:
    imp = metrics.get("feature_importance_strict_score", [])
    if not imp:
        return
    keys = FEATURE_KEYS[: len(imp)]
    order = np.argsort(imp)[::-1]
    keys = [keys[i] for i in order]
    vals = [imp[i] for i in order]
    fig, ax = plt.subplots(figsize=(11.0, 5.5))
    ax.bar(np.arange(len(keys)), vals, color="#d62728", alpha=0.85,
            edgecolor="#1f1f1f")
    ax.set_xticks(np.arange(len(keys)))
    ax.set_xticklabels(keys, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("strict_score head feature importance (06L RF)")
    ax.set_title("Stage 06L strict_score head feature importance "
                  "(higher = stronger signal for ranking)")
    ax.grid(True, alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_J1453_vs_G4867(rescored_06g: list[dict], rescored_06j: list[dict],
                           fd_truth: dict, out_path: Path) -> None:
    g_row = next((r for r in rescored_06g if r["recipe_id"] == "G_4867"), None)
    j_row = next((r for r in rescored_06j if r["recipe_id"] == "J_1453"), None)
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.5))
    metrics_06l = ["strict_score_06l_eval", "strict_score_06l_direct_mean",
                    "p_robust_valid_06l", "mean_cd_fixed_06l", "mean_ler_locked_06l",
                    "mean_p_line_margin_06l"]
    if g_row and j_row:
        x = np.arange(len(metrics_06l))
        ax = axes[0]
        ax.bar(x - 0.18, [_safe_float(g_row.get(m)) for m in metrics_06l],
                width=0.35, color="#1f77b4", alpha=0.85, label="G_4867 (Mode A)")
        ax.bar(x + 0.18, [_safe_float(j_row.get(m)) for m in metrics_06l],
                width=0.35, color="#d62728", alpha=0.85, label="J_1453 (Mode B)")
        ax.set_xticks(x)
        ax.set_xticklabels(metrics_06l, rotation=25, ha="right", fontsize=8)
        ax.set_title("06L surrogate metrics  (G_4867 vs J_1453)")
        ax.grid(True, alpha=0.25, axis="y")
        ax.legend(loc="best", fontsize=9)

    # Right: FD-truth strict_pass_prob bar (from 06H / 06J FD MC).
    ax = axes[1]
    truth_rows = [
        ("G_4867", fd_truth.get("G_4867", {}).get("fd_mc_strict_pass_prob", float("nan"))),
        ("J_1453", fd_truth.get("J_1453", {}).get("fd_mc_strict_pass_prob", float("nan"))),
    ]
    rids = [t[0] for t in truth_rows]
    vals = [t[1] for t in truth_rows]
    colors = ["#1f77b4", "#d62728"]
    ax.bar(rids, vals, color=colors, alpha=0.85, edgecolor="#1f1f1f")
    for i, v in enumerate(vals):
        ax.text(i, (v if np.isfinite(v) else 0.0) + 0.02,
                f"{v:.3f}" if np.isfinite(v) else "n/a",
                ha="center", fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.axhline(0.5, color="#1f1f1f", ls="--", lw=1.0,
                label="strict_pass = 0.5")
    ax.set_ylabel("FD MC strict_pass_prob")
    ax.set_title("FD MC truth  (G_4867 vs J_1453)")
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.25, axis="y")

    fig.suptitle("Stage 06L -- G_4867 (Mode A default) vs J_1453 (Mode B strict-best)",
                  fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------
def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str,
                   default=str(V3_DIR / "configs" / "yield_optimization.yaml"))
    p.add_argument("--strict_cfg_yaml", type=str,
                   default=str(V3_DIR / "outputs" / "yield_optimization"
                               / "stage06G_strict_score_config.yaml"))
    p.add_argument("--top_06g_csv", type=str,
                   default=str(V3_DIR / "outputs" / "yield_optimization"
                               / "stage06G_top_recipes.csv"))
    p.add_argument("--top_06j_csv", type=str,
                   default=str(V3_DIR / "outputs" / "yield_optimization"
                               / "stage06J_mode_b_top_recipes.csv"))
    p.add_argument("--summary_06j_csv", type=str,
                   default=str(V3_DIR / "outputs" / "yield_optimization"
                               / "stage06J_mode_b_recipe_summary.csv"))
    p.add_argument("--fd_06j_nominal_csv", type=str,
                   default=str(V3_DIR / "outputs" / "labels"
                               / "stage06J_mode_b_fd_sanity.csv"))
    p.add_argument("--fd_06j_mc_csv", type=str,
                   default=str(V3_DIR / "outputs" / "labels"
                               / "stage06J_mode_b_fd_mc_optional.csv"))
    p.add_argument("--summary_06h_json", type=str,
                   default=str(V3_DIR / "outputs" / "logs"
                               / "stage06H_fd_verification_summary.json"))
    p.add_argument("--metrics_06l_json", type=str,
                   default=str(V3_DIR / "outputs" / "logs" / "stage06L_model_metrics.json"))
    p.add_argument("--clf_06l", type=str,
                   default=str(V3_DIR / "outputs" / "models" / "stage06L_classifier.joblib"))
    p.add_argument("--reg_06l", type=str,
                   default=str(V3_DIR / "outputs" / "models" / "stage06L_regressor.joblib"))
    p.add_argument("--aux_06l", type=str,
                   default=str(V3_DIR / "outputs" / "models"
                               / "stage06L_aux_cd_fixed_regressor.joblib"))
    p.add_argument("--strict_06l", type=str,
                   default=str(V3_DIR / "outputs" / "models"
                               / "stage06L_strict_score_regressor.joblib"))
    p.add_argument("--n_var_eval", type=int, default=200)
    p.add_argument("--seed_eval",  type=int, default=8585)
    args = p.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    space = CandidateSpace.from_yaml(cfg["candidate_space_yaml"])
    var_spec = VariationSpec.from_yaml_dict(cfg["process_variation"])
    score_cfg = YieldScoreConfig.from_yaml_dict(cfg["yield_score"])

    strict_yaml = yaml.safe_load(Path(args.strict_cfg_yaml).read_text())
    cd_tol  = float(strict_yaml["thresholds"]["cd_tol_nm"])
    ler_cap = float(strict_yaml["thresholds"]["ler_cap_nm"])
    strict_cfg = StrictScoreConfig(cd_tol_nm=cd_tol, ler_cap_nm=ler_cap)

    # Load 06L joblibs.
    clf, _ = load_model(args.clf_06l)
    reg, _ = load_model(args.reg_06l)
    aux, _ = load_model(args.aux_06l)
    strict_reg, _ = load_model(args.strict_06l)

    # Read inputs.
    rows_06g = read_labels_csv(args.top_06g_csv)
    _coerce(rows_06g, ["rank_strict", "strict_score", "yield_score"] + FEATURE_KEYS)
    rows_06j = read_labels_csv(args.top_06j_csv)
    _coerce(rows_06j, ["strict_score", "yield_score"] + FEATURE_KEYS)

    rows_06j_summary = read_labels_csv(args.summary_06j_csv)
    _coerce(rows_06j_summary, ["strict_score", "yield_score"] + FEATURE_KEYS)

    fd_06j_nominal = read_labels_csv(args.fd_06j_nominal_csv) \
        if Path(args.fd_06j_nominal_csv).exists() else []
    _coerce(fd_06j_nominal, ["CD_final_nm", "CD_locked_nm",
                                "LER_CD_locked_nm", "area_frac",
                                "P_line_margin", "strict_score_surrogate"])

    fd_06j_mc = read_labels_csv(args.fd_06j_mc_csv) \
        if Path(args.fd_06j_mc_csv).exists() else []
    _coerce(fd_06j_mc, ["CD_final_nm", "CD_locked_nm", "LER_CD_locked_nm",
                          "area_frac", "P_line_margin", "variation_idx"])

    summary_06h = json.loads(Path(args.summary_06h_json).read_text())

    # ----- Re-score 06G top-100 (Mode A) and 06J top-100 (Mode B) -----
    print(f"  re-scoring 06G top-{len(rows_06g)} with 06L stack ...")
    rescored_06g = rescore_candidates(rows_06g, clf, reg, aux, strict_reg,
                                          var_spec, args.n_var_eval, space,
                                          score_cfg, strict_cfg,
                                          seed=args.seed_eval)
    for r in rescored_06g:
        r["mode"] = "mode_a"  # 06G recipes are Mode A by construction
    print(f"  re-scoring 06J top-{len(rows_06j)} with 06L stack ...")
    rescored_06j = rescore_candidates(rows_06j, clf, reg, aux, strict_reg,
                                          var_spec, args.n_var_eval, space,
                                          score_cfg, strict_cfg,
                                          seed=args.seed_eval + 1)
    for r in rescored_06j:
        r["mode"] = "mode_b"

    # ----- FD-truth strict_pass_prob lookup table -----
    fd_truth: dict[str, dict] = {}
    # Mode A: 06H FD top-10 MC + 6 reps MC (300 var each).
    for r in summary_06h.get("fd_top10_mc_aggr", []):
        fd_truth[r["recipe_id"]] = {
            "mode": "mode_a",
            "fd_mc_strict_pass_prob": float(r.get("p_strict_pass", float("nan"))),
            "fd_mc_n":                int(r.get("n_mc", 0)),
            "source":                  "06H_top10_mc_100var",
        }
    for r in summary_06h.get("fd_rep_mc_aggr", []):
        fd_truth[r["recipe_id"]] = {
            "mode": "mode_a",
            "fd_mc_strict_pass_prob": float(r.get("p_strict_pass", float("nan"))),
            "fd_mc_n":                int(r.get("n_mc", 0)),
            "source":                  "06H_rep_mc_300var",
            "rep_kind":                str(r.get("rep_kind", "")),
        }
    # Mode B: 06J FD MC (5 targets x 100 var each).
    by_recipe_mc: dict[str, list[dict]] = {}
    for r in fd_06j_mc:
        rid = str(r.get("source_recipe_id", ""))
        by_recipe_mc.setdefault(rid, []).append(r)
    for rid, rs in by_recipe_mc.items():
        agg = _mc_aggregate_strict_pass(rs, cd_tol, ler_cap, strict_yaml)
        fd_truth[rid] = {
            "mode": "mode_b",
            "fd_mc_strict_pass_prob": agg["strict_pass_prob"],
            "fd_mc_n":                agg["n"],
            "source":                  "06J_mc_100var",
        }

    # Pairs for the predicted-vs-FD plot.
    pairs = []
    by_id_06g = {r["recipe_id"]: r for r in rescored_06g}
    by_id_06j = {r["recipe_id"]: r for r in rescored_06j}
    for rid, truth in fd_truth.items():
        rs = by_id_06g.get(rid) or by_id_06j.get(rid)
        if rs is None:
            continue
        pairs.append({
            "recipe_id":                rid,
            "mode":                      truth["mode"],
            "pred_strict_06l":           float(rs["strict_score_06l_direct_mean"]),
            "fd_mc_strict_pass_prob":   truth["fd_mc_strict_pass_prob"],
            "fd_mc_n":                  truth["fd_mc_n"],
        })

    # ----- Spearman per mode -----
    if pairs:
        all_pred = np.array([p["pred_strict_06l"] for p in pairs])
        all_truth = np.array([p["fd_mc_strict_pass_prob"] for p in pairs])
        rho_overall = spearman(all_pred, all_truth)
        a_pairs = [p for p in pairs if p["mode"] == "mode_a"]
        b_pairs = [p for p in pairs if p["mode"] == "mode_b"]
        rho_a = spearman(np.array([p["pred_strict_06l"] for p in a_pairs]),
                            np.array([p["fd_mc_strict_pass_prob"] for p in a_pairs])) \
                            if len(a_pairs) >= 3 else float("nan")
        rho_b = spearman(np.array([p["pred_strict_06l"] for p in b_pairs]),
                            np.array([p["fd_mc_strict_pass_prob"] for p in b_pairs])) \
                            if len(b_pairs) >= 3 else float("nan")
    else:
        rho_overall = rho_a = rho_b = float("nan")
        a_pairs = []; b_pairs = []

    # ----- False-PASS demotion analysis on 06J nominal FD candidates -----
    by_id_06j_summary = {r["recipe_id"]: r for r in rows_06j_summary}
    # Re-score 06J nominal candidates (33 of them) so we can rank them under 06L.
    candidates_for_fp: list[dict] = []
    for r in fd_06j_nominal:
        rid = str(r.get("source_recipe_id", ""))
        sur = by_id_06j_summary.get(rid)
        if sur is not None:
            candidates_for_fp.append(sur)
    # Already in rescored_06j? Some may not be -- they may live deeper in
    # the 5,000 summary. Re-score the small set explicitly.
    if candidates_for_fp:
        rescored_fp = rescore_candidates(candidates_for_fp, clf, reg, aux,
                                              strict_reg, var_spec,
                                              args.n_var_eval, space, score_cfg,
                                              strict_cfg, seed=args.seed_eval + 7)
        # Build rank-under-06J and rank-under-06L over the same pool.
        pool_06j = sorted(candidates_for_fp,
                            key=lambda r: -_safe_float(r.get("strict_score")))
        rank_06j = {r["recipe_id"]: i + 1 for i, r in enumerate(pool_06j)}
        pool_06l = sorted(rescored_fp,
                            key=lambda r: -float(r["strict_score_06l_direct_mean"]))
        rank_06l = {r["recipe_id"]: i + 1 for i, r in enumerate(pool_06l)}
        demos: list[dict] = []
        for r in fd_06j_nominal:
            rid = str(r.get("source_recipe_id", ""))
            label = str(r.get("label", ""))
            if label not in HARD_FAIL_LABELS:
                continue
            if rid not in rank_06j or rid not in rank_06l:
                continue
            demos.append({
                "recipe_id":          rid,
                "fd_label":           label,
                "rank_06j":           int(rank_06j[rid]),
                "rank_06l":           int(rank_06l[rid]),
                "delta_rank":         int(rank_06l[rid] - rank_06j[rid]),
                "strict_score_06j_surrogate": float(by_id_06j_summary[rid].get("strict_score", float("nan"))),
                "strict_score_06l_direct":    float(next(
                    rs for rs in rescored_fp if rs["recipe_id"] == rid)["strict_score_06l_direct_mean"]),
            })
    else:
        demos = []

    # ----- AL targets -----
    # Tag rules:
    #   1. high 06L strict but small mean_cd_error margin -> CD-stress
    #   2. far from G_4867 in feature space but high 06L strict -> diverse coverage
    #   3. boundary candidates near 06L strict_pass = 0.5 surrogate proxy
    g4867 = next(
        rep for rep in yaml.safe_load(
            (V3_DIR / "outputs" / "yield_optimization"
             / "stage06I_mode_a_final_recipes.yaml").read_text())["representatives"]
        if rep["recipe_id"] == "G_4867")
    g4867_params = {k: float(v) for k, v in g4867["parameters"].items()}
    bounds = {"dose_mJ_cm2": (21.0, 60.0), "sigma_nm": (0.0, 3.0),
                "DH_nm2_s": (0.3, 0.8), "time_s": (20.0, 45.0),
                "Hmax_mol_dm3": (0.15, 0.22), "kdep_s_inv": (0.35, 0.65),
                "Q0_mol_dm3": (0.0, 0.03), "kq_s_inv": (0.5, 2.0),
                "abs_len_nm": (15.0, 100.0), "pitch_nm": (18.0, 32.0),
                "line_cd_ratio": (0.45, 0.60)}
    def _g4867_dist(r):
        d = 0.0; n = 0
        for k in FEATURE_KEYS:
            lo, hi = bounds.get(k, (0.0, 1.0))
            w = max(hi - lo, 1e-12)
            a = float(r.get(k, np.nan)); b = float(g4867_params.get(k, np.nan))
            if np.isfinite(a) and np.isfinite(b):
                d += ((a - b) / w) ** 2; n += 1
        return float(np.sqrt(d / max(n, 1)))

    al_targets = []
    sorted_06l = sorted(rescored_06j + rescored_06g,
                          key=lambda r: -float(r["strict_score_06l_direct_mean"]))
    for r in sorted_06l[:200]:
        cd_margin = abs(float(r["mean_cd_fixed_06l"]) - CD_TARGET_NM) - cd_tol
        ler_margin = ler_cap - float(r["mean_ler_locked_06l"])
        dist_g4867 = _g4867_dist(r)
        tags = []
        if -0.05 <= cd_margin <= 0.05:    # within 0.05 nm of strict CD boundary
            tags.append("cd_boundary")
        if 0.0 <= ler_margin <= 0.10:
            tags.append("ler_boundary")
        if r["mode"] == "mode_b" and dist_g4867 > 0.30:
            tags.append("mode_b_far_from_G4867")
        if r["mode"] == "mode_b" and -0.05 <= float(r["mean_p_line_margin_06l"]) - 0.20 <= 0.02:
            tags.append("mode_b_low_margin")
        if not tags:
            continue
        al_targets.append({
            "recipe_id":     r["recipe_id"],
            "mode":           r["mode"],
            "tags":           ",".join(tags),
            "strict_score_06l_direct_mean":  float(r["strict_score_06l_direct_mean"]),
            "mean_cd_fixed_06l":             float(r["mean_cd_fixed_06l"]),
            "mean_ler_locked_06l":           float(r["mean_ler_locked_06l"]),
            "mean_p_line_margin_06l":        float(r["mean_p_line_margin_06l"]),
            "g4867_normalised_distance":     float(dist_g4867),
        })

    # ----- Outputs -----
    yopt_dir = V3_DIR / "outputs" / "yield_optimization"
    logs_dir = V3_DIR / "outputs" / "logs"
    fig_dir  = V3_DIR / "outputs" / "figures" / "06_yield_optimization"
    fig_dir.mkdir(parents=True, exist_ok=True)

    rescore_cols = ["recipe_id", "mode",
                     "strict_score_06l_eval", "strict_score_06l_direct_mean",
                     "strict_score_06l_direct_std",
                     "yield_score_06l", "p_robust_valid_06l",
                     "mean_cd_fixed_06l", "std_cd_fixed_06l",
                     "mean_ler_locked_06l", "std_ler_locked_06l",
                     "mean_p_line_margin_06l"] + FEATURE_KEYS
    out_rows = list(rescored_06g) + list(rescored_06j)
    with (yopt_dir / "stage06L_mode_b_rescored_candidates.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rescore_cols, extrasaction="ignore")
        w.writeheader()
        for r in out_rows:
            w.writerow(r)

    if demos:
        cols = ["recipe_id", "fd_label", "rank_06j", "rank_06l", "delta_rank",
                "strict_score_06j_surrogate", "strict_score_06l_direct"]
        with (yopt_dir / "stage06L_false_pass_demotions.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            for r in demos:
                w.writerow(r)
    else:
        with (yopt_dir / "stage06L_false_pass_demotions.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["recipe_id", "fd_label",
                                                  "rank_06j", "rank_06l", "delta_rank"])
            w.writeheader()

    if al_targets:
        cols = ["recipe_id", "mode", "tags",
                "strict_score_06l_direct_mean",
                "mean_cd_fixed_06l", "mean_ler_locked_06l",
                "mean_p_line_margin_06l", "g4867_normalised_distance"]
        with (yopt_dir / "stage06L_al_targets.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            for r in al_targets:
                w.writerow(r)
    else:
        with (yopt_dir / "stage06L_al_targets.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["recipe_id", "mode", "tags"])
            w.writeheader()

    # Acceptance-aligned summary JSON.
    n_demoted = sum(1 for d in demos if d["delta_rank"] > 0)
    fp_payload = {
        "stage": "06L",
        "policy": cfg["policy"],
        "ranking_strict_score_spearman_overall_vs_fd_mc_strict_pass_prob": rho_overall,
        "ranking_strict_score_spearman_mode_a": rho_a,
        "ranking_strict_score_spearman_mode_b": rho_b,
        "n_pairs_total": len(pairs),
        "n_pairs_mode_a": len(a_pairs),
        "n_pairs_mode_b": len(b_pairs),
        "false_pass_06j_input_count":    sum(
            1 for r in fd_06j_nominal
            if str(r.get("label", "")) in HARD_FAIL_LABELS
        ),
        "false_pass_06j_demoted_count":  int(n_demoted),
        "false_pass_demotions":           demos,
        "al_target_count":                int(len(al_targets)),
        "g4867_in_06l_top10":             bool(any(
            r["recipe_id"] == "G_4867"
            for r in sorted(rescored_06g,
                              key=lambda r: -float(r["strict_score_06l_direct_mean"]))[:10]
        )),
        "j1453_in_06l_top10_mode_b":      bool(any(
            r["recipe_id"] == "J_1453"
            for r in sorted(rescored_06j,
                              key=lambda r: -float(r["strict_score_06l_direct_mean"]))[:10]
        )),
        "policy_v2_OP_frozen":            bool(cfg["policy"].get("v2_OP_frozen", True)),
        "policy_published_data_loaded":   bool(cfg["policy"].get("published_data_loaded", False)),
        "policy_external_calibration":    "none",
    }
    (logs_dir / "stage06L_false_pass_reduction.json").write_text(
        json.dumps(fp_payload, indent=2, default=float))

    # ----- Figures -----
    plot_strict_score_pred_vs_fd(pairs,
                                    fig_dir / "stage06L_strict_score_pred_vs_fd.png")
    plot_rank_before_after(rows_06g, rows_06j, rescored_06g, rescored_06j,
                              highlight_ids={"G_4867": "#1f77b4",
                                                "G_1096": "#1f77b4",
                                                "G_715":  "#1f77b4",
                                                "G_4299": "#1f77b4",
                                                "G_3691": "#1f77b4",
                                                "G_2311": "#1f77b4",
                                                "G_829":  "#1f77b4",
                                                "G_1226": "#1f77b4",
                                                "J_1453": "#d62728",
                                                "J_2261": "#d62728",
                                                "J_4793": "#d62728"},
                              out_path=fig_dir / "stage06L_strict_score_rank_before_after.png")
    plot_mode_a_vs_mode_b_distribution(rescored_06g, rescored_06j,
                                          fig_dir / "stage06L_mode_a_vs_mode_b_strict_score.png")
    plot_false_pass_demotions(demos,
                                  fig_dir / "stage06L_false_pass_demotions.png")
    metrics_06l = json.loads(Path(args.metrics_06l_json).read_text())
    plot_feature_importance(metrics_06l,
                                fig_dir / "stage06L_feature_importance_strict_score.png")
    plot_J1453_vs_G4867(rescored_06g, rescored_06j, fd_truth,
                            fig_dir / "stage06L_J1453_vs_G4867_ranking.png")

    # ----- Console summary -----
    print(f"\nStage 06L -- analysis summary")
    print(f"  Spearman strict_score head vs FD MC strict_pass_prob:")
    print(f"    overall (n={len(pairs)}):     {rho_overall:+.3f}")
    print(f"    Mode A  (n={len(a_pairs)}):   {rho_a:+.3f}")
    print(f"    Mode B  (n={len(b_pairs)}):   {rho_b:+.3f}")
    print(f"  06J nominal false-PASS recipes: {fp_payload['false_pass_06j_input_count']}")
    print(f"    of those, demoted under 06L:  {n_demoted}")
    if demos:
        for d in demos:
            print(f"      {d['recipe_id']}  fd_label={d['fd_label']:<22}  "
                  f"rank: 06J #{d['rank_06j']} -> 06L #{d['rank_06l']}  (delta = {d['delta_rank']:+d})")
    print(f"  G_4867 in 06L Mode A top-10: {fp_payload['g4867_in_06l_top10']}")
    print(f"  J_1453 in 06L Mode B top-10: {fp_payload['j1453_in_06l_top10_mode_b']}")
    print(f"  AL targets tagged: {len(al_targets)}")
    print(f"  outputs ->")
    print(f"    {yopt_dir / 'stage06L_mode_b_rescored_candidates.csv'}")
    print(f"    {yopt_dir / 'stage06L_false_pass_demotions.csv'}")
    print(f"    {yopt_dir / 'stage06L_al_targets.csv'}")
    print(f"    {logs_dir / 'stage06L_false_pass_reduction.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
