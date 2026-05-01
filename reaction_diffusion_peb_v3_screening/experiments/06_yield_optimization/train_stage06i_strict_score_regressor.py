"""Stage 06I -- optional diagnostic: direct strict_score regressor.

Tests whether the 06H surrogate's poor strict-ranking on the 06G
representatives (1 / 6 in 06H top-20) is caused by the 4-target
regressor's loss being mismatched with the strict_score formula. If
training a regressor *directly* on strict_score (a scalar derived from
the FD outputs) recovers the FD ranking, the answer is yes -- the
surrogate refresh path forward is to add a strict_score head.

Pipeline:
    1. Build per-row strict_score labels from the 06H training
       dataset using the Stage 06G strict-config thresholds. A single
       FD row is a one-hot MC sample, so the strict_score formula
       collapses to:
           +w[label] - w_cd_strict * max(0, |CD_final - 15| - cd_tol)/cd_tol
                     - w_ler_strict * max(0, LER - ler_cap)/1
                     + w_margin_bonus * P_line_margin
       (CD_std / LER_std penalties drop out because std = 0 for a
       single point; the MC aggregate is recovered by averaging
       per-row predictions across MC variations downstream.)
    2. Train an RF regressor on (FEATURE_KEYS) -> per-row
       strict_score.
    3. Score the 06G top-100 candidates by drawing 200 process
       variations per recipe and averaging the regressor's per-row
       predictions -- same MC pipeline shape the 06G optimisation
       used.
    4. Compare predicted MC strict_score to FD ground-truth strict
       metrics on the 13 recipes that have FD MC (top-10 + 6 reps,
       overlap on G_3691 / G_3185 / G_1226). Report Spearman vs FD
       MC strict_pass_prob and Spearman vs the FD MC strict_score
       formula.

Outputs:
    outputs/models/stage06I_strict_score_regressor.joblib
    outputs/logs/stage06I_strict_score_regressor_metrics.json

Policy
    v2_OP_frozen           = true
    published_data_loaded  = false
    No external calibration. This regressor is diagnostic only -- it
    does not replace FD as final ranking authority.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml
from sklearn.ensemble import RandomForestRegressor

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reaction_diffusion_peb_v3_screening.src.candidate_sampler import (
    CandidateSpace,
)
from reaction_diffusion_peb_v3_screening.src.fd_yield_score import (
    spearman, topk_overlap,
)
from reaction_diffusion_peb_v3_screening.src.metrics_io import (
    FEATURE_KEYS,
    read_labels_csv,
    save_model,
)
from reaction_diffusion_peb_v3_screening.src.process_variation import (
    VariationSpec,
    sample_variations,
)


V3_DIR = ROOT / "reaction_diffusion_peb_v3_screening"
CD_TARGET_NM = 15.0


def _safe_float(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def per_row_strict_score(row: dict, strict_cfg: dict) -> float:
    """One-hot strict_score on a single FD row (no std penalties)."""
    label = str(row.get("label", ""))
    w_class = strict_cfg["class_weights"]
    w = float(w_class.get(label, 0.0))
    cd = _safe_float(row.get("CD_final_nm"))
    ler = _safe_float(row.get("LER_CD_locked_nm"))
    margin = _safe_float(row.get("P_line_margin", 0.0))
    sp = strict_cfg["strict_penalties"]
    cd_tol = float(strict_cfg["thresholds"]["cd_tol_nm"])
    ler_cap = float(strict_cfg["thresholds"]["ler_cap_nm"])

    if np.isfinite(cd):
        cd_pen = max(0.0, abs(cd - CD_TARGET_NM) - cd_tol) / max(cd_tol, 1e-12)
    else:
        cd_pen = 0.0
    if np.isfinite(ler):
        ler_pen = max(0.0, ler - ler_cap) / max(float(sp["ler_std_norm_nm"]), 1e-12)
    else:
        ler_pen = 0.0

    score = (
        w
        - float(sp["cd_strict_weight"])  * cd_pen
        - float(sp["ler_strict_weight"]) * ler_pen
        + float(sp["margin_bonus"])      * (margin if np.isfinite(margin) else 0.0)
    )
    return float(score)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str,
                   default=str(V3_DIR / "configs" / "yield_optimization.yaml"))
    p.add_argument("--stage06h_csv", type=str,
                   default=str(V3_DIR / "outputs" / "labels"
                               / "stage06H_training_dataset.csv"))
    p.add_argument("--stage04c_csv", type=str,
                   default=str(V3_DIR / "outputs" / "labels"
                               / "stage04C_training_dataset.csv"))
    p.add_argument("--strict_cfg_yaml", type=str,
                   default=str(V3_DIR / "outputs" / "yield_optimization"
                               / "stage06G_strict_score_config.yaml"))
    p.add_argument("--top_06g_csv", type=str,
                   default=str(V3_DIR / "outputs" / "yield_optimization"
                               / "stage06G_top_recipes.csv"))
    p.add_argument("--fd_summary_json", type=str,
                   default=str(V3_DIR / "outputs" / "logs"
                               / "stage06H_fd_verification_summary.json"))
    p.add_argument("--n_estimators", type=int, default=200,
                   help="Match Stage 06H regressor4 default (keeps joblib "
                        "well under GitHub 100 MB push limit).")
    p.add_argument("--seed_train", type=int, default=7)
    p.add_argument("--seed_split", type=int, default=13)
    p.add_argument("--n_var_eval", type=int, default=200)
    p.add_argument("--seed_eval", type=int, default=6464)
    p.add_argument("--n_jobs", type=int, default=-1)
    args = p.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    strict_cfg = yaml.safe_load(Path(args.strict_cfg_yaml).read_text())

    rows_06h = read_labels_csv(args.stage06h_csv)
    rows_04c = read_labels_csv(args.stage04c_csv)
    print(f"  06H training pool: {len(rows_06h)}")

    # Reproduce 04C 80/20 holdout (seed=13).
    rng = np.random.default_rng(args.seed_split)
    idx = rng.permutation(len(rows_04c))
    cut = int(0.8 * len(rows_04c))
    test_keys = set()
    for i in idx[cut:]:
        r = rows_04c[i]
        try:
            test_keys.add(tuple(round(_safe_float(r[k]), 6) for k in FEATURE_KEYS)
                            + (str(r.get("_id", "")),))
        except Exception:
            pass

    # Build training pool (drop 04C-test rows).
    train: list[dict] = []
    for r in rows_06h:
        if str(r.get("source", "")).startswith("stage06C/stage04C"):
            key = tuple(round(_safe_float(r[k]), 6) for k in FEATURE_KEYS) \
                  + (str(r.get("_id", "")),)
            if key in test_keys:
                continue
        train.append(r)
    print(f"  training pool after 04C-test removal: {len(train)}")

    # Per-row strict_score labels.
    X = np.array([[_safe_float(r.get(k)) for k in FEATURE_KEYS] for r in train])
    y = np.array([per_row_strict_score(r, strict_cfg) for r in train])
    finite = np.isfinite(y)
    print(f"  strict_score labels: finite = {int(finite.sum())} / {len(y)}  "
          f"(min={y[finite].min():.3f}  max={y[finite].max():.3f}  "
          f"mean={y[finite].mean():.3f})")

    # Train RF regressor.
    print(f"  training direct strict_score regressor "
          f"(n_estimators={args.n_estimators})")
    reg = RandomForestRegressor(
        n_estimators=args.n_estimators, max_depth=None,
        random_state=args.seed_train, n_jobs=args.n_jobs,
    )
    reg.fit(X[finite], y[finite])

    # Eval on the same 04C 1,074 held-out test rows -- per-row strict score MAE.
    rows_04c_te = [rows_04c[i] for i in idx[cut:]]
    X_te = np.array([[_safe_float(r.get(k)) for k in FEATURE_KEYS] for r in rows_04c_te])
    y_te = np.array([per_row_strict_score(r, strict_cfg) for r in rows_04c_te])
    finite_te = np.isfinite(y_te)
    pred_te = reg.predict(X_te)
    d = pred_te[finite_te] - y_te[finite_te]
    mae = float(np.mean(np.abs(d)))
    rmse = float(np.sqrt(np.mean(d ** 2)))
    rho_test = spearman(y_te[finite_te], pred_te[finite_te])
    print(f"  test-set per-row MAE = {mae:.4f}  RMSE = {rmse:.4f}  "
          f"Spearman ρ = {rho_test:.4f}")

    save_model(reg,
                V3_DIR / "outputs" / "models"
                / "stage06I_strict_score_regressor.joblib",
                metadata={"n_train": int(finite.sum()),
                          "feature_keys": FEATURE_KEYS,
                          "stage": "06I",
                          "n_estimators": args.n_estimators,
                          "target": "per_row_strict_score",
                          "strict_thresholds": strict_cfg["thresholds"]})

    # ----- Score 06G top-100 with MC variations -----
    print(f"\n  scoring 06G top-100 with direct regressor "
          f"({args.n_var_eval} MC variations each) ...")
    space = CandidateSpace.from_yaml(cfg["candidate_space_yaml"])
    var_spec = VariationSpec.from_yaml_dict(cfg["process_variation"])

    rows_06g = read_labels_csv(args.top_06g_csv)
    num_keys_06g = FEATURE_KEYS + ["rank_strict", "strict_score"]
    for r in rows_06g:
        for k in num_keys_06g:
            if k in r: r[k] = _safe_float(r[k])

    base_rng = np.random.default_rng(args.seed_eval)
    pred_per_recipe: dict[str, float] = {}
    for r in rows_06g:
        rid = r["recipe_id"]
        base = {k: float(r[k]) for k in FEATURE_KEYS}
        base["pitch_nm"]    = float(base["pitch_nm"])
        base["line_cd_nm"]  = base["pitch_nm"] * base["line_cd_ratio"]
        base["domain_x_nm"] = base["pitch_nm"] * 5.0
        base["dose_norm"]   = base["dose_mJ_cm2"] / float(space.fixed["reference_dose_mJ_cm2"])
        for fk, fv in space.fixed.items():
            base.setdefault(fk, fv)
        sub_rng = np.random.default_rng(int(base_rng.integers(0, 2**31 - 1)))
        variations = sample_variations(base, var_spec, args.n_var_eval, space, rng=sub_rng)
        Xv = np.array([[_safe_float(v.get(k)) for k in FEATURE_KEYS] for v in variations])
        yv = reg.predict(Xv)
        pred_per_recipe[rid] = float(np.mean(yv))

    # ----- Compare against FD MC truth on the 13 recipes that have FD MC -----
    fd_summary = json.loads(Path(args.fd_summary_json).read_text())
    fd_truth: dict[str, dict] = {}
    for r in fd_summary.get("fd_top10_mc_aggr", []):
        fd_truth[r["recipe_id"]] = {
            "fd_strict_pass_prob": float(r.get("p_strict_pass", float("nan"))),
            "fd_FD_MC_strict_score": float(r.get("FD_MC_strict_score", float("nan"))),
            "source": "top10_mc_100var",
        }
    for r in fd_summary.get("fd_rep_mc_aggr", []):
        rid = r["recipe_id"]
        existing = fd_truth.get(rid, {})
        # Prefer rep_mc 300-var when both exist.
        rep_payload = {
            "fd_strict_pass_prob": float(r.get("p_strict_pass", float("nan"))),
            "fd_FD_MC_strict_score": float(r.get("mean_strict_score", float("nan"))),
            "source": "rep_mc_300var",
            "fd_top10_strict_pass_prob_100var": existing.get("fd_strict_pass_prob",
                                                                 float("nan")),
        }
        fd_truth[rid] = rep_payload

    # 06G surrogate strict_score and 06H rescore strict_score for context.
    rescored_path = V3_DIR / "outputs" / "yield_optimization" / "stage06H_06g_recipes_rescored_by_06h.csv"
    rescored = read_labels_csv(rescored_path) if rescored_path.exists() else []
    for r in rescored:
        for k in ["rank_strict_06g", "strict_score_06g", "strict_score_06h"]:
            if k in r: r[k] = _safe_float(r[k])
    rescored_lookup = {r["recipe_id"]: r for r in rescored}

    # Build comparison table.
    cmp_rows = []
    for rid, truth in fd_truth.items():
        sur = next((r for r in rows_06g if r["recipe_id"] == rid), {})
        rs = rescored_lookup.get(rid, {})
        cmp_rows.append({
            "recipe_id":   rid,
            "fd_strict_pass_prob":   truth["fd_strict_pass_prob"],
            "fd_FD_MC_strict_score": truth["fd_FD_MC_strict_score"],
            "06g_surrogate_strict_score": _safe_float(sur.get("strict_score")),
            "06h_surrogate_strict_score": _safe_float(rs.get("strict_score_06h")),
            "06i_direct_strict_score":     pred_per_recipe.get(rid, float("nan")),
            "rank_06g_surrogate":          int(_safe_float(sur.get("rank_strict", 0))),
        })

    arr = np.array
    fd_pass = arr([r["fd_strict_pass_prob"] for r in cmp_rows])
    fd_strict = arr([r["fd_FD_MC_strict_score"] for r in cmp_rows])
    s06g = arr([r["06g_surrogate_strict_score"] for r in cmp_rows])
    s06h = arr([r["06h_surrogate_strict_score"] for r in cmp_rows])
    s06i = arr([r["06i_direct_strict_score"]     for r in cmp_rows])

    rho_06g_pass = spearman(s06g, fd_pass)
    rho_06h_pass = spearman(s06h, fd_pass)
    rho_06i_pass = spearman(s06i, fd_pass)
    rho_06g_strict = spearman(s06g, fd_strict)
    rho_06h_strict = spearman(s06h, fd_strict)
    rho_06i_strict = spearman(s06i, fd_strict)

    # Top-k overlap (ranking by predicted strict, ground truth = strict_pass_prob desc).
    rids = [r["recipe_id"] for r in cmp_rows]
    fd_rank = [rids[i] for i in np.argsort(-fd_pass)]
    sur_06g_rank = [rids[i] for i in np.argsort(-s06g)]
    sur_06h_rank = [rids[i] for i in np.argsort(-s06h)]
    sur_06i_rank = [rids[i] for i in np.argsort(-s06i)]
    overlap = {
        "06g_surrogate_top3":  topk_overlap(sur_06g_rank, fd_rank, 3),
        "06g_surrogate_top5":  topk_overlap(sur_06g_rank, fd_rank, 5),
        "06h_surrogate_top3":  topk_overlap(sur_06h_rank, fd_rank, 3),
        "06h_surrogate_top5":  topk_overlap(sur_06h_rank, fd_rank, 5),
        "06i_direct_top3":     topk_overlap(sur_06i_rank, fd_rank, 3),
        "06i_direct_top5":     topk_overlap(sur_06i_rank, fd_rank, 5),
    }

    # Whether the 8 manifest recipes stay high under each predictor.
    MANIFEST_IDS = ["G_4867", "G_1096", "G_715", "G_3691",
                     "G_2311", "G_829", "G_4299", "G_1226"]

    # 06I rank of each manifest recipe within all 06G top-100 (by 06I score).
    pred_full = sorted(pred_per_recipe.items(), key=lambda kv: -kv[1])
    rank_full_06i = {rid: i + 1 for i, (rid, _) in enumerate(pred_full)}
    rep_rank_06i = {rid: rank_full_06i.get(rid, -1) for rid in MANIFEST_IDS}

    metrics = {
        "stage": "06I",
        "policy": {**cfg["policy"], "external_calibration": "none"},
        "diagnostic_status": "optional -- does NOT replace FD final ranking",
        "test_set_per_row_metrics": {
            "n":     int(finite_te.sum()),
            "mae":   mae,
            "rmse":  rmse,
            "spearman_rho": rho_test,
        },
        "ranking_correlation_vs_fd_mc": {
            "n_recipes_with_fd_mc":    len(cmp_rows),
            "spearman_06g_surrogate_strict_vs_fd_strict_pass_prob": rho_06g_pass,
            "spearman_06h_surrogate_strict_vs_fd_strict_pass_prob": rho_06h_pass,
            "spearman_06i_direct_vs_fd_strict_pass_prob":          rho_06i_pass,
            "spearman_06g_surrogate_strict_vs_fd_mc_strict_score": rho_06g_strict,
            "spearman_06h_surrogate_strict_vs_fd_mc_strict_score": rho_06h_strict,
            "spearman_06i_direct_vs_fd_mc_strict_score":           rho_06i_strict,
        },
        "topk_overlap_vs_fd_truth": overlap,
        "manifest_recipe_rank_under_06i": rep_rank_06i,
        "comparison_table": cmp_rows,
        "interpretation": (
            "If spearman_06i_direct_vs_fd_strict_pass_prob is meaningfully "
            "higher than the 06g/06h surrogate values (e.g. > 0.3 vs ~ -0.4), "
            "the 4-target loss IS the bottleneck and a strict_score head "
            "should be added to the 06H surrogate stack. Otherwise the FD "
            "rank flip is intrinsic to the saturated regime and the FD "
            "ranking authority decision in 06H stands."
        ),
    }
    out_json = V3_DIR / "outputs" / "logs" / "stage06I_strict_score_regressor_metrics.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(metrics, indent=2, default=float))

    print(f"\nStage 06I strict_score regressor diagnostic")
    print(f"  test-set per-row MAE = {mae:.4f}  Spearman ρ = {rho_test:.4f}")
    print(f"  ranking correlation vs FD MC strict_pass_prob:")
    print(f"    06G surrogate strict_score : ρ = {rho_06g_pass:+.3f}")
    print(f"    06H surrogate strict_score : ρ = {rho_06h_pass:+.3f}")
    print(f"    06I direct strict_score    : ρ = {rho_06i_pass:+.3f}")
    print(f"  ranking correlation vs FD MC strict_score:")
    print(f"    06G surrogate strict_score : ρ = {rho_06g_strict:+.3f}")
    print(f"    06H surrogate strict_score : ρ = {rho_06h_strict:+.3f}")
    print(f"    06I direct strict_score    : ρ = {rho_06i_strict:+.3f}")
    print(f"  top-3 / top-5 overlap with FD truth:")
    print(f"    06G surrogate: top3={overlap['06g_surrogate_top3']}  "
          f"top5={overlap['06g_surrogate_top5']}")
    print(f"    06H surrogate: top3={overlap['06h_surrogate_top3']}  "
          f"top5={overlap['06h_surrogate_top5']}")
    print(f"    06I direct   : top3={overlap['06i_direct_top3']}  "
          f"top5={overlap['06i_direct_top5']}")
    print(f"  manifest recipe ranks under 06I direct (lower = better):")
    for rid in MANIFEST_IDS:
        print(f"    {rid}: rank #{rep_rank_06i[rid]}")
    print(f"  metrics -> {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
