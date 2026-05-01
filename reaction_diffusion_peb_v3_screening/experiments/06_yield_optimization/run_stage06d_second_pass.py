"""Stage 06D — second-pass surrogate-driven yield optimisation using
the Stage 06C refreshed surrogate.

Pipeline
    1. Sobol-sample 5,000 fresh Mode A fixed-design candidates with a
       new seed (different from Stage 06A's seed=1011), keeping the
       candidate-space bounds identical.
    2. Score each candidate with the refreshed 06C surrogate using the
       same 200-variation MC pipeline, the same yield_score formula,
       and the same v2 frozen-OP baseline.
    3. Re-score the SAME 5,000 candidates with the existing Stage 06A
       surrogate so disagreement candidates (high 06C, low 06A) can
       be flagged for downstream inspection.
    4. Run nominal FD on the top-20 06D recipes — a light sanity check
       that does not promote this stage to a full FD verification.
    5. Compare to the Stage 06A top-100 (recipe-id overlap, knob-wise
       distribution shift, novelty), write artefacts.

Policy
    v2_OP_frozen           = true
    published_data_loaded  = false
    No external calibration. Closed Stage 04C dataset / joblibs are
    not used here (06C joblibs are used instead). Mode B remains out
    of scope.

Outputs
    outputs/yield_optimization/stage06D_recipe_summary.csv
    outputs/yield_optimization/stage06D_top_recipes.csv
    outputs/yield_optimization/stage06D_novelty_candidates.csv
    outputs/yield_optimization/stage06D_disagreement_candidates.csv
    outputs/yield_optimization/stage06D_top20_fd_check.csv
    outputs/logs/stage06D_summary.json
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reaction_diffusion_peb_v3_screening.src.candidate_sampler import (
    CandidateSpace,
    sample_candidates,
)
from reaction_diffusion_peb_v3_screening.src.fd_batch_runner import (
    run_one_candidate,
)
from reaction_diffusion_peb_v3_screening.src.fd_yield_score import (
    nominal_yield_score,
)
from reaction_diffusion_peb_v3_screening.src.labeler import LabelThresholds
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
    evaluate_single_recipe,
)


V3_DIR = ROOT / "reaction_diffusion_peb_v3_screening"


# Recipe knobs we report distributions for (drops geometry pins).
RECIPE_KNOBS = [
    "dose_mJ_cm2", "sigma_nm", "DH_nm2_s", "time_s",
    "Hmax_mol_dm3", "kdep_s_inv", "Q0_mol_dm3", "kq_s_inv",
]


def _build_fixed_design_space(space: CandidateSpace, fixed: dict) -> CandidateSpace:
    new_params = []
    for p in space.parameters:
        if p["name"] in fixed:
            new_params.append({"name": p["name"], "type": "choice",
                               "values": [fixed[p["name"]]]})
        else:
            new_params.append(p)
    return CandidateSpace(parameters=new_params,
                          derived=space.derived,
                          fixed=space.fixed)


def _v2_baseline_recipe(space: CandidateSpace, op: dict) -> dict:
    base = {p["name"]: (p["values"][0] if p["type"] == "choice" else p["low"])
            for p in space.parameters}
    base.update({k: float(v) if not isinstance(v, int) else int(v)
                 for k, v in op.items()})
    base["line_cd_nm"]   = float(base["pitch_nm"]) * float(base["line_cd_ratio"])
    base["domain_x_nm"]  = float(base["pitch_nm"]) * 5.0
    base["dose_norm"]    = float(base["dose_mJ_cm2"]) / float(space.fixed["reference_dose_mJ_cm2"])
    for fk, fv in space.fixed.items():
        base.setdefault(fk, fv)
    base["_id"] = "v2_frozen_op"
    return base


def _normalised_feature_matrix(rows: list[dict],
                               space: CandidateSpace) -> np.ndarray:
    """Map every recipe knob into [0, 1] using the candidate-space
    bounds. Used for novelty / distance computation against 06A top
    recipes."""
    bounds = {}
    for p in space.parameters:
        if p["type"] == "uniform":
            bounds[p["name"]] = (float(p["low"]), float(p["high"]))
        elif p["type"] == "choice":
            vals = [float(v) for v in p["values"]]
            bounds[p["name"]] = (min(vals), max(vals))

    def _norm(v: float, lo: float, hi: float) -> float:
        if hi - lo < 1e-12:
            return 0.0
        return float(np.clip((v - lo) / (hi - lo), 0.0, 1.0))

    M = np.zeros((len(rows), len(RECIPE_KNOBS)), dtype=np.float64)
    for i, r in enumerate(rows):
        for j, k in enumerate(RECIPE_KNOBS):
            lo, hi = bounds.get(k, (0.0, 1.0))
            try:
                M[i, j] = _norm(float(r[k]), lo, hi)
            except (TypeError, ValueError, KeyError):
                M[i, j] = np.nan
    return M


def _row_to_full_candidate(row: dict, space: CandidateSpace) -> dict:
    """Re-hydrate a recipe row to a full candidate dict suitable for
    fd_batch_runner. Mirrors run_fd_verification.py."""
    out = {}
    for k in FEATURE_KEYS:
        try:
            out[k] = float(row[k])
        except (TypeError, ValueError, KeyError):
            raise ValueError(f"recipe row missing or non-numeric {k}: {row.get(k)!r}")
    out["pitch_nm"] = float(out["pitch_nm"])
    out["line_cd_nm"]  = out["pitch_nm"] * out["line_cd_ratio"]
    out["domain_x_nm"] = out["pitch_nm"] * 5.0
    out["dose_norm"]   = out["dose_mJ_cm2"] / float(space.fixed["reference_dose_mJ_cm2"])
    for fk, fv in space.fixed.items():
        out.setdefault(fk, fv)
    out["_id"] = row.get("recipe_id", "?")
    return out


def _write_csv(rows: list[dict], path: Path,
               column_order: list[str] | None = None) -> None:
    if not rows:
        return
    cols = column_order if column_order else list(rows[0].keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------
def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str,
                   default=str(V3_DIR / "configs" / "yield_optimization.yaml"))
    p.add_argument("--clf_06c", type=str,
                   default=str(V3_DIR / "outputs" / "models"
                               / "stage06C_classifier.joblib"))
    p.add_argument("--reg_06c", type=str,
                   default=str(V3_DIR / "outputs" / "models"
                               / "stage06C_regressor.joblib"))
    p.add_argument("--aux_06c", type=str,
                   default=str(V3_DIR / "outputs" / "models"
                               / "stage06C_aux_cd_fixed_regressor.joblib"))
    p.add_argument("--clf_06a", type=str,
                   default=str(V3_DIR / "outputs" / "models"
                               / "stage04C_classifier.joblib"))
    p.add_argument("--reg_06a", type=str,
                   default=str(V3_DIR / "outputs" / "models"
                               / "stage04C_regressor.joblib"))
    p.add_argument("--aux_06a", type=str,
                   default=str(V3_DIR / "outputs" / "models"
                               / "06_yield_optimization_cd_fixed_aux.joblib"))
    p.add_argument("--top_recipes_06a_csv", type=str,
                   default=str(V3_DIR / "outputs" / "labels"
                               / "06_top_recipes_fixed_design_surrogate.csv"))
    p.add_argument("--n_candidates", type=int, default=5000)
    p.add_argument("--n_var", type=int, default=200)
    p.add_argument("--seed", type=int, default=4042,
                   help="Sobol seed — must be different from Stage 06A's 1011.")
    p.add_argument("--top_n_report", type=int, default=100)
    p.add_argument("--top_n_fd", type=int, default=20)
    p.add_argument("--novelty_distance_threshold", type=float, default=0.30,
                   help="A 06D top-N recipe is 'novel' if its min "
                        "Euclidean distance (normalised feature space) "
                        "to any 06A top-100 recipe exceeds this.")
    p.add_argument("--label_schema", type=str,
                   default=str(V3_DIR / "configs" / "label_schema.yaml"))
    args = p.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    space = CandidateSpace.from_yaml(cfg["candidate_space_yaml"])
    var_spec = VariationSpec.from_yaml_dict(cfg["process_variation"])
    score_cfg = YieldScoreConfig.from_yaml_dict(cfg["yield_score"])

    if int(args.seed) == int(cfg["run"]["seed"]):
        raise SystemExit(
            f"Stage 06D seed ({args.seed}) must differ from Stage 06A seed "
            f"({cfg['run']['seed']}). Pass --seed something else.")

    # Closed-state preservation note.
    print(f"Stage 06D — second-pass yield optimisation")
    print(f"  policy: v2_OP_frozen={cfg['policy']['v2_OP_frozen']}, "
          f"published_data_loaded={cfg['policy']['published_data_loaded']}")
    print(f"  Stage 06A seed (skipped): {cfg['run']['seed']}")
    print(f"  Stage 06D seed:           {args.seed}")
    print(f"  n_candidates={args.n_candidates}  n_var={args.n_var}")

    # Load both surrogates.
    clf_06c, _ = load_model(args.clf_06c)
    reg_06c, _ = load_model(args.reg_06c)
    aux_06c, _ = load_model(args.aux_06c)
    clf_06a, _ = load_model(args.clf_06a)
    reg_06a, _ = load_model(args.reg_06a)
    aux_06a, _ = load_model(args.aux_06a)

    # Mode A fixed-design space.
    fixed = cfg["mode_a_fixed_design"]["fixed"]
    space_a = _build_fixed_design_space(space, fixed)

    # Sobol sample with a new seed.
    candidates = sample_candidates(
        space_a, n=args.n_candidates,
        method=cfg["sampling"]["method"], seed=args.seed,
    )
    for c in candidates:
        c["_id"] = f"D_{c['_id']}"

    # ----------------------------------------------------------------
    # Score with 06C and 06A surrogates on the same pool.
    # ----------------------------------------------------------------
    op_base = _v2_baseline_recipe(space, cfg["v2_frozen_op"])
    baseline_06c = evaluate_single_recipe(
        op_base, clf_06c, reg_06c, aux_06c,
        var_spec, args.n_var, space, score_cfg, seed=args.seed,
    )
    baseline_06a = evaluate_single_recipe(
        op_base, clf_06a, reg_06a, aux_06a,
        var_spec, args.n_var, space, score_cfg, seed=args.seed,
    )
    print(f"  v2 frozen OP — 06C yield_score = {baseline_06c['yield_score']:.4f}  "
          f"|  06A yield_score = {baseline_06a['yield_score']:.4f}")

    print(f"\n  scoring with refreshed 06C surrogate ...")
    t0 = time.time()
    rows_06c = evaluate_recipes(
        candidates, clf_06c, reg_06c, aux_06c,
        var_spec, args.n_var, space, score_cfg, seed=args.seed + 1,
    )
    print(f"  06C scoring: {len(rows_06c)} recipes in {time.time()-t0:.1f}s")

    print(f"  scoring with Stage 06A surrogate ...")
    t0 = time.time()
    rows_06a_rescore = evaluate_recipes(
        candidates, clf_06a, reg_06a, aux_06a,
        var_spec, args.n_var, space, score_cfg, seed=args.seed + 1,
    )
    print(f"  06A re-scoring: {len(rows_06a_rescore)} recipes in {time.time()-t0:.1f}s")

    # Merge per-recipe.
    by_id_06a = {r["recipe_id"]: r for r in rows_06a_rescore}
    merged: list[dict] = []
    for r in rows_06c:
        rid = r["recipe_id"]
        s_06a = by_id_06a.get(rid, {})
        merged.append({
            "recipe_id":             rid,
            "yield_score_06c":       float(r["yield_score"]),
            "yield_score_06a":       float(s_06a.get("yield_score", "nan")),
            "p_robust_valid_06c":    float(r["p_robust_valid"]),
            "p_robust_valid_06a":    float(s_06a.get("p_robust_valid", "nan")),
            "mean_cd_fixed_06c":     float(r["mean_cd_fixed"]),
            "mean_cd_fixed_06a":     float(s_06a.get("mean_cd_fixed", "nan")),
            "mean_ler_locked_06c":   float(r["mean_ler_locked"]),
            "mean_ler_locked_06a":   float(s_06a.get("mean_ler_locked", "nan")),
            "cd_error_penalty_06c":  float(r["cd_error_penalty"]),
            "ler_penalty_06c":       float(r["ler_penalty"]),
            **{k: float(r[k]) for k in FEATURE_KEYS},
        })

    # Sort by 06C yield_score descending; assign ranks.
    merged.sort(key=lambda r: -r["yield_score_06c"])
    for i, r in enumerate(merged, start=1):
        r["rank_06c"] = i
    rank_06a_lookup = {r["recipe_id"]: i for i, r in
                       enumerate(sorted(merged, key=lambda r: -r["yield_score_06a"]),
                                  start=1)}
    for r in merged:
        r["rank_06a_rescore"] = rank_06a_lookup[r["recipe_id"]]

    # ----------------------------------------------------------------
    # Compare to Stage 06A top-100.
    # ----------------------------------------------------------------
    rows_06a_top100 = read_labels_csv(args.top_recipes_06a_csv)
    rows_06a_top100.sort(key=lambda r: -float(r["yield_score"]))

    norm_06a = _normalised_feature_matrix(rows_06a_top100, space)
    norm_06d = _normalised_feature_matrix(merged[:args.top_n_report], space)
    # min Euclidean distance per 06D top-N recipe to any 06A top-100 recipe.
    dist = np.zeros(norm_06d.shape[0])
    for i in range(norm_06d.shape[0]):
        d = np.linalg.norm(norm_06a - norm_06d[i], axis=1)
        dist[i] = float(np.nanmin(d))
    for i, r in enumerate(merged[:args.top_n_report]):
        r["min_dist_to_06a_top100"] = float(dist[i])

    # Knob-wise distribution shift (top-100 vs top-100).
    knob_shift = {}
    for k in RECIPE_KNOBS:
        v_a = np.array([float(r.get(k, np.nan)) for r in rows_06a_top100],
                       dtype=np.float64)
        v_d = np.array([float(r.get(k, np.nan)) for r in merged[:args.top_n_report]],
                       dtype=np.float64)
        knob_shift[k] = {
            "mean_06a_top100": float(np.nanmean(v_a)),
            "mean_06d_top100": float(np.nanmean(v_d)),
            "std_06a_top100":  float(np.nanstd(v_a)),
            "std_06d_top100":  float(np.nanstd(v_d)),
            "delta_mean":      float(np.nanmean(v_d) - np.nanmean(v_a)),
        }

    # Recipe-id overlap. The Sobol pools use different seeds, so the
    # candidate _ids differ. Match instead by exact (rounded) feature
    # tuple — extremely unlikely to overlap.
    def _key(r: dict) -> tuple:
        return tuple(round(float(r[k]), 6) for k in RECIPE_KNOBS)

    top100_06a_keys = {_key(r) for r in rows_06a_top100}
    top100_06d_overlap = sum(1 for r in merged[:args.top_n_report]
                              if _key(r) in top100_06a_keys)

    # ----------------------------------------------------------------
    # Novelty + disagreement candidate sets.
    # ----------------------------------------------------------------
    novelty_rows: list[dict] = []
    for r in merged[:args.top_n_report]:
        if r.get("min_dist_to_06a_top100", 0.0) >= args.novelty_distance_threshold:
            novelty_rows.append({**r, "novelty_distance": r["min_dist_to_06a_top100"]})

    # Disagreement: high 06C, low 06A. Define as 06C top-quartile (q≥0.75)
    # while 06A bottom-half (q≤0.50). Only use the top_n_report set.
    sc_06c = np.array([r["yield_score_06c"] for r in merged[:args.top_n_report]])
    sc_06a = np.array([r["yield_score_06a"] for r in merged[:args.top_n_report]])
    q06c75 = float(np.nanquantile(sc_06c, 0.75))
    q06a50 = float(np.nanquantile(sc_06a, 0.50))
    disagreement_rows = [
        {**r, "score_gap": float(r["yield_score_06c"] - r["yield_score_06a"])}
        for r in merged[:args.top_n_report]
        if r["yield_score_06c"] >= q06c75 and r["yield_score_06a"] <= q06a50
    ]

    # ----------------------------------------------------------------
    # Light FD check on top-20 06D recipes.
    # ----------------------------------------------------------------
    thresholds = LabelThresholds.from_yaml(args.label_schema)
    fd_rows: list[dict] = []
    print(f"\n  light FD check on top-{args.top_n_fd} 06D recipes ...")
    t0 = time.time()
    for r in merged[:args.top_n_fd]:
        cand = _row_to_full_candidate(r, space)
        cand["_id"] = f"{r['recipe_id']}__nom_06d"
        fd_row = run_one_candidate(cand, thresholds)
        nom = nominal_yield_score(fd_row, score_cfg)
        fd_rows.append({
            "recipe_id":             r["recipe_id"],
            "rank_06c":              int(r["rank_06c"]),
            "yield_score_06c":       float(r["yield_score_06c"]),
            "yield_score_06a":       float(r["yield_score_06a"]),
            "fd_label":              str(fd_row.get("label", "")),
            "FD_yield_score_nom":    float(nom["FD_yield_score"]),
            "FD_CD_final_nm":        float(fd_row.get("CD_final_nm", float("nan"))),
            "FD_CD_locked_nm":       float(fd_row.get("CD_locked_nm", float("nan"))),
            "FD_LER_CD_locked_nm":   float(fd_row.get("LER_CD_locked_nm", float("nan"))),
            "FD_P_line_margin":      float(fd_row.get("P_line_margin", float("nan"))),
            "FD_area_frac":          float(fd_row.get("area_frac", float("nan"))),
        })
    print(f"  light FD: {len(fd_rows)} runs in {time.time()-t0:.1f}s")

    # FD-side acceptance summary.
    n_fd_robust = sum(1 for r in fd_rows if r["fd_label"] == "robust_valid")
    HARD_FAIL = {"under_exposed", "merged",
                 "roughness_degraded", "numerical_invalid"}
    n_fd_hard_fail = sum(1 for r in fd_rows if r["fd_label"] in HARD_FAIL)
    n_fd_margin = sum(1 for r in fd_rows if r["fd_label"] == "margin_risk")
    fd_yield_mean = float(np.mean([r["FD_yield_score_nom"] for r in fd_rows])) if fd_rows else 0.0

    # ----------------------------------------------------------------
    # Write outputs.
    # ----------------------------------------------------------------
    yopt_dir = V3_DIR / "outputs" / "yield_optimization"
    logs_dir = V3_DIR / "outputs" / "logs"

    summary_cols = [
        "recipe_id", "rank_06c", "rank_06a_rescore",
        "yield_score_06c", "yield_score_06a",
        "p_robust_valid_06c", "p_robust_valid_06a",
        "mean_cd_fixed_06c", "mean_cd_fixed_06a",
        "mean_ler_locked_06c", "mean_ler_locked_06a",
        "cd_error_penalty_06c", "ler_penalty_06c",
    ] + FEATURE_KEYS

    _write_csv(merged, yopt_dir / "stage06D_recipe_summary.csv",
               column_order=summary_cols)
    _write_csv(merged[:args.top_n_report],
               yopt_dir / "stage06D_top_recipes.csv",
               column_order=summary_cols + ["min_dist_to_06a_top100"])
    if novelty_rows:
        _write_csv(novelty_rows, yopt_dir / "stage06D_novelty_candidates.csv",
                   column_order=summary_cols + ["min_dist_to_06a_top100",
                                                  "novelty_distance"])
    else:
        # Write an empty (header-only) CSV so downstream readers don't fail.
        _write_csv([{**{k: "" for k in summary_cols},
                     "min_dist_to_06a_top100": "",
                     "novelty_distance": ""}],
                   yopt_dir / "stage06D_novelty_candidates.csv",
                   column_order=summary_cols + ["min_dist_to_06a_top100",
                                                  "novelty_distance"])
        # remove the placeholder row
        # (re-open and rewrite header only)
        with (yopt_dir / "stage06D_novelty_candidates.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=summary_cols + [
                "min_dist_to_06a_top100", "novelty_distance"])
            w.writeheader()
    if disagreement_rows:
        _write_csv(disagreement_rows,
                   yopt_dir / "stage06D_disagreement_candidates.csv",
                   column_order=summary_cols + ["score_gap"])
    else:
        with (yopt_dir / "stage06D_disagreement_candidates.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=summary_cols + ["score_gap"])
            w.writeheader()
    _write_csv(fd_rows, yopt_dir / "stage06D_top20_fd_check.csv")

    # ----- Acceptance -----
    summary_06a = json.loads(
        (logs_dir / "06_yield_optimization_summary.json").read_text())
    best_06a = float(summary_06a["mode_summaries"]["fixed_design"]["best_yield_score"])
    best_06d = float(merged[0]["yield_score_06c"])

    acceptance = {
        "best_yield_score_06a": best_06a,
        "best_yield_score_06d": best_06d,
        "delta_best_yield_score": float(best_06d - best_06a),
        "best_06d_ge_best_06a": bool(best_06d >= best_06a),
        # "convergence" criterion: top-100 of 06D has at least one
        # exact-feature overlap with 06A top-100 (rare with different
        # seeds), or the 06D top-100 is well-clustered near the 06A top
        # cluster (median min_dist_to_06a_top100 < 0.20).
        "median_min_dist_06d_top100_to_06a": float(np.median(dist)),
        "top_n_overlap_06a_06d": int(top100_06d_overlap),
        "n_novelty_candidates": int(len(novelty_rows)),
        "n_disagreement_candidates": int(len(disagreement_rows)),
        "fd_top20_robust_valid_count": int(n_fd_robust),
        "fd_top20_hard_fail_count":    int(n_fd_hard_fail),
        "fd_top20_margin_risk_count":  int(n_fd_margin),
        "fd_top20_yield_score_mean":   float(fd_yield_mean),
        "policy_v2_OP_frozen":         True,
        "policy_published_data_loaded": False,
        "policy_external_calibration": "none",
    }

    payload = {
        "stage": "06D",
        "policy": cfg["policy"],
        "n_candidates": args.n_candidates,
        "n_var_per_candidate": args.n_var,
        "seed": int(args.seed),
        "v2_frozen_op_baseline": {
            "yield_score_06c": float(baseline_06c["yield_score"]),
            "yield_score_06a": float(baseline_06a["yield_score"]),
        },
        "best_recipe_06d": {
            "recipe_id":  merged[0]["recipe_id"],
            "yield_score_06c": float(merged[0]["yield_score_06c"]),
            "yield_score_06a": float(merged[0]["yield_score_06a"]),
            **{k: float(merged[0][k]) for k in FEATURE_KEYS},
        },
        "knob_shift": knob_shift,
        "novelty": {
            "distance_threshold": float(args.novelty_distance_threshold),
            "n_candidates": int(len(novelty_rows)),
            "median_min_dist_top100": float(np.median(dist)),
            "min_min_dist_top100":    float(np.min(dist)),
            "max_min_dist_top100":    float(np.max(dist)),
        },
        "fd_top20_check": {
            "n_runs":            int(len(fd_rows)),
            "n_robust_valid":    int(n_fd_robust),
            "n_hard_fail":       int(n_fd_hard_fail),
            "n_margin_risk":     int(n_fd_margin),
            "fd_yield_score_mean":  float(fd_yield_mean),
        },
        "acceptance": acceptance,
    }
    (logs_dir / "stage06D_summary.json").write_text(json.dumps(payload, indent=2))

    # ----- Console summary -----
    print(f"\nStage 06D summary")
    print(f"  best yield_score    06A={best_06a:.4f}    "
          f"06D={best_06d:.4f}    Δ={best_06d - best_06a:+.4f}")
    print(f"  06D top-100 median min-dist to 06A top-100: "
          f"{np.median(dist):.3f} (threshold {args.novelty_distance_threshold:.2f})")
    print(f"  novelty candidates:      {len(novelty_rows)}")
    print(f"  disagreement candidates: {len(disagreement_rows)}")
    print(f"  exact-feature overlap (06D top-100 ∩ 06A top-100): {top100_06d_overlap}")
    print(f"  FD top-20 check: {n_fd_robust}/{args.top_n_fd} robust_valid, "
          f"{n_fd_hard_fail} hard fail, {n_fd_margin} margin_risk, "
          f"mean FD yield_score = {fd_yield_mean:.4f}")
    print(f"\n  recipe summary  → {yopt_dir / 'stage06D_recipe_summary.csv'}")
    print(f"  top-{args.top_n_report} CSV     → {yopt_dir / 'stage06D_top_recipes.csv'}")
    print(f"  novelty CSV     → {yopt_dir / 'stage06D_novelty_candidates.csv'}")
    print(f"  disagreement CSV→ {yopt_dir / 'stage06D_disagreement_candidates.csv'}")
    print(f"  FD top-20 CSV   → {yopt_dir / 'stage06D_top20_fd_check.csv'}")
    print(f"  summary JSON    → {logs_dir / 'stage06D_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
