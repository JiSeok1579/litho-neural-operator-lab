"""Stage 06G -- data-driven stricter scoring + Mode A re-optimisation.

Stage 06F's threshold-survival table feeds the choice of strict
thresholds; we keep the original yield_score formula's class weights
but add CD / LER strict penalties (with mean and std components) and
a small P_line_margin bonus.

Primary strict config (chosen from stage06F_threshold_sensitivity.csv):
    CD tolerance = 0.5 nm   -> 36 / 100 nominal survivors at top-100
    LER cap      = 3.0 nm   -> survival is independent of LER cap in
                                this dataset (all survivors have
                                LER << 3.0), so 3.0 is the natural
                                non-arbitrary choice.
Backup config (less aggressive on CD):
    CD tolerance = 0.75 nm  -> 54 / 100 nominal survivors at top-100
    LER cap      = 3.0 nm

Pipeline
    1. Sobol-sample 5,000 fresh Mode A fixed-design candidates with a
       new seed (5151, distinct from 06A's 1011 and 06D's 4042).
    2. Score each candidate with the refreshed Stage 06C surrogate
       using the same 200-variation MC pipeline, plus the original
       yield_score (for back-comparison) and the new strict_score.
    3. Optionally run nominal FD on the top-20 by strict_score (light
       sanity, not promotion).

Policy
    v2_OP_frozen           = true
    published_data_loaded  = false
    No external calibration. Closed Stage 04C / 04D / 06B / 06E
    artefacts are not mutated. 06C joblibs are read-only.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import dataclass
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


@dataclass
class StrictScoreConfig:
    """Stage 06G strict scoring weights. Mirrors the spec; defaults are
    serialised to stage06G_strict_score_config.yaml so the run is
    reproducible."""
    cd_tol_nm: float
    ler_cap_nm: float
    # class weights (same as 06A yield_score for parity).
    w_robust_valid:       float = 1.0
    w_margin_risk:        float = -0.5
    w_under_exposed:      float = -2.0
    w_merged:             float = -2.0
    w_roughness_degraded: float = -1.5
    w_numerical_invalid:  float = -3.0
    # strict (mean) penalties.
    w_cd_strict_pen:    float = 1.5
    w_ler_strict_pen:   float = 1.5
    # std penalties.
    w_cd_std_pen:    float = 0.5
    w_ler_std_pen:   float = 0.5
    # margin bonus.
    w_margin_bonus:  float = 0.25
    # std-penalty normalisers (documented, not arbitrary): CD_std is
    # divided by cd_tol_nm so a std equal to the tolerance contributes 1
    # unit of penalty; LER_std is divided by 1 nm.
    cd_std_norm_nm:  float = 1.0   # set in __post_init__
    ler_std_norm_nm: float = 1.0

    def __post_init__(self) -> None:
        # Normalise CD_std penalty by cd_tol_nm. LER_std stays in
        # absolute nm (no implicit scaling -- the user can change
        # ler_std_norm_nm later if needed).
        self.cd_std_norm_nm = float(self.cd_tol_nm)

    def to_yaml_dict(self) -> dict:
        return {
            "thresholds": {
                "cd_tol_nm":  self.cd_tol_nm,
                "ler_cap_nm": self.ler_cap_nm,
            },
            "class_weights": {
                "robust_valid":       self.w_robust_valid,
                "margin_risk":        self.w_margin_risk,
                "under_exposed":      self.w_under_exposed,
                "merged":             self.w_merged,
                "roughness_degraded": self.w_roughness_degraded,
                "numerical_invalid":  self.w_numerical_invalid,
            },
            "strict_penalties": {
                "cd_strict_weight":  self.w_cd_strict_pen,
                "ler_strict_weight": self.w_ler_strict_pen,
                "cd_std_weight":     self.w_cd_std_pen,
                "ler_std_weight":    self.w_ler_std_pen,
                "margin_bonus":      self.w_margin_bonus,
                "cd_std_norm_nm":    self.cd_std_norm_nm,
                "ler_std_norm_nm":   self.ler_std_norm_nm,
            },
            "formula":
                "strict_score = "
                "+ w_robust_valid*P(robust_valid) "
                "+ w_margin_risk*P(margin_risk) "
                "+ w_under_exposed*P(under_exposed) "
                "+ w_merged*P(merged) "
                "+ w_roughness_degraded*P(roughness_degraded) "
                "+ w_numerical_invalid*P(numerical_invalid) "
                "- w_cd_strict_pen*max(0, |mean(CD_fixed_nm)-15| - cd_tol_nm)/cd_tol_nm "
                "- w_ler_strict_pen*max(0, mean(LER_CD_locked_nm) - ler_cap_nm)/1.0 "
                "- w_cd_std_pen*(std_cd_fixed_nm/cd_std_norm_nm) "
                "- w_ler_std_pen*(std_ler_locked_nm/ler_std_norm_nm) "
                "+ w_margin_bonus*mean(P_line_margin)",
            "rationale_source": "stage06F_threshold_sensitivity.csv",
        }


def compute_strict_score(row: dict, cfg: StrictScoreConfig) -> dict:
    """row comes from evaluate_recipes -- carries p_* + mean_* + std_*."""
    p_robust = float(row.get("p_robust_valid", 0.0))
    p_margin = float(row.get("p_margin_risk", 0.0))
    p_ue     = float(row.get("p_under_exposed", 0.0))
    p_mg     = float(row.get("p_merged", 0.0))
    p_rd     = float(row.get("p_roughness_degraded", 0.0))
    p_ni     = float(row.get("p_numerical_invalid", 0.0))
    cd_mean   = float(row.get("mean_cd_fixed", float("nan")))
    cd_std    = float(row.get("std_cd_fixed", 0.0))
    ler_mean  = float(row.get("mean_ler_locked", float("nan")))
    ler_std   = float(row.get("std_ler_locked", 0.0))
    margin    = float(row.get("mean_p_line_margin", 0.0))

    # CD strict penalty (dimensionless, divided by cd_tol_nm).
    if np.isfinite(cd_mean):
        cd_pen = max(0.0, abs(cd_mean - 15.0) - cfg.cd_tol_nm) / max(cfg.cd_tol_nm, 1e-12)
    else:
        cd_pen = 0.0
    # LER strict penalty (nm above cap, no rescale).
    if np.isfinite(ler_mean):
        ler_pen = max(0.0, ler_mean - cfg.ler_cap_nm) / max(cfg.ler_std_norm_nm, 1e-12)
    else:
        ler_pen = 0.0
    # std penalties (positive contribution = punish noisy recipes).
    cd_std_pen  = float(cd_std) / max(cfg.cd_std_norm_nm,  1e-12)
    ler_std_pen = float(ler_std) / max(cfg.ler_std_norm_nm, 1e-12)
    # margin bonus.
    margin_b    = float(margin)

    score = (
        cfg.w_robust_valid       * p_robust
        + cfg.w_margin_risk      * p_margin
        + cfg.w_under_exposed    * p_ue
        + cfg.w_merged           * p_mg
        + cfg.w_roughness_degraded * p_rd
        + cfg.w_numerical_invalid * p_ni
        - cfg.w_cd_strict_pen    * cd_pen
        - cfg.w_ler_strict_pen   * ler_pen
        - cfg.w_cd_std_pen       * cd_std_pen
        - cfg.w_ler_std_pen      * ler_std_pen
        + cfg.w_margin_bonus     * margin_b
    )
    return {
        "strict_score":      float(score),
        "strict_cd_pen":     float(cd_pen),
        "strict_ler_pen":    float(ler_pen),
        "strict_cd_std_pen": float(cd_std_pen),
        "strict_ler_std_pen": float(ler_std_pen),
        "strict_margin_bonus": float(margin_b),
    }


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
    base["pitch_nm"]    = float(base["pitch_nm"])
    base["line_cd_nm"]  = base["pitch_nm"] * float(base["line_cd_ratio"])
    base["domain_x_nm"] = base["pitch_nm"] * 5.0
    base["dose_norm"]   = float(base["dose_mJ_cm2"]) / float(space.fixed["reference_dose_mJ_cm2"])
    for fk, fv in space.fixed.items():
        base.setdefault(fk, fv)
    base["_id"] = "v2_frozen_op"
    return base


def _row_to_full_candidate(row: dict, space: CandidateSpace) -> dict:
    out = {}
    for k in FEATURE_KEYS:
        out[k] = float(row[k])
    out["pitch_nm"]    = float(out["pitch_nm"])
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
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=column_order or [])
            w.writeheader()
        return
    cols = column_order if column_order else list(rows[0].keys())
    extras = []
    for r in rows:
        for k in r.keys():
            if k not in cols and k not in extras:
                extras.append(k)
    cols = cols + extras
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


# --------------------------------------------------------------------------
# Main
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
    p.add_argument("--n_candidates", type=int, default=5000)
    p.add_argument("--n_var", type=int, default=200)
    p.add_argument("--seed", type=int, default=5151,
                   help="Sobol seed -- distinct from 06A's 1011 and 06D's 4042.")
    p.add_argument("--cd_tol_nm",  type=float, default=0.5,
                   help="Primary strict CD tolerance (chosen from "
                        "stage06F_threshold_sensitivity.csv -- 36/100 "
                        "nominal survival).")
    p.add_argument("--ler_cap_nm", type=float, default=3.0,
                   help="Primary strict LER cap (survival is independent "
                        "of LER cap in this dataset; 3.0 is the natural "
                        "non-arbitrary choice).")
    p.add_argument("--top_n_report", type=int, default=100)
    p.add_argument("--top_n_fd",     type=int, default=20)
    p.add_argument("--label_schema",
                   default=str(V3_DIR / "configs" / "label_schema.yaml"))
    p.add_argument("--skip_fd", action="store_true")
    args = p.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    space = CandidateSpace.from_yaml(cfg["candidate_space_yaml"])
    var_spec = VariationSpec.from_yaml_dict(cfg["process_variation"])
    score_cfg = YieldScoreConfig.from_yaml_dict(cfg["yield_score"])

    if int(args.seed) == int(cfg["run"]["seed"]):
        raise SystemExit(
            f"Stage 06G seed ({args.seed}) must differ from Stage 06A seed "
            f"({cfg['run']['seed']}).")

    strict_cfg = StrictScoreConfig(cd_tol_nm=args.cd_tol_nm,
                                    ler_cap_nm=args.ler_cap_nm)

    print(f"Stage 06G -- data-driven strict re-optimisation")
    print(f"  policy: v2_OP_frozen={cfg['policy']['v2_OP_frozen']}, "
          f"published_data_loaded={cfg['policy']['published_data_loaded']}")
    print(f"  strict CD tolerance: +/-{strict_cfg.cd_tol_nm:.2f} nm")
    print(f"  strict LER cap:      <= {strict_cfg.ler_cap_nm:.2f} nm")
    print(f"  Sobol seed:          {args.seed}  (06A {cfg['run']['seed']}, 06D 4042)")
    print(f"  n_candidates={args.n_candidates}  n_var={args.n_var}")

    # Load 06C surrogate.
    clf, _ = load_model(args.clf_06c)
    reg, _ = load_model(args.reg_06c)
    aux, _ = load_model(args.aux_06c)

    # Mode A fixed-design space.
    fixed = cfg["mode_a_fixed_design"]["fixed"]
    space_a = _build_fixed_design_space(space, fixed)

    candidates = sample_candidates(
        space_a, n=args.n_candidates,
        method=cfg["sampling"]["method"], seed=args.seed,
    )
    for c in candidates:
        c["_id"] = f"G_{c['_id']}"

    # ----- Baseline (v2 frozen OP) under same surrogate + strict_score -----
    op_base = _v2_baseline_recipe(space, cfg["v2_frozen_op"])
    baseline_row = evaluate_single_recipe(
        op_base, clf, reg, aux,
        var_spec, args.n_var, space, score_cfg, seed=args.seed,
    )
    bl_strict = compute_strict_score(baseline_row, strict_cfg)
    baseline_row.update(bl_strict)
    print(f"  v2 frozen OP under 06C surrogate: "
          f"yield_score = {baseline_row['yield_score']:.4f}, "
          f"strict_score = {baseline_row['strict_score']:.4f}")

    # ----- Score full pool -----
    print(f"\n  scoring with 06C surrogate (yield_score + strict_score) ...")
    t0 = time.time()
    rows = evaluate_recipes(
        candidates, clf, reg, aux,
        var_spec, args.n_var, space, score_cfg, seed=args.seed + 1,
    )
    for r in rows:
        r.update(compute_strict_score(r, strict_cfg))
    print(f"  scored {len(rows)} recipes in {time.time()-t0:.1f}s")

    # Sort by strict_score desc; assign rank.
    rows.sort(key=lambda r: -float(r["strict_score"]))
    for i, r in enumerate(rows, start=1):
        r["rank_strict"] = i

    # Survival counts under primary strict thresholds.
    n_hard_pass = sum(
        1 for r in rows
        if abs(float(r["mean_cd_fixed"]) - 15.0) <= strict_cfg.cd_tol_nm
           and float(r["mean_ler_locked"]) <= strict_cfg.ler_cap_nm
           and float(r["p_robust_valid"]) >= 0.5
    )

    # ----- Light FD sanity on top-20 -----
    fd_rows: list[dict] = []
    if not args.skip_fd:
        thresholds = LabelThresholds.from_yaml(args.label_schema)
        print(f"\n  light FD check on top-{args.top_n_fd} 06G strict_score recipes ...")
        t0 = time.time()
        for r in rows[:args.top_n_fd]:
            cand = _row_to_full_candidate(r, space)
            cand["_id"] = f"{r['recipe_id']}__nom_06g"
            fd_row = run_one_candidate(cand, thresholds)
            nom = nominal_yield_score(fd_row, score_cfg)
            fd_rows.append({
                "recipe_id":          r["recipe_id"],
                "rank_strict":        int(r["rank_strict"]),
                "strict_score":       float(r["strict_score"]),
                "yield_score":        float(r["yield_score"]),
                "fd_label":           str(fd_row.get("label", "")),
                "FD_yield_score_nom": float(nom["FD_yield_score"]),
                "FD_CD_final_nm":     float(fd_row.get("CD_final_nm", float("nan"))),
                "FD_CD_locked_nm":    float(fd_row.get("CD_locked_nm", float("nan"))),
                "FD_LER_CD_locked_nm": float(fd_row.get("LER_CD_locked_nm", float("nan"))),
                "FD_P_line_margin":   float(fd_row.get("P_line_margin", float("nan"))),
                "FD_area_frac":       float(fd_row.get("area_frac", float("nan"))),
                "FD_CD_error_nm":     abs(float(fd_row.get("CD_final_nm", float("nan"))) - 15.0),
                "FD_pass_strict":     bool(
                    str(fd_row.get("label", "")) == "robust_valid"
                    and abs(float(fd_row.get("CD_final_nm", float("nan"))) - 15.0) <= strict_cfg.cd_tol_nm
                    and float(fd_row.get("LER_CD_locked_nm", float("nan"))) <= strict_cfg.ler_cap_nm
                ),
            })
        print(f"  light FD: {len(fd_rows)} runs in {time.time()-t0:.1f}s")

    # ----- Outputs -----
    yopt_dir = V3_DIR / "outputs" / "yield_optimization"
    logs_dir = V3_DIR / "outputs" / "logs"

    summary_cols = [
        "recipe_id", "rank_strict",
        "strict_score", "yield_score",
        "p_robust_valid", "p_margin_risk", "p_under_exposed",
        "p_merged", "p_roughness_degraded", "p_numerical_invalid",
        "mean_cd_fixed", "std_cd_fixed",
        "mean_cd_locked", "std_cd_locked",
        "mean_ler_locked", "std_ler_locked",
        "mean_area_frac", "std_area_frac",
        "mean_p_line_margin", "std_p_line_margin",
        "strict_cd_pen", "strict_ler_pen",
        "strict_cd_std_pen", "strict_ler_std_pen",
        "strict_margin_bonus",
        "cd_error_penalty", "ler_penalty",
    ] + FEATURE_KEYS

    _write_csv(rows, yopt_dir / "stage06G_recipe_summary.csv",
                column_order=summary_cols)
    _write_csv(rows[:args.top_n_report],
                yopt_dir / "stage06G_top_recipes.csv",
                column_order=summary_cols)
    if fd_rows:
        _write_csv(fd_rows, yopt_dir / "stage06G_top20_fd_check.csv")

    # Strict-score config YAML (full reproducibility).
    yaml_cfg = strict_cfg.to_yaml_dict()
    yaml_cfg["selection_summary"] = {
        "primary_choice_cd_tol_nm":  strict_cfg.cd_tol_nm,
        "primary_choice_ler_cap_nm": strict_cfg.ler_cap_nm,
        "expected_top100_survival_06f": (
            "Selected from stage06F_threshold_sensitivity.csv -- the cell "
            f"(cd_tol={strict_cfg.cd_tol_nm}, ler_cap={strict_cfg.ler_cap_nm}) "
            "had n_survivors that was nonzero but selective (~30-50%)."
        ),
        "backup_cd_tol_nm":  0.75,
        "backup_ler_cap_nm": 3.0,
    }
    (yopt_dir / "stage06G_strict_score_config.yaml").write_text(
        yaml.safe_dump(yaml_cfg, sort_keys=False))

    # Light-FD sanity stats.
    n_fd_robust = sum(1 for r in fd_rows if r["fd_label"] == "robust_valid")
    n_fd_strict_pass = sum(1 for r in fd_rows if r["FD_pass_strict"])
    HARD_FAIL = {"under_exposed", "merged", "roughness_degraded", "numerical_invalid"}
    n_fd_hard_fail = sum(1 for r in fd_rows if r["fd_label"] in HARD_FAIL)
    n_fd_margin = sum(1 for r in fd_rows if r["fd_label"] == "margin_risk")

    payload = {
        "stage": "06G",
        "policy": cfg["policy"],
        "n_candidates": args.n_candidates,
        "n_var_per_candidate": args.n_var,
        "seed": int(args.seed),
        "strict_config": yaml_cfg,
        "v2_frozen_op_under_06c_surrogate": {
            "yield_score":  float(baseline_row["yield_score"]),
            "strict_score": float(baseline_row["strict_score"]),
            "p_robust_valid": float(baseline_row["p_robust_valid"]),
            "mean_cd_fixed":   float(baseline_row["mean_cd_fixed"]),
            "std_cd_fixed":    float(baseline_row["std_cd_fixed"]),
            "mean_ler_locked": float(baseline_row["mean_ler_locked"]),
            "std_ler_locked":  float(baseline_row["std_ler_locked"]),
            "mean_p_line_margin": float(baseline_row["mean_p_line_margin"]),
        },
        "best_recipe": {
            "recipe_id":   rows[0]["recipe_id"],
            "strict_score": float(rows[0]["strict_score"]),
            "yield_score":  float(rows[0]["yield_score"]),
            "p_robust_valid": float(rows[0]["p_robust_valid"]),
            "mean_cd_fixed":  float(rows[0]["mean_cd_fixed"]),
            "std_cd_fixed":   float(rows[0]["std_cd_fixed"]),
            "mean_ler_locked": float(rows[0]["mean_ler_locked"]),
            "std_ler_locked":  float(rows[0]["std_ler_locked"]),
            **{k: float(rows[0][k]) for k in FEATURE_KEYS},
        },
        "n_recipes_passing_strict_thresholds_at_surrogate": int(n_hard_pass),
        "fd_top20_check": {
            "n_runs":            int(len(fd_rows)),
            "n_robust_valid":    int(n_fd_robust),
            "n_strict_pass":     int(n_fd_strict_pass),
            "n_hard_fail":       int(n_fd_hard_fail),
            "n_margin_risk":     int(n_fd_margin),
        },
        "acceptance_pretest": {
            "best_strict_score":            float(rows[0]["strict_score"]),
            "v2_op_strict_score":           float(baseline_row["strict_score"]),
            "best_06g_beats_v2_op":         bool(float(rows[0]["strict_score"]) > float(baseline_row["strict_score"])),
            "policy_v2_OP_frozen":          bool(cfg["policy"].get("v2_OP_frozen", True)),
            "policy_published_data_loaded": bool(cfg["policy"].get("published_data_loaded", False)),
            "policy_external_calibration":  "none",
        },
    }
    (logs_dir / "stage06G_summary.json").write_text(
        json.dumps(payload, indent=2, default=float))

    # ----- Console summary -----
    print(f"\nStage 06G summary")
    print(f"  v2 frozen OP -- yield_score = {baseline_row['yield_score']:.4f}, "
          f"strict_score = {baseline_row['strict_score']:.4f}")
    print(f"  06G best     -- strict_score = {rows[0]['strict_score']:.4f}, "
          f"yield_score = {rows[0]['yield_score']:.4f}, "
          f"recipe = {rows[0]['recipe_id']}")
    print(f"  recipes passing strict thresholds at surrogate level: "
          f"{n_hard_pass} / {len(rows)}")
    if fd_rows:
        print(f"  light FD top-{args.top_n_fd}: {n_fd_robust}/{len(fd_rows)} robust_valid, "
              f"{n_fd_strict_pass}/{len(fd_rows)} pass strict thresholds, "
              f"{n_fd_hard_fail} hard fail, {n_fd_margin} margin_risk")
    print(f"  recipe summary -> {yopt_dir / 'stage06G_recipe_summary.csv'}")
    print(f"  top-{args.top_n_report} CSV    -> {yopt_dir / 'stage06G_top_recipes.csv'}")
    print(f"  strict-score config -> {yopt_dir / 'stage06G_strict_score_config.yaml'}")
    if fd_rows:
        print(f"  FD top-20 CSV -> {yopt_dir / 'stage06G_top20_fd_check.csv'}")
    print(f"  summary JSON  -> {logs_dir / 'stage06G_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
