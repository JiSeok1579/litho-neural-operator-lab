"""Stage 06A — surrogate-driven nominal-yield-proxy optimisation.

This is NOT real fab-yield prediction. The 04C surrogate is treated as
a fast oracle for the v2 nominal physics; the score is a *nominal-model
yield proxy*, not chip yield. v2_OP_frozen stays true and
published_data_loaded stays false throughout.

Pipeline
    1. (lazy-train if missing) auxiliary CD_fixed regressor.
    2. Mode A — fixed-design recipe optimisation: pitch=24, ratio=0.52,
       abs_len=50; sample N candidates over recipe knobs only.
    3. Mode B — open design-space exploration: every candidate-space axis
       is free.
    4. For each mode, evaluate yield_score over `variations_per_candidate`
       process variations using the closed Stage 04C classifier and 4-target
       regressor + auxiliary CD_fixed regressor.
    5. Score the v2 frozen OP under the same MC pipeline as a baseline.
    6. Write
         outputs/labels/06_yield_optimization_summary.csv
         outputs/labels/06_top_recipes_fixed_design_surrogate.csv
         outputs/labels/06_top_recipes_open_design_surrogate.csv
         outputs/logs/06_yield_optimization_summary.json
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from copy import deepcopy
from pathlib import Path

import joblib
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reaction_diffusion_peb_v3_screening.src.candidate_sampler import (
    CandidateSpace,
    sample_candidates,
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
    evaluate_single_recipe,
)


V3_DIR = ROOT / "reaction_diffusion_peb_v3_screening"


# --------------------------------------------------------------------------
# Auxiliary CD_fixed regressor — train once on the closed Stage 04C dataset.
# --------------------------------------------------------------------------
def ensure_cd_fixed_aux(cfg_aux: dict, model_path: Path) -> object:
    if model_path.exists():
        model, _ = load_model(model_path)
        print(f"  cd_fixed_aux: loaded {model_path}")
        return model

    print(f"  cd_fixed_aux: training (target = {cfg_aux['target_field']})")
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_absolute_error
    from sklearn.model_selection import train_test_split
    from reaction_diffusion_peb_v3_screening.src.metrics_io import (
        FEATURE_KEYS as FK, save_model,
    )

    rows = read_labels_csv(cfg_aux["source_csv"])
    X, y = [], []
    for r in rows:
        try:
            xi = [float(r[k]) for k in FK]
            yi = float(r[cfg_aux["target_field"]])
        except (KeyError, TypeError, ValueError):
            continue
        if not np.isfinite(yi):
            continue
        X.append(xi); y.append(yi)
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=float(cfg_aux.get("test_size", 0.2)),
        random_state=13,
    )
    est = cfg_aux["estimator"]
    model = RandomForestRegressor(
        n_estimators=int(est.get("n_estimators", 300)),
        max_depth=est.get("max_depth"),
        n_jobs=int(est.get("n_jobs", -1)),
        random_state=13,
    )
    model.fit(Xtr, ytr)
    yhat = model.predict(Xte)
    mae = float(mean_absolute_error(yte, yhat))
    print(f"  cd_fixed_aux: train_n={len(Xtr)} test_n={len(Xte)} test_MAE={mae:.3f} nm")
    save_model(model, model_path, metadata={
        "target_field": cfg_aux["target_field"],
        "feature_keys": FK,
        "test_mae_nm": mae,
        "n_train": int(len(Xtr)),
        "n_test":  int(len(Xte)),
    })
    return model


# --------------------------------------------------------------------------
# Sobol sampling helpers — Mode A pins three axes; Mode B is full-space.
# --------------------------------------------------------------------------
def _build_fixed_design_space(space: CandidateSpace, fixed: dict) -> CandidateSpace:
    """Return a CandidateSpace with the three fixed-design axes pinned."""
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
    """Build a v2-OP candidate dict that matches the layout produced by
    sample_candidates so the optimizer treats it identically."""
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


# --------------------------------------------------------------------------
# CSV writers
# --------------------------------------------------------------------------
def _write_summary_csv(rows: list[dict], path: Path,
                        extra_columns: list[str] | None = None) -> None:
    cols = ["mode"] + SUMMARY_COLUMNS
    if extra_columns:
        cols = cols + [c for c in extra_columns if c not in cols]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _stamp_mode(rows: list[dict], mode: str) -> None:
    for r in rows:
        r["mode"] = mode


# --------------------------------------------------------------------------
# Main driver
# --------------------------------------------------------------------------
def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str,
                   default=str(V3_DIR / "configs" / "yield_optimization.yaml"))
    p.add_argument("--n_candidates", type=int, default=None,
                   help="override sampling.n_candidates (for fast smoke runs)")
    p.add_argument("--n_var", type=int, default=None,
                   help="override sampling.variations_per_candidate")
    p.add_argument("--quick", action="store_true",
                   help="shortcut to (n_candidates=128, n_var=32)")
    args = p.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    space = CandidateSpace.from_yaml(cfg["candidate_space_yaml"])

    # Closed Stage 04C surrogate.
    classifier, _ = load_model(cfg["models"]["classifier"])
    regressor4, _ = load_model(cfg["models"]["regressor_4target"])
    cd_fixed_aux = ensure_cd_fixed_aux(
        cfg["aux_training"],
        Path(cfg["models"]["cd_fixed_aux"]),
    )

    var_spec = VariationSpec.from_yaml_dict(cfg["process_variation"])
    score_cfg = YieldScoreConfig.from_yaml_dict(cfg["yield_score"])

    n_cand = int(args.n_candidates if args.n_candidates is not None
                 else cfg["sampling"]["n_candidates"])
    n_var  = int(args.n_var if args.n_var is not None
                 else cfg["sampling"]["variations_per_candidate"])
    if args.quick:
        n_cand = 128
        n_var = 32
    seed = int(cfg["run"]["seed"])

    print(f"\nstage 06A — yield optimization")
    print(f"  classes available: {list(classifier.classes_)}")
    print(f"  n_candidates={n_cand}  n_variations={n_var}  seed={seed}")
    print(f"  candidate space: {[p['name'] for p in space.parameters]}")

    # ----- v2 frozen OP baseline (single recipe, same MC pipeline) -----
    op_base = _v2_baseline_recipe(space, cfg["v2_frozen_op"])
    t0 = time.time()
    baseline_row = evaluate_single_recipe(
        op_base, classifier, regressor4, cd_fixed_aux,
        var_spec, n_var, space, score_cfg, seed=seed,
    )
    print(f"  v2 frozen OP yield_score = {baseline_row['yield_score']:.4f}  "
          f"(P_robust = {baseline_row['p_robust_valid']:.3f}; "
          f"CD_fixed_mean = {baseline_row['mean_cd_fixed']:.2f} nm; "
          f"LER_locked_mean = {baseline_row['mean_ler_locked']:.2f} nm)")
    print(f"  baseline took {time.time() - t0:.1f}s")

    rows_all: list[dict] = []
    mode_summaries: dict = {}

    # ----- Mode A — fixed-design recipe optimisation -----
    if cfg["mode_a_fixed_design"].get("enabled", True):
        fixed = cfg["mode_a_fixed_design"]["fixed"]
        space_a = _build_fixed_design_space(space, fixed)
        print(f"\n  Mode A (fixed-design): pin {fixed}")
        cand_a = sample_candidates(space_a, n=n_cand, method=cfg["sampling"]["method"], seed=seed)
        for c in cand_a:
            c["_id"] = f"A_{c['_id']}"
        t0 = time.time()
        rows_a = evaluate_recipes(
            cand_a, classifier, regressor4, cd_fixed_aux,
            var_spec, n_var, space, score_cfg, seed=seed + 1,
        )
        _stamp_mode(rows_a, "fixed_design")
        rows_all.extend(rows_a)
        elapsed = time.time() - t0
        scores = np.array([r["yield_score"] for r in rows_a])
        beat = int(np.sum(scores > baseline_row["yield_score"]))
        print(f"  Mode A: {len(rows_a)} recipes scored in {elapsed:.1f}s "
              f"({len(rows_a)*n_var/elapsed:.0f} variations/s)")
        print(f"  Mode A: best score = {scores.max():.4f}  "
              f"|  beats baseline {beat} / {len(rows_a)} "
              f"({100.0*beat/len(rows_a):.1f} %)")
        mode_summaries["fixed_design"] = {
            "n_recipes": len(rows_a),
            "best_yield_score": float(scores.max()),
            "median_yield_score": float(np.median(scores)),
            "beats_baseline_count": beat,
            "beats_baseline_rate": float(beat / max(len(rows_a), 1)),
        }

    # ----- Mode B — open design-space exploration -----
    if cfg["mode_b_open_design"].get("enabled", True):
        print(f"\n  Mode B (open design): all candidate-space axes free")
        cand_b = sample_candidates(space, n=n_cand, method=cfg["sampling"]["method"], seed=seed + 100)
        for c in cand_b:
            c["_id"] = f"B_{c['_id']}"
        t0 = time.time()
        rows_b = evaluate_recipes(
            cand_b, classifier, regressor4, cd_fixed_aux,
            var_spec, n_var, space, score_cfg, seed=seed + 101,
        )
        _stamp_mode(rows_b, "open_design")
        rows_all.extend(rows_b)
        elapsed = time.time() - t0
        scores = np.array([r["yield_score"] for r in rows_b])
        beat = int(np.sum(scores > baseline_row["yield_score"]))
        print(f"  Mode B: {len(rows_b)} recipes scored in {elapsed:.1f}s "
              f"({len(rows_b)*n_var/elapsed:.0f} variations/s)")
        print(f"  Mode B: best score = {scores.max():.4f}  "
              f"|  beats baseline {beat} / {len(rows_b)} "
              f"({100.0*beat/len(rows_b):.1f} %)")
        mode_summaries["open_design"] = {
            "n_recipes": len(rows_b),
            "best_yield_score": float(scores.max()),
            "median_yield_score": float(np.median(scores)),
            "beats_baseline_count": beat,
            "beats_baseline_rate": float(beat / max(len(rows_b), 1)),
        }

    # ----- Outputs -----
    labels_dir = V3_DIR / "outputs" / "labels"
    logs_dir   = V3_DIR / "outputs" / "logs"

    # Stamp baseline so it sits in the summary CSV alongside candidates.
    baseline_for_csv = dict(baseline_row); baseline_for_csv["mode"] = "v2_frozen_op_baseline"
    _write_summary_csv([baseline_for_csv] + rows_all, labels_dir / "06_yield_optimization_summary.csv")

    top_n = int(cfg["reporting"]["top_n"])
    if cfg["mode_a_fixed_design"].get("enabled", True):
        rows_a_sorted = sorted(rows_a, key=lambda r: -r["yield_score"])[:top_n]
        _write_summary_csv(rows_a_sorted, labels_dir / "06_top_recipes_fixed_design_surrogate.csv")
    if cfg["mode_b_open_design"].get("enabled", True):
        rows_b_sorted = sorted(rows_b, key=lambda r: -r["yield_score"])[:top_n]
        _write_summary_csv(rows_b_sorted, labels_dir / "06_top_recipes_open_design_surrogate.csv")

    summary_payload = {
        "stage": "06A_yield_optimization",
        "policy": cfg["policy"],
        "n_candidates": n_cand,
        "variations_per_candidate": n_var,
        "seed": seed,
        "v2_frozen_op_baseline": {
            "yield_score":      baseline_row["yield_score"],
            "p_robust_valid":   baseline_row["p_robust_valid"],
            "p_margin_risk":    baseline_row["p_margin_risk"],
            "p_merged":         baseline_row["p_merged"],
            "p_under_exposed":  baseline_row["p_under_exposed"],
            "p_roughness":      baseline_row["p_roughness_degraded"],
            "p_numerical":      baseline_row["p_numerical_invalid"],
            "cd_error_penalty": baseline_row["cd_error_penalty"],
            "ler_penalty":      baseline_row["ler_penalty"],
            "mean_cd_fixed":    baseline_row["mean_cd_fixed"],
            "mean_cd_locked":   baseline_row["mean_cd_locked"],
            "mean_ler_locked":  baseline_row["mean_ler_locked"],
        },
        "mode_summaries": mode_summaries,
    }
    logs_dir.mkdir(parents=True, exist_ok=True)
    (logs_dir / "06_yield_optimization_summary.json").write_text(
        json.dumps(summary_payload, indent=2))

    print(f"\n  summary CSV  → {labels_dir / '06_yield_optimization_summary.csv'}")
    print(f"  top_fixed    → {labels_dir / '06_top_recipes_fixed_design_surrogate.csv'}")
    print(f"  top_open     → {labels_dir / '06_top_recipes_open_design_surrogate.csv'}")
    print(f"  summary JSON → {logs_dir   / '06_yield_optimization_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
