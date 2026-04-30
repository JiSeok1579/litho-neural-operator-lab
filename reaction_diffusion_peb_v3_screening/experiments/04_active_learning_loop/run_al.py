"""Active-learning iteration:
   sample fresh pool → prefilter → score with current surrogates →
   pick uncertain → FD → label → retrain → re-evaluate.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reaction_diffusion_peb_v3_screening.src.active_learning import acquisition_indices
from reaction_diffusion_peb_v3_screening.src.budget_prefilter import score_all, select_top_n
from reaction_diffusion_peb_v3_screening.src.candidate_sampler import (
    CandidateSpace, sample_candidates,
)
from reaction_diffusion_peb_v3_screening.src.fd_batch_runner import run_batch
from reaction_diffusion_peb_v3_screening.src.labeler import LABEL_ORDER, LabelThresholds
from reaction_diffusion_peb_v3_screening.src.metrics_io import (
    FEATURE_KEYS, REGRESSION_TARGETS, build_feature_matrix,
    build_regression_target_matrix, load_model, read_labels_csv, save_model,
)
from reaction_diffusion_peb_v3_screening.src.surrogate_classifier import (
    evaluate_classifier, train_classifier,
)
from reaction_diffusion_peb_v3_screening.src.surrogate_regressor import (
    evaluate_regressor, train_regressor,
)


V3_DIR = ROOT / "reaction_diffusion_peb_v3_screening"


def _coerce_float_keys(rows):
    numeric = set(FEATURE_KEYS) | {"CD_locked_nm", "LER_CD_locked_nm",
                                     "area_frac", "P_line_margin",
                                     "CD_final_nm", "CD_pitch_frac",
                                     "LER_after_PEB_P_nm", "LER_design_initial_nm",
                                     "P_space_center_mean", "P_line_center_mean",
                                     "contrast", "psd_locked_mid",
                                     "H_min", "P_min", "P_max", "prefilter_score",
                                     "_id", "line_cd_nm", "domain_x_nm"}
    out = []
    for r in rows:
        rr = dict(r)
        for k in numeric:
            if k in rr and rr[k] not in (None, "", "nan"):
                try:
                    rr[k] = float(rr[k])
                except (TypeError, ValueError):
                    rr[k] = float("nan")
        out.append(rr)
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--seed_labels_csv", type=str,
                   default=str(V3_DIR / "outputs" / "labels" / "02_monte_carlo_dataset.csv"))
    p.add_argument("--seed_classifier", type=str,
                   default=str(V3_DIR / "outputs" / "models" / "03_surrogate_screening_classifier.joblib"))
    p.add_argument("--seed_regressor", type=str,
                   default=str(V3_DIR / "outputs" / "models" / "03_surrogate_screening_regressor.joblib"))
    p.add_argument("--tag", type=str, default="04_active_learning_loop")
    args = p.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    space = CandidateSpace.from_yaml(cfg["candidate_space_yaml"])
    thresholds = LabelThresholds.from_yaml(cfg["label_schema_yaml"])
    al_cfg = cfg["active_learning"]

    seed_rows = _coerce_float_keys(read_labels_csv(args.seed_labels_csv))
    if not seed_rows:
        print(f"no seed labels found at {args.seed_labels_csv}")
        return 1

    clf, _ = load_model(args.seed_classifier)
    reg, _ = load_model(args.seed_regressor)

    # Generate AL pool, prefilter, then score uncertainty with current surrogates.
    pool = sample_candidates(space, n=int(al_cfg["pool_size"]),
                              method=cfg["sampling"]["method"],
                              seed=int(cfg["run"]["seed"]) + 1)
    pool_scored = score_all(pool)
    pool_retained = select_top_n(pool_scored, int(al_cfg["pool_prefilter_retain"]))

    X_pool = build_feature_matrix(pool_retained)
    selected_idx, diag = acquisition_indices(
        clf, reg, X_pool,
        classifier_unc_threshold=float(al_cfg["acquisition"]["classifier_uncertainty_threshold"]),
        regressor_std_min_nm=float(al_cfg["acquisition"]["regressor_std_min_nm"]),
        combine=str(al_cfg["combine"]),
        top_k=int(al_cfg["budget"]),
    )
    print(f"=== Stage 04 — active learning ===")
    print(json.dumps(diag, indent=2))

    al_candidates = [pool_retained[i] for i in selected_idx]
    out_csv = V3_DIR / "outputs" / "labels" / f"{args.tag}.csv"
    al_rows = run_batch(al_candidates, thresholds=thresholds, out_csv=out_csv)
    counts = Counter(r["label"] for r in al_rows)
    print("\nAL FD label histogram:")
    for k in LABEL_ORDER:
        print(f"  {k:<22} {counts.get(k, 0)}")

    # Combine seed + AL labels and retrain.
    combined = seed_rows + al_rows
    X = build_feature_matrix(combined)
    y_cls = [r["label"] for r in combined]
    clf2, (Xtr, Xte, ytr, yte) = train_classifier(
        X, y_cls,
        n_estimators=int(cfg["surrogate"]["classifier"]["n_estimators"]),
        max_depth=cfg["surrogate"]["classifier"]["max_depth"],
        test_size=float(cfg["surrogate"]["test_size"]),
        seed=int(cfg["run"]["seed"]),
        n_jobs=int(cfg["surrogate"]["classifier"]["n_jobs"]),
    )
    cls_eval2 = evaluate_classifier(clf2, Xte, yte, labels=LABEL_ORDER)
    print(f"\nClassifier after AL retrain → accuracy = {cls_eval2.accuracy:.4f}")
    print(cls_eval2.classification_report_str)

    # Regressor — drop numerical_invalid.
    train_rows = [r for r in combined if r["label"] != "numerical_invalid"]
    if train_rows:
        Xr = build_feature_matrix(train_rows)
        Yr = build_regression_target_matrix(train_rows, REGRESSION_TARGETS)
        reg2, (_a, Xr_te, _b, Yr_te) = train_regressor(
            Xr, Yr,
            n_estimators=int(cfg["surrogate"]["regressor"]["n_estimators"]),
            max_depth=cfg["surrogate"]["regressor"]["max_depth"],
            test_size=float(cfg["surrogate"]["test_size"]),
            seed=int(cfg["run"]["seed"]),
            n_jobs=int(cfg["surrogate"]["regressor"]["n_jobs"]),
        )
        reg_eval2 = evaluate_regressor(reg2, Xr_te, Yr_te, REGRESSION_TARGETS)
        print(f"\nRegressor after AL retrain — MAE per target:")
        for k, v in reg_eval2.mae_per_target.items():
            print(f"  {k:<22}  {v:.4f}")
    else:
        reg2 = None
        reg_eval2 = None

    models = V3_DIR / "outputs" / "models"
    save_model(clf2, models / f"{args.tag}_classifier.joblib",
                metadata={"feature_keys": FEATURE_KEYS, "label_order": LABEL_ORDER})
    if reg2 is not None:
        save_model(reg2, models / f"{args.tag}_regressor.joblib",
                    metadata={"feature_keys": FEATURE_KEYS,
                              "regression_targets": REGRESSION_TARGETS})

    summary = {
        "acquisition_diag": diag,
        "n_AL_FD_runs": len(al_rows),
        "AL_label_counts": dict(counts),
        "classifier_accuracy_after_AL": cls_eval2.accuracy,
        "regressor_mae_after_AL": reg_eval2.mae_per_target if reg_eval2 else None,
    }
    (V3_DIR / "outputs" / "logs" / f"{args.tag}_summary.json").write_text(
        json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
