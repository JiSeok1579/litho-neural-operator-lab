"""Stage 04C — roughness_degraded expansion + per-class regression acceptance.

Pipeline:
    1. Load every prior labelled CSV (02, 04, 04b iter1, 04b iter2).
    2. Re-label every row with the Stage-04C labeler (3 OR triggers
       and the under_exposed > merged precedence swap).
    3. Sample 3 000 candidates from `target_roughness_degraded_v2`
       (the bias preset bypasses the budget_prefilter so near-failure
       cases survive).
    4. Run FD on the first `--fd_budget` of those (default 1 500).
    5. Append to the re-labelled seed.
    6. Retrain the RF classifier + regressor.
    7. Evaluate per-class metrics; check the new per-class acceptance
       criteria (CD global, LER non-merged, LER merged-only, etc.).

Outputs (written under outputs/):
    labels/04c_roughness_expansion.csv          new FD batch only
    labels/stage04C_training_dataset.csv        seed + 04C combined
    models/stage04C_classifier.joblib
    models/stage04C_regressor.joblib
    logs/stage04C_summary.json                  acceptance + metrics
    figures/04c_roughness_expansion/{...}       confusion + reliability + MAE
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
    CandidateSpace, sample_with_bias,
)
from reaction_diffusion_peb_v3_screening.src.evaluation import (
    classifier_report, plot_confusion, plot_per_class_mae, plot_reliability_diagram,
    regressor_global_metrics, regressor_mae_by_class,
)
from reaction_diffusion_peb_v3_screening.src.fd_batch_runner import run_batch
from reaction_diffusion_peb_v3_screening.src.labeler import (
    LABEL_ORDER, LabelThresholds, label_one, roughness_triggers,
)
from reaction_diffusion_peb_v3_screening.src.metrics_io import (
    FEATURE_KEYS, REGRESSION_TARGETS, build_feature_matrix,
    build_regression_target_matrix, read_labels_csv, save_model,
)
from reaction_diffusion_peb_v3_screening.src.surrogate_classifier import train_classifier
from reaction_diffusion_peb_v3_screening.src.surrogate_regressor import train_regressor


V3_DIR = ROOT / "reaction_diffusion_peb_v3_screening"


def _coerce_floats(rows):
    numeric = set(FEATURE_KEYS) | set(REGRESSION_TARGETS) | {
        "CD_final_nm", "CD_pitch_frac",
        "LER_after_PEB_P_nm", "LER_design_initial_nm",
        "P_space_center_mean", "P_line_center_mean",
        "contrast", "psd_locked_mid", "psd_design_mid",
        "H_min", "P_min", "P_max", "prefilter_score",
        "_id", "line_cd_nm", "domain_x_nm",
    }
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


def _relabel_seed(rows, thresholds):
    """Apply the Stage-04C labeler to every seed row in place. Return the
    pre-04C label histogram and the post-relabel histogram (for the diff)."""
    pre_counter = Counter(r.get("label", "") for r in rows)
    for r in rows:
        new = label_one(r, t=thresholds)
        r["label"] = new
        fired = roughness_triggers(r, thresholds)
        r["roughness_trigger"] = "+".join(fired) if fired else r.get("roughness_trigger", "")
    post_counter = Counter(r["label"] for r in rows)
    return pre_counter, post_counter


def _train_pair(rows, cfg, seed):
    X = build_feature_matrix(rows)
    y_cls = [r["label"] for r in rows]
    clf, _ = train_classifier(
        X, y_cls,
        n_estimators=int(cfg["surrogate"]["classifier"]["n_estimators"]),
        max_depth=cfg["surrogate"]["classifier"]["max_depth"],
        test_size=float(cfg["surrogate"]["test_size"]),
        seed=seed,
        n_jobs=int(cfg["surrogate"]["classifier"]["n_jobs"]),
    )
    valid_rows = [r for r in rows if r["label"] != "numerical_invalid"]
    if valid_rows:
        Xr = build_feature_matrix(valid_rows)
        Yr = build_regression_target_matrix(valid_rows, REGRESSION_TARGETS)
        reg, _ = train_regressor(
            Xr, Yr,
            n_estimators=int(cfg["surrogate"]["regressor"]["n_estimators"]),
            max_depth=cfg["surrogate"]["regressor"]["max_depth"],
            test_size=float(cfg["surrogate"]["test_size"]),
            seed=seed,
            n_jobs=int(cfg["surrogate"]["regressor"]["n_jobs"]),
        )
    else:
        reg = None
    return clf, reg


def _evaluate(rows, clf, reg, label_order=LABEL_ORDER, seed=13):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(rows))
    cut = int(0.8 * len(rows))
    test_rows = [rows[i] for i in idx[cut:]]
    X_te = build_feature_matrix(test_rows)
    y_te = [r["label"] for r in test_rows]
    y_pred = clf.predict(X_te).tolist()
    cls_rep = classifier_report(y_te, y_pred, label_order)

    reg_global = None
    reg_by_class = None
    if reg is not None:
        valid_test = [r for r in test_rows if r["label"] != "numerical_invalid"]
        Xv = build_feature_matrix(valid_test)
        Yv = build_regression_target_matrix(valid_test, REGRESSION_TARGETS)
        finite_mask = np.isfinite(Yv).all(axis=1)
        if finite_mask.any():
            Xv_f = Xv[finite_mask]
            Yv_f = Yv[finite_mask]
            valid_test_f = [r for r, m in zip(valid_test, finite_mask) if m]
            Y_pred = reg.predict(Xv_f)
            reg_global = regressor_global_metrics(Yv_f, Y_pred, REGRESSION_TARGETS)
            reg_by_class = regressor_mae_by_class(
                Yv_f, Y_pred,
                class_labels=[r["label"] for r in valid_test_f],
                targets=REGRESSION_TARGETS,
            )

    proba = clf.predict_proba(X_te)
    proba_max = proba.max(axis=1)
    correct = (np.array(y_pred) == np.array(y_te))

    return {
        "classifier": cls_rep,
        "regressor_global": reg_global,
        "regressor_by_class": reg_by_class,
        "calibration": (proba_max, correct),
        "test_rows": test_rows,
        "y_pred": y_pred,
    }


def _non_merged_global_mae(reg_by_class, target_key):
    """Aggregate per-class MAE excluding the merged class (count-weighted)."""
    if reg_by_class is None:
        return float("nan")
    classes = [c for c in reg_by_class.keys() if c != "merged"]
    if not classes:
        return float("nan")
    vals = [reg_by_class[c].get(target_key, float("nan")) for c in classes]
    finite = [v for v in vals if np.isfinite(v)]
    return float(np.mean(finite)) if finite else float("nan")


def _feature_importances_for_class(clf, target_label: str) -> list[tuple[str, float]]:
    """RandomForest exposes a single feature_importances_ vector — return the
    sorted list. (Per-class importance would require shap or per-tree analysis.)"""
    if not hasattr(clf, "feature_importances_"):
        return []
    items = list(zip(FEATURE_KEYS, clf.feature_importances_))
    items.sort(key=lambda x: -x[1])
    return items


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--biases_yaml", type=str,
                   default=str(V3_DIR / "configs" / "failure_seeking.yaml"))
    p.add_argument("--n_candidates", type=int, default=3000)
    p.add_argument("--fd_budget",   type=int, default=1500)
    p.add_argument("--seed_csvs", type=str, nargs="+", default=[
        str(V3_DIR / "outputs" / "labels" / "02_monte_carlo_dataset.csv"),
        str(V3_DIR / "outputs" / "labels" / "04_active_learning_loop.csv"),
        str(V3_DIR / "outputs" / "labels" / "04b_iter1.csv"),
        str(V3_DIR / "outputs" / "labels" / "04b_iter2.csv"),
    ])
    p.add_argument("--tag", type=str, default="04c_roughness_expansion")
    args = p.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    biases = yaml.safe_load(Path(args.biases_yaml).read_text())
    thresholds = LabelThresholds.from_yaml(cfg["label_schema_yaml"])
    space = CandidateSpace.from_yaml(cfg["candidate_space_yaml"])

    # 1. Load every prior labelled CSV.
    seed_rows = []
    for path in args.seed_csvs:
        if Path(path).exists():
            seed_rows.extend(_coerce_floats(read_labels_csv(path)))
    print(f"Stage 04C — seed rows loaded: {len(seed_rows)} from "
          f"{[Path(p).name for p in args.seed_csvs if Path(p).exists()]}")

    # 2. Re-label with the Stage-04C criteria.
    pre_counter, post_counter = _relabel_seed(seed_rows, thresholds)
    print("\nseed re-label diff:")
    for k in LABEL_ORDER:
        delta = post_counter.get(k, 0) - pre_counter.get(k, 0)
        sign = ("+" if delta >= 0 else "")
        print(f"  {k:<22} {pre_counter.get(k, 0):>5} → {post_counter.get(k, 0):>5}  ({sign}{delta})")

    # Train an evaluation baseline on the re-labelled seed (so we can compare
    # the post-04C model against the same labelling convention).
    clf_pre, reg_pre = _train_pair(seed_rows, cfg, seed=int(cfg["run"]["seed"]))
    pre_eval = _evaluate(seed_rows, clf_pre, reg_pre)

    # 3. Sample roughness-targeted candidates with prefilter bypass.
    bias_spec = biases["target_roughness_degraded_v2"]
    print(f"\nsampling {args.n_candidates} candidates from target_roughness_degraded_v2")
    cs = sample_with_bias(
        space, bias_spec=bias_spec, n=int(args.n_candidates),
        method="latin_hypercube", seed=int(cfg["run"]["seed"]) + 1,
    )
    for j, c in enumerate(cs):
        c["_id"] = f"04c_{j}"
        c["_failure_target"] = "target_roughness_degraded_v2"
    selected = cs[: int(args.fd_budget)]
    print(f"  prefilter_bypass={bias_spec.get('prefilter_bypass', False)} → "
          f"FD on {len(selected)} candidates")

    # 4. FD batch.
    out_csv = V3_DIR / "outputs" / "labels" / f"{args.tag}.csv"
    new_rows = run_batch(selected, thresholds=thresholds, out_csv=out_csv)

    new_counts = Counter(r["label"] for r in new_rows)
    print(f"\nStage 04C new-row label histogram:")
    for k in LABEL_ORDER:
        print(f"  {k:<22} {new_counts.get(k, 0)}")

    # 5. Combined dataset + retrain.
    combined = seed_rows + new_rows
    cum_counts = Counter(r["label"] for r in combined)
    print(f"\ncumulative dataset rows: {len(combined)}")
    for k in LABEL_ORDER:
        print(f"  {k:<22} {cum_counts.get(k, 0)}")

    clf, reg = _train_pair(combined, cfg, seed=int(cfg["run"]["seed"]))

    # Save artefacts.
    models_dir = V3_DIR / "outputs" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    save_model(clf, models_dir / "stage04C_classifier.joblib",
                metadata={"feature_keys": FEATURE_KEYS, "label_order": LABEL_ORDER,
                          "stage": "04C"})
    if reg is not None:
        save_model(reg, models_dir / "stage04C_regressor.joblib",
                    metadata={"feature_keys": FEATURE_KEYS,
                              "regression_targets": REGRESSION_TARGETS,
                              "stage": "04C"})

    # Save combined training dataset.
    train_csv_path = V3_DIR / "outputs" / "labels" / "stage04C_training_dataset.csv"
    train_csv_path.parent.mkdir(parents=True, exist_ok=True)
    keys = list({k for r in combined for k in r.keys()})
    keys.sort()
    with train_csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in combined:
            w.writerow(r)
    print(f"\ntraining dataset saved → {train_csv_path}")

    # 6. Final evaluation.
    final = _evaluate(combined, clf, reg, seed=13)
    cls_rep = final["classifier"]
    reg_global = final["regressor_global"] or {}
    reg_by_class = final["regressor_by_class"] or {}

    fig_dir = V3_DIR / "outputs" / "figures" / args.tag
    fig_dir.mkdir(parents=True, exist_ok=True)
    plot_confusion(cls_rep.confusion, LABEL_ORDER, fig_dir / "confusion_after_04c.png")
    proba_max, correct = final["calibration"]
    plot_reliability_diagram(proba_max, correct, fig_dir / "calibration_after_04c.png")
    if reg_by_class:
        plot_per_class_mae(reg_by_class, REGRESSION_TARGETS,
                            fig_dir / "regressor_mae_by_class.png")

    # 7. Per-class regression aggregates.
    cd_global = reg_global.get("CD_locked_nm", {}).get("MAE", float("nan"))
    ler_global = reg_global.get("LER_CD_locked_nm", {}).get("MAE", float("nan"))
    ler_non_merged = _non_merged_global_mae(reg_by_class, "LER_CD_locked_nm")
    ler_merged = reg_by_class.get("merged", {}).get("LER_CD_locked_nm", float("nan"))

    rough_per_class = cls_rep.per_class.get("roughness_degraded", {})
    rough_count = cum_counts.get("roughness_degraded", 0)

    pre_macro_f1 = pre_eval["classifier"].macro_f1
    pre_balanced = pre_eval["classifier"].balanced_accuracy

    acceptance = {
        "CD_locked_global_MAE_nm": cd_global,
        "CD_locked_global_MAE_under_0p15": bool(np.isfinite(cd_global) and cd_global <= 0.15),
        "LER_CD_locked_non_merged_MAE_nm": ler_non_merged,
        "LER_CD_locked_non_merged_under_0p03": bool(
            np.isfinite(ler_non_merged) and ler_non_merged <= 0.03),
        "LER_CD_locked_merged_MAE_nm": ler_merged,
        "LER_CD_locked_merged_under_0p15": bool(
            np.isfinite(ler_merged) and ler_merged <= 0.15),
        "macro_F1_pre": pre_macro_f1,
        "macro_F1_post": cls_rep.macro_f1,
        "macro_F1_did_not_drop_more_than_0p03": bool(
            cls_rep.macro_f1 >= pre_macro_f1 - 0.03),
        "balanced_accuracy": cls_rep.balanced_accuracy,
        "balanced_accuracy_at_least_0p93": bool(cls_rep.balanced_accuracy >= 0.93),
        "roughness_degraded_count": int(rough_count),
        "roughness_degraded_count_at_least_100": bool(rough_count >= 100),
        "roughness_degraded_recall": float(rough_per_class.get("recall", 0.0)),
        "roughness_degraded_recall_at_least_0p50":
            bool(rough_count >= 100 and float(rough_per_class.get("recall", 0.0)) >= 0.50),
        "policy_unchanged": {"v2_OP_frozen": True, "published_data_loaded": False},
    }

    print("\n=== Final classifier metrics ===")
    print(f"  accuracy           : {cls_rep.accuracy:.4f}")
    print(f"  balanced accuracy  : {cls_rep.balanced_accuracy:.4f}")
    print(f"  macro F1           : {cls_rep.macro_f1:.4f}  (pre-04C: {pre_macro_f1:.4f})")
    for label, m in cls_rep.per_class.items():
        print(f"  {label:<22}  P={m['precision']:.3f} R={m['recall']:.3f} "
              f"F1={m['f1']:.3f}  support={m['support']}")

    print("\n=== Final regressor metrics ===")
    if reg_global:
        for t, m in reg_global.items():
            print(f"  {t:<22}  global MAE={m['MAE']:.4f}  R²={m['R2']:+.4f}")
    if reg_by_class:
        cls_keys = list(reg_by_class.keys())
        header = "  " + " ".join(f"{k:>22}" for k in ["target"] + cls_keys)
        print(header)
        for t in REGRESSION_TARGETS:
            row = [t] + [f"{reg_by_class[c].get(t, float('nan')):.4f}"
                          for c in cls_keys]
            print("  " + " ".join(f"{r:>22}" for r in row))
    print(f"\n  LER non-merged MAE = {ler_non_merged:.4f}")
    print(f"  LER merged-only MAE = {ler_merged:.4f}")

    print("\n=== Acceptance ===")
    for k, v in acceptance.items():
        if isinstance(v, bool):
            tag = "PASS" if v else "FAIL"
            print(f"  {k:<48} {tag}")
        elif isinstance(v, float):
            print(f"  {k:<48} {v:.4f}")
        elif isinstance(v, dict):
            print(f"  {k:<48} {v}")
        else:
            print(f"  {k:<48} {v}")

    print("\n=== Top feature importances ===")
    for name, imp in _feature_importances_for_class(clf, "roughness_degraded")[:10]:
        print(f"  {name:<24}  {imp:.4f}")

    summary = {
        "stage": "04C",
        "policy": {"v2_OP_frozen": True, "published_data_loaded": False},
        "seed_rows_loaded": len(seed_rows),
        "seed_relabel_pre": dict(pre_counter),
        "seed_relabel_post": dict(post_counter),
        "n_candidates_sampled": int(args.n_candidates),
        "fd_budget": int(args.fd_budget),
        "new_label_counts": dict(new_counts),
        "cumulative_label_counts": dict(cum_counts),
        "classifier": {
            "accuracy": cls_rep.accuracy,
            "balanced_accuracy": cls_rep.balanced_accuracy,
            "macro_f1": cls_rep.macro_f1,
            "per_class": cls_rep.per_class,
        },
        "classifier_pre_04c": {
            "macro_f1": pre_macro_f1,
            "balanced_accuracy": pre_balanced,
        },
        "regressor_global": reg_global,
        "regressor_by_class": reg_by_class,
        "regressor_LER_non_merged_MAE": ler_non_merged,
        "regressor_LER_merged_MAE": ler_merged,
        "feature_importances": [
            {"feature": k, "importance": float(v)}
            for k, v in _feature_importances_for_class(clf, "roughness_degraded")
        ],
        "acceptance": acceptance,
    }
    log_path = V3_DIR / "outputs" / "logs" / f"stage04C_summary.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps(summary, indent=2, default=lambda x: float(x)))
    print(f"\nstage04C summary → {log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
