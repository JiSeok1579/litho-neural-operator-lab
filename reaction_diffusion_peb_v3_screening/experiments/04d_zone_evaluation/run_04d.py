"""Stage 04D — operational-zone evaluation and v3 first-pass closeout.

This stage does *not* train new models or run new FD. It loads the
Stage 04C dataset and the Stage 04C classifier + regressor, replays the
80/20 evaluation split (seed=13, matching Stage 04C), and reports
zone-aware metrics that are the right thing to measure for a
personal-study screening surrogate:

    operational zone : robust_valid + margin_risk
    failure zone     : under_exposed + merged + roughness_degraded +
                       numerical_invalid

The acceptance bands move from the per-class split used in Stage 04C
to a zone split. Operational-zone bands are the ones we care about.
Failure-zone numbers are reported but not gated — failure cells are
intrinsically noisy in CD-lock / LER, and trying to drive them down
would force calibration we cannot do without external reference data.

Outputs:
    outputs/figures/04d_zone_evaluation/{...}
    outputs/logs/stage04D_summary.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reaction_diffusion_peb_v3_screening.src.evaluation import (
    FAILURE_ZONE,
    OPERATIONAL_ZONE,
    binary_zone_metrics,
    classifier_report,
    label_to_zone,
    per_trigger_analysis,
    plot_confusion,
    plot_per_class_mae,
    plot_reliability_diagram,
    plot_trigger_overlap,
    plot_zone_confusion,
    regressor_global_metrics,
    regressor_mae_by_class,
    regressor_zone_aggregates,
    robust_vs_all_metrics,
)
from reaction_diffusion_peb_v3_screening.src.labeler import LABEL_ORDER
from reaction_diffusion_peb_v3_screening.src.metrics_io import (
    FEATURE_KEYS,
    REGRESSION_TARGETS,
    build_feature_matrix,
    build_regression_target_matrix,
    load_model,
    read_labels_csv,
)


V3_DIR = ROOT / "reaction_diffusion_peb_v3_screening"


def _coerce_floats(rows):
    """CSV-loaded rows are all strings — promote the columns we need."""
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


def _split_test(rows, seed: int = 13, test_frac: float = 0.2):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(rows))
    cut = int((1.0 - test_frac) * len(rows))
    return [rows[i] for i in idx[cut:]]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--training_csv", type=str,
                   default=str(V3_DIR / "outputs" / "labels"
                               / "stage04C_training_dataset.csv"))
    p.add_argument("--classifier", type=str,
                   default=str(V3_DIR / "outputs" / "models"
                               / "stage04C_classifier.joblib"))
    p.add_argument("--regressor", type=str,
                   default=str(V3_DIR / "outputs" / "models"
                               / "stage04C_regressor.joblib"))
    p.add_argument("--seed", type=int, default=13,
                   help="evaluation split seed (Stage 04C used 13)")
    p.add_argument("--tag", type=str, default="04d_zone_evaluation")
    args = p.parse_args()

    # Load dataset and saved models. Nothing is retrained.
    rows = _coerce_floats(read_labels_csv(args.training_csv))
    print(f"Stage 04D — dataset rows: {len(rows)}")
    label_counts = Counter(r["label"] for r in rows)
    print("dataset label histogram:")
    for k in LABEL_ORDER:
        print(f"  {k:<22} {label_counts.get(k, 0)}")

    clf, _clf_meta = load_model(args.classifier)
    reg, _reg_meta = load_model(args.regressor)

    # 80/20 split — same RNG as Stage 04C `_evaluate(..., seed=13)`.
    test_rows = _split_test(rows, seed=args.seed, test_frac=0.2)
    print(f"\ntest split: {len(test_rows)} rows (seed={args.seed})")

    X_te = build_feature_matrix(test_rows)
    y_te = [r["label"] for r in test_rows]
    y_pred = clf.predict(X_te).tolist()

    # ---- 1. Six-class classifier report (replays Stage 04C numbers) ----
    cls_rep = classifier_report(y_te, y_pred, LABEL_ORDER)

    # ---- 2. Zone-aware classifier metrics ----
    zone_metrics  = binary_zone_metrics(y_te, y_pred)
    robust_vs_all = robust_vs_all_metrics(y_te, y_pred)

    # ---- 3. Regressor on the non-numerical-invalid test rows ----
    valid_test = [r for r in test_rows if r["label"] != "numerical_invalid"]
    Xv = build_feature_matrix(valid_test)
    Yv = build_regression_target_matrix(valid_test, REGRESSION_TARGETS)
    finite_mask = np.isfinite(Yv).all(axis=1)
    Xv_f = Xv[finite_mask]
    Yv_f = Yv[finite_mask]
    valid_test_f = [r for r, m in zip(valid_test, finite_mask) if m]
    Y_pred = reg.predict(Xv_f)

    reg_global  = regressor_global_metrics(Yv_f, Y_pred, REGRESSION_TARGETS)
    reg_by_class = regressor_mae_by_class(
        Yv_f, Y_pred,
        class_labels=[r["label"] for r in valid_test_f],
        targets=REGRESSION_TARGETS,
    )
    n_per_class_test = Counter(r["label"] for r in valid_test_f)
    reg_zone = regressor_zone_aggregates(
        reg_by_class, n_per_class_test, REGRESSION_TARGETS,
    )

    # ---- 4. Per-trigger analysis on roughness_degraded test rows ----
    rough_rows_test = [r for r in test_rows if r["label"] == "roughness_degraded"]
    per_trigger = per_trigger_analysis(rough_rows_test)

    # ---- 5. Figures ----
    fig_dir = V3_DIR / "outputs" / "figures" / args.tag
    fig_dir.mkdir(parents=True, exist_ok=True)
    plot_confusion(cls_rep.confusion, LABEL_ORDER,
                   fig_dir / "confusion_six_class.png")
    plot_zone_confusion(zone_metrics, fig_dir / "confusion_zone.png")
    proba = clf.predict_proba(X_te)
    proba_max = proba.max(axis=1)
    correct = (np.array(y_pred) == np.array(y_te))
    plot_reliability_diagram(proba_max, correct,
                             fig_dir / "calibration.png")
    if reg_by_class:
        plot_per_class_mae(reg_by_class, REGRESSION_TARGETS,
                           fig_dir / "regressor_mae_by_class.png")
    if per_trigger["n_roughness_rows_with_trigger"] > 0:
        plot_trigger_overlap(per_trigger, fig_dir / "trigger_overlap.png")

    # ---- 6. Acceptance — zone-based bands ----
    cd_op_mae   = reg_zone["operational"].get("CD_locked_nm",     float("nan"))
    ler_op_mae  = reg_zone["operational"].get("LER_CD_locked_nm", float("nan"))
    plm_op_mae  = reg_zone["operational"].get("P_line_margin",    float("nan"))
    cd_fail_mae  = reg_zone["failure"].get("CD_locked_nm",     float("nan"))
    ler_fail_mae = reg_zone["failure"].get("LER_CD_locked_nm", float("nan"))

    acceptance = {
        # hard gates: operational zone
        "CD_locked_operational_MAE_nm":  cd_op_mae,
        "CD_locked_operational_under_0p15":
            bool(np.isfinite(cd_op_mae) and cd_op_mae <= 0.15),
        "LER_CD_locked_operational_MAE_nm": ler_op_mae,
        "LER_CD_locked_operational_under_0p03":
            bool(np.isfinite(ler_op_mae) and ler_op_mae <= 0.03),
        "P_line_margin_operational_MAE": plm_op_mae,
        "P_line_margin_operational_under_0p03":
            bool(np.isfinite(plm_op_mae) and plm_op_mae <= 0.03),
        # macro-F1 / balanced-accuracy carry-over
        "macro_F1": cls_rep.macro_f1,
        "macro_F1_at_least_0p93":  bool(cls_rep.macro_f1 >= 0.93),
        "balanced_accuracy": cls_rep.balanced_accuracy,
        "balanced_accuracy_at_least_0p93":
            bool(cls_rep.balanced_accuracy >= 0.93),
        # informational: failure zone
        "CD_locked_failure_MAE_nm":     cd_fail_mae,
        "LER_CD_locked_failure_MAE_nm": ler_fail_mae,
        # informational: zone confusion
        "operational_precision": zone_metrics["operational_precision"],
        "operational_recall":    zone_metrics["operational_recall"],
        "false_robust_valid_rate": zone_metrics["false_robust_valid_rate"],
        "false_defect_rate":       zone_metrics["false_defect_rate"],
        # policy
        "policy_unchanged": {"v2_OP_frozen": True, "published_data_loaded": False},
    }
    hard_gates = [
        "CD_locked_operational_under_0p15",
        "LER_CD_locked_operational_under_0p03",
        "P_line_margin_operational_under_0p03",
        "macro_F1_at_least_0p93",
        "balanced_accuracy_at_least_0p93",
    ]
    all_gates_pass = all(acceptance[k] for k in hard_gates)

    # ---- 7. Print summary ----
    print("\n=== Six-class classifier (replay) ===")
    print(f"  accuracy           : {cls_rep.accuracy:.4f}")
    print(f"  balanced accuracy  : {cls_rep.balanced_accuracy:.4f}")
    print(f"  macro F1           : {cls_rep.macro_f1:.4f}")

    print("\n=== Operational vs failure (binary zone) ===")
    for k in [
        "tp_op_op", "fn_op_fail", "fp_fail_op", "tn_fail_fail",
        "operational_precision", "operational_recall", "operational_f1",
        "false_robust_valid_rate", "false_defect_rate",
        "n_predicted_robust_valid", "n_predicted_failure",
        "support_operational",      "support_failure",
    ]:
        v = zone_metrics[k]
        if isinstance(v, float):
            print(f"  {k:<28} {v:.4f}")
        else:
            print(f"  {k:<28} {v}")

    print("\n=== robust_valid one-vs-rest ===")
    for k, v in robust_vs_all.items():
        if isinstance(v, float):
            print(f"  {k:<28} {v:.4f}")
        else:
            print(f"  {k:<28} {v}")

    print("\n=== Regressor — zone aggregates (count-weighted MAE) ===")
    header = f"  {'target':<22} {'operational':>14} {'failure':>14}"
    print(header)
    for t in REGRESSION_TARGETS:
        op = reg_zone["operational"].get(t, float("nan"))
        fa = reg_zone["failure"].get(t, float("nan"))
        print(f"  {t:<22} {op:>14.4f} {fa:>14.4f}")

    print("\n=== Per-trigger analysis (roughness_degraded test rows) ===")
    print(f"  rows with at least one trigger : "
          f"{per_trigger['n_roughness_rows_with_trigger']}")
    for t, c in per_trigger["counts_per_trigger"].items():
        print(f"  trigger {t:<22} count {c}")
    print(f"  multiplicity (k → rows with k triggers fired) : "
          f"{per_trigger['multiplicity']}")

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
    print(f"\n  ALL HARD GATES PASS              : "
          f"{'YES' if all_gates_pass else 'NO'}")

    if all_gates_pass:
        verdict = (
            "v3 first-pass screening surrogate: COMPLETE. "
            "Operational-zone bands met. Stage 05 autoencoder remains "
            "optional future work."
        )
    else:
        failing = [k for k in hard_gates if not acceptance[k]]
        verdict = (
            "v3 first-pass screening surrogate: NOT YET CLOSEABLE. "
            f"Hard gates failing: {failing}. "
            "Investigate before declaring closeout."
        )
    print(f"\nverdict: {verdict}")

    # ---- 8. Save summary JSON ----
    summary = {
        "stage": "04D",
        "policy": {"v2_OP_frozen": True, "published_data_loaded": False},
        "external_references_used": False,
        "interpretation": (
            "Personal-study screening: external reference data "
            "intentionally unavailable. v3 is candidate screening on "
            "the v2 nominal model, not external calibration. "
            "Operational-zone metrics are the relevant evaluation "
            "target; failure-zone metrics are informational."
        ),
        "training_dataset_rows": len(rows),
        "training_dataset_label_counts": dict(label_counts),
        "test_split": {
            "seed": int(args.seed), "size": int(len(test_rows)),
        },
        "classifier_six_class": {
            "accuracy":          cls_rep.accuracy,
            "balanced_accuracy": cls_rep.balanced_accuracy,
            "macro_f1":          cls_rep.macro_f1,
            "per_class":         cls_rep.per_class,
        },
        "classifier_zone_binary":  zone_metrics,
        "classifier_robust_vs_all": robust_vs_all,
        "regressor_global":         reg_global,
        "regressor_by_class":       reg_by_class,
        "regressor_zone_aggregates": reg_zone,
        "regressor_test_rows_per_class": dict(n_per_class_test),
        "per_trigger_analysis": per_trigger,
        "acceptance":  acceptance,
        "hard_gates":  hard_gates,
        "all_hard_gates_pass": bool(all_gates_pass),
        "verdict": verdict,
        "operational_zone_labels": list(OPERATIONAL_ZONE),
        "failure_zone_labels":     list(FAILURE_ZONE),
    }
    log_path = V3_DIR / "outputs" / "logs" / "stage04D_summary.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps(summary, indent=2, default=lambda x: float(x)))
    print(f"\nstage04D summary → {log_path}")
    print(f"figures           → {fig_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
