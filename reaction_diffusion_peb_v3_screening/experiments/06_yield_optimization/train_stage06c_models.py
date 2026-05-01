"""Stage 06C — train the refreshed surrogate (classifier + 4-target
regressor + auxiliary CD_fixed regressor) on the merged training pool.

Train protocol
    1. Reproduce the Stage 04D 80/20 split on the *closed Stage 04C
       dataset only* (seed=13, np.random permutation). This pins the
       1,074-row test set so the 04C baseline numbers stay reproducible
       against `study_notes/03_v3_stage04d.md`.
    2. **Train a fair 04C baseline** on the 4,294 04D-train rows. The
       closed-state 04C joblibs were trained on the *full* 5,368 row
       dataset using sklearn's internal split, which leaks part of the
       04D test set into 04C training. To avoid this leakage in the
       06C-vs-04C comparison we train a fresh 04C-baseline on the same
       4,294 rows the 06C model uses (minus the 1,100 06B additions).
       The closed 04C joblibs on disk are not modified.
    3. Train the refreshed 06C model on
           Stage 04C train rows  (4,294 rows)
         + Stage 06B AL rows     (1,100 rows)
         = 5,394 rows.
    4. Evaluate the fair 04C baseline **and** the refreshed 06C model on
       the *same* 1,074-row 04D test set. Both models had zero overlap
       with the test rows during training, so any difference is the
       direct effect of including the 06B rows.

Outputs
    outputs/models/stage06C_classifier.joblib
    outputs/models/stage06C_regressor.joblib
    outputs/models/stage06C_aux_cd_fixed_regressor.joblib
    outputs/models/stage06C_fair_04c_baseline_classifier.joblib
    outputs/models/stage06C_fair_04c_baseline_regressor.joblib
    outputs/models/stage06C_fair_04c_baseline_aux_cd_fixed.joblib
    outputs/logs/stage06C_model_metrics.json

The `false_robust_valid_rate` definition follows Stage 04D /
yield-view: of all rows the classifier predicts as `robust_valid`,
what fraction has an actual *defect* label (under_exposed, merged,
roughness_degraded, numerical_invalid). margin_risk is excluded — it
is reported separately as `false_robust_valid_includes_margin_rate`.

Policy
    v2_OP_frozen           = true
    published_data_loaded  = false
    No external calibration. Closed Stage 04D / 04C training datasets
    are not mutated. The closed 04C joblibs on disk are not modified.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_fscore_support,
)

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reaction_diffusion_peb_v3_screening.src.labeler import (
    DEFECT_GROUP,
    LABEL_ORDER,
)
from reaction_diffusion_peb_v3_screening.src.metrics_io import (
    FEATURE_KEYS,
    REGRESSION_TARGETS,
    load_model,
    read_labels_csv,
    save_model,
)


V3_DIR = ROOT / "reaction_diffusion_peb_v3_screening"


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def _safe_float(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def _build_X(rows: list[dict]) -> np.ndarray:
    X = np.zeros((len(rows), len(FEATURE_KEYS)), dtype=np.float64)
    for i, r in enumerate(rows):
        for j, k in enumerate(FEATURE_KEYS):
            X[i, j] = _safe_float(r.get(k))
    return X


def _build_Y_regression(rows: list[dict],
                        keys: list[str] = REGRESSION_TARGETS) -> np.ndarray:
    Y = np.zeros((len(rows), len(keys)), dtype=np.float64)
    for i, r in enumerate(rows):
        for j, k in enumerate(keys):
            Y[i, j] = _safe_float(r.get(k))
    return Y


def _stage04d_split(rows: list[dict], seed: int = 13,
                    test_frac: float = 0.2) -> tuple[list[dict], list[dict]]:
    """Reproduce 04D's seed=13 80/20 permutation split."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(rows))
    cut = int((1.0 - test_frac) * len(rows))
    train_rows = [rows[i] for i in idx[:cut]]
    test_rows  = [rows[i] for i in idx[cut:]]
    return train_rows, test_rows


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def _classifier_metrics(y_true: list[str], y_pred: list[str],
                        labels: list[str] = LABEL_ORDER) -> dict:
    cm = confusion_matrix(y_true, y_pred, labels=labels).tolist()
    bal = float(balanced_accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, labels=labels,
                               average="macro", zero_division=0))
    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0)
    per_class = {
        labels[i]: {
            "precision": float(p[i]), "recall": float(r[i]),
            "f1": float(f[i]),         "support": int(s[i]),
        } for i in range(len(labels))
    }
    # false_robust_valid_rate (04D / yield-view definition): of all
    # rows the classifier predicted as robust_valid, what fraction had
    # an actual *defect* label (the four FAIL classes). margin_risk is
    # excluded — it lands in a separate, less-strict variant.
    DEFECT_LABELS = {"under_exposed", "merged",
                     "roughness_degraded", "numerical_invalid"}
    pred_pass = np.array([yp == "robust_valid" for yp in y_pred])
    actual_defect = np.array([yt in DEFECT_LABELS for yt in y_true])
    actual_non_robust = np.array([yt != "robust_valid" for yt in y_true])
    n_pred = int(pred_pass.sum())
    n_miss_defect = int((pred_pass & actual_defect).sum())
    n_miss_non_robust = int((pred_pass & actual_non_robust).sum())
    return {
        "accuracy":             float(np.mean(np.array(y_true) == np.array(y_pred))),
        "balanced_accuracy":    bal,
        "macro_f1":             macro_f1,
        "per_class":            per_class,
        "confusion_matrix":     cm,
        "labels":               list(labels),
        "n_predicted_robust_valid": n_pred,
        "n_missed_defect_in_pred_pass": n_miss_defect,
        "false_robust_valid_rate": float(n_miss_defect / n_pred) if n_pred > 0 else None,
        # Wider variant that includes margin_risk in the "false-robust" set.
        "n_missed_non_robust_in_pred_pass": n_miss_non_robust,
        "false_robust_valid_includes_margin_rate":
            float(n_miss_non_robust / n_pred) if n_pred > 0 else None,
    }


def _regression_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                        target_names: list[str] = REGRESSION_TARGETS) -> dict:
    out = {}
    for j, name in enumerate(target_names):
        finite = np.isfinite(y_true[:, j]) & np.isfinite(y_pred[:, j])
        if finite.sum() == 0:
            out[name] = {"mae": None, "rmse": None, "n": 0}
            continue
        d = y_true[finite, j] - y_pred[finite, j]
        out[name] = {
            "mae":  float(np.mean(np.abs(d))),
            "rmse": float(np.sqrt(np.mean(d ** 2))),
            "n":    int(finite.sum()),
        }
    return out


def _zone_breakdown_regression(rows: list[dict], y_pred: np.ndarray) -> dict:
    """Operational zone (label ∈ {robust_valid, margin_risk}) vs failure
    zone (label is one of the 4 defect / numerical labels). Computes
    per-target MAE inside each zone."""
    op_mask = np.array([
        DEFECT_GROUP.get(r.get("label", "?"), "defect") in ("normal", "marginal")
        for r in rows
    ])
    fail_mask = ~op_mask
    y_true = _build_Y_regression(rows, REGRESSION_TARGETS)

    def _zone(mask: np.ndarray) -> dict:
        if mask.sum() == 0:
            return {name: None for name in REGRESSION_TARGETS}
        return _regression_metrics(y_true[mask], y_pred[mask])

    return {
        "operational_zone": _zone(op_mask),
        "failure_zone":     _zone(fail_mask),
        "n_operational":    int(op_mask.sum()),
        "n_failure":        int(fail_mask.sum()),
    }


def _cd_fixed_metrics(rows: list[dict], y_pred: np.ndarray) -> dict:
    y_true = np.array([_safe_float(r.get("CD_final_nm")) for r in rows])
    finite = np.isfinite(y_true) & np.isfinite(y_pred)
    if finite.sum() == 0:
        return {"mae": None, "rmse": None, "n": 0}
    d = y_true[finite] - y_pred[finite]
    return {
        "mae":  float(np.mean(np.abs(d))),
        "rmse": float(np.sqrt(np.mean(d ** 2))),
        "n":    int(finite.sum()),
    }


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------
def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--stage04c_csv", type=str,
                   default=str(V3_DIR / "outputs" / "labels"
                               / "stage04C_training_dataset.csv"))
    p.add_argument("--stage06c_csv", type=str,
                   default=str(V3_DIR / "outputs" / "labels"
                               / "stage06C_training_dataset.csv"))
    p.add_argument("--baseline_classifier_joblib", type=str,
                   default=str(V3_DIR / "outputs" / "models"
                               / "stage04C_classifier.joblib"))
    p.add_argument("--baseline_regressor_joblib", type=str,
                   default=str(V3_DIR / "outputs" / "models"
                               / "stage04C_regressor.joblib"))
    p.add_argument("--baseline_aux_cd_fixed_joblib", type=str,
                   default=str(V3_DIR / "outputs" / "models"
                               / "06_yield_optimization_cd_fixed_aux.joblib"))
    p.add_argument("--n_estimators", type=int, default=300)
    p.add_argument("--seed_train", type=int, default=7)
    p.add_argument("--seed_split", type=int, default=13)
    p.add_argument("--n_jobs", type=int, default=-1)
    args = p.parse_args()

    rows_04c = read_labels_csv(args.stage04c_csv)
    rows_06c_all = read_labels_csv(args.stage06c_csv)

    # 04C-only train/test split — reproducible against 04D yield-view.
    rows_04c_tr, rows_04c_te = _stage04d_split(
        rows_04c, seed=args.seed_split, test_frac=0.2)
    print(f"  04C split (seed={args.seed_split}, 80/20): "
          f"train={len(rows_04c_tr)} test={len(rows_04c_te)}")

    # 06C training pool = 04C-train ∪ all 06B AL rows. The 06B rows are
    # those with source == "stage06B_yield_opt" in the merged CSV.
    rows_04c_te_keys = {id(r) for r in rows_04c_te}
    rows_06b_only = [r for r in rows_06c_all
                     if r.get("source") == "stage06B_yield_opt"]
    # 04C-train rows: identify by row content rather than identity since
    # rows_06c_all comes from a different file read.
    test_keys = set()
    for r in rows_04c_te:
        try:
            test_keys.add(tuple(round(_safe_float(r[k]), 6) for k in FEATURE_KEYS)
                          + (str(r.get("_id", "")),))
        except Exception:
            pass
    train_06c: list[dict] = []
    for r in rows_06c_all:
        if r.get("source") == "stage04C":
            key = tuple(round(_safe_float(r[k]), 6) for k in FEATURE_KEYS) \
                  + (str(r.get("_id", "")),)
            if key in test_keys:
                continue   # this 04C row is in the held-out 04C test set
        train_06c.append(r)
    print(f"  06C training pool: {len(train_06c)} rows  "
          f"(= 04C-train {len(rows_04c_tr)} + 06B {len(rows_06b_only)})")
    if len(train_06c) != len(rows_04c_tr) + len(rows_06b_only):
        print(f"  WARNING: 06C training pool size ({len(train_06c)}) does not "
              f"equal 04C-train + 06B ({len(rows_04c_tr) + len(rows_06b_only)})")

    # ----------------------------------------------------------------
    # 1. Classifier
    # ----------------------------------------------------------------
    X_tr = _build_X(train_06c); y_tr = [r["label"] for r in train_06c]
    X_te = _build_X(rows_04c_te); y_te = [r["label"] for r in rows_04c_te]

    print(f"\n  training 06C classifier (n_estimators={args.n_estimators})")
    clf06c = RandomForestClassifier(
        n_estimators=args.n_estimators, max_depth=None,
        random_state=args.seed_train, n_jobs=args.n_jobs,
    )
    clf06c.fit(X_tr, y_tr)
    save_model(clf06c, V3_DIR / "outputs" / "models" / "stage06C_classifier.joblib",
               metadata={"n_train": len(train_06c), "feature_keys": FEATURE_KEYS,
                         "labels": LABEL_ORDER})

    # ----------------------------------------------------------------
    # 2. 4-target regressor — train on rows where every target is finite.
    # ----------------------------------------------------------------
    Y_full = _build_Y_regression(train_06c)
    finite_mask = np.all(np.isfinite(Y_full), axis=1)
    print(f"  training 06C 4-target regressor on {int(finite_mask.sum())} "
          f"rows (drop {int((~finite_mask).sum())} with NaN targets)")
    reg06c = RandomForestRegressor(
        n_estimators=args.n_estimators, max_depth=None,
        random_state=args.seed_train, n_jobs=args.n_jobs,
    )
    reg06c.fit(X_tr[finite_mask], Y_full[finite_mask])
    save_model(reg06c, V3_DIR / "outputs" / "models" / "stage06C_regressor.joblib",
               metadata={"n_train": int(finite_mask.sum()),
                         "feature_keys": FEATURE_KEYS,
                         "targets": REGRESSION_TARGETS})

    # ----------------------------------------------------------------
    # 3. Auxiliary CD_fixed regressor.
    # ----------------------------------------------------------------
    y_cd = np.array([_safe_float(r.get("CD_final_nm")) for r in train_06c])
    cd_finite = np.isfinite(y_cd)
    print(f"  training 06C aux CD_fixed regressor on {int(cd_finite.sum())} rows")
    aux06c = RandomForestRegressor(
        n_estimators=args.n_estimators, max_depth=None,
        random_state=args.seed_train, n_jobs=args.n_jobs,
    )
    aux06c.fit(X_tr[cd_finite], y_cd[cd_finite])
    save_model(aux06c, V3_DIR / "outputs" / "models"
               / "stage06C_aux_cd_fixed_regressor.joblib",
               metadata={"n_train": int(cd_finite.sum()),
                         "target_field": "CD_final_nm",
                         "feature_keys": FEATURE_KEYS})

    # ----------------------------------------------------------------
    # 4. Train a *fair* 04C baseline on the 4,294 04D-train rows.
    #
    #    The closed-state 04C joblibs at outputs/models/stage04C_*.joblib
    #    were trained on the full 5,368-row dataset using sklearn's
    #    internal split, which leaks ~half of the 04D test set into 04C
    #    training. Comparing the closed joblibs to the strictly-
    #    holdout-trained 06C model would penalise 06C unfairly.
    #
    #    The fix: retrain a 04C baseline on the same 4,294 rows that
    #    feed the 06C model (minus the 1,100 06B additions). The closed
    #    04C joblibs on disk are NOT modified — we save the new fair
    #    baseline under stage06C_fair_04c_baseline_*.joblib.
    # ----------------------------------------------------------------
    X_tr_04c = _build_X(rows_04c_tr); y_tr_04c = [r["label"] for r in rows_04c_tr]

    print(f"\n  training fair 04C-baseline classifier "
          f"(n_train={len(rows_04c_tr)})")
    clf04c = RandomForestClassifier(
        n_estimators=args.n_estimators, max_depth=None,
        random_state=args.seed_train, n_jobs=args.n_jobs,
    )
    clf04c.fit(X_tr_04c, y_tr_04c)
    save_model(clf04c, V3_DIR / "outputs" / "models"
               / "stage06C_fair_04c_baseline_classifier.joblib",
               metadata={"n_train": len(rows_04c_tr),
                         "feature_keys": FEATURE_KEYS,
                         "labels": LABEL_ORDER,
                         "note": "fair-baseline trained on 04D-train rows only"})

    Y_full_04c = _build_Y_regression(rows_04c_tr)
    reg_mask_04c = np.all(np.isfinite(Y_full_04c), axis=1)
    print(f"  training fair 04C-baseline 4-target regressor "
          f"(n_train={int(reg_mask_04c.sum())})")
    reg04c = RandomForestRegressor(
        n_estimators=args.n_estimators, max_depth=None,
        random_state=args.seed_train, n_jobs=args.n_jobs,
    )
    reg04c.fit(X_tr_04c[reg_mask_04c], Y_full_04c[reg_mask_04c])
    save_model(reg04c, V3_DIR / "outputs" / "models"
               / "stage06C_fair_04c_baseline_regressor.joblib",
               metadata={"n_train": int(reg_mask_04c.sum()),
                         "feature_keys": FEATURE_KEYS,
                         "targets": REGRESSION_TARGETS,
                         "note": "fair-baseline trained on 04D-train rows only"})

    y_cd_04c = np.array([_safe_float(r.get("CD_final_nm")) for r in rows_04c_tr])
    cd_mask_04c = np.isfinite(y_cd_04c)
    print(f"  training fair 04C-baseline aux CD_fixed regressor "
          f"(n_train={int(cd_mask_04c.sum())})")
    aux04c = RandomForestRegressor(
        n_estimators=args.n_estimators, max_depth=None,
        random_state=args.seed_train, n_jobs=args.n_jobs,
    )
    aux04c.fit(X_tr_04c[cd_mask_04c], y_cd_04c[cd_mask_04c])
    save_model(aux04c, V3_DIR / "outputs" / "models"
               / "stage06C_fair_04c_baseline_aux_cd_fixed.joblib",
               metadata={"n_train": int(cd_mask_04c.sum()),
                         "target_field": "CD_final_nm",
                         "feature_keys": FEATURE_KEYS,
                         "note": "fair-baseline trained on 04D-train rows only"})

    y_pred_04c   = clf04c.predict(X_te)
    Y_pred_04c   = reg04c.predict(X_te)
    cd_pred_04c  = aux04c.predict(X_te)

    y_pred_06c   = clf06c.predict(X_te)
    Y_pred_06c   = reg06c.predict(X_te)
    cd_pred_06c  = aux06c.predict(X_te)

    metrics = {
        "stage": "06C",
        "policy": {"v2_OP_frozen": True, "published_data_loaded": False,
                   "external_calibration": "none"},
        "split": {
            "seed":             int(args.seed_split),
            "n_train_04c_only": len(rows_04c_tr),
            "n_train_06c_pool": len(train_06c),
            "n_test_04c_only":  len(rows_04c_te),
            "note": ("Fair-baseline 04C and refreshed 06C are both trained "
                     "with strict 04D-train holdout (same 1,074-row test "
                     "set, no leakage). The closed Stage 04C joblibs on "
                     "disk are not used in this comparison and are not "
                     "modified."),
        },
        "feature_keys": FEATURE_KEYS,
        "labels":       LABEL_ORDER,
        "regression_targets": REGRESSION_TARGETS,
        "fair_baseline_04c": {
            "n_train":          len(rows_04c_tr),
            "classifier":       _classifier_metrics(y_te, y_pred_04c.tolist()),
            "regressor4":       _regression_metrics(_build_Y_regression(rows_04c_te), Y_pred_04c),
            "regressor4_zones": _zone_breakdown_regression(rows_04c_te, Y_pred_04c),
            "aux_cd_fixed":     _cd_fixed_metrics(rows_04c_te, cd_pred_04c),
        },
        "refreshed_06c": {
            "n_train":          len(train_06c),
            "classifier":       _classifier_metrics(y_te, y_pred_06c.tolist()),
            "regressor4":       _regression_metrics(_build_Y_regression(rows_04c_te), Y_pred_06c),
            "regressor4_zones": _zone_breakdown_regression(rows_04c_te, Y_pred_06c),
            "aux_cd_fixed":     _cd_fixed_metrics(rows_04c_te, cd_pred_06c),
        },
    }

    logs_dir = V3_DIR / "outputs" / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    (logs_dir / "stage06C_model_metrics.json").write_text(
        json.dumps(metrics, indent=2))

    # ----- Console summary -----
    def _bullet_clf(name: str, payload: dict) -> str:
        return (f"  {name}: bal_acc={payload['balanced_accuracy']:.3f}  "
                f"macro_F1={payload['macro_f1']:.3f}  "
                f"false_robust_valid_rate="
                f"{(payload['false_robust_valid_rate'] or float('nan')) * 100:.2f}%")

    def _bullet_reg(name: str, payload: dict, key: str) -> str:
        e = payload[key]
        return f"  {name}/{key}: MAE={e['mae']:.3f}  RMSE={e['rmse']:.3f}  n={e['n']}"

    print(f"\nStage 06C model metrics (test set: 04D 1,074 rows, no leakage)")
    print(_bullet_clf("fair 04C baseline", metrics["fair_baseline_04c"]["classifier"]))
    print(_bullet_clf("refreshed 06C   ", metrics["refreshed_06c"]["classifier"]))
    for k in REGRESSION_TARGETS:
        print(_bullet_reg("fair 04C baseline", metrics["fair_baseline_04c"]["regressor4"], k))
        print(_bullet_reg("refreshed 06C   ", metrics["refreshed_06c"]["regressor4"], k))
    print(f"  fair 04C baseline aux_cd_fixed: "
          f"MAE={metrics['fair_baseline_04c']['aux_cd_fixed']['mae']:.3f}  "
          f"RMSE={metrics['fair_baseline_04c']['aux_cd_fixed']['rmse']:.3f}")
    print(f"  refreshed 06C    aux_cd_fixed: "
          f"MAE={metrics['refreshed_06c']['aux_cd_fixed']['mae']:.3f}  "
          f"RMSE={metrics['refreshed_06c']['aux_cd_fixed']['rmse']:.3f}")
    print(f"\n  metrics → {logs_dir / 'stage06C_model_metrics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
