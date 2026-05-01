"""Stage 06H -- train the refreshed surrogate (classifier + 4-target
regressor + auxiliary CD_fixed regressor) on the merged 06C + 06E
disagreement + 06H FD pool.

Train protocol (mirrors Stage 06C)
    1. The closed Stage 04C dataset has a fixed seed=13 80/20 split that
       defines a reusable held-out test set (1,074 rows). Both the
       fair-04C baseline and 06C used that test set.
    2. Stage 06H is trained on:
         06C training rows  (after removing the 04C test rows)
       + Stage 06E disagreement rows
       + Stage 06H FD rows from Parts 1-3
    3. Evaluation is reported on the SAME 1,074-row 04C test set, so
       06H vs 06C numbers are directly comparable.

Outputs
    outputs/models/stage06H_classifier.joblib
    outputs/models/stage06H_regressor.joblib
    outputs/models/stage06H_aux_cd_fixed_regressor.joblib
    outputs/logs/stage06H_surrogate_refresh_summary.json

Policy
    v2_OP_frozen           = true
    published_data_loaded  = false
    No external calibration. Closed Stage 04C / 04D / 06C training
    datasets and joblibs are not mutated.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
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
    read_labels_csv,
    save_model,
)


V3_DIR = ROOT / "reaction_diffusion_peb_v3_screening"


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


def _build_Y(rows: list[dict],
              keys: list[str] = REGRESSION_TARGETS) -> np.ndarray:
    Y = np.zeros((len(rows), len(keys)), dtype=np.float64)
    for i, r in enumerate(rows):
        for j, k in enumerate(keys):
            Y[i, j] = _safe_float(r.get(k))
    return Y


def _stage04d_split(rows: list[dict], seed: int = 13,
                     test_frac: float = 0.2) -> tuple[list[dict], list[dict]]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(rows))
    cut = int((1.0 - test_frac) * len(rows))
    return [rows[i] for i in idx[:cut]], [rows[i] for i in idx[cut:]]


def _classifier_metrics(y_true: list[str], y_pred: list[str],
                         labels: list[str] = LABEL_ORDER) -> dict:
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
    DEFECT_LABELS = {"under_exposed", "merged",
                     "roughness_degraded", "numerical_invalid"}
    pred_pass = np.array([yp == "robust_valid" for yp in y_pred])
    actual_defect = np.array([yt in DEFECT_LABELS for yt in y_true])
    actual_non_robust = np.array([yt != "robust_valid" for yt in y_true])
    n_pred = int(pred_pass.sum())
    n_miss_defect = int((pred_pass & actual_defect).sum())
    n_miss_non_robust = int((pred_pass & actual_non_robust).sum())
    return {
        "accuracy": float(np.mean(np.array(y_true) == np.array(y_pred))),
        "balanced_accuracy": bal,
        "macro_f1": macro_f1,
        "per_class": per_class,
        "n_predicted_robust_valid": n_pred,
        "n_missed_defect_in_pred_pass": n_miss_defect,
        "false_robust_valid_rate": float(n_miss_defect / n_pred) if n_pred > 0 else None,
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


def _zone_breakdown_regression(rows: list[dict],
                                 y_pred: np.ndarray) -> dict:
    op_mask = np.array([
        DEFECT_GROUP.get(r.get("label", "?"), "defect") in ("normal", "marginal")
        for r in rows
    ])
    fail_mask = ~op_mask
    y_true = _build_Y(rows, REGRESSION_TARGETS)
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


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--stage04c_csv", type=str,
                   default=str(V3_DIR / "outputs" / "labels"
                               / "stage04C_training_dataset.csv"))
    p.add_argument("--stage06h_csv", type=str,
                   default=str(V3_DIR / "outputs" / "labels"
                               / "stage06H_training_dataset.csv"))
    p.add_argument("--n_estimators",           type=int, default=300)
    p.add_argument("--n_estimators_regressor", type=int, default=200,
                    help="The 4-target RandomForestRegressor on the "
                         "8,311-row 06H pool produces a >100 MB joblib at "
                         "n_estimators=300, which exceeds GitHub's push "
                         "limit. 200 trees fit in ~80 MB with negligible "
                         "metric loss. Override only if you do not need "
                         "to push the joblib through the GitHub HTTPS API.")
    p.add_argument("--seed_train", type=int, default=7)
    p.add_argument("--seed_split", type=int, default=13)
    p.add_argument("--n_jobs", type=int, default=-1)
    args = p.parse_args()

    rows_04c = read_labels_csv(args.stage04c_csv)
    rows_06h_all = read_labels_csv(args.stage06h_csv)

    # 04C-only split for held-out test (matches 06C's protocol).
    _, rows_04c_te = _stage04d_split(rows_04c,
                                       seed=args.seed_split, test_frac=0.2)
    print(f"  04C held-out test rows (seed={args.seed_split}): {len(rows_04c_te)}")

    # Build a feature key for the 04C test rows.
    test_keys = set()
    for r in rows_04c_te:
        try:
            test_keys.add(tuple(round(_safe_float(r[k]), 6) for k in FEATURE_KEYS)
                          + (str(r.get("_id", "")),))
        except Exception:
            pass

    # 06H training pool = everything in stage06H_training_dataset.csv
    # MINUS any row whose feature tuple is in the 04C held-out test set.
    train_06h: list[dict] = []
    n_dropped_test = 0
    for r in rows_06h_all:
        if str(r.get("source", "")).startswith("stage06C/stage04C"):
            key = tuple(round(_safe_float(r[k]), 6) for k in FEATURE_KEYS) \
                  + (str(r.get("_id", "")),)
            if key in test_keys:
                n_dropped_test += 1
                continue
        train_06h.append(r)
    print(f"  06H training pool: {len(train_06h)} rows  "
          f"(dropped {n_dropped_test} rows that are in the 04C held-out test)")

    # Source counts for the kept training pool.
    by_source = {}
    for r in train_06h:
        by_source[str(r.get("source", "?"))] = by_source.get(str(r.get("source", "?")), 0) + 1
    print(f"  by source (train pool):")
    for k, v in sorted(by_source.items()):
        print(f"    {k:<42} {v:>6}")

    # ----- Train classifier -----
    X_tr = _build_X(train_06h)
    y_tr = [r.get("label", "?") for r in train_06h]
    X_te = _build_X(rows_04c_te)
    y_te = [r.get("label", "?") for r in rows_04c_te]

    print(f"\n  training 06H classifier (n_estimators={args.n_estimators})")
    clf = RandomForestClassifier(
        n_estimators=args.n_estimators, max_depth=None,
        random_state=args.seed_train, n_jobs=args.n_jobs,
    )
    clf.fit(X_tr, y_tr)
    save_model(clf, V3_DIR / "outputs" / "models" / "stage06H_classifier.joblib",
                metadata={"n_train": len(train_06h),
                          "feature_keys": FEATURE_KEYS,
                          "labels": LABEL_ORDER,
                          "stage": "06H"})

    # ----- Train 4-target regressor -----
    Y_full = _build_Y(train_06h)
    finite_mask = np.all(np.isfinite(Y_full), axis=1)
    print(f"  training 06H 4-target regressor on "
          f"{int(finite_mask.sum())} rows "
          f"(drop {int((~finite_mask).sum())}, n_estimators={args.n_estimators_regressor})")
    reg = RandomForestRegressor(
        n_estimators=args.n_estimators_regressor, max_depth=None,
        random_state=args.seed_train, n_jobs=args.n_jobs,
    )
    reg.fit(X_tr[finite_mask], Y_full[finite_mask])
    save_model(reg, V3_DIR / "outputs" / "models" / "stage06H_regressor.joblib",
                metadata={"n_train": int(finite_mask.sum()),
                          "feature_keys": FEATURE_KEYS,
                          "targets": REGRESSION_TARGETS,
                          "stage": "06H"})

    # ----- Train aux CD_fixed regressor -----
    y_cd = np.array([_safe_float(r.get("CD_final_nm")) for r in train_06h])
    cd_finite = np.isfinite(y_cd)
    print(f"  training 06H aux CD_fixed regressor on {int(cd_finite.sum())} rows")
    aux = RandomForestRegressor(
        n_estimators=args.n_estimators, max_depth=None,
        random_state=args.seed_train, n_jobs=args.n_jobs,
    )
    aux.fit(X_tr[cd_finite], y_cd[cd_finite])
    save_model(aux, V3_DIR / "outputs" / "models"
                / "stage06H_aux_cd_fixed_regressor.joblib",
                metadata={"n_train": int(cd_finite.sum()),
                          "target_field": "CD_final_nm",
                          "feature_keys": FEATURE_KEYS,
                          "stage": "06H"})

    # ----- Evaluate on held-out 04C test rows -----
    y_pred = clf.predict(X_te)
    Y_pred = reg.predict(X_te)
    cd_pred = aux.predict(X_te)

    metrics = {
        "stage": "06H",
        "policy": {"v2_OP_frozen": True, "published_data_loaded": False,
                   "external_calibration": "none"},
        "split": {
            "seed":            int(args.seed_split),
            "n_test_04c_only": len(rows_04c_te),
            "note": ("06H trained on 06C pool ∪ 06E disagreement ∪ 06H FD "
                      "rows, with the 04C held-out test set strictly "
                      "removed. Closed Stage 04C / 04D joblibs are not "
                      "modified."),
        },
        "feature_keys":       FEATURE_KEYS,
        "labels":             LABEL_ORDER,
        "regression_targets": REGRESSION_TARGETS,
        "n_train":            len(train_06h),
        "by_source_train":    by_source,
        "classifier":         _classifier_metrics(y_te, y_pred.tolist()),
        "regressor4":         _regression_metrics(_build_Y(rows_04c_te), Y_pred),
        "regressor4_zones":   _zone_breakdown_regression(rows_04c_te, Y_pred),
        "aux_cd_fixed":       _cd_fixed_metrics(rows_04c_te, cd_pred),
        "feature_importance_classifier": [
            float(v) for v in clf.feature_importances_
        ],
        "feature_importance_regressor4_avg": [
            float(np.mean(reg.feature_importances_)),
        ],
    }

    logs_dir = V3_DIR / "outputs" / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    (logs_dir / "stage06H_surrogate_refresh_summary.json").write_text(
        json.dumps(metrics, indent=2))

    print(f"\nStage 06H surrogate metrics (test: 04C 1,074 holdout rows)")
    c = metrics["classifier"]
    print(f"  classifier: bal_acc={c['balanced_accuracy']:.3f}  "
          f"macro_F1={c['macro_f1']:.3f}  "
          f"false_robust_valid_rate="
          f"{(c['false_robust_valid_rate'] or 0.0)*100:.2f}%")
    for k, e in metrics["regressor4"].items():
        print(f"  regressor4/{k:<18} MAE={e['mae']:.3f}  RMSE={e['rmse']:.3f}  n={e['n']}")
    a = metrics["aux_cd_fixed"]
    print(f"  aux_cd_fixed: MAE={a['mae']:.3f}  RMSE={a['rmse']:.3f}  n={a['n']}")
    print(f"\n  metrics -> {logs_dir / 'stage06H_surrogate_refresh_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
