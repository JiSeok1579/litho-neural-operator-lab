"""Stage 06P -- train the AL-refresh surrogate stack:
  classifier + 4-target regressor + aux CD_fixed regressor +
  direct strict_score regressor.

Train protocol (mirrors Stage 06L)
    1. Reproduce the closed Stage 04C 80/20 split (seed=13). Hold
       out the same 1,074 04C test rows so 06L vs 06P numbers stay
       directly comparable on a fixed ground truth.
    2. Train on the 06P pool (14,194 rows) MINUS the 1,074 04C-test
       rows = ~13,120 train rows.
    3. Save 4 joblibs and one metrics JSON.
    4. Per-mode + per-family + per-tag evaluation on the test set.

Joblib size guard: the 4-target regressor is trained at
n_estimators=200 to stay under GitHub's 100 MB push limit. The strict
head matches at 200 trees.

Outputs
    outputs/models/stage06P_classifier.joblib
    outputs/models/stage06P_regressor.joblib                 (n=200)
    outputs/models/stage06P_aux_cd_fixed_regressor.joblib
    outputs/models/stage06P_strict_score_regressor.joblib    (n=200)
    outputs/logs/stage06P_model_metrics.json

Policy
    v2_OP_frozen           = true
    published_data_loaded  = false
    No external calibration. Closed Stage 04C / 04D / 06C / 06L joblibs
    are not modified.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import yaml
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    precision_recall_fscore_support,
)

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reaction_diffusion_peb_v3_screening.src.fd_yield_score import spearman
from reaction_diffusion_peb_v3_screening.src.labeler import (
    DEFECT_GROUP, LABEL_ORDER,
)
from reaction_diffusion_peb_v3_screening.src.metrics_io import (
    FEATURE_KEYS, REGRESSION_TARGETS, read_labels_csv, save_model,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_stage06l_dataset import per_row_strict_score  # noqa: E402


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


def _stage04d_split(rows, seed=13, test_frac=0.2):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(rows))
    cut = int((1.0 - test_frac) * len(rows))
    return [rows[i] for i in idx[:cut]], [rows[i] for i in idx[cut:]]


def _classifier_metrics(y_true, y_pred, labels=LABEL_ORDER):
    bal = float(balanced_accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, labels=labels,
                                average="macro", zero_division=0))
    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0)
    per_class = {labels[i]: {"precision": float(p[i]), "recall": float(r[i]),
                                  "f1": float(f[i]), "support": int(s[i])}
                  for i in range(len(labels))}
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
        "false_robust_valid_rate":
            float(n_miss_defect / n_pred) if n_pred > 0 else None,
        "n_missed_non_robust_in_pred_pass": n_miss_non_robust,
        "false_robust_valid_includes_margin_rate":
            float(n_miss_non_robust / n_pred) if n_pred > 0 else None,
    }


def _regression_metrics(y_true, y_pred, target_names=REGRESSION_TARGETS):
    out = {}
    for j, name in enumerate(target_names):
        finite = np.isfinite(y_true[:, j]) & np.isfinite(y_pred[:, j])
        if finite.sum() == 0:
            out[name] = {"mae": None, "rmse": None, "n": 0}
            continue
        d = y_true[finite, j] - y_pred[finite, j]
        out[name] = {"mae":  float(np.mean(np.abs(d))),
                     "rmse": float(np.sqrt(np.mean(d ** 2))),
                     "n":    int(finite.sum())}
    return out


def _scalar_regression_metrics(y_true: np.ndarray,
                                  y_pred: np.ndarray) -> dict:
    finite = np.isfinite(y_true) & np.isfinite(y_pred)
    if finite.sum() == 0:
        return {"mae": None, "rmse": None, "n": 0,
                "spearman_rho": None}
    d = y_true[finite] - y_pred[finite]
    rho = spearman(y_true[finite], y_pred[finite])
    return {"mae":  float(np.mean(np.abs(d))),
            "rmse": float(np.sqrt(np.mean(d ** 2))),
            "n":    int(finite.sum()),
            "spearman_rho": rho}


def _zone_breakdown_regression(rows, y_pred):
    op_mask = np.array([
        DEFECT_GROUP.get(r.get("label", "?"), "defect") in ("normal", "marginal")
        for r in rows
    ])
    fail_mask = ~op_mask
    y_true = _build_Y(rows, REGRESSION_TARGETS)
    def _zone(mask):
        if mask.sum() == 0:
            return {name: None for name in REGRESSION_TARGETS}
        return _regression_metrics(y_true[mask], y_pred[mask])
    return {
        "operational_zone": _zone(op_mask),
        "failure_zone":     _zone(fail_mask),
        "n_operational":    int(op_mask.sum()),
        "n_failure":        int(fail_mask.sum()),
    }


def _cd_fixed_metrics(rows, y_pred):
    y_true = np.array([_safe_float(r.get("CD_final_nm")) for r in rows])
    finite = np.isfinite(y_true) & np.isfinite(y_pred)
    if finite.sum() == 0:
        return {"mae": None, "rmse": None, "n": 0}
    d = y_true[finite] - y_pred[finite]
    return {"mae": float(np.mean(np.abs(d))),
            "rmse": float(np.sqrt(np.mean(d ** 2))),
            "n":   int(finite.sum())}


def _slice_eval(rows: list[dict], mask: np.ndarray, *,
                  clf, reg, aux, strict_reg,
                  strict_yaml: dict) -> dict:
    """Compute classifier + regressor + strict-head metrics over a
    boolean slice of `rows`."""
    sub = [rows[i] for i in range(len(rows)) if bool(mask[i])]
    if not sub:
        return {"n": 0}
    Xs = _build_X(sub)
    ys = [r.get("label", "?") for r in sub]
    yp = clf.predict(Xs).tolist()
    Yp = reg.predict(Xs)
    cd_pred = aux.predict(Xs)
    strict_true = np.array([per_row_strict_score(r, strict_yaml) for r in sub])
    strict_pred = strict_reg.predict(Xs)
    return {
        "n": int(len(sub)),
        "classifier": _classifier_metrics(ys, yp),
        "regressor4": _regression_metrics(_build_Y(sub), Yp),
        "aux_cd_fixed": _cd_fixed_metrics(sub, cd_pred),
        "strict_score_regressor":
            _scalar_regression_metrics(strict_true, strict_pred),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--stage04c_csv", type=str,
                   default=str(V3_DIR / "outputs" / "labels"
                               / "stage04C_training_dataset.csv"))
    p.add_argument("--stage06p_csv", type=str,
                   default=str(V3_DIR / "outputs" / "labels"
                               / "stage06P_training_dataset.csv"))
    p.add_argument("--strict_cfg_yaml", type=str,
                   default=str(V3_DIR / "outputs" / "yield_optimization"
                               / "stage06G_strict_score_config.yaml"))
    p.add_argument("--n_estimators",          type=int, default=300,
                   help="classifier; small file even at 300.")
    p.add_argument("--n_estimators_regressor", type=int, default=140,
                   help="4-target regressor; 140 trees keeps the joblib "
                        "comfortably under 100 MB at ~13k training rows.")
    p.add_argument("--n_estimators_aux",       type=int, default=200,
                   help="aux CD_fixed regressor; 200 trees keeps the "
                        "joblib under 100 MB at ~13k training rows.")
    p.add_argument("--n_estimators_strict",    type=int, default=200,
                   help="strict_score head; pinned consistent with regressor.")
    p.add_argument("--seed_train", type=int, default=7)
    p.add_argument("--seed_split", type=int, default=13)
    p.add_argument("--n_jobs", type=int, default=-1)
    args = p.parse_args()

    strict_yaml = yaml.safe_load(Path(args.strict_cfg_yaml).read_text())

    rows_04c = read_labels_csv(args.stage04c_csv)
    rows_06p_all = read_labels_csv(args.stage06p_csv)

    _, rows_04c_te = _stage04d_split(rows_04c,
                                       seed=args.seed_split, test_frac=0.2)
    print(f"  04C held-out test rows (seed={args.seed_split}): {len(rows_04c_te)}")
    test_keys = set()
    for r in rows_04c_te:
        try:
            test_keys.add(tuple(round(_safe_float(r[k]), 6) for k in FEATURE_KEYS)
                            + (str(r.get("_id", "")),))
        except Exception:
            pass

    train_06p: list[dict] = []
    n_drop = 0
    for r in rows_06p_all:
        # 06L base rows that originated from closed 04C carry source
        # like "stage06H/stage06C/stage04C". Drop the held-out 04C test
        # rows from the training pool by (feature, _id) match.
        src = str(r.get("source", ""))
        if src.startswith("stage06H/stage06C/stage04C"):
            key = tuple(round(_safe_float(r[k]), 6) for k in FEATURE_KEYS) \
                  + (str(r.get("_id", "")),)
            if key in test_keys:
                n_drop += 1
                continue
        train_06p.append(r)
    print(f"  06P training pool: {len(train_06p)} rows  "
          f"(dropped {n_drop} 04C-test rows)")

    by_source = Counter(r.get("source", "?") for r in train_06p)
    by_mode   = Counter(r.get("mode", "?")   for r in train_06p)
    by_family = Counter(r.get("recipe_family", "?") for r in train_06p)
    print(f"  by source (training pool, top 10):")
    for k, v in by_source.most_common(10):
        print(f"    {k:<54} {v:>6}")
    print(f"  by mode (training pool):    {dict(by_mode)}")
    print(f"  by family (training pool):  {dict(by_family)}")

    # ----- Train -----
    X_tr = _build_X(train_06p); y_tr = [r.get("label", "?") for r in train_06p]
    X_te = _build_X(rows_04c_te); y_te = [r.get("label", "?") for r in rows_04c_te]

    print(f"\n  training 06P classifier (n_estimators={args.n_estimators})")
    clf = RandomForestClassifier(
        n_estimators=args.n_estimators, max_depth=None,
        random_state=args.seed_train, n_jobs=args.n_jobs,
    )
    clf.fit(X_tr, y_tr)
    save_model(clf, V3_DIR / "outputs" / "models" / "stage06P_classifier.joblib",
                metadata={"n_train": len(train_06p), "feature_keys": FEATURE_KEYS,
                          "labels": LABEL_ORDER, "stage": "06P"})

    Y_full = _build_Y(train_06p)
    finite_mask = np.all(np.isfinite(Y_full), axis=1)
    print(f"  training 06P 4-target regressor on {int(finite_mask.sum())} rows  "
          f"(n_estimators={args.n_estimators_regressor})")
    reg = RandomForestRegressor(
        n_estimators=args.n_estimators_regressor, max_depth=None,
        random_state=args.seed_train, n_jobs=args.n_jobs,
    )
    reg.fit(X_tr[finite_mask], Y_full[finite_mask])
    save_model(reg, V3_DIR / "outputs" / "models" / "stage06P_regressor.joblib",
                metadata={"n_train": int(finite_mask.sum()),
                          "feature_keys": FEATURE_KEYS,
                          "targets": REGRESSION_TARGETS,
                          "n_estimators": args.n_estimators_regressor,
                          "stage": "06P"})

    y_cd = np.array([_safe_float(r.get("CD_final_nm")) for r in train_06p])
    cd_finite = np.isfinite(y_cd)
    print(f"  training 06P aux CD_fixed regressor on {int(cd_finite.sum())} rows  "
          f"(n_estimators={args.n_estimators_aux})")
    aux = RandomForestRegressor(
        n_estimators=args.n_estimators_aux, max_depth=None,
        random_state=args.seed_train, n_jobs=args.n_jobs,
    )
    aux.fit(X_tr[cd_finite], y_cd[cd_finite])
    save_model(aux, V3_DIR / "outputs" / "models"
                / "stage06P_aux_cd_fixed_regressor.joblib",
                metadata={"n_train": int(cd_finite.sum()),
                          "target_field": "CD_final_nm",
                          "feature_keys": FEATURE_KEYS,
                          "stage": "06P"})

    y_strict = np.array([_safe_float(r.get("strict_score_per_row"))
                            for r in train_06p])
    strict_finite = np.isfinite(y_strict)
    print(f"  training 06P STRICT_SCORE regressor on {int(strict_finite.sum())} rows  "
          f"(n_estimators={args.n_estimators_strict})")
    strict_reg = RandomForestRegressor(
        n_estimators=args.n_estimators_strict, max_depth=None,
        random_state=args.seed_train, n_jobs=args.n_jobs,
    )
    strict_reg.fit(X_tr[strict_finite], y_strict[strict_finite])
    save_model(strict_reg,
                V3_DIR / "outputs" / "models"
                / "stage06P_strict_score_regressor.joblib",
                metadata={"n_train": int(strict_finite.sum()),
                          "target_field": "strict_score_per_row",
                          "feature_keys": FEATURE_KEYS,
                          "n_estimators": args.n_estimators_strict,
                          "stage": "06P"})

    # ----- Evaluate on the 04C 1,074 holdout -----
    y_pred = clf.predict(X_te)
    Y_pred = reg.predict(X_te)
    cd_pred = aux.predict(X_te)
    y_strict_te = np.array([per_row_strict_score(r, strict_yaml)
                             for r in rows_04c_te])
    strict_pred_te = strict_reg.predict(X_te)

    # ----- Per-slice evaluation on the 06P pool itself -----
    # Use the full 06P pool (not training-pool only) for these slice
    # diagnostics: J_1453 / G_4867 / time_boundary / strict-top etc.
    # are only present in the 06P additions.
    rows_pool = list(rows_06p_all)
    Xp = _build_X(rows_pool)
    yp_pool = clf.predict(Xp)
    Yp_pool = reg.predict(Xp)
    cd_pred_pool = aux.predict(Xp)
    strict_pred_pool = strict_reg.predict(Xp)
    strict_true_pool = np.array(
        [_safe_float(r.get("strict_score_per_row")) for r in rows_pool]
    )

    def _slice(mask: np.ndarray) -> dict:
        idx = np.where(mask)[0]
        if idx.size == 0:
            return {"n": 0}
        sub = [rows_pool[i] for i in idx]
        ys = [r.get("label", "?") for r in sub]
        return {
            "n": int(idx.size),
            "classifier": _classifier_metrics(ys, [yp_pool[i] for i in idx]),
            "regressor4": _regression_metrics(_build_Y(sub), Yp_pool[idx]),
            "aux_cd_fixed": _cd_fixed_metrics(sub, cd_pred_pool[idx]),
            "strict_score_regressor":
                _scalar_regression_metrics(strict_true_pool[idx],
                                                strict_pred_pool[idx]),
        }

    mode_a_mask = np.array([r.get("mode") == "mode_a" for r in rows_pool])
    mode_b_mask = np.array([r.get("mode") == "mode_b" for r in rows_pool])
    fam_g_mask  = np.array([r.get("recipe_family") == "G4867_family" for r in rows_pool])
    fam_j_mask  = np.array([r.get("recipe_family") == "J1453_family" for r in rows_pool])
    time_mask   = np.array(["time_boundary" in str(r.get("boundary_tags", ""))
                              for r in rows_pool])
    strict_top_mask = np.array(
        [_safe_float(r.get("strict_score_per_row")) >= 0.7 for r in rows_pool]
    )
    false_promise_mask = np.array(
        [r.get("recipe_family") == "false_promise" for r in rows_pool]
    )
    disagreement_mask = np.array(
        ["disagreement_candidate" in str(r.get("boundary_tags", ""))
         for r in rows_pool]
    )

    metrics = {
        "stage": "06P",
        "policy": {"v2_OP_frozen": True, "published_data_loaded": False,
                   "external_calibration": "none"},
        "split": {
            "seed":            int(args.seed_split),
            "n_test_04c_only": len(rows_04c_te),
            "note": ("06P trained on 06L pool union 06J-B FD additions "
                      "(top-100 nominal, top-10 MC, representative MC) "
                      "and 06M-B J_1453 time-window FD additions "
                      "(deterministic offsets and Gaussian time smearing). "
                      "The 04C held-out test set is strictly removed. "
                      "Closed Stage 04C / 04D / 06C / 06L joblibs are not "
                      "modified."),
        },
        "feature_keys":       FEATURE_KEYS,
        "labels":             LABEL_ORDER,
        "regression_targets": REGRESSION_TARGETS,
        "n_train":            len(train_06p),
        "by_source_train":    dict(by_source),
        "by_mode_train":      dict(by_mode),
        "by_family_train":    dict(by_family),
        "classifier":         _classifier_metrics(y_te, y_pred.tolist()),
        "regressor4":         _regression_metrics(_build_Y(rows_04c_te), Y_pred),
        "regressor4_zones":   _zone_breakdown_regression(rows_04c_te, Y_pred),
        "aux_cd_fixed":       _cd_fixed_metrics(rows_04c_te, cd_pred),
        "strict_score_regressor": _scalar_regression_metrics(
            y_strict_te, strict_pred_te,
        ),
        "strict_score_regressor_n_train": int(strict_finite.sum()),
        "feature_importance_classifier":
            [float(v) for v in clf.feature_importances_],
        "feature_importance_strict_score":
            [float(v) for v in strict_reg.feature_importances_],
        "slice_metrics_06p_pool": {
            "mode_a":              _slice(mode_a_mask),
            "mode_b":              _slice(mode_b_mask),
            "G4867_family":        _slice(fam_g_mask),
            "J1453_family":        _slice(fam_j_mask),
            "time_boundary":       _slice(time_mask),
            "strict_top_tier_ge_0p7": _slice(strict_top_mask),
            "false_promise":       _slice(false_promise_mask),
            "disagreement_candidate": _slice(disagreement_mask),
        },
    }

    logs_dir = V3_DIR / "outputs" / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    (logs_dir / "stage06P_model_metrics.json").write_text(
        json.dumps(metrics, indent=2))

    # ----- Console summary -----
    print(f"\nStage 06P model metrics (test: 04C 1,074 holdout)")
    c = metrics["classifier"]
    print(f"  classifier: bal_acc={c['balanced_accuracy']:.3f}  "
          f"macro_F1={c['macro_f1']:.3f}  "
          f"false_robust_valid_rate="
          f"{(c['false_robust_valid_rate'] or 0.0)*100:.2f}%")
    for k, e in metrics["regressor4"].items():
        print(f"  regressor4/{k:<18} MAE={e['mae']:.3f}  RMSE={e['rmse']:.3f}  n={e['n']}")
    a = metrics["aux_cd_fixed"]
    print(f"  aux_cd_fixed: MAE={a['mae']:.3f}  RMSE={a['rmse']:.3f}  n={a['n']}")
    s = metrics["strict_score_regressor"]
    print(f"  strict_score_regressor: MAE={s['mae']:.3f}  RMSE={s['rmse']:.3f}  "
          f"Spearman_rho={s['spearman_rho']:.3f}  n={s['n']}")
    print(f"  metrics -> {logs_dir / 'stage06P_model_metrics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
