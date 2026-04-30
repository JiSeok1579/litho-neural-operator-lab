"""Stage 04B — active-learning expansion + class-balanced failure sampling.

Two iterations. Each iteration:
    1. Sample failure-seeking candidates per target label
       (target_under_exposed, target_merged, target_margin_risk,
        target_roughness_degraded). Per-target batch defaults to 200.
    2. Sample a fresh Sobol pool (n=10 000) for AL acquisition.
    3. Apply the budget_prefilter and retain the top 3 000.
    4. Score uncertain candidates with the four-signal acquisition
       (classifier max-prob, regressor per-tree std, predicted margin
       inside band, predicted near-merged / near-under_exposed).
    5. Run FD on the selected (failure-seeking + AL).
    6. Label, append to the dataset, retrain classifier + regressor.
    7. Save iter-tagged models + per-iteration counts.

Final evaluation:
    - confusion matrix, per-class P/R/F1, macro-F1, balanced accuracy
    - regressor MAE / R² globally and by class
    - calibration reliability diagram
    - acceptance check vs Stage 04B targets
"""
from __future__ import annotations

import argparse
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

from reaction_diffusion_peb_v3_screening.src.active_learning import acquisition_indices_v2
from reaction_diffusion_peb_v3_screening.src.budget_prefilter import score_all, select_top_n
from reaction_diffusion_peb_v3_screening.src.candidate_sampler import (
    CandidateSpace, sample_candidates, sample_margin_perturbation, sample_with_bias,
)
from reaction_diffusion_peb_v3_screening.src.evaluation import (
    classifier_report, plot_confusion, plot_per_class_mae, plot_reliability_diagram,
    regressor_global_metrics, regressor_mae_by_class,
)
from reaction_diffusion_peb_v3_screening.src.fd_batch_runner import run_batch
from reaction_diffusion_peb_v3_screening.src.labeler import LABEL_ORDER, LabelThresholds
from reaction_diffusion_peb_v3_screening.src.metrics_io import (
    FEATURE_KEYS, REGRESSION_TARGETS, build_feature_matrix,
    build_regression_target_matrix, read_labels_csv, save_model,
)
from reaction_diffusion_peb_v3_screening.src.surrogate_classifier import (
    train_classifier,
)
from reaction_diffusion_peb_v3_screening.src.surrogate_regressor import (
    train_regressor,
)


V3_DIR = ROOT / "reaction_diffusion_peb_v3_screening"


def _coerce_float_keys(rows):
    numeric = set(FEATURE_KEYS) | set(REGRESSION_TARGETS) | {
        "CD_final_nm", "CD_pitch_frac",
        "LER_after_PEB_P_nm", "LER_design_initial_nm",
        "P_space_center_mean", "P_line_center_mean",
        "contrast", "psd_locked_mid",
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


def _load_seed_dataset(args) -> list[dict]:
    rows: list[dict] = []
    for path in [args.seed_csv_02, args.seed_csv_04]:
        if Path(path).exists():
            rows.extend(_coerce_float_keys(read_labels_csv(path)))
    print(f"  seed dataset: {len(rows)} rows from "
          f"{[Path(p).name for p in [args.seed_csv_02, args.seed_csv_04] if Path(p).exists()]}")
    return rows


def _train_pair(rows, cfg, seed, label_order=LABEL_ORDER):
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


def _evaluate(rows, clf, reg, label_order=LABEL_ORDER):
    """Evaluate on a single 80/20 split for stable comparison across iters."""
    rng = np.random.default_rng(13)
    idx = rng.permutation(len(rows))
    cut = int(0.8 * len(rows))
    test_rows = [rows[i] for i in idx[cut:]]
    X_te = build_feature_matrix(test_rows)
    y_te = [r["label"] for r in test_rows]

    y_pred = clf.predict(X_te).tolist()
    cls_rep = classifier_report(y_te, y_pred, label_order)

    if reg is not None:
        valid_test = [r for r in test_rows if r["label"] != "numerical_invalid"]
        Xv = build_feature_matrix(valid_test)
        Yv = build_regression_target_matrix(valid_test, REGRESSION_TARGETS)
        # Drop rows where any regression target is NaN (e.g., merged rows where
        # CD-lock could not converge → CD_locked / LER_locked are NaN).
        finite_mask = np.isfinite(Yv).all(axis=1)
        if finite_mask.any():
            Xv_f = Xv[finite_mask]
            Yv_f = Yv[finite_mask]
            valid_test_f = [r for r, m in zip(valid_test, finite_mask) if m]
            Yv_pred = reg.predict(Xv_f)
            reg_global = regressor_global_metrics(Yv_f, Yv_pred, REGRESSION_TARGETS)
            reg_by_class = regressor_mae_by_class(
                Yv_f, Yv_pred,
                class_labels=[r["label"] for r in valid_test_f],
                targets=REGRESSION_TARGETS,
            )
        else:
            reg_global = None
            reg_by_class = None
    else:
        reg_global = None
        reg_by_class = None

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


def _iteration(it_idx, rows, clf, reg, cfg, biases, thresholds, args):
    """Run one Stage-04B iteration. Returns updated rows + retrained models."""
    print(f"\n========== Stage 04B iteration {it_idx} ==========")

    space = CandidateSpace.from_yaml(cfg["candidate_space_yaml"])

    # 1. failure-seeking batches
    fs_targets = ["target_under_exposed", "target_merged",
                   "target_margin_risk", "target_roughness_degraded"]
    fs_n = int(args.failure_seek_per_label)
    fs_candidates = []
    for t in fs_targets:
        spec = biases.get(t, {})
        if "perturbation" in spec:
            cs = sample_margin_perturbation(
                space, spec, seed_rows=rows, n=fs_n, seed=cfg["run"]["seed"] + it_idx,
            )
        else:
            cs = sample_with_bias(
                space, spec, n=fs_n,
                method="latin_hypercube",
                seed=cfg["run"]["seed"] + it_idx,
            )
        for j, c in enumerate(cs):
            c["_id"] = f"iter{it_idx}_{t}_{j}"
            c["_failure_target"] = t
        fs_candidates.extend(cs)
        print(f"  failure-seek [{t}]: {len(cs)} candidates")

    # 2. AL pool
    pool = sample_candidates(
        space, n=int(cfg["active_learning"]["pool_size"]),
        method=cfg["sampling"]["method"],
        seed=cfg["run"]["seed"] + 100 + it_idx,
    )
    pool_scored = score_all(pool)
    pool_retained = select_top_n(pool_scored, int(cfg["active_learning"]["pool_prefilter_retain"]))

    X_pool = build_feature_matrix(pool_retained)
    selected_idx, diag = acquisition_indices_v2(
        clf, reg, X_pool,
        classifier_unc_threshold=float(cfg["active_learning"]["acquisition"]["classifier_uncertainty_threshold"]),
        regressor_std_min_nm=float(cfg["active_learning"]["acquisition"]["regressor_std_min_nm"]),
        margin_band=tuple(args.margin_band),
        top_k=int(cfg["active_learning"]["budget"]),
    )
    print(f"  AL acquisition diag: {json.dumps({k: v for k, v in diag.items() if not isinstance(v, dict)})}")
    al_candidates = [pool_retained[i] for i in selected_idx]
    for j, c in enumerate(al_candidates):
        c["_id"] = f"iter{it_idx}_AL_{j}"
        c["_failure_target"] = "AL"

    # 3. FD on the union of failure-seek + AL.
    all_candidates = fs_candidates + al_candidates
    out_csv = V3_DIR / "outputs" / "labels" / f"04b_iter{it_idx}.csv"
    new_rows = run_batch(all_candidates, thresholds=thresholds, out_csv=out_csv)

    counts = Counter(r["label"] for r in new_rows)
    print(f"  iter {it_idx} new-rows label histogram:")
    for k in LABEL_ORDER:
        print(f"    {k:<22} {counts.get(k, 0)}")

    # Append to the rolling dataset.
    combined = rows + new_rows

    # 4. Retrain on the combined dataset.
    clf2, reg2 = _train_pair(combined, cfg, seed=int(cfg["run"]["seed"]))

    # Save iter-tagged models.
    models_dir = V3_DIR / "outputs" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    save_model(clf2, models_dir / f"04b_iter{it_idx}_classifier.joblib",
                metadata={"feature_keys": FEATURE_KEYS, "label_order": LABEL_ORDER})
    if reg2 is not None:
        save_model(reg2, models_dir / f"04b_iter{it_idx}_regressor.joblib",
                    metadata={"feature_keys": FEATURE_KEYS,
                              "regression_targets": REGRESSION_TARGETS})

    return combined, clf2, reg2, {
        "iter": it_idx,
        "new_rows": len(new_rows),
        "new_label_counts": dict(counts),
        "cumulative_total": len(combined),
        "acquisition_diag": diag,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--biases_yaml", type=str,
                   default=str(V3_DIR / "configs" / "failure_seeking.yaml"))
    p.add_argument("--seed_csv_02", type=str,
                   default=str(V3_DIR / "outputs" / "labels" / "02_monte_carlo_dataset.csv"))
    p.add_argument("--seed_csv_04", type=str,
                   default=str(V3_DIR / "outputs" / "labels" / "04_active_learning_loop.csv"))
    p.add_argument("--n_iterations", type=int, default=2)
    p.add_argument("--failure_seek_per_label", type=int, default=200)
    p.add_argument("--margin_band", type=float, nargs=2, default=[-0.02, 0.07])
    p.add_argument("--tag", type=str, default="04b_balanced_active_learning")
    args = p.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    biases = yaml.safe_load(Path(args.biases_yaml).read_text())
    thresholds = LabelThresholds.from_yaml(cfg["label_schema_yaml"])

    rows = _load_seed_dataset(args)
    if not rows:
        print("no seed dataset; run Stage 02 first.")
        return 1

    # Train a fresh seed pair on the seed dataset for a stable evaluation.
    clf, reg = _train_pair(rows, cfg, seed=int(cfg["run"]["seed"]))

    eval_history = []
    eval_history.append({"label": "before_04b",
                          "report": _evaluate(rows, clf, reg)})

    iter_summaries = []
    for it in range(1, int(args.n_iterations) + 1):
        rows, clf, reg, summary = _iteration(it, rows, clf, reg, cfg, biases, thresholds, args)
        iter_summaries.append(summary)
        eval_history.append({"label": f"after_iter{it}",
                              "report": _evaluate(rows, clf, reg)})

    # ---------- final evaluation + figures ----------
    fig_dir = V3_DIR / "outputs" / "figures" / args.tag
    fig_dir.mkdir(parents=True, exist_ok=True)
    final_eval = eval_history[-1]["report"]
    cls_rep = final_eval["classifier"]
    plot_confusion(cls_rep.confusion, LABEL_ORDER, fig_dir / "confusion_after_04b.png")
    proba_max, correct = final_eval["calibration"]
    plot_reliability_diagram(proba_max, correct, fig_dir / "calibration_after_04b.png")
    if final_eval["regressor_by_class"]:
        plot_per_class_mae(final_eval["regressor_by_class"], REGRESSION_TARGETS,
                            fig_dir / "regressor_mae_by_class.png")

    # Class counts per iteration table.
    print("\n========== Per-iteration class counts ==========")
    full_counts = Counter([r["label"] for r in rows])
    print(f"  cumulative dataset rows: {len(rows)}")
    print(f"  cumulative class counts:")
    for k in LABEL_ORDER:
        print(f"    {k:<22} {full_counts.get(k, 0)}")

    print("\n========== Final classifier metrics ==========")
    print(f"  accuracy           : {cls_rep.accuracy:.4f}")
    print(f"  balanced accuracy  : {cls_rep.balanced_accuracy:.4f}")
    print(f"  macro F1           : {cls_rep.macro_f1:.4f}")
    for label, metrics in cls_rep.per_class.items():
        print(f"  {label:<22} P={metrics['precision']:.3f} "
              f"R={metrics['recall']:.3f} F1={metrics['f1']:.3f} "
              f"support={metrics['support']}")

    if final_eval["regressor_global"]:
        print("\n========== Final regressor metrics (global) ==========")
        for t, m in final_eval["regressor_global"].items():
            print(f"  {t:<22} MAE={m['MAE']:.4f}  R²={m['R2']:+.4f}")
    if final_eval["regressor_by_class"]:
        print("\n========== Regressor MAE by class ==========")
        cls_keys = list(final_eval["regressor_by_class"].keys())
        header = "  " + " ".join(f"{k:>22}" for k in ["target"] + cls_keys)
        print(header)
        for t in REGRESSION_TARGETS:
            row = [t] + [f"{final_eval['regressor_by_class'][c].get(t, float('nan')):.4f}"
                          for c in cls_keys]
            print("  " + " ".join(f"{r:>22}" for r in row))

    # Acceptance check.
    rg = final_eval["regressor_global"] or {}
    cd_mae = rg.get("CD_locked_nm", {}).get("MAE", float("nan"))
    ler_mae = rg.get("LER_CD_locked_nm", {}).get("MAE", float("nan"))
    minority_growth = (full_counts.get("under_exposed", 0)
                        + full_counts.get("merged", 0)
                        + full_counts.get("margin_risk", 0)
                        + full_counts.get("roughness_degraded", 0))
    seed_minority = sum(1 for r in _load_seed_dataset(args)
                         if r["label"] in {"under_exposed", "merged",
                                              "margin_risk", "roughness_degraded"})
    pre_macro_f1 = eval_history[0]["report"]["classifier"].macro_f1

    acceptance = {
        "minority_growth": {
            "before": int(seed_minority),
            "after":  int(minority_growth),
            "ok":     bool(minority_growth > seed_minority),
        },
        "macro_f1_improved": {
            "before": float(pre_macro_f1),
            "after":  float(cls_rep.macro_f1),
            "ok":     bool(cls_rep.macro_f1 > pre_macro_f1),
        },
        "regressor_stable": {
            "CD_locked_MAE":      float(cd_mae),
            "LER_CD_locked_MAE":  float(ler_mae),
            "ok": bool(cd_mae <= 0.15 and ler_mae <= 0.03),
        },
        "policy_unchanged": {
            "v2_OP_frozen":          True,
            "published_data_loaded": False,
        },
    }

    summary = {
        "n_iterations":   int(args.n_iterations),
        "iter_summaries": iter_summaries,
        "cumulative_class_counts": dict(full_counts),
        "final_classifier": {
            "accuracy":           cls_rep.accuracy,
            "balanced_accuracy":  cls_rep.balanced_accuracy,
            "macro_f1":           cls_rep.macro_f1,
            "per_class":          cls_rep.per_class,
        },
        "final_regressor_global":   final_eval["regressor_global"],
        "final_regressor_by_class": final_eval["regressor_by_class"],
        "acceptance": acceptance,
    }
    log_path = V3_DIR / "outputs" / "logs" / f"{args.tag}_summary.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps(summary, indent=2, default=lambda x: float(x)))

    # Acceptance verdict.
    print("\n========== Acceptance ==========")
    for k, v in acceptance.items():
        ok = v.get("ok") if isinstance(v, dict) else None
        print(f"  {k:<32} {'PASS' if ok else ('—' if ok is None else 'FAIL')}  {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
