"""Train classifier + regressor on the labelled dataset from Stage 02
and report confusion matrix + per-target MAE."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reaction_diffusion_peb_v3_screening.src.metrics_io import (
    FEATURE_KEYS, REGRESSION_TARGETS, build_feature_matrix,
    build_regression_target_matrix, read_labels_csv, save_model,
)
from reaction_diffusion_peb_v3_screening.src.surrogate_classifier import (
    evaluate_classifier, train_classifier,
)
from reaction_diffusion_peb_v3_screening.src.surrogate_regressor import (
    evaluate_regressor, train_regressor,
)
from reaction_diffusion_peb_v3_screening.src.labeler import LABEL_ORDER


V3_DIR = ROOT / "reaction_diffusion_peb_v3_screening"


def _coerce_float_keys(rows: list[dict]) -> list[dict]:
    """CSV reader gives strings — cast known numeric keys."""
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


def plot_confusion(cm: np.ndarray, labels: list[str], out_path: Path):
    fig, ax = plt.subplots(figsize=(6.5, 5.0))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
    ax.set_xlabel("predicted")
    ax.set_ylabel("actual")
    ax.set_title("Status confusion matrix")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(int(cm[i, j])), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                    fontsize=9)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_regression_scatter(y_true: np.ndarray, y_pred: np.ndarray,
                              targets: list[str], out_path: Path):
    n = len(targets)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4.0))
    if n == 1:
        axes = [axes]
    for j, name in enumerate(targets):
        ax = axes[j]
        ax.scatter(y_true[:, j], y_pred[:, j], s=8, alpha=0.6)
        lo = min(np.min(y_true[:, j]), np.min(y_pred[:, j]))
        hi = max(np.max(y_true[:, j]), np.max(y_pred[:, j]))
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=0.8)
        ax.set_xlabel(f"{name} (FD)")
        ax.set_ylabel(f"{name} (RF)")
        ax.set_title(name)
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--labels_csv", type=str,
                   default=str(V3_DIR / "outputs" / "labels" / "02_monte_carlo_dataset.csv"))
    p.add_argument("--tag", type=str, default="03_surrogate_screening")
    args = p.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    rows = _coerce_float_keys(read_labels_csv(args.labels_csv))
    if not rows:
        print(f"no rows found in {args.labels_csv}; run Stage 02 first.")
        return 1

    # Drop numerical_invalid rows from the regressor training (they have NaNs).
    train_rows = [r for r in rows if r["label"] != "numerical_invalid"]
    X = build_feature_matrix(rows)
    y_cls = [r["label"] for r in rows]

    clf, (X_tr, X_te, y_tr, y_te) = train_classifier(
        X, y_cls,
        n_estimators=int(cfg["surrogate"]["classifier"]["n_estimators"]),
        max_depth=cfg["surrogate"]["classifier"]["max_depth"],
        test_size=float(cfg["surrogate"]["test_size"]),
        seed=int(cfg["run"]["seed"]),
        n_jobs=int(cfg["surrogate"]["classifier"]["n_jobs"]),
    )
    cls_eval = evaluate_classifier(clf, X_te, y_te, labels=LABEL_ORDER)
    print(f"\n=== Stage 03 — classifier ===")
    print(f"  accuracy = {cls_eval.accuracy:.4f}")
    print(cls_eval.classification_report_str)

    figs = V3_DIR / "outputs" / "figures" / args.tag
    figs.mkdir(parents=True, exist_ok=True)
    plot_confusion(cls_eval.confusion, LABEL_ORDER, figs / "confusion_matrix.png")

    # Regressor on non-invalid rows.
    if train_rows:
        Xr = build_feature_matrix(train_rows)
        Yr = build_regression_target_matrix(train_rows, REGRESSION_TARGETS)
        reg, (Xr_tr, Xr_te, Yr_tr, Yr_te) = train_regressor(
            Xr, Yr,
            n_estimators=int(cfg["surrogate"]["regressor"]["n_estimators"]),
            max_depth=cfg["surrogate"]["regressor"]["max_depth"],
            test_size=float(cfg["surrogate"]["test_size"]),
            seed=int(cfg["run"]["seed"]),
            n_jobs=int(cfg["surrogate"]["regressor"]["n_jobs"]),
        )
        reg_eval = evaluate_regressor(reg, Xr_te, Yr_te, REGRESSION_TARGETS)
        print(f"\n=== Stage 03 — regressor ===")
        print(f"  MAE per target:")
        for k, v in reg_eval.mae_per_target.items():
            print(f"    {k:<22}  {v:.4f}")
        print(f"  R² per target:")
        for k, v in reg_eval.r2_per_target.items():
            print(f"    {k:<22}  {v:+.4f}")

        Y_pred_te = reg.predict(Xr_te)
        plot_regression_scatter(Yr_te, Y_pred_te, REGRESSION_TARGETS,
                                  figs / "regression_scatter.png")
    else:
        reg = None
        reg_eval = None

    models = V3_DIR / "outputs" / "models"
    models.mkdir(parents=True, exist_ok=True)
    save_model(clf, models / f"{args.tag}_classifier.joblib",
                metadata={"feature_keys": FEATURE_KEYS,
                          "label_order": LABEL_ORDER,
                          "test_size": cfg["surrogate"]["test_size"]})
    if reg is not None:
        save_model(reg, models / f"{args.tag}_regressor.joblib",
                    metadata={"feature_keys": FEATURE_KEYS,
                              "regression_targets": REGRESSION_TARGETS})

    summary = {
        "n_rows_total": len(rows),
        "n_rows_valid_for_regression": len(train_rows),
        "classifier_accuracy": cls_eval.accuracy,
        "regressor_mae": reg_eval.mae_per_target if reg_eval else None,
        "regressor_r2": reg_eval.r2_per_target if reg_eval else None,
    }
    (V3_DIR / "outputs" / "logs" / f"{args.tag}_summary.json").write_text(
        json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
