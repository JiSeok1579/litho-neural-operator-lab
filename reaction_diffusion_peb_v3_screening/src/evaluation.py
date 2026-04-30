"""Imbalance-aware evaluation helpers for v3 surrogates.

Reports that *do not* lean on plain accuracy:
  - macro_f1, balanced_accuracy, per-class precision/recall/F1
  - regressor MAE / R² broken down by class label
  - reliability diagram (calibration of max-class probability)
"""
from __future__ import annotations

from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    precision_recall_fscore_support,
    r2_score,
)


@dataclass
class ClassMetricsReport:
    accuracy: float
    balanced_accuracy: float
    macro_f1: float
    per_class: dict[str, dict]
    confusion: np.ndarray
    label_order: list[str]


def classifier_report(y_true: list[str], y_pred: list[str],
                       labels: list[str]) -> ClassMetricsReport:
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    p, r, f, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0,
    )
    macro = float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))
    bal = float(balanced_accuracy_score(y_true, y_pred))
    acc = float(np.mean(np.array(y_pred) == np.array(y_true)))
    per_class = {
        labels[i]: {
            "precision": float(p[i]),
            "recall":    float(r[i]),
            "f1":        float(f[i]),
            "support":   int(support[i]),
        } for i in range(len(labels))
    }
    return ClassMetricsReport(
        accuracy=acc, balanced_accuracy=bal, macro_f1=macro,
        per_class=per_class, confusion=cm, label_order=labels,
    )


def regressor_mae_by_class(
    y_true: np.ndarray, y_pred: np.ndarray,
    class_labels: list[str], targets: list[str],
) -> dict[str, dict[str, float]]:
    """Returns {class_label: {target: MAE}}. MAE = NaN if class has no rows."""
    out: dict[str, dict[str, float]] = {}
    arr_labels = np.array(class_labels)
    for cls in sorted(set(class_labels)):
        mask = arr_labels == cls
        if not mask.any():
            out[cls] = {t: float("nan") for t in targets}
            continue
        per_target = {}
        for j, t in enumerate(targets):
            per_target[t] = float(mean_absolute_error(y_true[mask, j], y_pred[mask, j]))
        out[cls] = per_target
    return out


def regressor_global_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, targets: list[str],
) -> dict[str, dict[str, float]]:
    out = {}
    for j, t in enumerate(targets):
        try:
            r2 = float(r2_score(y_true[:, j], y_pred[:, j]))
        except Exception:  # noqa: BLE001
            r2 = float("nan")
        out[t] = {
            "MAE": float(mean_absolute_error(y_true[:, j], y_pred[:, j])),
            "R2":  r2,
        }
    return out


def plot_confusion(cm: np.ndarray, labels: list[str], out_path):
    fig, ax = plt.subplots(figsize=(7.5, 6.0))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
    ax.set_xlabel("predicted"); ax.set_ylabel("actual"); ax.set_title("Status confusion matrix")
    vmax = max(cm.max(), 1)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            v = int(cm[i, j])
            ax.text(j, i, str(v), ha="center", va="center",
                    color="white" if v > vmax / 2 else "black", fontsize=9)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_reliability_diagram(
    proba_max: np.ndarray, correct: np.ndarray, out_path,
    n_bins: int = 10,
) -> None:
    """Reliability diagram of max-class probability vs accuracy in that bin."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    centers = 0.5 * (bins[:-1] + bins[1:])
    accs = []
    counts = []
    for i in range(n_bins):
        mask = (proba_max >= bins[i]) & (proba_max < bins[i + 1])
        if mask.any():
            accs.append(float(correct[mask].mean()))
            counts.append(int(mask.sum()))
        else:
            accs.append(np.nan)
            counts.append(0)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="perfect")
    ax.plot(centers, accs, "o-", label="observed")
    for c, x, a in zip(counts, centers, accs):
        if c > 0 and np.isfinite(a):
            ax.text(x, a + 0.02, str(c), ha="center", fontsize=7)
    ax.set_xlabel("predicted max class probability")
    ax.set_ylabel("empirical accuracy in bin")
    ax.set_title("Classifier reliability diagram")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_per_class_mae(
    mae_by_class: dict[str, dict[str, float]], targets: list[str], out_path,
) -> None:
    classes = list(mae_by_class.keys())
    fig, axes = plt.subplots(1, len(targets), figsize=(4.0 * len(targets), 3.5))
    if len(targets) == 1:
        axes = [axes]
    for j, t in enumerate(targets):
        ax = axes[j]
        vals = [mae_by_class[c].get(t, float("nan")) for c in classes]
        bars = ax.bar(classes, vals, color="#4472c4")
        ax.set_ylabel("MAE")
        ax.set_title(t)
        ax.tick_params(axis="x", rotation=30)
        for b, v in zip(bars, vals):
            if np.isfinite(v):
                ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}",
                        ha="center", va="bottom", fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
