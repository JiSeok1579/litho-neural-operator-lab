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


# --------------------------------------------------------------------------
# Stage 04D — zone-aware evaluation
# --------------------------------------------------------------------------

OPERATIONAL_ZONE = ("robust_valid", "margin_risk")
FAILURE_ZONE     = ("under_exposed", "merged", "roughness_degraded", "numerical_invalid")


def label_to_zone(label: str) -> str:
    if label in OPERATIONAL_ZONE:
        return "operational"
    if label in FAILURE_ZONE:
        return "failure"
    return "unknown"


def binary_zone_metrics(y_true: list[str], y_pred: list[str]) -> dict:
    """Operational-zone vs failure-zone binary classifier metrics.

    Treats the operational zone (robust_valid + margin_risk) as the
    "positive" class for the operational-precision / operational-recall
    pair. Returns the four 2×2 cells, derived rates, and the auxiliary
    error fractions the user asked for:
        false_robust_valid_rate = P(actual ∈ failure_zone | predicted == robust_valid)
        false_defect_rate       = P(actual == robust_valid | predicted ∈ failure_zone)
    """
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)

    is_op_true = np.isin(y_true_arr, OPERATIONAL_ZONE)
    is_op_pred = np.isin(y_pred_arr, OPERATIONAL_ZONE)
    is_fail_true = np.isin(y_true_arr, FAILURE_ZONE)
    is_fail_pred = np.isin(y_pred_arr, FAILURE_ZONE)

    tp = int(np.sum(is_op_true & is_op_pred))
    fn = int(np.sum(is_op_true & ~is_op_pred))
    fp = int(np.sum(~is_op_true & is_op_pred))
    tn = int(np.sum(~is_op_true & ~is_op_pred))

    op_precision = tp / max(tp + fp, 1)
    op_recall    = tp / max(tp + fn, 1)
    op_f1        = (2 * op_precision * op_recall) / max(op_precision + op_recall, 1e-12)

    # The two diagnostic rates the user emphasised.
    pred_robust = (y_pred_arr == "robust_valid")
    n_pred_robust = int(pred_robust.sum())
    if n_pred_robust > 0:
        false_robust_valid_rate = float(np.sum(pred_robust & is_fail_true) / n_pred_robust)
    else:
        false_robust_valid_rate = float("nan")

    pred_failure = is_fail_pred
    n_pred_failure = int(pred_failure.sum())
    if n_pred_failure > 0:
        false_defect_rate = float(np.sum(pred_failure & (y_true_arr == "robust_valid"))
                                    / n_pred_failure)
    else:
        false_defect_rate = float("nan")

    return {
        "tp_op_op": tp, "fn_op_fail": fn, "fp_fail_op": fp, "tn_fail_fail": tn,
        "operational_precision": float(op_precision),
        "operational_recall":    float(op_recall),
        "operational_f1":        float(op_f1),
        "false_robust_valid_rate": false_robust_valid_rate,   # high → dangerous
        "false_defect_rate":       false_defect_rate,         # high → conservative
        "n_predicted_robust_valid": n_pred_robust,
        "n_predicted_failure":      n_pred_failure,
        "support_operational":      int(is_op_true.sum()),
        "support_failure":          int(is_fail_true.sum()),
    }


def robust_vs_all_metrics(y_true: list[str], y_pred: list[str]) -> dict:
    """One-vs-rest binary metrics with `robust_valid` as the positive class."""
    y_t = np.array(y_true)
    y_p = np.array(y_pred)
    is_pos_t = (y_t == "robust_valid")
    is_pos_p = (y_p == "robust_valid")
    tp = int(np.sum(is_pos_t & is_pos_p))
    fn = int(np.sum(is_pos_t & ~is_pos_p))
    fp = int(np.sum(~is_pos_t & is_pos_p))
    tn = int(np.sum(~is_pos_t & ~is_pos_p))
    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1)
    f1 = (2 * p * r) / max(p + r, 1e-12)
    return {
        "tp": tp, "fn": fn, "fp": fp, "tn": tn,
        "precision": float(p), "recall": float(r), "f1": float(f1),
    }


def regressor_zone_aggregates(
    mae_by_class: dict[str, dict[str, float]],
    n_per_class: dict[str, int],
    targets: list[str],
) -> dict[str, dict[str, float]]:
    """Count-weighted MAE aggregates over operational and failure zones."""
    out: dict[str, dict[str, float]] = {"operational": {}, "failure": {}}
    for zone_name, zone_classes in [("operational", OPERATIONAL_ZONE),
                                       ("failure",     FAILURE_ZONE)]:
        for t in targets:
            num = 0.0; denom = 0
            for c in zone_classes:
                v = mae_by_class.get(c, {}).get(t, float("nan"))
                n = int(n_per_class.get(c, 0))
                if not np.isfinite(v) or n <= 0:
                    continue
                num += v * n
                denom += n
            out[zone_name][t] = float(num / denom) if denom > 0 else float("nan")
    return out


def per_trigger_analysis(rough_rows: list[dict]) -> dict:
    """For roughness_degraded rows, count which triggers fired and their
    overlap. Each row's `roughness_trigger` is a "+"-joined list."""
    triggers = ["ler_locked_max", "ler_design_excess", "psd_mid_increase"]
    counts = {t: 0 for t in triggers}
    overlap = {f"{a}+{b}": 0 for a in triggers for b in triggers}
    multiplicity = {1: 0, 2: 0, 3: 0}
    n_rows = 0
    for r in rough_rows:
        raw = r.get("roughness_trigger") or ""
        fired = [t for t in raw.split("+") if t in triggers]
        if not fired:
            continue
        n_rows += 1
        for t in fired:
            counts[t] += 1
        multiplicity[len(fired)] = multiplicity.get(len(fired), 0) + 1
        for a in fired:
            for b in fired:
                overlap[f"{a}+{b}"] += 1
    return {
        "n_roughness_rows_with_trigger": n_rows,
        "counts_per_trigger": counts,
        "multiplicity": multiplicity,
        "overlap_matrix": overlap,
        "trigger_order": triggers,
    }


def plot_zone_confusion(metrics: dict, out_path) -> None:
    cm = np.array([[metrics["tp_op_op"],     metrics["fn_op_fail"]],
                   [metrics["fp_fail_op"],   metrics["tn_fail_fail"]]])
    labels = ["operational", "failure"]
    fig, ax = plt.subplots(figsize=(4.8, 3.8))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(2)); ax.set_xticklabels(labels)
    ax.set_yticks(range(2)); ax.set_yticklabels(labels)
    ax.set_xlabel("predicted zone")
    ax.set_ylabel("actual zone")
    ax.set_title("Operational vs failure (binary)")
    vmax = max(cm.max(), 1)
    for i in range(2):
        for j in range(2):
            v = int(cm[i, j])
            ax.text(j, i, str(v), ha="center", va="center",
                    color="white" if v > vmax / 2 else "black", fontsize=11)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_trigger_overlap(per_trigger: dict, out_path) -> None:
    triggers = per_trigger["trigger_order"]
    counts = [per_trigger["counts_per_trigger"][t] for t in triggers]
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.6))
    axes[0].bar(triggers, counts, color="#4472c4")
    axes[0].set_ylabel("count of roughness rows where this trigger fired")
    axes[0].set_title("Per-trigger count")
    axes[0].tick_params(axis="x", rotation=20)
    for x, y in zip(triggers, counts):
        axes[0].text(x, y, str(y), ha="center", va="bottom", fontsize=8)

    mult = per_trigger["multiplicity"]
    keys = sorted(mult.keys())
    axes[1].bar([str(k) for k in keys], [mult[k] for k in keys], color="#ed7d31")
    axes[1].set_xlabel("number of triggers fired")
    axes[1].set_ylabel("count")
    axes[1].set_title("Multiplicity")
    for x, y in zip(keys, [mult[k] for k in keys]):
        axes[1].text(str(x), y, str(y), ha="center", va="bottom", fontsize=8)

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
