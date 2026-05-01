"""Stage 04D — yield-management view.

Produces a fab-style yield dashboard from the same Stage 04C dataset and
classifier the operational-zone evaluation already uses. No retraining,
no new FD. The point is purely visualisation: make 양품 / 불량 obvious at
a glance the way a fab yield engineer expects to see it.

Class → yield bucket mapping (semiconductor convention):

    PASS      (양품)      robust_valid
    MARGINAL  (한계품)    margin_risk
    FAIL      (불량)      under_exposed, merged, roughness_degraded,
                          numerical_invalid

Outputs (figures only, no model artefacts):
    outputs/figures/04d_zone_evaluation/yield_view/
        01_yield_summary.png        big-number headline card
        02_defect_pareto.png        descending bars + cumulative %
        03_pass_fail_confusion.png  3x3 colour-coded confusion
        04_process_window.png       CD_locked vs P_line_margin scatter
        05_yield_by_pitch.png       yield % per pitch_nm
        yield_summary.json          machine-readable counts + rates
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reaction_diffusion_peb_v3_screening.src.metrics_io import (
    FEATURE_KEYS,
    REGRESSION_TARGETS,
    build_feature_matrix,
    load_model,
    read_labels_csv,
)


V3_DIR = ROOT / "reaction_diffusion_peb_v3_screening"

# Korean font — Noto Sans CJK KR ships in the NotoSansCJK-Regular.ttc on
# this box. Register explicitly so Hangul renders even though matplotlib
# enumerates the .ttc as "Noto Sans CJK JP".
_CJK_TTC = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
if _CJK_TTC.exists():
    from matplotlib import font_manager
    font_manager.fontManager.addfont(str(_CJK_TTC))
    plt.rcParams["font.family"] = ["Noto Sans CJK JP", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


PASS_LABEL     = "robust_valid"
MARGINAL_LABEL = "margin_risk"
FAIL_LABELS    = ("under_exposed", "merged", "roughness_degraded", "numerical_invalid")

BUCKETS = ["PASS", "MARGINAL", "FAIL"]
BUCKET_KO = {"PASS": "양품", "MARGINAL": "한계", "FAIL": "불량"}
BUCKET_COLOR = {"PASS": "#2ca02c", "MARGINAL": "#ffbf00", "FAIL": "#d62728"}

DEFECT_KO = {
    "under_exposed":      "노광 부족 (under-exposed)",
    "merged":             "라인 병합 (merged)",
    "roughness_degraded": "거칠기 불량 (roughness)",
    "numerical_invalid":  "수치 무효 (numerical)",
    "margin_risk":        "마진 부족 (margin-risk)",
    "robust_valid":       "양품 (robust)",
}


def label_to_bucket(label: str) -> str:
    if label == PASS_LABEL:
        return "PASS"
    if label == MARGINAL_LABEL:
        return "MARGINAL"
    return "FAIL"


def _coerce_floats(rows):
    numeric = set(FEATURE_KEYS) | set(REGRESSION_TARGETS) | {
        "CD_final_nm", "CD_locked_nm", "CD_pitch_frac",
        "P_line_center_mean", "P_space_center_mean",
        "P_line_margin", "area_frac",
        "LER_CD_locked_nm", "LER_design_initial_nm",
        "pitch_nm", "line_cd_ratio", "dose_mJ_cm2",
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


# --------------------------------------------------------------------------
# 1. Headline yield summary card
# --------------------------------------------------------------------------
def plot_yield_summary(
    n_total: int, bucket_counts: dict[str, int],
    false_pass_rate: float, missed_defect_count: int,
    out_path: Path,
) -> None:
    yield_strict   = 100.0 * bucket_counts["PASS"] / max(n_total, 1)
    yield_inclusive = 100.0 * (bucket_counts["PASS"] + bucket_counts["MARGINAL"]) / max(n_total, 1)
    fail_rate      = 100.0 * bucket_counts["FAIL"] / max(n_total, 1)

    fig = plt.figure(figsize=(12.0, 6.5))
    gs = fig.add_gridspec(3, 3, height_ratios=[0.6, 1.4, 1.0],
                          hspace=0.45, wspace=0.25)

    ax_title = fig.add_subplot(gs[0, :]); ax_title.axis("off")
    ax_title.text(0.5, 0.55,
                  "PEB v3 — Yield Summary  (수율 요약)",
                  ha="center", va="center", fontsize=20, fontweight="bold")
    ax_title.text(0.5, 0.05,
                  f"전체 후보(샘플): N = {n_total:,}   "
                  f"|   robust_valid = 양품, margin_risk = 한계, 그 외 = 불량",
                  ha="center", va="center", fontsize=10, color="#555")

    # Big-number tiles for each bucket
    for j, b in enumerate(BUCKETS):
        ax = fig.add_subplot(gs[1, j]); ax.axis("off")
        ax.add_patch(plt.Rectangle((0.02, 0.05), 0.96, 0.90,
                                   facecolor=BUCKET_COLOR[b], alpha=0.18,
                                   edgecolor=BUCKET_COLOR[b], linewidth=2))
        ax.text(0.5, 0.78, f"{b}  ({BUCKET_KO[b]})",
                ha="center", fontsize=13, fontweight="bold",
                color=BUCKET_COLOR[b])
        cnt = bucket_counts[b]
        pct = 100.0 * cnt / max(n_total, 1)
        ax.text(0.5, 0.45, f"{cnt:,}",
                ha="center", fontsize=32, fontweight="bold",
                color=BUCKET_COLOR[b])
        ax.text(0.5, 0.18, f"{pct:.1f} %",
                ha="center", fontsize=14, color="#333")

    # Bottom row: yield + risk metrics
    ax_y = fig.add_subplot(gs[2, 0]); ax_y.axis("off")
    ax_y.text(0.05, 0.85, "Yield (양품률, strict)",
              fontsize=11, color="#555")
    ax_y.text(0.05, 0.45, f"{yield_strict:.2f} %",
              fontsize=24, fontweight="bold", color="#2ca02c")
    ax_y.text(0.05, 0.10,
              f"PASS / N = {bucket_counts['PASS']:,} / {n_total:,}",
              fontsize=9, color="#666")

    ax_y2 = fig.add_subplot(gs[2, 1]); ax_y2.axis("off")
    ax_y2.text(0.05, 0.85, "Yield (PASS+MARGINAL, inclusive)",
               fontsize=11, color="#555")
    ax_y2.text(0.05, 0.45, f"{yield_inclusive:.2f} %",
               fontsize=24, fontweight="bold", color="#888c2c")
    ax_y2.text(0.05, 0.10,
               f"한계품을 양품으로 간주했을 때",
               fontsize=9, color="#666")

    ax_r = fig.add_subplot(gs[2, 2]); ax_r.axis("off")
    ax_r.text(0.05, 0.85, "Surrogate 위험 지표",
              fontsize=11, color="#555")
    ax_r.text(0.05, 0.55,
              f"불량 누락률(false-PASS) : {false_pass_rate*100:.2f} %",
              fontsize=11, color="#d62728")
    ax_r.text(0.05, 0.30,
              f"  test split, predicted=robust_valid 중 실제 불량",
              fontsize=8, color="#777")
    ax_r.text(0.05, 0.05,
              f"누락 건수: {missed_defect_count}",
              fontsize=9, color="#444")

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------
# 2. Defect Pareto
# --------------------------------------------------------------------------
def plot_defect_pareto(label_counts: dict[str, int], n_total: int,
                       out_path: Path) -> None:
    defects = {k: v for k, v in label_counts.items() if k in FAIL_LABELS}
    items = sorted(defects.items(), key=lambda kv: kv[1], reverse=True)
    names = [DEFECT_KO.get(k, k) for k, _ in items]
    counts = [v for _, v in items]
    n_defect = sum(counts)
    cum = np.cumsum(counts)
    cum_pct = 100.0 * cum / max(n_defect, 1)

    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    bars = ax.bar(names, counts, color="#d62728", alpha=0.85,
                  edgecolor="#a01010")
    ax.set_ylabel("불량 건수 (count)")
    fail_pct = 100.0 * n_defect / max(n_total, 1)
    ax.set_title(f"Defect Pareto — 불량 모드별 분포  "
                 f"(전체 N={n_total:,}, 불량 {n_defect:,} / {fail_pct:.1f} %)")
    ax.tick_params(axis="x", rotation=15)

    for b, v in zip(bars, counts):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:,}",
                ha="center", va="bottom", fontsize=9)

    ax2 = ax.twinx()
    ax2.plot(names, cum_pct, "o-", color="#1f1f1f", lw=1.5)
    ax2.set_ylabel("누적 불량 비율 (cumulative %)")
    ax2.set_ylim(0, 105)
    for x, y in zip(names, cum_pct):
        ax2.text(x, y + 2, f"{y:.0f}%", ha="center",
                 fontsize=8, color="#333")
    ax2.axhline(80, color="#888", lw=0.7, ls="--", alpha=0.7)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------
# 3. PASS / MARGINAL / FAIL confusion (colour-coded)
# --------------------------------------------------------------------------
def plot_pass_fail_confusion(y_true_bucket: list[str], y_pred_bucket: list[str],
                             out_path: Path) -> None:
    cm = np.zeros((3, 3), dtype=int)
    idx = {b: i for i, b in enumerate(BUCKETS)}
    for t, p in zip(y_true_bucket, y_pred_bucket):
        cm[idx[t], idx[p]] += 1
    row_sums = cm.sum(axis=1, keepdims=True).clip(min=1)
    rates = cm / row_sums

    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    ax.imshow(np.zeros_like(cm), cmap="Greys", vmin=0, vmax=1)
    cell_colors = {
        ("PASS", "PASS"):         BUCKET_COLOR["PASS"],
        ("MARGINAL", "MARGINAL"): BUCKET_COLOR["MARGINAL"],
        ("FAIL", "FAIL"):         BUCKET_COLOR["FAIL"],
        ("PASS", "FAIL"):         "#f6b8b8",
        ("PASS", "MARGINAL"):     "#fce6a3",
        ("MARGINAL", "PASS"):     "#cfe6c8",
        ("MARGINAL", "FAIL"):     "#f6b8b8",
        ("FAIL", "PASS"):         "#7a0a0a",  # 가장 위험: 불량을 양품으로
        ("FAIL", "MARGINAL"):     "#f6b8b8",
    }
    for i, t in enumerate(BUCKETS):
        for j, p in enumerate(BUCKETS):
            color = cell_colors.get((t, p), "#dddddd")
            alpha = 0.35 if (i != j) else 0.55
            if (t, p) == ("FAIL", "PASS"):
                alpha = 0.85
            ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                       facecolor=color, alpha=alpha,
                                       edgecolor="white", linewidth=2))
            txt_color = "white" if (t, p) == ("FAIL", "PASS") else "black"
            ax.text(j, i - 0.08, f"{cm[i, j]:,}",
                    ha="center", va="center", fontsize=14,
                    fontweight="bold", color=txt_color)
            ax.text(j, i + 0.22, f"{rates[i, j]*100:.1f} %",
                    ha="center", va="center", fontsize=9, color=txt_color)

    ax.set_xticks(range(3))
    ax.set_xticklabels([f"{b}\n({BUCKET_KO[b]})" for b in BUCKETS])
    ax.set_yticks(range(3))
    ax.set_yticklabels([f"{b}\n({BUCKET_KO[b]})" for b in BUCKETS])
    ax.set_xlabel("Surrogate 예측 (predicted)")
    ax.set_ylabel("실제 라벨 (actual)")
    ax.set_title("PASS / MARGINAL / FAIL  (양품 / 한계 / 불량) — surrogate 판정 정합성\n"
                 "← 위험: 불량을 양품으로 흘려보내는 경우 (어두운 빨강)",
                 fontsize=11)
    ax.set_xlim(-0.5, 2.5); ax.set_ylim(2.5, -0.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------
# 4. Process-window scatter (CD_locked vs P_line_margin)
# --------------------------------------------------------------------------
def plot_process_window(rows: list[dict], out_path: Path) -> None:
    cd, mg, bk = [], [], []
    for r in rows:
        try:
            x = float(r.get("CD_locked_nm", "nan"))
            y = float(r.get("P_line_margin", "nan"))
        except (TypeError, ValueError):
            continue
        if not (np.isfinite(x) and np.isfinite(y)):
            continue
        cd.append(x); mg.append(y); bk.append(label_to_bucket(r["label"]))
    cd = np.array(cd); mg = np.array(mg); bk = np.array(bk)

    fig, ax = plt.subplots(figsize=(11.0, 7.5))
    for b in BUCKETS:
        m = (bk == b)
        ax.scatter(cd[m], mg[m], s=14, alpha=0.55,
                   color=BUCKET_COLOR[b],
                   label=f"{b} ({BUCKET_KO[b]}) n={int(m.sum()):,}")
    ax.axhline(0.05, color="#2ca02c", lw=1.0, ls="--", alpha=0.7,
               label="margin = 0.05 (robust threshold)")
    ax.axhline(0.0, color="#d62728", lw=1.0, ls=":", alpha=0.7,
               label="margin = 0 (process gate)")
    ax.set_xlabel("CD_locked_nm  (locked critical dimension)")
    ax.set_ylabel("P_line_margin")
    ax.set_title("Process-window 산점도 — 양품 / 한계 / 불량 분포")
    ax.legend(loc="lower right", fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------
# 5. Yield by pitch_nm
# --------------------------------------------------------------------------
def plot_yield_by_pitch(rows: list[dict], out_path: Path) -> None:
    by_pitch: dict[float, Counter] = defaultdict(Counter)
    for r in rows:
        try:
            p = float(r.get("pitch_nm", "nan"))
        except (TypeError, ValueError):
            continue
        if not np.isfinite(p):
            continue
        by_pitch[p][label_to_bucket(r["label"])] += 1

    pitches = sorted(by_pitch.keys())
    if not pitches:
        return
    pass_cnt   = [by_pitch[p].get("PASS", 0)     for p in pitches]
    marg_cnt   = [by_pitch[p].get("MARGINAL", 0) for p in pitches]
    fail_cnt   = [by_pitch[p].get("FAIL", 0)     for p in pitches]
    totals     = [pc + mc + fc for pc, mc, fc in zip(pass_cnt, marg_cnt, fail_cnt)]
    yield_pct  = [100.0 * pc / max(t, 1) for pc, t in zip(pass_cnt, totals)]

    fig, ax = plt.subplots(figsize=(9.0, 5.5))
    x = np.arange(len(pitches))
    ax.bar(x, pass_cnt, color=BUCKET_COLOR["PASS"], alpha=0.85, label="PASS (양품)")
    ax.bar(x, marg_cnt, bottom=pass_cnt,
           color=BUCKET_COLOR["MARGINAL"], alpha=0.85, label="MARGINAL (한계)")
    bottom2 = [pc + mc for pc, mc in zip(pass_cnt, marg_cnt)]
    ax.bar(x, fail_cnt, bottom=bottom2,
           color=BUCKET_COLOR["FAIL"], alpha=0.85, label="FAIL (불량)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(p)} nm\n(N={t:,})" for p, t in zip(pitches, totals)])
    ax.set_xlabel("pitch_nm")
    ax.set_ylabel("count")
    ax.set_title("Pitch별 분포 — stacked PASS / MARGINAL / FAIL")
    ax.legend(loc="upper right", fontsize=9)

    ax2 = ax.twinx()
    ax2.plot(x, yield_pct, "o-", color="#1f1f1f", lw=1.5,
             label="Yield (PASS only, %)")
    ax2.set_ylabel("Yield (%)")
    ax2.set_ylim(0, 100)
    for xi, yi in zip(x, yield_pct):
        ax2.text(xi, yi + 2, f"{yi:.0f}%", ha="center", fontsize=9,
                 color="#1f1f1f", fontweight="bold")
    ax2.legend(loc="upper left", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------
def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--training_csv", type=str,
                   default=str(V3_DIR / "outputs" / "labels"
                               / "stage04C_training_dataset.csv"))
    p.add_argument("--classifier", type=str,
                   default=str(V3_DIR / "outputs" / "models"
                               / "stage04C_classifier.joblib"))
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--tag", type=str, default="04d_zone_evaluation/yield_view")
    args = p.parse_args()

    rows = _coerce_floats(read_labels_csv(args.training_csv))
    n_total = len(rows)
    label_counts = Counter(r["label"] for r in rows)
    bucket_counts = Counter(label_to_bucket(r["label"]) for r in rows)
    print(f"yield-view — dataset rows: {n_total}")
    for b in BUCKETS:
        print(f"  {b:<10} ({BUCKET_KO[b]}) {bucket_counts.get(b, 0):,}")
    for k in sorted(label_counts):
        print(f"    {k:<22} {label_counts[k]:,}")

    # Surrogate risk: false-PASS rate on the 04C/04D test split
    clf, _ = load_model(args.classifier)
    test_rows = _split_test(rows, seed=args.seed, test_frac=0.2)
    X_te = build_feature_matrix(test_rows)
    y_te = [r["label"] for r in test_rows]
    y_pred = clf.predict(X_te).tolist()
    pred_pass = np.array([yp == PASS_LABEL for yp in y_pred])
    actual_fail = np.array([label_to_bucket(yt) == "FAIL" for yt in y_te])
    n_pred_pass = int(pred_pass.sum())
    n_missed   = int(np.sum(pred_pass & actual_fail))
    false_pass_rate = (n_missed / n_pred_pass) if n_pred_pass > 0 else float("nan")
    print(f"\ntest split: {len(test_rows)} rows  "
          f"|  predicted=robust_valid: {n_pred_pass}  "
          f"|  실제 불량(누락): {n_missed}  "
          f"|  false-PASS rate: {false_pass_rate*100:.2f} %")

    fig_dir = V3_DIR / "outputs" / "figures" / args.tag
    fig_dir.mkdir(parents=True, exist_ok=True)

    plot_yield_summary(
        n_total=n_total,
        bucket_counts={b: int(bucket_counts.get(b, 0)) for b in BUCKETS},
        false_pass_rate=float(false_pass_rate) if np.isfinite(false_pass_rate) else 0.0,
        missed_defect_count=n_missed,
        out_path=fig_dir / "01_yield_summary.png",
    )
    plot_defect_pareto(label_counts, n_total, fig_dir / "02_defect_pareto.png")

    y_true_bucket = [label_to_bucket(yt) for yt in y_te]
    y_pred_bucket = [label_to_bucket(yp) for yp in y_pred]
    plot_pass_fail_confusion(y_true_bucket, y_pred_bucket,
                             fig_dir / "03_pass_fail_confusion.png")

    plot_process_window(rows, fig_dir / "04_process_window.png")
    plot_yield_by_pitch(rows, fig_dir / "05_yield_by_pitch.png")

    summary = {
        "stage": "04D-yield-view",
        "n_total": n_total,
        "bucket_counts": dict(bucket_counts),
        "label_counts": dict(label_counts),
        "yield_strict_pct": 100.0 * bucket_counts.get("PASS", 0) / max(n_total, 1),
        "yield_inclusive_pct": 100.0 * (bucket_counts.get("PASS", 0)
                                         + bucket_counts.get("MARGINAL", 0))
                                / max(n_total, 1),
        "fail_rate_pct": 100.0 * bucket_counts.get("FAIL", 0) / max(n_total, 1),
        "test_split": {
            "seed": int(args.seed), "size": int(len(test_rows)),
            "n_predicted_robust_valid": n_pred_pass,
            "missed_defects": n_missed,
            "false_pass_rate": float(false_pass_rate)
                                if np.isfinite(false_pass_rate) else None,
        },
        "bucket_definition": {
            "PASS":     [PASS_LABEL],
            "MARGINAL": [MARGINAL_LABEL],
            "FAIL":     list(FAIL_LABELS),
        },
    }
    (fig_dir / "yield_summary.json").write_text(
        json.dumps(summary, indent=2))
    print(f"\nfigures → {fig_dir}")
    print(f"summary → {fig_dir / 'yield_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
