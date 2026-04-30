# Phase 5 — Stage 5: pitch × dose process window

## 0. TL;DR

Starting from the Stage-4 balanced OP (σ=2, Q0=0.02, kq=1.0, DH=0.5, t=30): 6 pitch × 6 dose = 36 primary runs + 72 control runs = **108 runs**. **At pitch=16 the process window is closed**, pitch=18 has a single valid-only point, and pitch ≥ 24 has a 4-dose-wide robust_valid window. An interesting finding: **the quencher narrows the small-pitch window** (a trade-off between Stage 4's LER improvement and small-pitch tolerance). The recommended dose for every pitch ≥ 20 is dose=40 (consistent with Stage 4).

---

## 1. Goal

- Plan §5 Stage 5: process-window shape and pitch-dependent operating point.
- Additionally answer:
  - At small pitch (16, 18), where does the chemistry collapse?
  - How does the quencher impact small-pitch tolerance?
  - What is the difference in process windows between σ-only, quencher-only, and both?

---

## 2. Steps taken

1. Wrote the **Stage 5 base config** (`configs/v2_stage5_pitch_dose.yaml`) using the Stage-4 balanced OP as-is.
2. Wrote the **Stage 5 sweep script** (`experiments/05_pitch_dose/run_pitch_dose_sweep.py`):
   - 6 pitch × 6 dose × 3 blocks (primary + 2 controls) = 108 runs.
   - Per pitch: domain_x_nm = pitch * 5 (`n_periods_x=5`), domain_y_nm = 120 nm (consistent LER y-sample count).
   - Per run: deepcopy cfg → update pitch / domain_x / dose → call `run_one_with_overrides`.
   - Classification: unstable / merged / under_exposed / low_contrast / valid / robust_valid (precedence order).
3. **Heatmap figures** (per block × 6 metrics): status, CD_shift, total_LER%, P_line_margin, area_frac, contrast.
4. Saved 36 **primary contour overlays** (controls keep heatmaps only).
5. **Recommendation** per pitch: robust_valid first → min |CD_shift| → max LER% → max margin.

---

## 3. Problems and resolutions

### Problem 1 — At pitch=16, every dose is merged or under_exposed

**Symptom**: Primary OP, pitch=16, dose ∈ {21..60} → only dose=21 is under_exposed; the other 5 doses are all merged. Not a single valid point.

**Cause analysis**:

- pitch=16 / line_cd=12.5 → space=3.5 nm.
- diffusion length √(2·DH·t) = √(2·0.5·30) ≈ 5.5 nm > space.
- That is, normal acid spread from the line centre fills the space.
- On top of that, σ=2 nm electron blur smooths the space further and pushes P_space up.
- Result: lower the dose and the lines themselves drop to P_line < 0.65 (under_exposed); raise the dose and the space hits P > 0.5 (merged). No middle ground.

**Interpretation**: The "DH·t = 15 nm² = (line_cd/2)²" budget that worked at 24 nm pitch is unworkable at 16 nm pitch — duty line_cd / pitch is 0.78 and the space is too narrow. The process window genuinely closes.

**Resolution (in this stage)**:

- Mark pitch=16 as no recommendation. We acknowledge the user's sweep range while documenting the chemistry limit.
- To rescue pitch=16 we need either (a) a line_cd scaled to ~50% of pitch, or (b) a smaller DH·t budget (e.g. t=15 or DH=0.3). Stage 5 preserves the original setup; this is split out as a Stage 6/7 follow-up.

---

### Problem 2 — Negative total_LER_reduction_pct at pitch ≤ 20

**Symptom**: pitch=20, dose=40 → robust_valid (every gate passes) but total_LER% = -26.22%. pitch=18, dose=28.4 → valid with -34.18%.

**Cause**: The small-pitch version of the contour-displacement artefact diagnosed in Stages 3/4. At small pitch the CD_shift / pitch ratio looks larger, so the contour drifts further from the design edge.

```text
pitch=20, dose=40: CD_shift = +4.10 nm  →  CD_final/pitch = 0.83
                     contour is 4.1 nm away from the design edge
                     LER is measured at a position unrelated to the design edge
```

**Resolution**:

- The robust_valid classification itself is correct — the gates (contour, area, P) all pass.
- Absolute LER reduction is misleading at small pitch.
- A meaningful comparison requires a **same-pitch** baseline delta. This stage only reports absolute values and notes the caveat.
- Stage 4B (CD-locked LER) will tackle this directly.

---

### Problem 3 — The quencher narrows the small-pitch process window

**Symptom**: control_sigma2_no_q (quencher off) vs primary (Q0=0.02, kq=1.0):

| pitch | control_σ2_no_q robust_valid count | primary robust_valid count |
|---|---|---|
| 16 | 0 | 0 |
| 18 | 0 | 0 |
| 20 | 1 (dose=28.4) | 1 (dose=40) |
| 24 | 5 | 4 |
| 28 | 5 | 5 |
| 32 | 5 | 5 |

Adding the quencher leaves the robust window comparable or slightly smaller. In particular at pitch=18, control_sigma0_no_q (σ=0, quencher off) has dose=28.4 robust_valid, but primary and control_sigma2_no_q are valid only.

**Cause hypothesis**:

- In Stage 4 the quencher's effect was to reduce line widening, keeping the contour close. At 24 nm pitch this reads as LER improvement.
- At small pitch, however, the line is already narrow and the contour is close (CD/p large). Adding quencher consumes more H, so P_line drops and the P_line_margin shrinks (`pitch=18 dose=28.4`: control_σ0_noq margin=+0.070 → primary margin=+0.012).
- That is, **the quencher's LER benefit is biased toward large pitch; at small pitch its P_line cost shrinks the robust window**.

**Interpretation**: The Stage 4 balanced OP (Q0=0.02, kq=1.0) should not be applied as-is to small pitch. At pitch ≤ 20 we need either (a) a weaker quencher (Q0 ≤ 0.005 or kq ≤ 0.5) or (b) a dose correction.

**Resolution (in this stage)**: Recorded as a study-note finding. A pitch-dependent quencher tuning is a lesson for downstream stages or for application.

---

### Problem 4 — 100% LER reduction in the heatmap is a merged artefact

**Symptom**: For pitch=16/18 merged cells, `total_LER_reduction_pct = +100.00%`.

**Cause**: The Stage 2 diagnosis: when `LER_after_PEB_P = 0` (lines merged → edges extract NaN/0), the reduction is computed as 100%. These cells are classified as merged, so the selection algorithm never picks them.

**Resolution**: The heatmap title and study note explicitly call these "merged-cell LER values are artefacts." Heatmap vmin/vmax is set to [-30, +15] so the 100% cells render as saturated red and stand out visually.

---

## 4. Decision log

| Decision | Adopted | Reason |
|---|---|---|
| OP choice | Stage-4 balanced OP only | User-specified. The algorithmic best is demoted from Stage 3 onward. |
| line_cd handling | 12.5 nm fixed | User-specified. Plan §4.1 as-is. We accept duty changing across pitch. |
| domain_x_nm | pitch * 5 | User-specified n_periods_x=5. Avoids FFT seam artefact. |
| domain_y_nm | 120 nm fixed | Keeps LER y-sample count consistent. Other options would make ny vary with pitch and shift the LER noise floor. |
| Run controls | Run both control blocks (skip option preserved) | Isolate the standalone effects of quencher and e-blur. |
| Classification precedence | unstable → merged → under_exposed → low_contrast → valid → robust_valid | User-specified. low_contrast is not in the user spec but is added as a fallback when the contrast gate fails. |
| Recommendation algorithm | robust_valid first → min |CD_shift| → max LER% → max margin | User-specified. |
| Negative / 100% LER artefacts | Reporting only; classification untouched | Gates and classification are contour/area-based, not affected. |
| pitch=16 recommendation | Marked "no recommendation" explicitly | Reports honestly that the user's proposed process window is outside the chemistry limit. |
| Stage 4B / 3B | Still on hold | Trigger not raised. |

---

## 5. Verified results

### Primary OP (36 of 108 runs)

```text
Status counts (primary):
  unstable      :  0
  merged        : 17  (essentially all of pitch 16/18, dose ≥ 44.2 of pitch 20, dose 28.4 of pitch 16)
  under_exposed :  6  (every pitch, dose=21)
  low_contrast  :  0
  valid         :  5  (dose=28.4 across pitch 18-24)
  robust_valid  : 14
```

### Recommended dose per pitch (primary)

| pitch | rec dose | status | margin | CD_shift | LER% | comment |
|---|---|---|---|---|---|---|
| 16 | — | none | — | — | — | process window closed |
| 18 | 28.4 | valid | 0.012 | +1.99 | -34.18 | margin tight; LER artefact |
| 20 | 40 | robust_valid | 0.098 | +4.10 | -26.22 | LER artefact (CD widening) |
| 24 | 40 | robust_valid | 0.096 | +2.07 | +8.77 | matches Stage 4 |
| 28 | 40 | robust_valid | 0.095 | +1.81 | +14.33 | comfortable |
| 32 | 40 | robust_valid | 0.095 | +1.78 | +15.03 | comfortable |

**Every pitch ≥ 20 prefers dose=40**, insensitive to dose changes. Read this as Stage 4's quencher having stabilized the acid budget.

### Control comparison (robust_valid cell counts)

```text
                     pitch=16  18  20  24  28  32
  primary               0      0   1   4   5   5
  control_σ0_no_q       0      1   3   5   5   5
  control_σ2_no_q       0      0   1   5   5   5
```

- σ=0 no-quencher gives the widest window. Both e-blur and quencher push small-pitch tolerance the wrong way.
- Adding e-blur (control_σ0_no_q → control_σ2_no_q): pitch=18 loses 1 robust_valid, pitch=20 loses 2, pitch=24+ unchanged.
- Adding quencher (control_σ2_no_q → primary): pitch=24 loses 1, the rest unchanged.

### Qualitative trends (plan §5 expected vs observed)

| plan §5 expected | observed |
|---|---|
| Stable contour at 20–24 nm pitch | yes — 1 robust_valid at pitch=20, 4 at pitch=24 |
| Process window narrows at 16–18 nm pitch | yes — pitch=16 closed, pitch=18 valid only at 1 dose |
| CD widening / over-deprotection at high dose | yes — every pitch ≤ 20 with dose ≥ 44 is merged |
| Under-deprotection at low dose | yes — every pitch at dose=21 is under_exposed |

---

## 6. Follow-up work

- **Stage 6 (x-z standing wave)**: the next main stage. Plan §5 Stage 6 as-is.
- **Stage 4B (CD-locked LER)**: the small-pitch LER artefacts at pitch ≤ 20 affect decisions → trigger fired. Stage 6 takes priority; run after Stage 6 or as a Stage 5 follow-up.
- **Pitch-dependent quencher** (hypothesis test): a mini-sweep at pitch ≤ 20 to check whether weakening the quencher (e.g. Q0=0.005) restores the robust window. This was a follow-up that emerged from this stage's lessons, not added to the plan.
- **line_cd scaling** (alternative 1): scale line_cd ≈ pitch/2 to rescue pitch=16. Could be split into its own stage.
- **DH·t budget reduction at small pitch** (alternative 2): at pitch=16, shrink the budget with DH=0.3 or t=15. A separate sweep dimension from this stage's dose.
- **Stage 3B** (σ=5/8 compatible budget): still on hold.

---

## 7. Artefacts

```text
configs/v2_stage5_pitch_dose.yaml
experiments/05_pitch_dose/
  __init__.py
  run_pitch_dose_sweep.py

outputs/
  figures/05_pitch_dose/
    primary/                              # 6 heatmaps
    control_sigma0_no_q/                  # 6 heatmaps
    control_sigma2_no_q/                  # 6 heatmaps
    primary_contours/                     # 36 contour overlays
  logs/05_pitch_dose_summary.csv          # 108 rows, full metric set
  logs/05_pitch_dose_summary.json
  logs/05_pitch_dose_recommendation.json  # per-pitch recommendation

EXPERIMENT_PLAN.md
  §Stage 5 updated (OP, sweep, classification, recommendation algorithm,
  verified results, control comparison, mapping to plan §5 success criteria)

study_notes/
  05_stage5_pitch_dose.md  (this file)
  README.md  index updated
```
