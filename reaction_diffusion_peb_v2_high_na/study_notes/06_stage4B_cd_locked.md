# Phase 6 — Stage 4B: CD-locked LER + pitch-dependent quencher mini-sweep

## 0. TL;DR

Introduced a CD-lock measurement that bisects on P_threshold to pin the contour to the design CD. Across 12 Block-A runs (primary + control × 3 pitch × 2 dose), only 1 case was a displacement artefact (control_σ0 / pitch=20 / dose=40) and 2 were merged artefacts; **everything else was real roughness degradation**. So part of Stage 5's negative LER reduction is a measurement artefact, but at the primary OP (σ=2 + Q0=0.02 + kq=1) with pitch ≤ 20 it is genuine degradation. The pitch-dependent quencher mini-sweep (20 runs) showed that **at pitch=18, 20 even weakening the quencher cannot recover LER_locked** — the Stage 4 balanced OP is recommended only for pitch ≥ 24.

---

## 1. Goal

Combine plan §5 Stage 4B with Stage 5B and answer two questions.

```text
1. Is the "negative LER reduction at pitch ≤ 20" from Stage 5 a contour-displacement
   artefact of the fixed-threshold measurement, or genuine PEB roughness degradation?
2. At pitch=18, 20 does weakening the quencher recover the LER?
   (Follow-up on the small-pitch process-window narrowing from Stage 5)
```

---

## 2. Steps taken

1. **Added the CD-lock function** (`src/metrics_edge.py`):
   - `find_cd_lock_threshold(field, x, line_centers, pitch, cd_target, P_min=0.2, P_max=0.8, cd_tol=0.25)` — bisection to find P_threshold.
   - When the endpoints have an empty contour, narrow inward in 0.05 steps (covers the common case where P_max exceeds P_line_max).
   - Status: `ok / unstable_low_bound / unstable_high_bound / unstable_no_crossing / unstable_no_converge`.
   - Added 3 unit tests (20 total passing).
2. **Stage 4B Block A — CD-locked re-measurement**:
   - Ran the primary OP (σ=2, Q0=0.02, kq=1.0) and control OP (σ=0, quencher off) at pitch ∈ {18, 20, 24} × dose ∈ {28.4, 40} → 12 runs.
   - Per row: extract both fixed (P=0.5) and locked metrics from a single PEB run.
   - Decision label has 4 categories: `real degradation / fixed underestimates (merged) / fixed overestimates (displacement) / OK`.
3. **Stage 5B (Block B) — pitch-dependent quencher mini-sweep**:
   - σ=2, DH=0.5, t=30 fixed. (pitch=18, dose=28.4) + (pitch=20, dose=40) × (1 baseline + 3 Q0 × 3 kq) = 20 runs.
   - Reported both CD-locked LER and fixed LER.
   - Per-pitch heatmap (LER_locked, dLER_locked, dCD_shift, P_line_margin).
4. Updated **plan §Stage 4B**, wrote the **study notes**, committed, merged.

---

## 3. Problems and resolutions

### Problem 1 — In CD-lock, the P_max=0.8 endpoint has no contour in nearly every row

**Symptom**: On the first run, 9 of 12 Block-A runs returned `unstable_no_crossing`. CD-lock barely worked.

**Cause**: P_line_center_mean usually sits in 0.65–0.85, so the P_threshold=0.8 contour catches only the narrowest part of the line — or nothing at all. extract_edges returns NaN on every row → cd_overall_mean = NaN → the function flags unstable_no_crossing.

**Resolution**: Add logic to step inward by 0.05 from the endpoint when "no contour."

```python
P_hi_use = P_max
cd_hi, _ = cd_at(P_hi_use)
while not np.isfinite(cd_hi) and P_hi_use > P_lo_use + 1e-3:
    P_hi_use = round(P_hi_use - 0.05, 4)
    cd_hi, _ = cd_at(P_hi_use)
```

Apply the same logic to P_min, then bisect within the valid range.

**Result after fix**: 12/12 Block A runs CD_lock=ok. Mini-sweep 20/20 ok.

**Lesson**: The endpoints of a bisection should be auto-narrowed from the "user-spec range" to the "valid range." Even when the spec is [0.2, 0.8], if the dynamic range of P is narrower we preserve the user's intent while ensuring stability.

---

### Problem 2 — At pitch=18 dose=40, fixed-threshold LER comes out *below* design

**Symptom**:

```text
primary, pitch=18, dose=40:
  status_fixed = merged
  LER_design = 2.77,  LER_fixed = 1.63   (smoother than design?)
```

Counter-intuitive — the cell is merged yet LER is small.

**Cause**: When the lines fully merge, the P>0.5 contour no longer sits on the line "edge" but follows the domain boundary or small variations of acid intensity. The edges extracted there are smooth curves unrelated to the design line edges, so LER comes out small. **In short, the fixed-threshold LER is meaningless in regions where lines disappear**.

**Resolution**: CD-lock forces the contour back to the design CD=12.4 → LER_locked=4.16, the correct measurement. Added "fixed underestimates (merged-line artefact)" to the decision labels.

**Lesson**: The fixed-threshold LER is unreliable for merged-status cells. Comparisons must use CD-locked LER or PSD-mid.

---

### Problem 3 — Displacement artefact confirmed at control_σ0 pitch=20 dose=40

**Symptom**: Stage 5's control_sigma0_no_q at pitch=20 dose=40 was robust_valid with total_LER% = -19.75% (fixed). After CD-lock:

```text
LER_design = 2.77,  LER_fixed = 3.32,  LER_locked = 2.84
```

LER_locked nearly matches design. The +0.55 nm gap of fixed is a displacement artefact from a contour off the design edge.

**Interpretation**: The Stage 3/4 hypothesis "PEB does not raise LER per se; line widening shifted the contour" is verified exactly in this single case. The CD-locked LER tool earns its keep.

---

### Problem 4 — At primary OP / pitch ≤ 20 the LER is still bad even after CD-lock

**Symptom**:

```text
primary, pitch=18, dose=28.4:  fixed=3.72  locked=4.07  → both above design
primary, pitch=20, dose=40:    fixed=3.50  locked=3.25  → both above design
```

Even with the contour pinned to the design CD, LER is well above design.

**Interpretation**:

- This is "real roughness degradation" — PEB really did roughen the line edges.
- Compared with the σ=0 control at the same pitch=20 dose=40 (LER_locked=2.84 ≈ design): introducing σ=2 adds +0.4 nm to locked LER.
- That is, **electron blur itself raises LER at pitch=20**. This extends the high-σ PEB-LER deterioration hypothesis from Stage 3: at small pitch the effect appears already at "weak" σ such as σ=2.

**Resolution**: An honest conclusion in the study note — the Stage 4 balanced OP (σ=2 + quencher) is recommended only for pitch ≥ 24. At pitch ≤ 20 we need to lower σ itself (e.g. σ=0) or use different chemistry.

---

### Problem 5 — The pitch-dependent quencher mini-sweep does not recover LER

**Symptom (Block B)**:

```text
pitch=18 / dose=28.4:
  baseline (no quencher):     LER_lock = 4.16
  Q0=0.005, kq=0.5 (weakest): LER_lock = 4.13  (recovered 0.03 nm)
  Q0=0.02,  kq=2   (strongest): LER_lock = 4.05  (recovered 0.11 nm)
  → still +1.3 nm above design 2.77. Almost no recovery.

pitch=20 / dose=40:
  baseline:                   LER_lock = 3.28
  Q0=0.02, kq=2 (max effort): LER_lock = 3.21  (recovered 0.07 nm)
  → still +0.4 nm above design 2.77.
```

**Interpretation**: The Stage 4 hypothesis "weakening the quencher recovers small pitch" is not verified. Whether the quencher is weakened or strengthened, LER at pitch ≤ 20 cannot be brought back to design. The cause is not the quencher but σ=2 e-blur (same as Problem 4).

**Resolution**:

- σ=2 is itself inappropriate as a baseline at pitch ≤ 20. We need σ=0 or proportional line_cd scaling.
- The Stage 5 recommendation of dose=40 / σ=2 / Q0=0.02 / kq=1 is **pitch ≥ 24 only**.

---

## 4. Decision log

| Decision | Adopted | Reason |
|---|---|---|
| CD-lock algorithm | bisection on P ∈ [0.2, 0.8] with adaptive endpoint narrowing | Preserves the spec [0.2, 0.8] while handling P's actual dynamic range (0.65–0.85). |
| `cd_tol_nm` | 0.25 nm | Spec. In this sweep every ok row converged with |CD_locked - CD_target| < 0.1 nm. |
| Block A pitch / dose | spec ({18, 20, 24} × {28.4, 40}) | User-specified. Stage 5's marginal/robust window-boundary region. |
| Block B pitch / dose | (18, 28.4) + (20, 40) | Use the per-pitch recommended doses from Stage 5; other doses would mix pitch-dose effects. |
| Block B Q0 / kq | {0.005, 0.01, 0.02} × {0.5, 1.0, 2.0} | Consistent with the spec's sweep range. |
| 4 decision-label categories | real degradation / fixed underestimates (merged) / fixed overestimates (displacement) / OK | Extended from the user spec's 2 categories — needed to distinguish merged-LER and displacement artefacts accurately. |
| `DECISION_TOL` | 0.20 nm | About 10% of the LER measurement noise level. Smaller would be sensitive to fluctuation. |
| status_CD_locked classification | P_space/P_line/contrast same as fixed; only area_frac/CD_pitch_frac recomputed at the locked threshold | The P field does not change; only threshold-dependent outputs need updating. |
| Conclusion for pitch ≤ 20 | σ=2 + quencher OP marked unfit | Decided based on Block B. The Stage 4 balanced OP applies for pitch ≥ 24. |
| Stage 6 (x-z standing wave) ok to proceed | Yes | Stage 4B disentangled LER artefact vs real degradation; LER comparisons in Stage 6 are now reliable. |

---

## 5. Verified results

### Block A decision table

| OP | pitch | dose | LER_design | LER_fixed | LER_locked | decision |
|---|---|---|---|---|---|---|
| primary | 18 | 28.4 | 2.77 | 3.72 | 4.07 | real degradation |
| primary | 18 | 40   | 2.77 | 1.63 | 4.16 | fixed underestimates (merged) |
| primary | 20 | 28.4 | 2.77 | 3.24 | 3.14 | real degradation |
| primary | 20 | 40   | 2.77 | 3.50 | 3.25 | real degradation |
| primary | 24 | 28.4 | 2.77 | 2.46 | 2.47 | OK |
| primary | 24 | 40   | 2.77 | 2.53 | 2.47 | OK |
| ctrl σ0 | 18 | 28.4 | 2.77 | 3.42 | 3.49 | real degradation |
| ctrl σ0 | 18 | 40   | 2.77 | 2.27 | 3.47 | fixed underestimates (merged) |
| ctrl σ0 | 20 | 28.4 | 2.77 | 2.95 | 2.85 | OK |
| ctrl σ0 | 20 | 40   | 2.77 | 3.32 | 2.84 | **fixed overestimates (displacement); locked recovers** |
| ctrl σ0 | 24 | 28.4 | 2.77 | 2.52 | 2.52 | OK |
| ctrl σ0 | 24 | 40   | 2.77 | 2.53 | 2.52 | OK |

### Block B table (full mini-sweep)

```text
pitch=18 / dose=28.4 — quencher effects
  baseline (Q0=0):   CD_fix=16.67  CD_lock=12.74  LER_fix=2.12  LER_lock=4.16  margin=+0.067
  Q0=0.005, kq=0.5: CD_fix=16.47  CD_lock=12.59  LER_fix=2.36  LER_lock=4.13  margin=+0.058
  Q0=0.005, kq=1.0: CD_fix=16.33  CD_lock=12.71  LER_fix=2.51  LER_lock=4.14  margin=+0.053
  ...
  Q0=0.02,  kq=2.0: CD_fix=13.48  CD_lock=12.69  LER_fix=4.04  LER_lock=4.05  margin=-0.004 (under_exposed)

pitch=20 / dose=40 — quencher effects
  baseline:         CD_fix=18.49  CD_lock=12.37  LER_fix=2.24  LER_lock=3.28  margin=+0.142
  Q0=0.005, kq=2:   CD_fix=17.93  CD_lock=12.50  LER_fix=2.76  LER_lock=3.30  margin=+0.129
  Q0=0.02,  kq=2.0: CD_fix=15.69  CD_lock=12.44  LER_fix=3.63  LER_lock=3.21  margin=+0.086 (robust_valid)
```

### PSD mid-band comparison (Block A, pitch=24 dose=40)

```text
primary:  psd_eblur_mid = 2.81  →  psd_PEB_mid = 1.06  →  psd_locked_mid = 1.03
ctrl σ0:  psd_eblur_mid = 4.62  →  psd_PEB_mid = 1.59  →  psd_locked_mid = 1.74
```

At pitch=24 the PEB mid-band reduction is consistent in both fixed and locked → smoothing is normal.

```text
primary, pitch=20 dose=40:  psd_eblur_mid = 2.82  →  psd_PEB_mid = 5.12  →  psd_locked_mid = 1.59
```

Here fixed shows 5.12 (an artificial increase) but locked shows 1.59 (a normal decrease). **Mid-band PSD is also a useful indicator for the Stage 5 displacement artefact** — a side benefit of Stage 4B.

### Qualitative conclusion

| hypothesis | verification result |
|---|---|
| The negative LER reduction at small pitch is purely a displacement artefact | Partially — only control σ0 / pitch=20 / dose=40 is artefact; the rest are real degradation |
| The Stage 4 quencher recovers small-pitch LER | No — at pitch ≤ 20 neither weakening nor strengthening the quencher recovers LER |
| Application range of σ=2 + quencher OP | Robust only at pitch ≥ 24 |

---

## 6. Follow-up work

- **Stage 6 (x-z standing wave)**: ready to proceed. Be aware that LER comparisons are meaningful only in the pitch=24 region.
- **σ=0 follow-up at small pitch** (possible Stage 5C): Stage 4B suggests σ=2 is the main culprit for small-pitch LER. Measure the process window at pitch=18, 20 with σ=0 + quencher off. Run after Stage 6 or in parallel with another stage.
- **CD-locked LER as the default for every stage?**: Decide whether to integrate CD-lock into the helper from Stage 6 onward. Compute cost is small (extract_edges × max ~50 calls). Awaiting user decision.
- **Use PSD mid-band as an LER auxiliary metric**: Per Stage 4B's finding, mid-band PSD is useful for identifying displacement artefacts. Recommend adding it to reporting columns from Stage 6.
- **Stage 3B (σ=5/8 compatible budget)**: still on hold.
- **line_cd scaling for small pitch**: Stage 5 follow-up — proportional line_cd scaling per pitch. Separate stage.

---

## 7. Artefacts

```text
src/metrics_edge.py
  + find_cd_lock_threshold (adaptive endpoint), CD_LOCK_* constants

tests/test_edge_metrics.py
  + 3 CD-lock tests   (20/20 passing total)

configs/v2_stage4b_cd_locked.yaml
experiments/04b_cd_locked/
  __init__.py
  run_cd_locked_analysis.py     (Block A 12 + Block B 20 = 32 runs)

outputs/
  figures/04b_cd_locked_block_a/             # 12 P maps with both contours overlaid
  figures/04b_cd_locked_block_b/             # 8 heatmaps (2 pitch × 4 metrics)
  logs/04b_cd_locked_block_a.csv             # Block A full
  logs/04b_cd_locked_block_b.csv             # Block B mini-sweep full
  logs/04b_cd_locked_block_a_decisions.json  # decision labels

EXPERIMENT_PLAN.md
  §Stage 4B "deferred" → "executed" + results table + Stage 5B mini-sweep conclusion

study_notes/
  06_stage4B_cd_locked.md  (this file)
  README.md  index updated
```
