# Phase 3 — Stage 3: electron-blur isolation + redefined measurement convention

## 0. TL;DR

Split the LER measurement into three stages — design / e-blur / PEB — and ran σ ∈ {0,1,2,3} at the two Stage-2 OPs. The **robust OP (DH=0.5, t=30)** passed the Stage-3 strengthened gate at all four σ; the **algorithmic-best OP (DH=0.8, t=20)** failed at all four σ (P_line_margin < 0.03). As σ grows the LER reduction from electron blur is monotonically increasing (+0% → +11%), but the LER reduction from PEB decreases monotonically and goes negative (+8.7% → −37.7%), so the total is largest at σ=0 (+8.7%). σ=5/8 is split out as plan §Stage 3B.

---

## 1. Goal

- Plan §5 Stage 3: separate the smoothing effect of electron blur from that of PEB acid diffusion, quantitatively.
- Answer two quantitative questions.
  - As electron-blur σ grows, how much does the LER drop already at the H0 stage?
  - On top of that, how much additional LER reduction does PEB add?
- At the same time, clean up the σ-dependence of the measurement convention found in Stages 1/2.

---

## 2. Steps taken

1. **Plan revision** (option b): shrink the Stage 3 sweep from `[0,2,5,8]` → `[0,1,2,3]`. Stage 1A confirmed there is no compatible budget for σ=5,8 within the 24 nm pitch / kdep=0.5 / Hmax≤0.2 spec; the search for a compatible budget is split into plan §Stage 3B (future).
2. **Redefined measurement convention**: added three-stage LER measurement to `run_one_with_overrides` in `run_sigma_sweep_helpers.py`.
   - `LER_design_initial_nm`     ← `extract_edges(I, threshold=0.5)`      (σ-independent)
   - `LER_after_eblur_H0_nm`     ← `extract_edges(I_blurred, 0.5)`
   - `LER_after_PEB_P_nm`        ← `extract_edges(P, 0.5)`
   - The three reduction percentages are reported as well.
3. **Strengthened gate**: always emit `P_line_margin = P_line_center_mean − 0.65`. The Stage 3 sweep script requires `P_line_margin >= 0.03` on top of the existing interior gate (the constraint is applied only inside the sweep script; the helper's `passed` is unchanged).
4. **Sweep**: 2 OPs × 4 σ points = 8 runs.
   - robust OP             : DH=0.5, t=30
   - algorithmic-best OP   : DH=0.8, t=20
5. **Result analysis + figures + CSV/JSON saved**.

---

## 3. Problems encountered and resolutions

### Problem 1 — `_stage3_passed` vs `passed` column-name mismatch → CSV writer error

**Symptom**: The first run died with

```text
ValueError: dict contains fields not in fieldnames: 'passed'
```

**Cause**: `_stage3_passed` was placed in the CSV `fieldnames`, but the row dict did `pop("_stage3_passed")` and then re-inserted as `row["passed"]`, so the keys disagreed.

**Resolution**: Separate `src_keys` (for reading) and `out_keys` (for writing = `src_keys + ["passed", "fail_reason"]`). At row build time, assign explicitly: `row["passed"] = r["_stage3_passed"]`. The JSON dump was reorganized the same way.

**Lesson**: For dict→csv mappings, fill the final-name keys at build time. Pop+rename schemes have to be kept in sync with fieldnames.

---

### Problem 2 — The algorithmic-best OP fails the P_line_margin gate already at σ=0

**Symptom**: At DH=0.8, t=20 with σ=0, `P_line=0.6534, margin=0.003 < 0.03` → fail. σ=1, 2, 3 also fail.

**Cause**: The Stage 2 algorithmic best already sat right on the P_line gate boundary (0.65) with margin 0.003 (the Stage 2 selection rule had no margin condition). As σ grows the line-centre mean drops slightly more, so the margin tightens further or goes negative (at σ=3, P_line=0.641 < 0.65).

**Resolution**: This is the intended outcome of strengthening the Stage-3 gate. The user-specified `P_line_margin >= 0.03` correctly filters out the Stage-2 boundary OP.

**Decision**: Conclude that the algorithmic-best OP is unfit for the Stage 3 σ sweep. From this point on, when later stages cite Stage 2 results, the robust OP is the default.

---

### Problem 3 — As σ grows, the PEB LER reduction goes negative

**Symptom**: At the robust OP

```text
σ=0: PEB_LER_reduction = +8.7%
σ=1: PEB_LER_reduction = +5.7%
σ=2: PEB_LER_reduction = -2.7%
σ=3: PEB_LER_reduction = -37.7%
```

PEB *increases* LER. This contradicts the expected trend in plan §8 ("longer PEB time → smaller LER").

**Cause analysis**:

- A larger electron blur smooths the `I_blurred` edge, so `LER_after_eblur_H0` drops (σ=0:2.77 → σ=3:2.47).
- After PEB, acid spreads further outside the line, the line widens, and the contour ends up further from the design edge (σ=3 has CD_shift=+5.85, CD/p=0.76).
- The new contour is no longer on the acid-ridge but on the outer edge of broad diffusion, so the local noise is relatively larger.
- The std of the post-PEB contour (= LER) therefore exceeds the post-e-blur LER.

In other words, in the high-σ regime PEB does mostly "line widening" rather than "acid diffusion smoothing," so the LER measurement is taken at a different position.

**Resolution**: The data itself is correct. The qualitative interpretation is captured in the study note, and total LER reduction (design → PEB) is used as the comparison reference. Total is maximized at σ=0 (+8.7%).

**Lesson**: Comparing absolute LER values is meaningful only when the contours sit at the same position. With different σ the contour position (CD) differs, so comparing `LER_after_PEB_P` alone is misleading. From the next stage we should compare at equal CD, or work in the PSD domain.

---

### Side problem — σ=2,3 show a non-linear bump in `electron_blur_LER_reduction_pct` relative to σ=1

**Symptom**:

```text
σ=1: e-blur reduction = +2.2%
σ=2: e-blur reduction = +6.1%
σ=3: e-blur reduction = +11.1%
```

Monotonically increasing, but σ and LER reduction are not proportional.

**Cause**: The PSD of the edge roughness reacts non-linearly to a Gaussian filter of width σ. Short-correlation-length (5 nm) noise is essentially erased at σ ≥ 5 nm but only partially attenuated at σ=1,2. PSD analysis would make this explicit.

**Future work**: After implementing the edge PSD metric in plan §6.4, draw the σ-dependent attenuation curve.

---

## 4. Decision log

| Decision | Adopted | Reason |
|---|---|---|
| Stage 3 σ range | [0,2,5,8] → [0,1,2,3] | No compatible budget for σ=5,8 (Stage 1A). Adding σ=1 makes the [0,3] range denser. |
| σ=5/8 handling | Split into plan §Stage 3B (future / optional) | Requires extending the dose, kdep, Hmax search. Starting now would add 24+ runs; defer with explicit trigger conditions. |
| Measurement convention | 3-stage LER measurement (design / e-blur / PEB) | Removes σ dependence and lets us decompose the effects. |
| Threshold for `LER_design_initial` | binary I @ 0.5 | 0.5 is the natural edge for a binary mask. σ-independent. |
| Scope of the Stage-3 strengthened gate | Only inside the sweep script (helper's `passed` unchanged) | Stage 1/2 backward compatibility. The helper stays a reusable component. |
| Algorithmic-best OP handling | Concluded unfit for Stage 3; from now on the robust OP is the default | Fails P_line_margin at every σ. Not used as the OP for downstream stages. |
| Reference baseline for total_LER_reduction | `LER_design_initial` (σ-independent) | Only base that can be compared across σ. |

---

## 5. Verified results

### Robust OP (DH=0.5, t=30) — passes the Stage-3 gate at every σ

| σ (nm) | P_space | P_line | margin | CD_shift | LER_design | LER_eblur | LER_PEB | e-blur% | PEB% | total% | passed |
|--------|---------|--------|--------|----------|------------|-----------|---------|---------|------|--------|--------|
| 0      | 0.210   | 0.792  | 0.142  | +1.79    | 2.772      | 2.772     | 2.531   | +0.0    | +8.7 | **+8.7** | ✓ |
| 1      | 0.245   | 0.794  | 0.144  | +2.62    | 2.772      | 2.711     | 2.556   | +2.2    | +5.7 | +7.8     | ✓ |
| 2      | 0.311   | 0.790  | 0.140  | +3.83    | 2.772      | 2.604     | 2.673   | +6.1    | -2.7 | +3.6     | ✓ |
| 3      | 0.398   | 0.780  | 0.130  | +5.85    | 2.772      | 2.465     | 3.396   | +11.1   | -37.7 | -22.5   | ✓ |

### Algorithmic-best OP (DH=0.8, t=20) — fails at every σ (Stage-3 strengthened gate)

| σ (nm) | P_line | margin | passed | reason |
|--------|--------|--------|--------|--------|
| 0      | 0.653  | 0.003  | ✗      | margin < 0.03 |
| 1      | 0.656  | 0.006  | ✗      | margin < 0.03 |
| 2      | 0.651  | 0.001  | ✗      | margin < 0.03 |
| 3      | 0.641  | -0.009 | ✗      | P_line < 0.65 |

### Qualitative trends (vs. plan §8)

| plan §8 expected | observed (robust OP) |
|---|---|
| σ up → I_blurred edge smoother | yes — `LER_after_eblur_H0` decreases monotonically (2.77 → 2.47) |
| σ up → additional smoothing after PEB | **no** — `LER_after_PEB_P` rises from σ ≥ 2. The reason is "PEB acts mostly as line widening and the contour drifts away from the design edge." |
| total LER is largest at σ=0 | yes — total +8.7% is the maximum |
| σ up → CD shift up | yes — +1.79 → +5.85 |

### Key finding

**The PEB-only smoothing at σ=0 outperforms the combined e-blur + PEB at σ=3.** That is, at narrow pitches such as 24 nm, increasing electron blur makes it harder for PEB to add LER reduction. e-blur and PEB are *not complementary* in this regime — they actually *compete* in part.

---

## 6. Follow-up work

- **Stage 4 (weak quencher)**: start from the robust OP (DH=0.5, t=30, σ=0) as the default. Check whether quencher cuts the acid tail enough to cure the σ-sweep PEB-LER-rise problem.
- **Edge PSD** (plan §6.4): plot σ-dependent attenuation curves in the frequency domain to quantify which spatial frequencies σ removes.
- **Stage 3B** (optional): if the trigger is met, run a `(dose × kdep × Hmax × σ ∈ {5,8} × t × DH)` factorial sample. On hold for now.
- **CD-locked LER comparison** (in Stage 4 or Stage 5): introduce an option to auto-tune `P_threshold` to equalize CD across σ, fixing the "different CD positions" issue.
- **Algorithmic-best OP**: when later stages cite a Stage 2 result, default to the robust OP. The algorithmic best is cited only as a "P_line margin too small" lesson.

---

## 7. Artefacts

```text
configs/v2_stage3_electron_blur.yaml
experiments/03_electron_blur/
  __init__.py
  run_eblur_sweep.py
experiments/run_sigma_sweep_helpers.py     # 3-stage LER measurement added
outputs/
  figures/03_electron_blur/                # 8 P maps + 8 contour overlays
  logs/03_electron_blur.csv                # full metric rows
  logs/03_electron_blur_summary.csv        # core columns + reasons
  logs/03_electron_blur_summary.json       # JSON twin
study_notes/03_stage3_electron_blur.md     # this note
EXPERIMENT_PLAN.md
  §5 Stage 3 sweep [0,2,5,8] → [0,1,2,3]
  §5 Stage 3 measurement convention redefined (3-stage LER)
  §5 Stage 3 gate strengthened (P_line_margin >= 0.03)
  §5 Stage 3B (future) added
```
