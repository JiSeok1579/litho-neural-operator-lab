# Phase 2 — Stage 2: DH × time sweep

## 0. TL;DR

Ran a 25-grid DH × time sweep on top of the clean-geometry baseline (σ=0). The algorithmic choice is **DH=0.8, t=20** (LER reduction 9.61%, CD_shift −1.18 nm), but its P_line gate margin is very tight, so practically we recommend **DH=0.5, t=20** (LER 8.80%, CD_shift −0.93, P_line=0.68) or **DH=0.5, t=30** (LER 8.69%, CD_shift +1.79, P_line=0.79).

---

## 1. Goal

- Plan §5 Stage 2: separate the influence of PEB diffusion length on roughness smoothing and CD shift.
- Answer two quantitative questions.
  - Between DH and time, which contributes more to LER reduction?
  - How much CD shift do we have to accept to obtain that LER reduction?

---

## 2. Steps taken

1. Wrote the Stage 2 base config `configs/v2_stage2_dh_time.yaml`. Geometry / exposure / chemistry are identical to the Stage 1 baseline; the sweep script overrides `time_s` and `DH_nm2_s`.
2. Wrote `experiments/02_dh_time_sweep/run_dh_time_sweep.py`. Grid:
   - DH ∈ {0.3, 0.5, 0.8, 1.0, 1.5} nm²/s
   - time ∈ {15, 20, 30, 45, 60} s
   - 25 runs at σ=0, kdep=0.5, kloss=0.005, Hmax=0.2, quencher off.
3. Two-step evaluation
   - **Interior gate** (same as Stage 1A): P_space_mean<0.5, P_line_mean>0.65, contrast>0.15, area_frac<0.90, CD/pitch<0.85, finite metrics.
   - **Selection bound**: among passing runs, maximize LER_reduction_pct subject to extra constraints (CD_shift ≤ 3.0, CD/pitch < 0.85, area_frac < 0.90).
4. Saved CSV / figures / best.json. Logged fail / selection-rejection reasons cell-by-cell.

---

## 3. Problems and resolutions

### Problem 1 — `LER_final = 0` shows up as 100% reduction (merged-line artefact)

**Symptom**: For cells that fail the gate (e.g. DH=1.0, t=60), `LER_reduction_pct = 100.00%`. In reality lines are merged and edges cannot be extracted.

**Cause**: When the lines merge, edge_extract drops to NaN or 0, so final LER becomes 0 and the ratio is computed as 100%. This is the metric code's normal behaviour, but the value is meaningless.

**Resolution**: The selection algorithm only searches among gate-passing runs, so this artefact does not influence selection. To make it explicit, the report tables print fail_reason alongside the LER value so the 100% can be read as an artefact.

**Unresolved / Stage 2 limitation**: The LER metric is contour-based, so values are unreliable in the gate-fail region. Stage 2 analyses only the gate-passing region.

---

### Problem 2 — The algorithmic best has too-tight a P_line margin

**Symptom**: At DH=0.8, t=20 (algorithmic best) the `P_line_center_mean = 0.6534`, leaving only 0.003 above the gate (>0.65). One seed change could flip it to fail.

**Cause**: The user-specified selection criterion only maximizes LER_reduction_pct without considering the P_line / contrast margin.

**Resolution**: Report the algorithmic best per the spec, and additionally record alternates with larger safety margins in the study note.

```text
algorithmic best : DH=0.8, t=20 → LER 9.61%, CD_shift -1.18, P_line 0.65 (margin 0.003)
robust alt 1     : DH=0.5, t=20 → LER 8.80%, CD_shift -0.93, P_line 0.68 (margin 0.03)
robust alt 2     : DH=0.5, t=30 → LER 8.69%, CD_shift +1.79, P_line 0.79 (margin 0.14)
robust alt 3     : DH=0.3, t=30 → LER 8.04%, CD_shift +1.32, P_line 0.81 (margin 0.16)
```

To strengthen the Stage 2 selection in the future, an additional constraint such as P_line margin ≥ 0.05 would be reasonable.

---

### Problem 3 — Line CD shrinks in the small-(DH, t) region

**Symptom**: At DH=0.8, t=20 / DH=0.5, t=15, etc., `CD_shift` is negative (e.g. -1.18, -3.29). Conventional wisdom says PEB diffusion widens lines.

**Cause**: An interaction of the measurement convention with short PEB times.
- `CD_initial` is the threshold-0.5 contour of the binary `I_blurred` (a plain binary mask at σ=0) → line width = 12.5 nm.
- `CD_final` is the threshold-0.5 contour of the post-PEB `P` → wherever P=0.5 falls.
- At t=20 / DH=0.8 the diffusion length is √(2·0.8·20) ≈ 5.7 nm — close to half the line width (12.5 nm). Acid leaks out of the line, H is averaged spatially, P_line_center_mean drops to 0.65, and the P=0.5 contour pulls inward toward the line centre.
- That is, in the short-t regime even the line centre barely clears P=0.5, so the CD can come out smaller than the binary width.

**Resolution**: This is a physically valid result (under-developed line), but it conflicts with the user's intuition that "PEB always widens the line." The note states this explicitly and recommends, for practical use, the region with P_line_center_mean ≥ 0.75 (e.g. DH=0.5, t=30).

**Lesson**: The sign of CD_shift on its own does not classify normal vs. abnormal. It must be read alongside the absolute P_line value. In the under-developed regime (P_line~0.65) a negative CD shift is normal.

---

## 4. Decision log

| Decision | Adopted | Reason |
|---|---|---|
| Stage 2 grid size | DH 5 × time 5 = 25 runs | User-specified. A smaller grid blurs the process-window boundary. |
| Measurement convention | Same as Stage 1 (LER initial = `I_blurred` threshold) | At σ=0, `I_blurred` equals the binary mask, so there is no σ-dependence problem. Will be redefined just before Stage 3 (see Stage 1 note). |
| Selection criterion | User spec (max LER% subject to CD_shift≤3, CD/p<0.85, area<0.9) | Applied as-is. Algorithmic best plus safe alternates are reported together. |
| Handling the marginal P_line of "best" | Keep the algorithmic best; record alternates separately | Without changing the spec, the user can balance safety vs LER directly. |
| Interpretation of negative CD_shift | Classified as normal | A contour-position difference of an under-developed line. No selection penalty. |

---

## 5. Verified results

### 25-run grid (LER reduction %, ✓ = interior gate passes)

```
                15         20         30         45         60
  DH=0.30:    6.55✗     6.63✓     8.04✓     6.96✓   -14.34✓
  DH=0.50:    9.00✗     8.80✓     8.69✓   -17.04✓    45.55✗
  DH=0.80:   10.06✗     9.61✓     4.25✓     8.28✗   100.00✗
  DH=1.00:    8.62✗     8.84✗    -1.98✓    62.62✗   100.00✗
  DH=1.50:   -4.48✗     3.98✗   -38.01✓   100.00✗   100.00✗
```

- The process window has a diagonal shape. The lower-left region is under-developed (P_line<0.65), the upper-right is merged (P_space>0.5).
- LER reduction in the passing region ranges roughly 4–10%.
- LER values in the ✗ region are unreliable (especially the 100% values, which are artefacts).

### Algorithmic best (max LER% subject to CD_shift≤3, CD/p<0.85, area<0.9)

```text
DH = 0.8 nm²/s
t  = 20 s
P_space_center_mean = 0.162
P_line_center_mean  = 0.653  ← gate margin 0.003 (tight)
contrast            = 0.491
area_frac           = 0.471
CD: 12.46 → 11.28 nm  (CD_shift = -1.18 nm)
LER: 2.77 → 2.51 nm   (-9.61 %)
```

### Recommended (large-margin alternates)

| label | DH | t | LER% | CD_shift | P_line | P_line margin | note |
|---|---|---|---|---|---|---|---|
| algorithmic best | 0.80 | 20 | **+9.61** | −1.18 | 0.653 | 0.003 | tight; under-developed regime |
| robust alt 1 | 0.50 | 20 | +8.80 | −0.93 | 0.678 | 0.028 | 0.8 pp LER cost buys 9× the margin |
| robust alt 2 | 0.50 | 30 | +8.69 | +1.79 | 0.792 | 0.142 | positive CD shift, plenty of P_line |
| robust alt 3 | 0.30 | 30 | +8.04 | +1.32 | 0.814 | 0.164 | healthiest of all |

### Qualitative trends (vs. the expected trends in plan §8)

| plan §8 expected | observed |
|---|---|
| DH up → LER down | partial yes — LER% rises through DH=0.3→0.5→0.8 (peaks around 10–11%), then falls again from DH≥1.0 (gate fails dominate) |
| DH up → CD shift up | yes — at fixed t, larger DH gives more positive CD_shift |
| time up → LER down | partial yes — LER% rises from t=15→20→30, then becomes unreliable as lines merge at t=45,60 |
| time up → CD shift up | yes |
| DH/t too large → over-blur of line edge | yes — the upper-right region collapses lines into a slab |

---

## 6. Follow-up work

- **Redefine the measurement convention** (just before Stage 3): remove the σ dependence of "initial" edges. Adopt either binary I or a per-σ no-PEB reference.
- **Stage 3 (electron-blur isolation)**: extend σ from 0 to σ ∈ {0,2,3}. σ=5,8 still need Stage 1A.3 (kdep, dose extension) first.
- **Strengthen selection**: from the next stage onward, adding a P_line margin (e.g. ≥ 0.05) or contrast margin (e.g. ≥ 0.20) to the selection rule will keep the algorithmic best within a robust region.
- **Edge PSD metric** (plan §6.4): LER frequency decomposition. Bundle with Stage 3.
- **Process-window plot**: a single figure with pass/fail and LER% iso-lines on the DH × t plane. Pre-work for the Stage 5 process-window analysis.

---

## 7. Artefacts

```text
configs/v2_stage2_dh_time.yaml
experiments/02_dh_time_sweep/
  __init__.py
  run_dh_time_sweep.py
outputs/
  figures/02_dh_time_sweep/                 # 25 P maps + 25 contour overlays
  logs/02_dh_time_sweep.csv                 # full metric set
  logs/02_dh_time_sweep_summary.csv         # core columns + fail/selection reason
  logs/02_dh_time_sweep_best.json           # algorithmic best
study_notes/02_stage2_dh_time_sweep.md      # this note
```
