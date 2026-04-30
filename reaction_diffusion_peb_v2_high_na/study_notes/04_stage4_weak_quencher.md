# Phase 4 — Stage 4: weak quencher + PSD band metric + Stage 4B deferral

## 0. TL;DR

Ran a 52-run σ × Q0 × kq sweep on the robust OP (DH=0.5, t=30). **All 52 runs passed the Stage-3 gate**. 51/52 are robust candidates. The biggest finding: **at σ=3, Stage 3's PEB-LER deterioration (-22.5%) recovers to +6.6% with a weak quencher** — quencher prevents line widening, so the contour stays near the design edge and the LER measurement is restored. The PSD high band is already 99 %+ removed at baseline, so quencher differences only show up in the low/mid bands.

---

## 1. Goal

Plan §5 Stage 4's nominal goals plus follow-up questions from Stage 3.

```text
1. Does a weak quencher (Q0 ≤ 0.03, kq ≤ 2) reduce the acid tail and CD shift?
2. Does the quencher mitigate the PEB-LER deterioration at higher σ
   (where line widening pulls the contour off the design edge)?
3. In which spatial-frequency band does the quencher's effect appear?
   (PSD low / mid / high decomposition)
```

---

## 2. Steps taken

1. **Added the edge PSD metric** (`src/metrics_edge.py`):
   - `compute_edge_band_powers(edges, dy_nm, bands)` — for each edge track, residual rfft → sum band powers → average across all tracks.
   - Default bands: low [0, 0.05), mid [0.05, 0.20), high [0.20, ∞) nm⁻¹ (≈ λ > 20 nm / 5–20 nm / < 5 nm).
   - `stack_lr_edges` helper — concatenates left+right edges into a (2*n_lines, ny) array.
   - 3 unit tests added (sinusoid concentrate / zero / shape).

2. **Extended the helper** (`experiments/run_sigma_sweep_helpers.py`):
   - Added quencher overrides (`quencher_enabled`, `Q0_mol_dm3`, `DQ_nm2_s`, `kq_s_inv`) to `run_one_with_overrides`.
   - PSD bands are computed automatically at all three stages (design / e-blur / PEB).
   - A single `psd_high_band_reduction_pct = 100*(design_high - PEB_high)/design_high` reduction is also emitted.

3. **Stage 4 sweep run** (`experiments/04_weak_quencher/run_quencher_sweep.py`):
   - Robust OP fixed (DH=0.5, t=30, kdep=0.5, Hmax=0.2, kloss=0.005, pitch=24, CD=12.5).
   - σ ∈ {0,1,2,3} × (1 baseline + 4 Q0 × 3 kq) = 52 runs.
   - Compare each row against the same-σ no-quencher baseline (`dCD_shift_nm`, `darea_frac`, `dtotal_LER_pp`).
   - Stage-3 gate + Stage-4 robust criteria (`P_line_margin ≥ 0.05`, `dCD_shift < 0`, `darea_frac < 0`, `dtotal_LER_pp ≥ -1.0`).
   - Saved 13 σ=2 contour overlays + 4 Q0×kq metric heatmaps.

4. Updated **plan §Stage 4**, with a results table + an explicit deferral of Stage 4B (CD-locked LER).

---

## 3. Problems and resolutions

### Problem 1 — Identifying the σ-driven PEB-LER deterioration from Stage 3

**Symptom (recap from Stage 3)**: `PEB_LER_reduction_pct` worsens monotonically from +8.7% to -37.7% across σ ∈ {0,1,2,3}.

**Stage 4 verification**: At the σ=3 baseline, `total_LER_reduction = -22.5%` (LER grew). Adding a weak quencher (e.g. Q0=0.03, kq=1) restores it to `+6.6%` (dLER = +29.15 pp).

```text
dCD_shift = -3.79 nm   ← line is 3.8 nm narrower than the baseline
darea_frac = -0.159    ← over-deprotect area drops by 16%
dtotal_LER_pp = +29.15 ← LER reduction recovers from -22.5% to +6.6%
```

**Interpretation**:

- The hypothesis from Stage 3 ("PEB does not raise LER per se; line widening pulls the contour off the design edge so the LER measurement becomes inaccurate") is verified.
- When the quencher caps the acid tail, the line narrows and the contour stays near the design edge → LER measurement is restored.
- That is, the negative PEB_LER_reduction in Stage 3 was not a problem with PEB physics but a "contour position vs design-edge position" mismatch.

**Future handling**: Stage 4B's CD-locked LER (auto-adjusted P_threshold) will pin the contour to the design CD, so we can isolate the true PEB-only smoothing. For now we have only the fixed-threshold measurement.

---

### Problem 2 — PSD high band already saturates at ~100% reduction in baseline

**Symptom**: At σ=0,1,2 every row shows `psd_high_band_reduction_pct ≥ 99.9%`. Quencher differences are essentially invisible in the high band.

**Cause**: The high band (f ≥ 0.20 nm⁻¹, λ < 5 nm) of the design noise is a white-noise tail shorter than the edge correlation length (5 nm), so PEB diffusion alone (scale ~ √(2DH·t) = 5.5 nm at the robust OP) wipes it out almost completely. There is no further headroom for the quencher to reduce.

**Interpretation**: The high band is inappropriate for analysing quencher effects. Use the mid band (5–20 nm) or total LER for comparison.

**Resolution**: Documented in the study note and plan. The mid-band reduction will be the main PSD metric in subsequent stages.

---

### Problem 3 — Fixed-threshold comparison is contaminated by σ/quencher CD changes

**Symptom**: When σ or the quencher changes, the line CD changes (e.g. σ=3 baseline CD/p = 0.76, σ=3 + Q0=0.03 + kq=2 → CD/p = 0.55), so the P_threshold=0.5 contour is taken at a different position. Comparing absolute LER values can be misleading.

**Resolution (in this stage)**:

- Replaced absolute comparisons (`LER_after_PEB_P_nm`) with **deltas vs the same-σ baseline** (`dtotal_LER_pp`). Since σ is fixed within each row, this removes the σ effect and leaves only the quencher effect.
- Cross-σ comparisons (e.g. σ=0 vs σ=3) are not done with absolute LER; deferred to the Stage 5 process window or Stage 4B's CD-locked measurement.

**Stage 4B trigger**: Triggered when CD-shift contamination of LER materially changes a decision in Stage 5 or in an external reference comparison.

---

## 4. Decision log

| Decision | Adopted | Reason |
|---|---|---|
| OP choice | robust OP (DH=0.5, t=30) only | The algorithmic-best OP failed every P_line_margin gate in Stage 3. From Stage 4 the algorithmic-best OP is no longer cited. |
| σ range | {0, 1, 2, 3} | Same as Stage 3. σ=5/8 remain on hold under plan §Stage 3B. |
| Q0 range | {0, 0.005, 0.01, 0.02, 0.03} | User-specified. Denser at the low end than the plan's `[0,0.01,0.02,0.03,0.05]`; 0.05 was the v1 issue zone and is excluded. |
| kq range | {0.5, 1.0, 2.0} | User-specified. The lower 3 of plan §4.4 `kq_sweep_safe`. |
| DQ | 0 fixed | Adding quencher diffusion increases the variable count. For Stage 4 the quencher starts spatially uniform and changes only via reaction with H. |
| baseline definition | per-σ no-quencher (Q0=0) | Separates the σ effect from the quencher effect. |
| kq irrelevance at Q0=0 | 1 baseline row only | When Q0=0, kq does not affect the result, so the row count drops to 4 × (1 + 12) = 52. |
| Comparison metrics | dCD_shift, darea_frac, dtotal_LER_pp | User-specified. The PSD high band saturates already at baseline, so it is supplementary. |
| Robust criterion | P_line_margin ≥ 0.05 + dCD<0 + darea<0 + dLER≥-1pp | User-specified. The dLER ≥ -1 pp threshold is the quantitative definition of "materially worsen." |
| dLER threshold | -1.0 pp | An ad-hoc bound for "materially." 1 pp ≈ 12% of the Stage 1 baseline LER reduction (8.7%), the smallest change clearly above noise. |
| σ=2 as primary | User-specified | σ=0 has no e-blur effect, σ=3 baseline is itself abnormal (negative LER). σ=2 is the zone where e-blur, PEB, and quencher all matter. |
| Stage 4B (CD-locked) deferral | Trigger conditions specified, then on hold | The fixed-threshold Stage 4 produced robust enough results; not needed right now. |

---

## 5. Verified results

### Gate / robust statistics

```text
All 52 runs:
  Stage-3 strengthened gate pass : 52 / 52
  Stage-4 robust candidate       : 51 / 52   (only σ=3, Q0=0.03, kq=2 with margin=0.039 < 0.05)
```

### σ=2 (primary): dtotal_LER_pp heatmap

```text
                kq=0.5    kq=1.0    kq=2.0
  Q0=0.030      +4.90     +6.47     +7.64
  Q0=0.020      +3.74     +5.21     +6.44
  Q0=0.010      +2.18     +3.24     +4.23
  Q0=0.005      +1.18     +1.84     +2.49

baseline (Q0=0): total_LER_reduction = +3.56%
adding quencher gives +1.18 pp ~ +7.64 pp of additional reduction.
```

### σ=2 recommended candidates

| label | Q0 | kq | dCD | darea | dLER pp | margin | total LER% (final) |
|---|---|---|---|---|---|---|---|
| balanced | 0.020 | 1.0 | -1.76 | -0.073 | +5.21 | 0.096 | +8.77 |
| max-LER | 0.030 | 2.0 | -3.54 | -0.147 | +7.64 | 0.053 | +11.19 |
| max-margin | 0.005 | 0.5 | -0.29 | -0.012 | +1.18 | 0.132 | +4.74 |

### LER recovery at σ=3 (verification of the Stage 3 finding)

```text
σ=3 baseline (no quencher):
  total_LER_reduction = -22.51%   (LER grows after PEB)
  CD/p = 0.76, area_frac = 0.77
σ=3 + Q0=0.03, kq=1.0:
  total_LER_reduction = +6.64%    (LER drops normally)
  CD/p = 0.61, area_frac = 0.61
  dLER = +29.15 pp, dCD = -3.79 nm, darea = -0.159
```

The quencher prevents line widening, the contour stays near the design edge, and the LER measurement is normalized.

### PSD analysis

```text
psd_high_band_reduction_pct ≈ 99.7% – 100%  (every row, including baseline)
→ high-band noise (λ < 5 nm) is essentially erased by PEB diffusion length (~5 nm).
→ quencher LER differences live in the mid band (low/mid power tracking is more useful for further analysis).
```

PSD low/mid columns are already in `outputs/logs/04_weak_quencher_summary.csv` and will be revisited in Stage 4B or in a PSD-focused analysis.

### Qualitative trends (plan §8 expected vs observed)

| plan §8 expected | observed |
|---|---|
| Q0 or kq up → smaller acid tail | yes — `dCD_shift` decreases monotonically |
| moderate quencher → smaller CD shift | yes — every quencher row has dCD<0 |
| too-strong quencher → smaller Pmax / area | partial — at Q0=0.03, kq=2.0 area drops to 0.49–0.62 but P_line_margin sits at 0.04–0.05, on the edge of robust. The sweep is just shy of the "too-strong" regime. |
| Q0=0.01, kq=1 keeps the P contour | yes — at every σ (margin 0.077 – 0.144) |

---

## 6. Follow-up work

- **Stage 5 (pitch / dose process window)**: start from σ=2 + Q0=0.02, kq=1.0 (balanced), or from the σ=0 baseline. Measure how the process-window shape changes across pitch ∈ {16,18,20,24,28,32}.
- **Stage 4B (CD-locked LER)**: run when the trigger fires. Decide after seeing Stage 5 results.
- **PSD mid-band analysis**: plot the quencher dependence of the low/mid bands separately. The columns are in CSV but no figure exists yet.
- **plan §Stage 3B (σ=5,8 compatible budget)**: still on hold; trigger not met.

---

## 7. Artefacts

```text
src/metrics_edge.py
  + DEFAULT_PSD_BANDS, compute_edge_band_powers, stack_lr_edges

experiments/run_sigma_sweep_helpers.py
  + quencher overrides, PSD band columns

configs/v2_stage4_weak_quencher.yaml
experiments/04_weak_quencher/
  __init__.py
  run_quencher_sweep.py

outputs/
  figures/04_weak_quencher_sigma2/        # 13 contour overlays for sigma=2
  figures/04_weak_quencher_summary/       # 4 Q0×kq heatmaps for sigma=2
  logs/04_weak_quencher_summary.csv       # 52 rows, full metric set
  logs/04_weak_quencher_summary.json

tests/
  test_edge_metrics.py + 3 PSD tests   (17/17 passing)

EXPERIMENT_PLAN.md
  §5 Stage 4 spec updated (sweep, gate, comparison criteria, results)
  §5 Stage 4B (CD-locked LER) deferral added

study_notes/
  04_stage4_weak_quencher.md  (this file)
  README.md  index updated
```
