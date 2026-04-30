# Phase 1 — Stage 1 + Stage 1A: clean geometry baseline and the σ-compatible budget

## 0. TL;DR

The original nominal in EXPERIMENT_PLAN §4 (`σ=5 nm, t=60 s`) cannot produce line/space at 24 nm pitch / 12.5 nm CD — the entire domain over-deprotects. Stage 1 baseline was redefined as the **clean-geometry condition (σ=0, t=30)**, and the operating range where σ ≠ 0 is admissible was found by calibration (σ ∈ [0, 3] nm at kdep=0.5, Hmax=0.2).

---

## 1. Goal

### Stage 1 plan-stated goal

```text
At 24 nm pitch / 12.5 nm CD line-space,
build an H0 with initial edge roughness,
take the P>0.5 contour after PEB without quencher,
and quantify CD_shift while ensuring LER_before > LER_after.
```

### Additional questions this phase had to answer

- Do the nominal parameters in plan §4 actually produce a line-space pattern?
- If not, what parameter set is compatible?
- What is the operating range of σ that lets us isolate electron-blur effects per stage?

---

## 2. Steps taken

1. Scaffolded the v2 folder: `configs/`, `src/`, `experiments/01_lspace_baseline/`, `tests/`, `outputs/`.
2. Implemented the core modules
   - `src/geometry.py`: line-space mask + edge roughness
   - `src/roughness.py`: 1D Gaussian-correlated edge perturbation (FFT-based)
   - `src/electron_blur.py`: 2D periodic Gaussian blur
   - `src/exposure_high_na.py`: Dill-style `H0 = Hmax (1 - exp(-η · dose_norm · I))`
   - `src/fd_solver_2d.py`: operator splitting. H, Q use a spectral exact step (`exp(-(D|k|² + k_decay) dt)`), the reaction term uses explicit Euler. P is updated explicitly via `P += dt · k_dep · H · (1-P)`.
   - `src/metrics_edge.py`: per-line left/right edge interpolation, LER/LWR/CD
   - `src/visualization.py`: field/contour overlay
3. Wrote 5 test modules; 14/14 pass.
4. Ran the first baseline using the plan §10 nominal (`σ=5, t=60, DH=0.8, kdep=0.5, Hmax=0.2`).
5. Result was abnormal → proceeded to gate definition / sweeps / calibration.

---

## 3. Problems encountered and resolutions

### Problem 1 — Output-path bug (`parents[2]` misuse)

**Symptom**: After the first baseline run, metrics printed to stdout but no actual files were under `outputs/`. `find` revealed an oddly nested `reaction_diffusion_peb_v2_high_na/reaction_diffusion_peb_v2_high_na/outputs/...` had been created in the wrong place.

**Cause**: The runner script had

```python
V2_DIR = Path(__file__).resolve().parents[2] / "reaction_diffusion_peb_v2_high_na"
```

`parents[2]` is already the `reaction_diffusion_peb_v2_high_na` directory, so appending `/ "reaction_diffusion_peb_v2_high_na"` produces a fake one-level-deeper directory.

**Resolution**:

```python
V2_DIR = Path(__file__).resolve().parents[2]
DEFAULT_OUT = V2_DIR / "outputs"
```

The wrongly-created directory was removed with `rm -rf`.

**Lesson**: Always count `parents[N]` by stepping from `__file__` one level at a time. It is especially confusing when the directory name and the package name are identical.

---

### Problem 2 — Lines and spaces merge under the plan nominal

**Symptom**: With `σ=5, t=60, DH=0.8, kdep=0.5, Hmax=0.2`,

```text
P_min = 0.60   (entire domain has P > 0.5)
area(P>0.5) = 16384 nm² = whole domain
CD_initial = 12.5 nm  →  CD_final = 23.5 nm  (almost the full pitch=24)
LER_final = 0.0  ← false reduction; edges have escaped to the domain boundary
```

This precisely matches the immediate-stop criterion in plan §8 ("entire domain has P > threshold for every dose").

**Cause analysis (four factors acting simultaneously)**:

1. σ=5 nm is large compared to 24 nm pitch.
   The fundamental Fourier-mode attenuation at this pitch is `exp(-(2π/p)²·σ²/2) = exp(-1.71) ≈ 0.18`.
   With duty=12.5/24=0.52 the post-blur intensity sits at `I_blurred ∈ [0.34, 0.70]` — even the space is not zero.
2. `Hmax · η · dose = 0.2` is enough to convert even a weak I into H ≈ 0.058.
3. `DH · t = 0.8 × 60 = 48 nm²` → diffusion length √(2·48) ≈ 9.8 nm, almost equal to the half-pitch (12 nm). Spatial averaging dominates.
4. `kdep · H_min · t ≈ 0.5 × 0.038 × 60 = 1.14` → P_min in the space ≈ 1 - e^-1.14 ≈ 0.68.

In short, the exposure / diffusion / reaction budget exceeds the line-space discrimination capacity (= I_blurred contrast).

**Resolution**: Move on to a σ-sweep aimed at shrinking the budget (which keeps stacking up through Problem 4).

---

### Problem 3 — `P_min` gate falsely passes due to FFT seam artefact

**Symptom**: Sweep over σ ∈ {0,1,2,3,4,5} at t=60:

```text
σ=1, t=60 →  P_min=0.384 (< 0.45 passes)  passed=True
            yet the contour plot shows the lines merged
```

The P_min gate passes but the figure shows no line/space.

**Cause**: Domain `domain_x_nm = 128`, `pitch = 24`. Five lines sit at x = 12, 36, 60, 84, 108 and the next line position is 132, outside the domain. The FFT's periodic BC connects x=128 ↔ x=0, so the seam there has an effective space of 19.5 nm vs. the interior 11.5 nm. P drops below 0.5 only in that wider seam, pulling the global P_min down enough to pass the gate. **In every interior space, P is at or above 0.5** — the lines really are merged.

**Resolution**: Two-step fix.

1. **Align the domain to an integer multiple of the pitch**: `domain_x_nm = 5 × 24 = 120 nm`. The seam coincides exactly with one period.
   Added a helper `ensure_pitch_aligned_domain()` that snaps automatically to the nearest integer multiple.
2. **Replace the gate with interior-only measurements**: instead of a global `P_min`, use the mean/minimum on the mid-strip between lines (`P_space_center_mean`) and the strip mean over the line centre (`P_line_center_mean`). Plus:
   - `P_space_center_mean < 0.50`
   - `P_line_center_mean > 0.65`
   - `contrast = P_line_mean - P_space_mean > 0.15`
   - `area_frac < 0.90`
   - `CD_final / pitch < 0.85` (lines must not merge)

Both changes are required together. Aligning the domain alone leaves the area gate fuzzy; only replacing the gate leaves boundary effects polluting LER measurement.

**Lesson**: For an FFT-based spectral solver **the domain must be an integer multiple of the pattern period**; otherwise some seam artefact is hiding inside one of the metrics.

---

### Problem 4 — Every σ fails at t=60

**Symptom**: With domain alignment + interior gate, the σ sweep at t=60 looked like:

```text
σ=0 → P_space=0.65, area_frac=1.00, CD/pitch=0.98  (fail)
σ=5 → P_space=0.83, area_frac=1.00, CD/pitch=0.98  (fail)
all σ collapse the lines
```

t=60 itself is too long.

**Resolution**: Use a fallback time sweep `[30, 45, 60]` at the σ with the highest contrast.

```text
σ=0, t=30  →  P_space=0.31, P_line=0.76, contrast=0.45, CD/pitch=0.625, passed=True
σ=0, t=45  →  P_space=0.51, area_frac=0.94, CD/pitch=0.93, passed=False
σ=0, t=60  →  P_space=0.65, area_frac=1.00, CD/pitch=0.98, passed=False
```

t=30 is the first passing point.

---

### Problem 5 — Only clean geometry passes (σ=0); σ ≠ 0 keeps failing

**Symptom**: The above (σ=0, t=30) result passes, but the user intent was that electron-blur effects (σ ≠ 0) should also be handled.

**Resolution**: Split via option C.

- Stage 1 = clean-geometry baseline (σ=0, t=30) is the new definition.
- Add Stage 1A: σ-compatible budget calibration.
- Add Stage 1B: keep σ=5/t=60 nominal as an over-budget stress case.

Re-ran the σ sweep with t fixed to 30.

```text
σ=0, t=30 → passed (margin: P_space 0.31, contrast 0.45, area 0.625, CD/p 0.625)
σ=1, t=30 → passed (P_space 0.34, contrast 0.42, area 0.667, CD/p 0.667)
σ=2, t=30 → passed (P_space 0.40, contrast 0.37, area 0.739, CD/p 0.74)
σ=3, t=30 → passed but margin tight (P_space 0.46, area 0.852, CD/p 0.85 ← right at the limit)
σ=4, t=30 → fail (area_frac 0.961)
σ=5, t=30 → fail (contrast 0.16, area 1.00)
```

---

### Problem 6 — σ=5 has no passing budget within the user-specified search space

**Symptom**: σ=5 budget search

```text
Stage A: σ=5 × time ∈ {10,15,20,30} × DH ∈ {0.3,0.8} × kdep=0.5 × Hmax=0.2  → 8 / 8 fail
Stage B: σ=5 × time=20  × DH=0.3      × Hmax ∈ {0.1, 0.15, 0.2}             → 3 / 3 fail
closest miss: σ=5, t=20, DH=0.3, Hmax=0.2 → P_line = 0.630 (0.02 short of the 0.65 gate)
```

**Cause**: At σ=5 / 24 nm pitch,

- the I_blurred amplitude is too small (interior I roughly 0.34–0.70),
- the saturated H0 amplitude with Hmax=0.2 is also small,
- shrinking the kdep=0.5 budget further drops P_line too, failing the line gate,
- raising kdep or dose can lift P_line, but that is outside the search space.

**Conclusion**: Within this search space σ=5 is not compatible. A real σ=5-compatible budget would require one of:

```text
- extend kdep ∈ {0.5, 1.0}
- raise dose_mJ_cm2 ∈ {40, 50, 60} for stronger contrast
- or accept σ_max = 3 as the effective upper bound
```

This decision is revisited just before Stage 3 (electron-blur isolation).

---

### Side problem — LER_initial shrinks as σ grows

**Symptom**: The σ=3 result shows `LER_initial=2.47, LER_final=3.67` — a non-physical "LER grew after diffusion" outcome.

**Cause**: A measurement-convention issue. "Initial" edges are defined as the threshold contour of `I_blurred` (intensity with blur already applied), so as σ grows I_blurred itself is smoother and LER_initial drops. The post-PEB P contour also includes acid diffusion, so the σ effect does not show up cleanly in the LER value.

**Stop-gap**: The gate ignores absolute LER and is judged purely on `P_space`/`P_line`/`contrast`/`area`/`CD/pitch`. LER reduction is reported only.

**TODO (just before Stage 2 or Stage 3)**: Change the "initial edge" definition either to the threshold contour of the binary I (pre-blur), or to a per-σ "no-PEB" reference run that isolates the PEB effect. The PSD analysis in plan §6.4 will revisit this.

---

## 4. Decision log

| Decision | Option chosen | Reason |
|---|---|---|
| Domain size | 128 → 120 nm | Integer multiple of pitch → eliminates the FFT seam artefact. `ensure_pitch_aligned_domain()` snaps automatically. |
| Gate definition | global `P_min < 0.45` → interior gate (P_space/P_line/contrast/area/CD/pitch) | Global P_min can false-pass through boundary effects; only interior measurements certify line-space separation. |
| Stage 1 nominal | (σ=5, t=60) → (σ=0, t=30) | (σ=5, t=60) is incompatible with 24 nm pitch; (σ=0, t=30) has the largest interior-gate margin. |
| (σ=5, t=60) handling | Discard? No — keep as an over-budget stress case ✓ | Useful as a reference data point for "this combination is the abnormal regime"; will be reused in the Stage 5 process-window analysis. |
| When to stop the σ=5 budget search | Do not extend the search space | Within the user spec ("kdep=0.5 first, then Hmax sweep") the failure is conclusive. Re-decide just before Stage 3. |
| σ operating range | σ ∈ [0, 3] nm at (t=30, DH=0.8, kdep=0.5, Hmax=0.2) | From σ=4, area_frac > 0.95 and lines merge. σ=3 also sits at the CD/pitch=0.85 limit, so the practical upper bound is 3. |
| Measurement convention | Knowingly keep using the `I_blurred` initial contour despite its σ dependence | Will be redefined formally in Stage 2/3; changing now would break the gate's coherence. |

---

## 5. Verified results

### Stage 1 baseline (`configs/v2_stage1_clean_geometry.yaml`)

```text
σ = 0 nm
t = 30 s
DH = 0.8 nm²/s
kdep = 0.5 s⁻¹
kloss = 0.005 s⁻¹
Hmax = 0.2 mol/dm³
quencher = off
domain = 120 × 120 nm  (5 pitches)

Outcome:
  H_peak = 0.077, H_min = 0.034
  P_max  = 0.787, P_min = 0.219
  P_space_center_mean = 0.311
  P_line_center_mean  = 0.763
  contrast            = 0.452
  area_frac           = 0.625
  CD_initial = 12.46 nm  →  CD_final = 15.01 nm  (CD_shift = +2.55 nm)
  LER_initial = 2.77 nm  →  LER_final = 2.65 nm  (-4.3 %)
  every interior gate PASS, no NaN, bounds OK
```

### σ-compatible operating range (Stage 1A)

```text
fix:  t=30, DH=0.8, kdep=0.5, Hmax=0.2
σ ∈ [0, 3] nm  → interior gate PASS
σ ≥ 4 nm       → fail (area_frac or contrast)
```

### Over-budget stress (Stage 1B)

```text
σ=5, t=60 (the original plan §4 nominal):
  P_space_mean=0.83, contrast=0.07, area_frac=1.00, CD/pitch=0.98
  → lines collapse into a slab
```

---

## 6. Follow-up work

- **Stage 2** (DH × time sweep): start from the σ=0 baseline. Plan §5.2 can be executed as written.
- **Just before Stage 3** (electron-blur isolation): to keep the σ ∈ {0, 2, 5, 8} intent, σ=5/8 need a compatible budget. Decide whether to run Stage 1A.3 (extend kdep, dose) first or to rewrite the plan with σ_max=3.
- **Measurement convention**: remove the σ dependence of "initial" edges. Two candidates — (a) use the threshold contour of the binary I, (b) use a per-σ no-PEB reference. Decide before Stage 2 begins.
- **PSD-based edge metric**: plan §6.4's edge PSD is not yet implemented; needed for Stage 2's LER-smoothing analysis.

---

## 7. Artefacts

```text
configs/
  v2_stage1_clean_geometry.yaml      # Stage 1 baseline (σ=0, t=30)
  v2_baseline_lspace.yaml            # Stage 1B over-budget reference (σ=5, t=60)

src/
  geometry.py                        # line-space mask + edge roughness
  roughness.py                       # 1D Gaussian-correlated noise
  electron_blur.py                   # 2D Gaussian blur
  exposure_high_na.py                # Dill-style H0
  fd_solver_2d.py                    # operator-split spectral + explicit reaction
  metrics_edge.py                    # LER / LWR / CD / PSD
  visualization.py                   # plot helpers

experiments/
  run_sigma_sweep_helpers.py         # shared helper (interior gate included)
  01_lspace_baseline/
    run_baseline_no_quencher.py      # single baseline run
    run_sigma_sweep.py               # σ sweep + time fallback
    run_calibration_sigma5.py        # σ=5 stage A/B budget search

tests/
  test_geometry.py
  test_exposure_high_na.py
  test_edge_metrics.py
  test_solver_bounds.py
  test_mass_budget.py                # 14 / 14 PASS

outputs/
  figures/01_clean_geometry/         # Stage 1 final figures
  figures/01_sigma_sweep_t30/        # Stage 1A σ sweep
  figures/01_calibration_sigma5_stageA/  # σ=5 budget grid
  figures/01_calibration_sigma5_stageB/  # σ=5 Hmax fallback
  logs/01_*.csv, 01_*.json

EXPERIMENT_PLAN.md
  §5  Stage 1 redefined (clean geometry)
  §5  Stage 1A added (σ-compatible budget calibration)
  §5  Stage 1B added (over-budget reference)
  §9  Stage 1 stop criteria updated
  §10 Stage 1 baseline config updated
  §13 final write-up reflects calibration findings
```
