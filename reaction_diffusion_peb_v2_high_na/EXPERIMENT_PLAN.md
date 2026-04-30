# High-NA EUV PEB v2 Experiment Plan

## Status — first-pass closeout

| Stage | Status | Note |
|---|---|---|
| 1   | ✅ complete | Stage-1 clean geometry baseline (σ=0, t=30) |
| 1A  | ✅ complete | σ-compatible budget calibration; σ ∈ [0, 3] usable at kdep=0.5 / Hmax≤0.2 |
| 1B  | ✅ complete | over-budget reference (σ=5/t=60 = lines merge) |
| 2   | ✅ complete | DH × time process window (algorithmic best vs robust alternates) |
| 3   | ✅ complete | electron blur decomposition + 3-stage LER measurement convention |
| 3B  | ⏸ deferred | σ=5/8 compatible budget search (search-space expansion required) |
| 4   | ✅ complete | weak quencher 52-run sweep + balanced OP (Q0=0.02, kq=1) |
| 4B  | ✅ complete | CD-locked LER + Stage-5B mini-sweep; CD-locked adopted as helper default |
| 5   | ✅ complete | pitch × dose process window (108 runs, primary + 2 controls) |
| 5C  | ⏸ deferred | σ=0 small-pitch follow-up |
| 6   | ✅ complete | x-z standing wave (12 runs, 3 thickness × 4 amplitude) |
| 6B  | ⏸ deferred | full 3D x-y-z (large compute cost) |

### Recommended v2 operating point (verified robust zone)

```yaml
pitch_nm:           24
line_cd_nm:         12.5
domain_x_nm:        120          # = pitch * 5, FFT-seam-safe
dose_mJ_cm2:        40
electron_blur_sigma_nm: 2
DH_nm2_s:           0.5
time_s:             30
kdep_s_inv:         0.5
Hmax_mol_dm3:       0.2
kloss_s_inv:        0.005
quencher:           Q0=0.02, kq=1.0, DQ=0     # Stage-4 balanced
P_threshold:        0.5                       # the CD-locked variant is also auto-computed by the helper
```

This OP starts from Stage 2's `robust alt 2`, adds σ=2 in Stage 3, balances the quencher in Stage 4, is verified as robust at pitch ≥ 24 in Stage 5, and is confirmed to absorb z-modulation in Stage 6.

### Next milestone

The next step is calibration against literature or measurement data.

```text
- Compare against external reference CD / LER / process-window shapes
- If a systematic offset is found, correct kdep, Hmax, or dose
- After calibration, decide on extensions such as Stage 6B (3D), Stage 3B (σ=5/8), Stage 5C (small pitch)
- Until then, hold off on adding new chemistry or starting new stages
```

For detailed findings see `STUDY_SUMMARY.md` and `study_notes/`.

---

## 0. Purpose

The current `reaction_diffusion_peb/` folder is kept as is.
A new second experiment folder is created, repurposing the existing toy/sandbox PEB model into a **process-oriented experiment based on High-NA EUV line/space patterns**.

Recommended name for the new folder:

```text
reaction_diffusion_peb_v2_high_na/
```

The goals of v2 are as follows.

```text
Under High-NA EUV conditions,
incrementally add line/space pitch, CD, edge roughness, electron blur, film thickness,
z-direction standing wave, and weak quencher conditions
to quantify the impact of PEB on LER/LWR/CD shift.
```

The key questions are as follows.

```text
1. By how much does initial line edge roughness shrink after PEB diffusion?
2. What is the trade-off between roughness smoothing and CD shift?
3. Can electron blur and PEB acid-diffusion blur be separated?
4. When a weak quencher is added, do the acid tail and CD shift decrease?
5. By how much are z-direction film thickness / standing-wave modulation relaxed after PEB?
```

---

## 1. Limitations of the existing v1 folder

The existing `reaction_diffusion_peb/` is successful as a PEB physics sandbox. However, viewed as an actual High-NA EUV process experiment, it has the following limitations.

### 1.1 Geometry is weak

The existing setup mostly uses a synthetic Gaussian spot or a simple line-space exposure map.

```text
Gaussian / simple synthetic I(x,y)
→ H0(x,y)
→ diffusion / reaction
```

But what is actually needed is the following.

```text
pitch
line CD
half pitch
edge roughness amplitude
edge roughness correlation length
LWR / LER metric
CD shift metric
```

In other words, v1 is fine for "does the field diffuse well?", but lacks the geometry information needed to see "how does a High-NA line/space pattern change after PEB?"

### 1.2 No z-direction information

v1's core model is essentially 2D.

```text
H(x,y,t)
Q(x,y,t)
P(x,y,t)
```

As a result, the following phenomena cannot be observed directly.

```text
resist film thickness effect
standing wave along z
top / bottom boundary effect
x-z cross-section striation
```

In v2 we do not jump straight to full 3D; first we set up an `x-z` cross-section experiment as a separate stage.

### 1.3 Exposure uses a normalized toy dose

v1's exposure has roughly the following form.

```text
H0 = Hmax * (1 - exp(-eta * dose * I))
```

Here `dose=1.0` is a normalized dose, not an actual dose in `mJ/cm^2`.
In v2, the actual dose value and the normalized dose are separated.

```text
dose_mJ_cm2
reference_dose_mJ_cm2
dose_norm = dose_mJ_cm2 / reference_dose_mJ_cm2
```

### 1.4 Quencher parameters fall into too strong a regime

The key combination that caused problems in v1 is as follows.

```text
H0_peak ≈ 0.126
Q0 = 0.1
kq >= 5 s^-1
kdep = 0.05 or 0.5 s^-1
P_threshold = 0.5
```

In this combination, acid is captured by quencher too quickly before it can sufficiently deprotect the polymer.
As a result, the `P > 0.5` contour disappears.

In v2 we must proceed in the following order.

```text
1. quencher off baseline
2. weak quencher only
3. medium quencher
4. stiff quencher only as a separate stress-test
```

### 1.5 PINN/FNO are unsuitable as default solvers

In v1, PINN had low quantitative accuracy on diffusion-only and deprotection.
In particular, the Phase 5 PINN had a `P_min < 0` problem.

In v2 we fix the truth solver for the initial experiments to the following.

```text
primary solver: FD / FFT
secondary analysis: optional PINN/FNO only after FD baseline is verified
```

PINN/FNO are not used in early v2 experiments.

---

## 2. Direction of v2 experiments

The goal of v2 is not "to add many phases".
The goal is to **make the geometry and process conditions realistic, and to separate the effect of each physical term**.

Basic v2 strategy:

```text
Step A: change the geometry to real line/space scale.
Step B: add electron blur and initial roughness.
Step C: confirm a no-quencher PEB baseline.
Step D: add a weak quencher.
Step E: add the x-z standing wave experiment.
Step F: perform pitch / dose / DH / time sweeps.
```

Things that should not be added from the start in v2:

```text
strong kq = 100~1000 s^-1
large Q0 = 0.1
full 3D solver
PINN / FNO surrogate
complex development rate model
```

---

## 3. New folder structure

Recommended layout:

```text
reaction_diffusion_peb_v2_high_na/
  README.md
  EXPERIMENT_PLAN.md

  configs/
    v2_baseline_lspace.yaml
    v2_weak_quencher.yaml
    v2_pitch_sweep.yaml
    v2_dose_sweep.yaml
    v2_xz_standing_wave.yaml

  src/
    geometry.py
    roughness.py
    exposure_high_na.py
    electron_blur.py
    metrics_edge.py
    fd_solver_2d.py
    fd_solver_xz.py
    visualization.py

  experiments/
    01_lspace_baseline/
      run_baseline_no_quencher.py
    02_roughness_smoothing/
      run_ler_sweep.py
    03_weak_quencher/
      run_weak_quencher_sweep.py
    04_pitch_dose_sweep/
      run_pitch_sweep.py
      run_dose_sweep.py
    05_xz_standing_wave/
      run_xz_standing_wave.py

  outputs/
    figures/
    logs/
    fields/

  tests/
    test_geometry.py
    test_exposure_high_na.py
    test_edge_metrics.py
    test_solver_bounds.py
    test_mass_budget.py
```

---

## 4. v2 input parameters

The values below organize the uploaded High-NA EUV PEB quantitative parameter table for v2 experiments.

### 4.1 Geometry parameters

```yaml
geometry:
  pattern: line_space

  # Start with one stable baseline before sweeping.
  pitch_nm: 24.0
  half_pitch_nm: 12.0
  line_cd_nm: 12.5

  # Later sweep.
  pitch_sweep_nm: [16, 18, 20, 24, 28, 32]
  half_pitch_sweep_nm: [8, 9, 10, 12, 14, 16]

  # Simulation grid.
  grid_spacing_nm: 0.5
  domain_x_nm: 128.0
  domain_y_nm: 128.0

  # Edge roughness model.
  edge_roughness_enabled: true
  edge_roughness_amp_nm: 1.0
  edge_roughness_corr_nm: 5.0
  edge_roughness_seed: 7
```

The initial baseline starts at `pitch=24 nm`, `line_cd=12.5 nm`.
Do not jump straight to 16 nm pitch. Small pitches have a narrow process window which makes cause isolation difficult.

### 4.2 Exposure / aerial image parameters

```yaml
exposure:
  wavelength_nm: 13.5

  # Actual dose value and normalized conversion.
  dose_mJ_cm2: 40.0
  reference_dose_mJ_cm2: 40.0
  dose_norm: 1.0

  dose_sweep_mJ_cm2: [21, 28.4, 40, 44.2, 59, 60]

  # Dill-style acid generation.
  eta: 1.0
  Hmax_mol_dm3: 0.2

  # EUV electron blur approximation.
  electron_blur_enabled: true
  electron_blur_sigma_nm: 5.0

  # Optional aerial contrast quality metric.
  target_NILS: 1.5
```

Note:

```text
dose_mJ_cm2 is the actual physical-unit value,
dose_norm is the normalized scale fed into the acid-generation formula.
```

Initially we set:

```text
dose_norm = dose_mJ_cm2 / 40.0
```

Later, `eta` or `Hmax` is calibrated against experimental values.

### 4.3 PEB reaction parameters

```yaml
peb:
  time_s: 60.0
  time_sweep: [30, 45, 60, 75, 90]

  temperature_C: 100.0
  temperature_sweep_C: [80, 90, 100, 110, 120]

  DH_nm2_s: 0.8
  DH_sweep_nm2_s: [0.3, 0.8, 1.5]

  kloss_s_inv: 0.005

  # Use stronger deprotection than the weak v1 config.
  kdep_s_inv: 0.5
```

The initial baseline uses `kdep=0.5`.
`kdep=0.05` proved too weak in v1 to form a threshold contour, so it is not used as the v2 default.

### 4.4 Quencher parameters

```yaml
quencher:
  enabled: false

  # Weak quencher stage only.
  Q0_mol_dm3: 0.01
  Q0_sweep_mol_dm3: [0.0, 0.01, 0.02, 0.03, 0.05]

  DQ_nm2_s: 0.0
  DQ_ratio_to_DH: 0.0

  kq_s_inv: 1.0
  kq_sweep_safe_s_inv: [0.5, 1.0, 2.0, 5.0]

  # Stiff values are not baseline. Use only after weak stage works.
  kq_sweep_stiff_s_inv: [100.0, 300.0, 1000.0]
```

Important:

```text
Do not use Q0=0.1 with kq>=5 as a baseline.
Strong quencher destroys the P contour, so it is separated into a stand-alone stress-test.
```

### 4.5 Development / threshold parameters

```yaml
development:
  method: threshold
  P_threshold: 0.5
  P_threshold_sweep: [0.3, 0.4, 0.5, 0.6]

metrics:
  compute_LER: true
  compute_LWR: true
  compute_CD_shift: true
  compute_edge_PSD: true
```

Initially, no full dissolution model is included.
The developable region is defined by the `P > P_threshold` contour.

### 4.6 z direction / standing wave parameters

The z direction is added as a separate stage later in v2.

```yaml
film:
  enabled_z: false
  film_thickness_nm: 20.0
  film_thickness_sweep_nm: [15.0, 20.0, 30.0]
  dz_nm: 0.5

standing_wave:
  enabled: false
  period_nm: 6.75
  amplitude: 0.10
  amplitude_sweep: [0.0, 0.05, 0.10, 0.20]
  phase_rad: 0.0

  # Simple absorption envelope for first implementation.
  absorption_enabled: true
  absorption_length_nm: 30.0
```

Initial x-z exposure model:

```text
I(x,z) = I_xy(x) * [1 + A*cos(2*pi*z/period + phase)] * exp(-z/absorption_length)
```

---

## 5. Experiment stages

## Stage 1 — 2D line/space baseline, no quencher (CLEAN GEOMETRY)

### Purpose

Replace the previous Gaussian toy map with an H0 built from realistic line/space pitch/CD, and confirm that PEB smoothing behaves normally without a quencher.

### Important — why the nominal changed

Calibration confirmed that the original nominal in §4.2 / §4.3 (`electron_blur_sigma_nm=5, time_s=60`) **cannot pass the interior gate** at 24 nm pitch / 12.5 nm CD (see Stage 1A below).
- σ=5, t=60 → P_space_center_mean=0.83, area_frac=1.0, CD_final≈pitch (lines fully merge).
- For σ=5, the entire t={10,15,20,30} × DH={0.3,0.8} × kdep=0.5 × Hmax={0.1,0.15,0.2} grid fails.
- Therefore σ=5/t=60 is **not** the Stage 1 baseline. It is moved to the §6.4 over-budget stress case.

The Stage 1 baseline is redefined as a **clean geometry** condition with electron-blur effects removed.
Electron-blur effects are evaluated separately in Stage 3 together with the σ-compatible budget.

### Setup (clean geometry baseline)

```yaml
geometry.pitch_nm: 24
geometry.line_cd_nm: 12.5
geometry.domain_x_nm: 120          # 5 * pitch — pitch-aligned (prevents FFT seam artifact)
geometry.domain_y_nm: 120
exposure.dose_mJ_cm2: 40
exposure.electron_blur_sigma_nm: 0  # NO e-blur for Stage 1
peb.DH_nm2_s: 0.8
peb.time_s: 30                      # half of original nominal — at 60 s every σ merges
peb.kdep_s_inv: 0.5
peb.kloss_s_inv: 0.005
quencher.enabled: false
development.P_threshold: 0.5
```

config: `configs/v2_stage1_clean_geometry.yaml`

### Outputs

```text
H0(x,y)
H_final(x,y)
P_final(x,y)
P_threshold contour
initial edge contour
final edge contour
LER_before / LER_after
CD_before / CD_after
```

### Success criteria — interior gate (seam-artifact-resistant)

```text
H >= 0
0 <= P <= 1
interior P_space_center_mean < 0.50    # the strip-mean between lines is below threshold
interior P_line_center_mean  > 0.65    # the strip-mean at line center is above threshold
contrast = P_line_mean - P_space_mean > 0.15
area_frac (P>=threshold) < 0.90        # prevents whole-domain over-deprotection
CD_final / pitch < 0.85                # lines do not merge
CD shift and LER are measurable
```

Global P_min is not used. If the domain is not an integer multiple of the pitch, the FFT-seam wider-space artifact lowers P_min artificially.

### Verified results (σ=0, t=30)

```text
P_space_center_mean = 0.31
P_line_center_mean  = 0.76
contrast            = 0.45
area_frac           = 0.625
CD_initial / final  = 12.46 / 15.01 nm  (CD_shift = +2.55 nm)
CD/pitch            = 0.625
LER_initial / final = 2.77 / 2.65 nm
all interior gates  = PASS
```

---

## Stage 1A — σ-compatible exposure / PEB budget calibration

### Purpose

To run the §4.2 sweep `electron_blur_sigma_nm: [0,2,5,8]` meaningfully, each σ needs a (t, DH, Hmax, kdep) budget that preserves line-space separation.
With the original `kdep=0.5, time=60, DH=0.8, Hmax=0.2`, lines merge at σ≥4 on 24 nm pitch, making the σ-sweep impossible.
Stage 1A finds a compatible budget per σ.

### Procedure

```text
Stage 1A.1  σ sweep at fixed t=30, DH=0.8, kdep=0.5, Hmax=0.2:
              σ ∈ {0,1,2,3,4,5}
              gate = interior gate above, unchanged
              result: σ ∈ {0,1,2,3} pass, σ ∈ {4,5} fail.

Stage 1A.2  σ=5 budget search (within existing spec range):
              time × DH grid: time ∈ {10,15,20,30}, DH ∈ {0.3,0.8}, kdep=0.5
              if needed, Hmax sweep ∈ {0.1,0.15,0.2} at the highest-contrast (t,DH)
              result: complete failure (σ=5,t=20,DH=0.3,Hmax=0.2 came closest with cond_line 0.630<0.65).
              conclusion: no σ=5 compatible budget within the search range.
              interpretation: at σ=5/24 nm pitch, I_blurred contrast is too weak.

Stage 1A.3  expand search space if needed:
              kdep ∈ {0.5, 1.0}                 # P_line lift
              dose_mJ_cm2 ∈ {40, 50, 60}        # H0 contrast lift
              or fix the effective upper bound of σ at σ_max=3.
```

### Stage 1A observations

| σ (nm) | t (s) | DH | Hmax | P_space | P_line | contrast | area_frac | CD/p | passed |
|--------|-------|-----|------|---------|--------|----------|-----------|------|--------|
| 0      | 30    | 0.8 | 0.2  | 0.31    | 0.76   | 0.45     | 0.625     | 0.63 | ✅ |
| 1      | 30    | 0.8 | 0.2  | 0.34    | 0.77   | 0.42     | 0.667     | 0.67 | ✅ |
| 2      | 30    | 0.8 | 0.2  | 0.40    | 0.76   | 0.37     | 0.739     | 0.74 | ✅ |
| 3      | 30    | 0.8 | 0.2  | 0.46    | 0.76   | 0.30     | 0.852     | 0.85 | ✅ (limit) |
| 4      | 30    | 0.8 | 0.2  | 0.53    | 0.75   | 0.22     | 0.961     | 0.94 | ❌ |
| 5      | 30    | 0.8 | 0.2  | 0.58    | 0.73   | 0.16     | 1.000     | 0.98 | ❌ |
| 5      | 20    | 0.3 | 0.2  | 0.38    | 0.63   | 0.25     | 0.598     | 0.60 | ❌ (P_line 0.02 low) |

### Stage 1A decision

```text
σ-compatible operating range at 24 nm pitch = σ ∈ [0, 3] nm (within kdep=0.5, Hmax=0.2 spec)
σ ≥ 4 is revisited after expanding the budget search space (Stage 1A.3).
σ=5 and σ=5/t=60 are moved to the §6.4 over-budget stress case.
```

---

## Stage 1B — over-budget stress reference (σ=5, t=60)

### Purpose

Explicitly record that the original nominal in plan §4 causes line merge.
Used in process-window evaluation as a "this combination is in the abnormal region" reference.

### Setup

config: `configs/v2_baseline_lspace.yaml` (the header carries an OVER-BUDGET warning)

### Observed results

```text
P_space_center_mean = 0.83
P_line_center_mean  = 0.90
contrast            = 0.07
area_frac           = 1.000
CD_final / pitch    = 0.98
All interior gates fail. Lines merge into a single slab.
```

This result is used as a data point for the trend "high-σ × long-t × strong-kdep combinations cause lines to collapse" in the Stage 5 process-window analysis.

---

## Stage 2 — DH / PEB time smoothing sweep

### Purpose

Separate the contributions of PEB diffusion length to roughness and to CD shift.

### Sweep

```yaml
DH_sweep_nm2_s: [0.3, 0.5, 0.8, 1.0, 1.5]    # actually executed: 5 points (3 from the original plan + 0.5/1.0 added)
time_sweep:     [15, 20, 30, 45, 60]          # actually executed: lower half added
```

### Trends to look for

```text
DH increases → LER decreases
DH increases → CD shift increases
PEB time increases → LER decreases
PEB time increases → CD shift increases
Excessively large DH/time → line edges blur too much
```

### Verified results

config: `configs/v2_stage2_dh_time.yaml` (Stage 1 baseline as is, only the sweep variables override it)

```text
LER reduction (%) — DH (rows) × time (cols), ✓ = interior gate pass

                15         20         30         45         60
  DH=0.30:    6.55✗     6.63✓     8.04✓     6.96✓   -14.34✓
  DH=0.50:    9.00✗     8.80✓     8.69✓   -17.04✓    45.55✗
  DH=0.80:   10.06✗     9.61✓     4.25✓     8.28✗   100.00✗
  DH=1.00:    8.62✗     8.84✗    -1.98✓    62.62✗   100.00✗
  DH=1.50:   -4.48✗     3.98✗   -38.01✓   100.00✗   100.00✗

LER% in the ✗ region is not reliable (lines merged → edge extraction NaN/0).
100% values in ✗ rows are artifacts; only ✓ rows are meaningful.
```

Algorithmic best (max LER% subject to CD_shift ≤ 3, CD/p < 0.85, area_frac < 0.9):
**DH=0.8 nm²/s, t=20 s** — LER reduced by 9.61%, CD_shift = −1.18 nm, P_line=0.65 (margin 0.003).

Practical recommendation (candidates with larger P_line margin): **DH=0.5, t=20** (LER 8.80%, P_line=0.68) or **DH=0.5, t=30** (LER 8.69%, P_line=0.79).

For detailed analysis see `study_notes/02_stage2_dh_time_sweep.md`.

---

## Stage 3 — electron blur decomposition experiment

### Purpose

Separate EUV secondary-electron blur from PEB acid-diffusion blur.

### Sweep (revised)

```yaml
electron_blur_sigma_nm: [0, 1, 2, 3]      # range compatible with 24 nm pitch / kdep=0.5 / Hmax<=0.2
DH_nm2_s, time_s: the two operating points from Stage 2
                  - robust OP             : DH=0.5, t=30
                  - algorithmic-best OP   : DH=0.8, t=20
```

In the original plan `[0, 2, 5, 8]`, σ=5 and σ=8 are not compatible with the 24 nm pitch / kdep=0.5 / Hmax≤0.2 budget (Stage 1A's full search space fails). Finding a compatible budget requires raising dose / kdep / Hmax, which is moved into a separate stage (Stage 3B).

### Measurement convention (redefined)

The "initial edge" definition in Stage 1/2 has σ-dependence (`I_blurred` smooths out as σ grows), making it unfair across a σ-sweep. From Stage 3, LER is measured in three separated stages.

```text
LER_design_initial    : threshold 0.5 contour of binary I (before blur)
LER_after_eblur_H0    : threshold 0.5 contour of I_blurred (after blur, before PEB)
LER_after_PEB_P       : threshold 0.5 contour of P (after PEB)

electron_blur_LER_reduction_pct = 100 * (LER_design_initial - LER_after_eblur_H0) / LER_design_initial
PEB_LER_reduction_pct           = 100 * (LER_after_eblur_H0 - LER_after_PEB_P)   / LER_after_eblur_H0
total_LER_reduction_pct         = 100 * (LER_design_initial - LER_after_PEB_P)   / LER_design_initial
```

`LER_design_initial` is σ-independent, so it is a consistent baseline across the σ-sweep.

### Gate (Stage 3, tightened)

The Stage 1/2 interior gate plus an added `P_line_margin >= 0.03`.

```text
P_space_center_mean < 0.50
P_line_center_mean  > 0.65
P_line_margin = P_line_center_mean - 0.65 >= 0.03   # newly added
contrast > 0.15
area_frac < 0.90
CD_final / pitch < 0.85
CD_final, LER_after_PEB_P are finite
```

The P_line_margin gate prevents the issue from Stage 2 where the algorithmic best (DH=0.8, t=20) sat right on the boundary at P_line=0.6534.

### Analysis

```text
Electron-blur effect alone (before PEB): I_blurred → LER reduction = electron_blur_LER_reduction_pct
PEB effect alone (after electron blur):  I_blurred → P → LER reduction = PEB_LER_reduction_pct
Combined effect:                         binary I  → P → LER reduction = total_LER_reduction_pct
```

### Success criteria

```text
σ increases → LER_after_eblur_H0 decreases (primary smoothing from electron blur)
σ increases → LER_after_PEB_P decreases or stays similar (PEB adds further smoothing)
The two effects can be separated into independent columns
At least one gate-passing σ exists per OP
```

---

## Stage 3B — σ=5/8 compatible budget search (future / optional)

### Purpose

Stage 1A confirmed that no compatible budget for `σ=5` exists within the spec range (`kdep=0.5`, `Hmax∈{0.1,0.15,0.2}`). Stage 3's results allow the separated analysis for σ ∈ {0,1,2,3}, but real High-NA EUV e-beam blur is around σ ≈ 5 nm as reference. Recovering this σ value requires expanding the search space.

### Whether to proceed is decided after seeing the Stage 3 results

Trigger conditions (Stage 3B starts if any of the following is satisfied):

```text
- Stage 3's electron_blur_LER_reduction_pct trend up to σ=3 cannot estimate the σ=5/8 effect
- PSD decomposition cannot quantify the high-frequency cutoff with only σ=3
- A comparison with external references (literature / measurement) at σ=5 is essential
```

### Search space (Stage 3B)

```text
variable 1: dose_mJ_cm2          ∈ {40, 50, 60}    # boost H0 contrast
variable 2: kdep_s_inv           ∈ {0.5, 1.0}      # P_line lift
variable 3: Hmax_mol_dm3         ∈ {0.1, 0.15, 0.2}
variable 4: σ                    ∈ {5, 8}
variable 5: time_s               ∈ {15, 20, 30}
variable 6: DH_nm2_s             ∈ {0.3, 0.5, 0.8}
```

The full grid (3×2×3×2×3×3 = 324) is too large, so it is reduced by factorial design or Latin-Hypercube sampling.

### Success criteria (Stage 3B)

The Stage 3 interior gate (including P_line_margin) is passed by **at least one** (dose, kdep, Hmax, t, DH) combination for both σ=5 and σ=8.

---

## Stage 4 — weak quencher experiment

### Purpose

Re-introduce the quencher that was too strong in v1, but weakly. Verify whether it reduces the acid tail and suppresses CD shift, and whether it relaxes the σ-induced PEB LER degradation found in Stage 3 (line widening at high σ → the contour drifts away from the design edge, increasing LER).

### Operating point

Use **only** the robust OP from Stage 2. The algorithmic-best OP from Stage 3 fails the P_line_margin gate for every σ, so it is not adopted downstream.

```yaml
DH_nm2_s    : 0.5
time_s      : 30
kdep_s_inv  : 0.5
Hmax_mol_dm3: 0.2
kloss_s_inv : 0.005
pitch_nm    : 24
line_cd_nm  : 12.5
```

### Sweep (revised)

```yaml
sigma_nm       : [0, 1, 2, 3]            # primary analysis: sigma=2
Q0_mol_dm3     : [0.0, 0.005, 0.01, 0.02, 0.03]   # 0.0 = no-quencher baseline
kq_s_inv       : [0.5, 1.0, 2.0]
DQ_nm2_s       : 0.0
quencher_enabled : Q0=0 → false, Q0>0 → true
```

Total 4 × (1 + 4×3) = 52 runs.

### Gate (same as Stage 3)

```text
P_space_center_mean < 0.50
P_line_center_mean  > 0.65
P_line_margin       >= 0.03
contrast > 0.15
area_frac < 0.90
CD_final / pitch < 0.85
CD_final, LER_after_PEB_P are finite
```

### Stage 4 comparison criteria (vs same-σ no-quencher baseline)

```text
dCD_shift_nm      < 0     # quencher reduces line widening
darea_frac        < 0     # quencher reduces over-deprotect area
dtotal_LER_pp     >= -1.0 # total LER reduction must not drop by more than 1pp
P_line_margin     >= 0.05 # extra safety margin for robust candidates
```

Rows that satisfy all of the above are flagged as robust candidates.

### Measurement (same as Stage 3 measurement convention + PSD)

```text
LER_design_initial  / LER_after_eblur_H0  / LER_after_PEB_P
electron_blur_LER_reduction_pct
PEB_LER_reduction_pct
total_LER_reduction_pct

PSD bands (default):
  low  : f ∈ [0,    0.05) nm^-1   (λ > 20 nm, long-range wiggle)
  mid  : f ∈ [0.05, 0.20) nm^-1   (λ 5–20 nm, main correlation regime)
  high : f ∈ [0.20, ∞)   nm^-1   (λ < 5 nm, sub-correlation noise)
psd_high_band_reduction_pct = 100*(design_high - PEB_high)/design_high
```

### Verified results

config: `configs/v2_stage4_weak_quencher.yaml`

**All 52 runs pass the Stage-3 gate.** Only one row, σ=3, Q0=0.03, kq=2, falls short of the robust margin (P_line_margin = 0.039 < 0.05).

Effect of introducing the quencher (vs σ-matched baseline):

```text
For every σ:
  dCD_shift   < 0   (line widening decreases)
  darea_frac  < 0   (over-deprotect area decreases)
  dtotal_LER ≥ 0   (LER reduction is greater than or equal to the baseline)

In particular at σ=3 (the case where Stage 3 saw PEB worsen LER to -22.5%):
  Q0=0.03, kq=1.0  →  total LER reduction = +6.64% (baseline -22.5%, dLER=+29.15pp)
  Q0=0.02, kq=2.0  →  total LER reduction = +5.95% (dLER=+28.47pp)
  → the quencher prevents line widening, so the contour stays near the design edge
    and the LER measurement is normalised.

σ=2 (primary analysis) dtotal_LER_pp heatmap (Q0 rows, kq cols):
              kq=0.5  kq=1.0  kq=2.0
  Q0=0.030    +4.90   +6.47   +7.64
  Q0=0.020    +3.74   +5.21   +6.44
  Q0=0.010    +2.18   +3.24   +4.23
  Q0=0.005    +1.18   +1.84   +2.49
```

PSD analysis:
- Every row shows `psd_high_band_reduction_pct ≈ 99–100%` — high-frequency edge noise is essentially fully removed by PEB even in the baseline.
- The LER differences originate in the low/mid band (long-range wiggle, main correlation regime).
- That is, the quencher's effect comes from suppressing the mid-frequency line wiggle, not the high-frequency noise.

Recommended σ=2 robust candidates:

```text
Q0=0.02, kq=1.0  (balanced)         : dCD=-1.76, darea=-0.073, dLER=+5.21pp, margin=0.096
Q0=0.03, kq=2.0  (max LER reduction): dCD=-3.54, darea=-0.147, dLER=+7.64pp, margin=0.053
```

For detailed analysis see `study_notes/04_stage4_weak_quencher.md`.

### Failure criteria (preserving the original plan's definition)

```text
If the P>0.5 contour disappears starting from Q0=0.01, kq=1,
then one of Hmax/dose/kdep is too weak or Q0 is still too strong
→ this did not occur in this sweep (the P contour is healthy in every row).
```

### Stage 4B — CD-locked LER (executed)

The LER comparisons in Stages 3/4 are measured on the fixed P_threshold=0.5 contour, so when σ or the quencher changes CD, the contour location shifts with it and LER is measured at a different place. The negative-LER-reduction artifact at pitch ≤ 20 in Stage 5 triggered the CD-lock re-measurement.

#### Measurement convention

```text
Bisect P_threshold within [0.2, 0.8] to match CD_overall_mean ≈ CD_initial (tol 0.25 nm).
If an endpoint is contour-empty, narrow inward in 0.05 steps.
status ∈ ok / unstable_low_bound / unstable_high_bound / unstable_no_crossing / unstable_no_converge.
```

#### Block A results (12 runs)

| OP | pitch | dose | LER_design | LER_fixed | LER_locked | decision |
|---|---|---|---|---|---|---|
| primary | 18 | 28.4 | 2.77 | 3.72 | 4.07 | real degradation |
| primary | 18 | 40   | 2.77 | 1.63 | 4.16 | fixed underestimates (merged-line artifact) |
| primary | 20 | 28.4 | 2.77 | 3.24 | 3.14 | real degradation |
| primary | 20 | 40   | 2.77 | 3.50 | 3.25 | real degradation |
| primary | 24 | 28.4 | 2.77 | 2.46 | 2.47 | OK |
| primary | 24 | 40   | 2.77 | 2.53 | 2.47 | OK |
| ctrl σ0 | 18 | 28.4 | 2.77 | 3.42 | 3.49 | real degradation |
| ctrl σ0 | 18 | 40   | 2.77 | 2.27 | 3.47 | fixed underestimates (merged-line artifact) |
| ctrl σ0 | 20 | 28.4 | 2.77 | 2.95 | 2.85 | OK |
| ctrl σ0 | 20 | 40   | 2.77 | 3.32 | 2.84 | **fixed overestimates (displacement artifact); locked recovers** |
| ctrl σ0 | 24 | 28.4 | 2.77 | 2.52 | 2.52 | OK |
| ctrl σ0 | 24 | 40   | 2.77 | 2.53 | 2.52 | OK |

#### Key findings

```text
1. LER values at pitch=24 (both fixed and locked ≈ design) → PEB smoothing is normal.
2. Only the control σ=0 / pitch=20 / dose=40 row shows a displacement artifact (fixed=3.32, locked=2.84).
   This artifact accounts for part of the "negative LER reduction at pitch=20" seen in Stage 5.
3. However, the primary OP (σ=2 + Q0=0.02 + kq=1) at pitch=20 still has LER = 3.14–3.25
   even after CD-lock — higher than design 2.77 → diagnosed as "real roughness degradation".
4. At pitch=18, dose=40 the lines are fully merged so the fixed LER is artificially low.
   CD-lock moves the contour back onto the real line position and gives LER=4.16, which is correct.
```

→ **The Stage 4 quencher actually worsens LER at pitch=20.** The Stage 4 conclusion that "the quencher rescues σ-induced LER degradation" only holds at pitch=24. This is the same mechanism as the small-pitch process-window shrinkage seen in Stage 5.

#### Block B — pitch-dependent weak-quencher mini-sweep (Stage-5 follow-up)

**Purpose**: at pitch ∈ {18, 20}, verify whether weakening the quencher (Q0 ≤ 0.005, kq ≤ 0.5) recovers LER_locked.

**Setup**: σ=2, DH=0.5, t=30. Two (pitch, dose) pairs (pitch=18, dose=28.4) + (pitch=20, dose=40) × (Q0=0 baseline + Q0 ∈ {0.005, 0.01, 0.02} × kq ∈ {0.5, 1.0, 2.0}) = 20 runs.

**Result summary**:

```text
pitch=18 / dose=28.4:
  baseline (no q):  LER_lock = 4.16
  best (Q0=0.005, kq=0.5): LER_lock = 4.13   (ΔLER_lock = -0.02 nm, almost no recovery)
  strongest (Q0=0.02, kq=2.0): LER_lock = 4.05   (ΔLER_lock = -0.11)
  → in every quencher setting LER_lock is more than +1.3 nm above design 2.77 → cannot be recovered.

pitch=20 / dose=40:
  baseline (no q):  LER_lock = 3.28
  best (Q0=0.02, kq=2.0): LER_lock = 3.21   (ΔLER_lock = -0.08)
  → recovery is small. design 2.77 cannot be reached.

→ The LER degradation at pitch ≤ 20 cannot be recovered by the quencher. This is a different regime than 24 nm pitch.
```

**Conclusion**: the LER degradation at pitch=18, 20 is (a) dominated by e-blur σ=2, and (b) not recovered by weakening the quencher. The Stage 4 balanced OP is recommended only at pitch ≥ 24.

config: `configs/v2_stage4b_cd_locked.yaml`
detailed analysis: `study_notes/06_stage4B_cd_locked.md`

---

## Stage 5 — pitch / dose process window

### Purpose

Confirm whether the balanced operating point from Stage 4 holds across pitch / dose.
For each pitch, quantify the size of the robust_valid dose window and the recommended dose.

### Operating point (primary)

```yaml
sigma_nm    : 2.0
DH_nm2_s    : 0.5
time_s      : 30
kdep_s_inv  : 0.5
kloss_s_inv : 0.005
Hmax_mol_dm3: 0.2
quencher    : Q0=0.02, kq=1.0, DQ=0    # Stage 4 balanced
line_cd_nm  : 12.5                       # fixed across all pitches
```

### Sweep

```yaml
pitch_sweep_nm:    [16, 18, 20, 24, 28, 32]
dose_sweep_mJ_cm2: [21, 28.4, 40, 44.2, 59, 60]
domain_x_nm  = pitch_nm * 5     # n_periods_x = 5
domain_y_nm  = 120              # fixed (so the LER y-sample count is constant)
```

Total 36 runs (primary). With additional controls: 36 + 36 = 108 runs.

### Optional controls

```text
control_sigma0_no_q : sigma=0, quencher disabled  → σ-independent measurement-convention reference
control_sigma2_no_q : sigma=2, quencher disabled  → isolates the quencher-only effect
```

### Per-run classification (in precedence order)

```text
unstable      : NaN / Inf / bounds violation / contour-extraction failure
merged        : P_space_mean >= 0.50  OR  area_frac >= 0.90  OR  CD/pitch >= 0.85
under_exposed : P_line_mean  < 0.65
low_contrast  : contrast      <= 0.15  (rarely appears; 0 cases in this sweep)
valid         : passes the interior gate but lacks margin
robust_valid  : passes the interior gate AND P_line_margin >= 0.05
```

### Recommended-dose selection algorithm

For each pitch:

```text
1. prefer the robust_valid pool
2. otherwise use the valid pool
3. minimise |CD_shift_nm| within the pool
4. break ties by maximising total_LER_reduction_pct
5. further break ties by maximising P_line_margin
```

### Verified results (primary OP)

config: `configs/v2_stage5_pitch_dose.yaml`

```text
status heatmap (primary):
                21    28.4    40    44.2    59     60
  pitch=32     unde   vali   robu   robu   robu   robu
  pitch=28     unde   vali   robu   robu   robu   robu
  pitch=24     unde   vali   robu   robu   robu   robu
  pitch=20     unde   vali   robu   merg   merg   merg
  pitch=18     unde   vali   merg   merg   merg   merg
  pitch=16     unde   merg   merg   merg   merg   merg

→ At pitch=16, no dose yields a valid region — the process window is closed.
→ At pitch ≥ 24, four dose points are robust_valid (workable window).
```

Recommended dose per pitch:

```text
pitch=16: no recommendation (process window closed)
pitch=18: dose=28.4 (valid only, margin 0.012, CD_shift +1.99, LER -34.18%)
pitch=20: dose=40   (robust_valid, margin 0.098, CD_shift +4.10, LER -26.22%)
pitch=24: dose=40   (robust_valid, margin 0.096, CD_shift +2.07, LER  +8.77%)
pitch=28: dose=40   (robust_valid, margin 0.095, CD_shift +1.81, LER +14.33%)
pitch=32: dose=40   (robust_valid, margin 0.095, CD_shift +1.78, LER +15.03%)
```

The negative LER reductions at pitch ≤ 20 are the contour-displacement artifact diagnosed in Stages 3 / 4 (line widening pulls the contour off the design edge). It will be corrected in Stage 4B (CD-locked LER), but the robust_valid classification itself is correct.

### Control comparison (vs primary)

```text
control_sigma0_no_q: only pitch=16 has a closed window; pitch ≥ 18 is mostly robust_valid
control_sigma2_no_q: process window is narrow up to pitch=20, mostly robust_valid at pitch ≥ 24
primary (sigma=2 + Q0=0.02, kq=1): only one point (pitch=20, dose=40) is robust_valid
```

Interesting finding: **adding the quencher narrows the small-pitch process window.** This is a trade-off with the Stage 4 LER improvement (less line widening = contour pulled in tighter, hitting the merge threshold sooner at small pitch).

### Success criteria

```text
✓ stable contour at 20~24 nm pitch → primary has robust_valid at pitch=20 (dose=40) and pitch=24 (4 doses)
✓ process window narrows at 16~18 nm pitch → pitch=16 closed, pitch=18 valid at only one dose
✓ CD widening / over-deprotection at high dose → merged at pitch ≤ 20, dose ≥ 44
✓ under-deprotection at low dose → under_exposed at every pitch for dose=21
```

For detailed analysis see `study_notes/05_stage5_pitch_dose.md`.

---

## Stage 6 — x-z standing wave experiment (executed)

### Purpose

Verify whether z-direction film thickness and standing-wave modulation are relaxed after PEB. The robust OP at pitch=24 (Stage 4 balanced + Stage 5 verified) is used as is.

### Operating point (fixed)

```yaml
pitch_nm    : 24
line_cd_nm  : 12.5
sigma_nm    : 2          # e-blur (1D, along x)
DH_nm2_s    : 0.5
time_s      : 30
kdep_s_inv  : 0.5
kloss_s_inv : 0.005
Hmax        : 0.2
quencher    : Q0=0.02, kq=1.0, DQ=0
```

### Sweep

```yaml
film.thickness_sweep_nm     : [15.0, 20.0, 30.0]
film.dz_nm                  : 0.5
standing_wave.period_nm     : 6.75
standing_wave.amplitude_sweep: [0.0, 0.05, 0.10, 0.20]
standing_wave.absorption_length_nm: 30.0
top / bottom BC             : no-flux (Neumann), implemented via even-mirror FFT
```

Total 12 runs.

### Measurement (per run)

```text
H0_z_modulation_pct          (peak-to-peak / mean of H0[:, ix_line])
H0_z_modulation_sw_only_pct  (= H0_z_modulation_pct - same-thickness A=0 baseline)
H_final_z_modulation_pct
P_final_z_modulation_pct
modulation_reduction_pct     (z-mod reduction from H0 to P_final)
top_bottom_asymmetry         (|P[top] - P[bot]| / max at line-center column)
LER_fixed_threshold_nm       (extract_edges on P(x,z) treating z as the edge-track axis;
                              measures sidewall x-position variation)
LER_CD_locked_nm             (same with P_threshold bisected to design CD)
psd_mid_fixed / psd_mid_locked (Stage 4 PSD mid-band on z-tracks)
H_min, P_min, P_max          (bounds)
mass_budget_drift_pct        (relative change of the trapezoidal integral of H from H0 to H_final;
                              normal H consumption from kloss + quencher → about -35% expected)
```

### Gate (per row)

```text
no NaN / no Inf
H_min >= -1e-6, P_min >= -1e-6, P_max <= 1+1e-6
A == 0  → H0_z_modulation_sw_only_pct < 1 %     (excluding absorption contribution)
A > 0   → H0_z_modulation_sw_only_pct increases monotonically with A
PEB     → P_final_z_modulation_pct < H0_z_modulation_pct (every row)
```

### Verified results

config: `configs/v2_stage6_xz_standing_wave.yaml`

**12/12 rows PASS** all gates (per-row + cross-row).

```text
H0_z_modulation_sw_only_pct (standing wave only, absorption excluded):

                A=0.00   A=0.05   A=0.10   A=0.20
  thick=15      +0.00    +2.70    +5.26   +15.56
  thick=20      +0.00    +1.04    +7.12   +20.27
  thick=30      +0.00    +6.51   +12.93   +25.54

  → monotonic in A, monotonic in thickness (longer films accumulate more
    standing-wave effect). Every A=0 row has sw_only ≡ 0 (by definition).

P_final_z_modulation_pct (full post-PEB z-mod, absorption included):

                A=0.00   A=0.05   A=0.10   A=0.20
  thick=15       9.86    10.00    10.06    10.20
  thick=20      20.05    20.19    20.34    20.71
  thick=30      38.17    38.24    38.31    38.42

  → almost the entire P_final z-mod (95%) is determined by absorption, even at large A.
    The residual standing-wave effect after PEB is only 0.14–0.85%, which is very small.

modulation_reduction_pct = (H0_zmod - P_zmod) / H0_zmod:
  thick=15, A=0.20: +78.89 % (H0 48.31 → Pf 10.20)
  thick=20, A=0.20: +68.24 % (H0 65.20 → Pf 20.71)
  thick=30, A=0.20: +60.00 % (H0 96.06 → Pf 38.42)

  → the thicker the film, the smaller the z-mod attenuation by PEB.
    The diffusion length is shorter than the film thickness, so z cannot be uniformised.

top_bottom_asymmetry (line-center column):
  thick=15: 0.10
  thick=20: 0.18-0.19
  thick=30: 0.32

  → thicker films have a larger top/bottom gap because of absorption.

LER_CD_locked_nm (sidewall x-displacement std):
  thick=15: 1.32 (very small; PEB is highly effective in thin films)
  thick=20: 2.32-2.40 (slightly increases with A)
  thick=30: 3.80-3.87 (large; PEB is weak in thick films)

H_min ≈ 0.014–0.022, P_min ≈ 0.13–0.21, P_max ≈ 0.73 → all bounds healthy.
mass_budget_drift_pct ≈ -35% → kloss(0.005)*30s ≈ 14% + quencher (Q0=0.02
  consumes proportional H) ≈ 21%, summed. Normal H consumption.
```

### Success criteria mapping (plan §6 expected vs observed)

| plan expected | observed |
|---|---|
| At A=0 there is no standing-wave-only z-modulation | yes — `H0_zmod_sw_only_pct` ≡ 0 at A=0 (by definition) |
| At A>0, z-direction layered modulation appears in H0/P | yes — sw_only +1.04 ~ +25.54 %, monotonic |
| Modulation amplitude decreases after PEB | yes — `P_zmod < H0_zmod` in 12/12 rows |
| Thickness 15 / 20 / 30 nm shows a clear effect difference | yes — modulation_reduction (15: 79% > 20: 68% > 30: 60%), top/bottom asymmetry (0.10 < 0.19 < 0.32) |

### Candidate next steps

- **Stage 6B (full 3D)**: integrate x-y-z. This stage is (x, z) only, so it does not capture coupling with y-roughness. Compute cost is large.
- **Plotting / heatmap of CD-locked LER + PSD mid-band**: the stage CSV is rich but only thick=20 has a figure; other thicknesses can be added.
- **Stage 4B sigma=0 follow-up**, **Stage 3B σ=5/8 compatible budget**: can remain deferred.

For detailed analysis see `study_notes/07_stage6_xz_standing_wave.md`.

---

## 6. Implementation details

### 6.1 Preserve the existing folder

Never modify any file in the existing `reaction_diffusion_peb/`.

```text
Do not edit:
reaction_diffusion_peb/
```

All new experiments are conducted under:

```text
reaction_diffusion_peb_v2_high_na/
```

### 6.2 v1 code-reuse policy

Copy v1's verified solvers for use, but avoid import coupling.

Recommended:

```text
copy selected FD/FFT utilities into v2 src/
```

Discouraged:

```text
from reaction_diffusion_peb.src... import ...
```

The reason is that v2 must be an independently reproducible experiment.

### 6.3 First implementation targets

Files to implement first:

```text
src/geometry.py
src/roughness.py
src/electron_blur.py
src/metrics_edge.py
experiments/01_lspace_baseline/run_baseline_no_quencher.py
```

### 6.4 Edge-metric implementation

Extract edge positions from the `P_threshold` contour.

Recommended approach:

```text
For each y row, find the x-position where P(x,y)=P_threshold by interpolation
Obtain edge_x(y)
LER = 3*sigma(edge_x - mean(edge_x)), or report alongside an RMS-based variant
CD(y) = right_edge_x(y) - left_edge_x(y)
LWR = 3*sigma(CD(y))
```

PSD metric:

```text
edge_residual(y) = edge_x(y) - smooth_mean_edge
FFT(edge_residual)
Compare PSD_before / PSD_after
```

### 6.5 Physics bounds tests

Enforce the following as tests at every stage.

```text
H >= -1e-8
Q >= -1e-8, if Q enabled
0 <= P <= 1
no NaN
no Inf
mass budget reasonable
```

---

## 7. Output artefact rules

Every experiment must save figures and CSV together.

```text
outputs/figures/*.png
outputs/logs/*.csv
outputs/fields/*.npz
```

Recommended figures:

```text
01_lspace_H0.png
01_lspace_P_final.png
01_lspace_contour_overlay.png
02_DH_time_sweep_LER_CD.png
03_electron_blur_decomposition.png
04_weak_quencher_sweep.png
05_pitch_dose_window.png
06_xz_standing_wave_before_after.png
```

Recommended CSV columns:

```text
run_id
pitch_nm
line_cd_nm
dose_mJ_cm2
dose_norm
DH_nm2_s
time_s
T_C
Q0
kq
kdep
kloss
P_threshold
H_peak
P_max
P_mean
area_P_gt_threshold
CD_initial_nm
CD_final_nm
CD_shift_nm
LER_initial_nm
LER_final_nm
LER_reduction_pct
LWR_initial_nm
LWR_final_nm
status
notes
```

---

## 8. Judgement criteria

### Normal physical trends

```text
DH increases → H peak decreases
DH increases → LER decreases
DH increases → CD shift increases
PEB time increases → Pmax increases
PEB time increases → LER decreases
temperature increases → reaction rate increases
weak quencher increases → acid tail decreases
strong quencher increases → Pmax decreases
standing-wave amplitude increases → z modulation increases
PEB diffusion → z modulation decreases
```

### Abnormal trends that must halt the run

```text
H becomes a large negative number
P below 0 or above 1
Q goes negative
DH increases but peak increases
time increases but Pmax decreases — investigate the strong-acid-loss exception separately
contour fully disappears at Q0=0.01, kq=1
P>threshold area is 0 at every dose
The whole domain has P>threshold at every dose
```

---

## 9. v2 first-target result (Stage 1 closure criterion)

v2's first success target is not a grandiose full High-NA simulation.
Only the following needs to succeed first.

```text
At 24 nm pitch, 12.5 nm CD line-space (domain = 5 * pitch = 120 nm),
construct an H0 with initial edge roughness,
and after PEB without a quencher confirm that the P>0.5 contour preserves line-space separation
(interior gate),
that LER_before > LER_after, and that CD_shift can be quantified.
```

**This target was achieved at σ=0, t=30, DH=0.8, kdep=0.5, Hmax=0.2** (see the §Stage 1 verified-results table).
The σ=5/t=60 nominal has been moved to the §Stage 1B over-budget stress case.

Until this result is in hand, the following must not be done.

```text
strong quencher
PINN/FNO
full 3D
complex developer model
```

---

## 10. Stage 1 baseline config — `configs/v2_stage1_clean_geometry.yaml`

```yaml
run:
  name: v2_stage1_clean_geometry
  seed: 7

geometry:
  pattern: line_space
  pitch_nm: 24.0
  half_pitch_nm: 12.0
  line_cd_nm: 12.5
  grid_spacing_nm: 0.5
  # pitch-aligned domain (5 * 24 = 120) prevents the FFT-seam wider-space
  # artifact that otherwise lowers P_min near the boundary and misleads gates.
  domain_x_nm: 120.0
  domain_y_nm: 120.0
  edge_roughness_enabled: true
  edge_roughness_amp_nm: 1.0
  edge_roughness_corr_nm: 5.0

exposure:
  wavelength_nm: 13.5
  dose_mJ_cm2: 40.0
  reference_dose_mJ_cm2: 40.0
  dose_norm: 1.0
  eta: 1.0
  Hmax_mol_dm3: 0.2
  # Stage-1 baseline studies clean geometry only. The σ-compatible budget for
  # nonzero electron blur is found in Stage 1A (calibration) before Stage 3.
  electron_blur_enabled: false
  electron_blur_sigma_nm: 0.0
  target_NILS: 1.5

peb:
  time_s: 30.0
  temperature_C: 100.0
  DH_nm2_s: 0.8
  kloss_s_inv: 0.005
  kdep_s_inv: 0.5

quencher:
  enabled: false
  Q0_mol_dm3: 0.0
  DQ_nm2_s: 0.0
  kq_s_inv: 0.0

development:
  method: threshold
  P_threshold: 0.5

outputs:
  save_fields: true
  save_figures: true
  save_metrics_csv: true
```

---

## 11. Recommended weak quencher config: `configs/v2_weak_quencher.yaml`

```yaml
run:
  name: v2_weak_quencher_sweep
  seed: 7

base_config: configs/v2_baseline_lspace.yaml

quencher:
  enabled: true
  DQ_nm2_s: 0.0
  Q0_sweep_mol_dm3: [0.0, 0.01, 0.02, 0.03, 0.05]
  kq_sweep_s_inv: [0.5, 1.0, 2.0, 5.0]

criteria:
  require_contour_at:
    Q0_mol_dm3: 0.01
    kq_s_inv: 1.0
    P_threshold: 0.5
```

---

## 12. Recommended x-z config: `configs/v2_xz_standing_wave.yaml`

```yaml
run:
  name: v2_xz_standing_wave
  seed: 7

geometry:
  pattern: line_space_xz
  pitch_nm: 24.0
  line_cd_nm: 12.5
  dx_nm: 0.5
  dz_nm: 0.5
  domain_x_nm: 128.0

film:
  film_thickness_nm: 20.0
  thickness_sweep_nm: [15.0, 20.0, 30.0]
  top_bc: no_flux
  bottom_bc: no_flux

exposure:
  dose_mJ_cm2: 40.0
  reference_dose_mJ_cm2: 40.0
  dose_norm: 1.0
  eta: 1.0
  Hmax_mol_dm3: 0.2

standing_wave:
  enabled: true
  period_nm: 6.75
  amplitude: 0.10
  amplitude_sweep: [0.0, 0.05, 0.10, 0.20]
  phase_rad: 0.0
  absorption_enabled: true
  absorption_length_nm: 30.0

peb:
  time_s: 60.0
  DH_nm2_s: 0.8
  kloss_s_inv: 0.005
  kdep_s_inv: 0.5

quencher:
  enabled: false

development:
  P_threshold: 0.5
```

---

## 13. Final summary

v2 does not replace v1. v1 is kept for per-term solver verification.
v2 is a process-oriented experiment that adds geometry, roughness, electron blur, film thickness, and weak quencher under High-NA EUV conditions.

The most important principle:

```text
Do not turn on all physics at once.
Proceed in the order geometry → blur → diffusion/deprotection → weak quencher → pitch/dose sweep → x-z standing wave.
```

v2 first-completion criterion:

```text
At 24 nm pitch / 12.5 nm CD / clean geometry (σ=0) / no quencher,
the P>0.5 contour preserves line-space separation (interior gate),
and LER reduction and CD shift can both be quantified.
```

**Stage 1 status (finalised after calibration):**

```text
✅ Achieved — σ=0, t=30, DH=0.8, kdep=0.5, Hmax=0.2 (configs/v2_stage1_clean_geometry.yaml)
   P_space_mean=0.31, P_line_mean=0.76, contrast=0.45
   CD: 12.46 → 15.01 nm (+2.55 nm)
   LER: 2.77 → 2.65 nm
   All interior gates PASS

⚠ Calibration finding — the σ=5/t=60 nominal is over-budget at 24 nm pitch.
   This nominal is moved to the Stage 1B stress case.
   σ-compatible operating range (within kdep=0.5, Hmax=0.2 spec): σ ∈ [0, 3] nm.
   The σ ≥ 4 compatible budget is revisited after the search space is expanded (Stage 1A.3).
```
