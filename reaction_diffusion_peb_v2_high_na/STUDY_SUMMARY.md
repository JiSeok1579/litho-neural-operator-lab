# PEB v2 — first-pass study summary

High-NA EUV PEB v2 simulation work — Stage 1 through Stage 6 are complete.
This document gathers the design intent of [`EXPERIMENT_PLAN.md`](./EXPERIMENT_PLAN.md) and the verified results in one place.

For the per-stage detailed analysis see `study_notes/0[1-7]_*.md`.

---

## Claims boundary (read before citing any number)

```text
✓ The model is internally consistent and physically plausible.
  - All sweeps respect the documented bounds (H ≥ 0, 0 ≤ P ≤ 1).
  - All trends (PEB smoothing thin > thick, σ ↓ → wider small-pitch
    window, dose ↑ → process-window CD shift, etc.) match the
    qualitative expectations of the equations and BCs implemented.
  - The Phase 2B sensitivity atlas provides a controllability map ready
    to drive future calibration once external data lands.

✗ The model is NOT externally calibrated.
  - The internal calibration targets (CD ≈ 15 nm, LER ≈ 2.6 nm) are
    derived from v2's own first-pass observations.
  - calibration_status.published_data_loaded is FALSE.
  - calibration_status.v2_OP_frozen is TRUE — see calibration/calibration_targets.yaml.

✗ Quantitative agreement with High-NA EUV experiments is NOT claimed.
  - No published or measured CD / LER / process-window values have
    been loaded into calibration_targets.yaml.
  - All sweeps post-freeze are labelled sensitivity / controllability /
    hypothesis study, never "calibration" or "calibrated to real".

To convert this into externally-calibrated work, follow `FUTURE_WORK.md`
Gate A (load measured / published references → re-run Phase 1 / 2A / 2B
→ only then refer to "external calibration").
```

---

## Status

| Stage | Headline outcome | Note |
|---|---|---|
| **1**   | clean geometry baseline passes | σ=0, t=30, DH=0.8 |
| **1A**  | σ ∈ [0, 3] usable at kdep=0.5, Hmax≤0.2 | σ=4, 5 are budget-incompatible |
| **1B**  | σ=5 / t=60 = lines merge (over-budget reference) | original plan nominal demoted |
| **2**   | algorithmic best DH=0.8 / t=20 (margin 0.003) → robust alt DH=0.5 / t=30 chosen | margin vs LER trade-off |
| **3**   | σ ∈ {0, 1, 2, 3} cleanly separable; σ ↑ → e-blur LER ↓ but PEB-LER goes negative at σ ≥ 2 | "displacement artifact" hypothesis raised |
| **3B**  | σ=5 / 8 budget search → deferred | requires search-space expansion |
| **4**   | weak quencher: 52 / 52 cells pass the gate; balanced OP Q0=0.02, kq=1 | σ=3 LER recovery +29 pp |
| **4B**  | **CD-locked LER tool added**; pitch ≤ 20 LER worsening is real, not artifact | three-label diagnosis |
| **5**   | process window 36 + 72 runs; pitch=16 closed, pitch ≥ 24 wide window | recommended dose=40 for every workable pitch ≥ 20 |
| **5C**  | σ=0 small-pitch follow-up | partially addressed in Phase 2B Part C; further work deferred |
| **6**   | x-z standing wave 12 runs; PEB absorbs z-modulation (thin 79 % > thick 60 %) | Neumann-z mirror FFT |
| **6B**  | full 3D x-y-z | deferred (compute cost) |

---

## Per-stage findings

### Stage 1 — clean geometry baseline

**Conditions**: σ=0, t=30, DH=0.8, kdep=0.5, Hmax=0.2, no quencher, pitch=24 nm, CD=12.5 nm.

**Result**:
```text
P_space_center_mean = 0.31     P_line_center_mean  = 0.76
contrast            = 0.45     area_frac           = 0.625
CD: 12.46 → 15.01 nm  (CD_shift = +2.55)
LER: 2.77 → 2.65 nm   (-4.3 %)
```

**Key insight**: the original plan's nominal (σ=5, t=60) is incompatible with the 24 nm pitch / 12.5 nm CD geometry. The domain has to be aligned to an integer multiple of pitch (120 = 5 × 24) to avoid an FFT seam artifact. False-pass cases at the boundary are ruled out by the interior gate (P_space_center, P_line_center, contrast, area, CD/pitch).

config: `configs/v2_stage1_clean_geometry.yaml`. Notes: `study_notes/01_stage1_clean_geometry.md`.

---

### Stage 2 — DH × time process window

**Conditions**: 25-cell grid (DH ∈ {0.3, 0.5, 0.8, 1.0, 1.5} × time ∈ {15, 20, 30, 45, 60}). All other variables fixed at the Stage-1 baseline.

**Result**: 9 / 25 cells pass the interior gate, forming a diagonal process window.

```text
LER reduction (%):              t=15      20     30     45     60
                  DH=0.30:    6.55✗   6.63✓  8.04✓  6.96✓  −14.34✓
                  DH=0.50:    9.00✗   8.80✓  8.69✓ −17.04✓  45.55✗
                  DH=0.80:   10.06✗   9.61✓  4.25✓   8.28✗ 100.00✗
                  DH=1.00:    8.62✗   8.84✗ −1.98✓  62.62✗ 100.00✗
                  DH=1.50:   −4.48✗   3.98✗ −38.01✓ 100.00✗ 100.00✗
```

**Algorithmic best**: DH=0.8, t=20 (LER −9.6 %, but margin 0.003 — boundary).
**Robust alternative**: DH=0.5, t=30 (LER +8.69 %, margin 0.142). **This OP becomes the default for Stage 3 onward.**

**Key insight**: the selection criterion lacks a P_line_margin clause, so the algorithmic best lands on the boundary. From Stage 3 onward we add `P_line_margin ≥ 0.05`.

config: `configs/v2_stage2_dh_time.yaml`. Notes: `study_notes/02_stage2_dh_time_sweep.md`.

---

### Stage 3 — electron blur separation

**Conditions**: σ ∈ {0, 1, 2, 3} on two operating points × 4 σ = 8 runs. The plan's original [0, 2, 5, 8] sweep is reduced because σ=5, 8 are not budget-compatible (Stage 1A finding).

**Measurement convention** (redefined here):

```text
LER_design_initial    = LER on binary I @ 0.5     (σ-independent baseline)
LER_after_eblur_H0    = LER on I_blurred @ 0.5
LER_after_PEB_P       = LER on P @ 0.5
electron_blur_LER_reduction_pct  = 100 * (design - eblur)/design
PEB_LER_reduction_pct            = 100 * (eblur - PEB)/eblur
total_LER_reduction_pct          = 100 * (design - PEB)/design
```

**Result (robust OP DH=0.5, t=30)**:

| σ | total LER % | e-blur % | PEB % | CD_shift |
|---|---|---|---|---|
| 0 | **+8.7** | +0.0 | +8.7 | +1.79 |
| 1 | +7.8 | +2.2 | +5.7 | +2.62 |
| 2 | +3.6 | +6.1 | −2.7 | +3.83 |
| 3 | −22.5 | +11.1 | −37.7 | +5.85 |

The algorithmic-best OP (DH=0.8, t=20) fails `P_line_margin ≥ 0.03` for every σ → demoted from downstream use.

**Key insight**: e-blur and PEB do not stack — they *compete*. As σ rises the line widens, the contour drifts off the design edge, and PEB-LER turns negative. The next stage (4B) confirms this is partly contour displacement, not a PEB physics failure. From here on we use the `P_line_margin ≥ 0.03` gate.

config: `configs/v2_stage3_electron_blur.yaml`. Notes: `study_notes/03_stage3_electron_blur.md`.

---

### Stage 4 — weak quencher

**Conditions**: 52 runs at the robust OP × σ ∈ {0, 1, 2, 3} × (Q0=0 baseline + Q0 ∈ {0.005, 0.01, 0.02, 0.03} × kq ∈ {0.5, 1.0, 2.0}).

**Result**: 52 / 52 pass the Stage-3 gate; 51 / 52 satisfy the Stage-4 robust criterion.

```text
σ=2 dtotal_LER_pp (vs σ-matched baseline):
              kq=0.5    kq=1.0    kq=2.0
  Q0=0.030    +4.90     +6.47     +7.64
  Q0=0.020    +3.74     +5.21     +6.44
  Q0=0.010    +2.18     +3.24     +4.23
  Q0=0.005    +1.18     +1.84     +2.49
```

**Balanced OP**: σ=2, Q0=0.02, kq=1.0 → dCD=−1.76, darea=−0.073, dLER=+5.21 pp, margin=0.096.

**σ=3 LER recovery**: baseline total_LER = −22.5 % → adding Q0=0.03 / kq=1 lifts it to +6.6 % (dLER = +29.15 pp).

**Key insight**: the PSD high-band is already ~99.9 % suppressed by PEB even without quencher, so quencher's effect is almost entirely in the mid-band. The Stage 3 "PEB-LER negative" hypothesis is partly verified (full diagnosis lands in Stage 4B).

config: `configs/v2_stage4_weak_quencher.yaml`. Notes: `study_notes/04_stage4_weak_quencher.md`.

---

### Stage 5 — pitch × dose process window

**Conditions**: 108 runs (primary + 2 controls). 6 pitches × 6 doses. `domain_x_nm = pitch × 5` to keep the FFT seam aligned.

**Status heatmap (primary OP)**:

```text
              dose:  21    28.4    40    44.2    59     60
  pitch=32          unde   vali   robu   robu   robu   robu
  pitch=28          unde   vali   robu   robu   robu   robu
  pitch=24          unde   vali   robu   robu   robu   robu
  pitch=20          unde   vali   robu   merg   merg   merg
  pitch=18          unde   vali   merg   merg   merg   merg
  pitch=16          unde   merg   merg   merg   merg   merg
```

**Recommended dose**: dose=40 for every workable pitch ≥ 20. The pitch=16 process window is closed at the v2 chemistry (line_cd=12.5 / pitch=16 → duty 0.78 + diffusion length 5.5 nm > inter-line space).

**Control comparison**: σ=0 with quencher off has the widest window. Adding quencher actually narrows the small-pitch tolerance.

**Key insight**: the Stage 4 LER benefit lives at the large-pitch end. Some of the small-pitch (≤20) "LER worsening" is contour displacement (separated by Stage 4B); the rest is real degradation.

config: `configs/v2_stage5_pitch_dose.yaml`. Notes: `study_notes/05_stage5_pitch_dose.md`.

---

### Stage 4B — CD-locked LER

**Trigger**: Stage 5's negative LER reduction at pitch ≤ 20 needs to be split into "displacement artifact" vs "real roughness degradation".

**Tool**: `find_cd_lock_threshold` — bisect P ∈ [0.2, 0.8] (with adaptive endpoint narrowing) to lock the contour to the design CD before measuring LER.

**Block A decisions (12 cells)**:

| OP | pitch | dose | label |
|---|---|---|---|
| primary | 18 | 28.4 | real degradation |
| primary | 18 | 40   | merged-line artifact (fixed underestimates) |
| primary | 20 | 28.4 | real degradation |
| primary | 20 | 40   | real degradation |
| primary | 24 | 28.4 | OK |
| primary | 24 | 40   | OK |
| ctrl σ0 | 18 | 28.4 | real degradation |
| ctrl σ0 | 18 | 40   | merged-line artifact |
| ctrl σ0 | 20 | 28.4 | OK |
| **ctrl σ0** | **20** | **40** | **displacement artifact (locked recovers)** |
| ctrl σ0 | 24 | 28.4 | OK |
| ctrl σ0 | 24 | 40   | OK |

**Block B (mini-sweep)**: at pitch ∈ {18, 20}, weakening the quencher (Q0 ≤ 0.005, kq ≤ 0.5) does **not** recover LER_locked. The best small-pitch case shrinks LER by only 0.11 nm (still ≈ 1.3 nm above design).

**Key insight**: pitch ≤ 20 LER worsening is real. The Stage-4 balanced OP is robust at pitch ≥ 24 only. **CD-locked LER is integrated as the helper default from Stage 6 onward** so every later sweep emits both fixed and locked metrics.

config: `configs/v2_stage4b_cd_locked.yaml`. Notes: `study_notes/06_stage4B_cd_locked.md`.

---

### Stage 6 — x-z standing wave

**Conditions**: 12 runs. thickness ∈ {15, 20, 30} nm × amplitude ∈ {0, 0.05, 0.10, 0.20}. period=6.75 nm, abs_len=30 nm. Neumann-z BC enforced via even-mirror FFT.

**Result (12 / 12 PASS)**:

```text
PEB modulation reduction:
  thick=15: 79 %   (most effective)
  thick=20: 68 %
  thick=30: 60 %   (diffusion length 5.5 nm < thickness)

H0_z_modulation_sw_only_pct (excluding the absorption envelope):
              A=0.05   A=0.10   A=0.20
  thick=15    +2.70    +5.26   +15.56
  thick=20    +1.04    +7.12   +20.27
  thick=30    +6.51   +12.93   +25.54

Side-wall LER (CD-locked, z as the track axis):
  thick=15: 1.32 nm
  thick=20: 2.32–2.40 nm
  thick=30: 3.80–3.87 nm

Top/bottom asymmetry: 0.10 → 0.18 → 0.32 (driven by absorption envelope)
```

**Key insight**: PEB nearly fully absorbs the period-6.75-nm standing wave. The residual sw component on P is < 1 %. In thick films the absorption envelope dominates the remaining z-modulation. Summary plots: `outputs/figures/06_xz_standing_wave/summary/`.

config: `configs/v2_stage6_xz_standing_wave.yaml`. Notes: `study_notes/07_stage6_xz_standing_wave.md`.

---

## Recommended v2 operating point

The robust standard OP validated across the previous stages:

```yaml
geometry:
  pitch_nm:        24                    # Stage 5 robust window centre
  line_cd_nm:      12.5
  grid_spacing_nm: 0.5
  domain_x_nm:     120                   # = pitch * 5 (Stage 1A FFT-seam-safe)
  domain_y_nm:     120                   # constant LER y-sample count
  edge_roughness:  amp=1.0, corr=5.0     # reasonable design noise

exposure:
  dose_mJ_cm2:           40              # Stage 5 recommended dose for pitch ≥ 20
  reference_dose_mJ_cm2: 40
  Hmax_mol_dm3:          0.2
  eta:                   1.0
  electron_blur_sigma_nm: 2              # midpoint of the σ-compatible range [0, 3] (Stage 1A)
  electron_blur_enabled: true

peb:
  time_s:        30                      # Stage 2 robust window
  DH_nm2_s:      0.5
  kdep_s_inv:    0.5
  kloss_s_inv:   0.005

quencher:
  enabled:     true                      # Stage 4 balanced
  Q0_mol_dm3:  0.02
  kq_s_inv:    1.0
  DQ_nm2_s:    0.0

development:
  P_threshold: 0.5                       # the helper also computes the CD-locked variant
```

**Validated zone**:
- pitch=24 nm, every dose 28.4–60 mJ/cm² is robust_valid (Stage 5).
- x-z standing wave amplitudes up to 0.20 are absorbed by PEB (Stage 6).
- standard metrics include `LER_CD_locked` and `psd_mid_band_reduction`.

**Out-of-zone**:
- pitch ≤ 20: real LER degradation (Stage 4B). Either lower σ to 0 or scale `line_cd` proportionally — both belong in a separate stage.
- σ ≥ 4: budget incompatible (Stage 1A). Needs dose / kdep / Hmax extension (Stage 3B).
- film thickness > 30 nm: PEB modulation reduction extrapolates below 60 % (Stage 6 trend).

---

## Next milestone — calibration against external reference

Before adding new physics or starting another stage, the next substantive step is to compare the current results to **literature or measurement data**.

### Comparison targets

```text
1. Published measurements at 24 nm pitch / 12.5 nm CD / 40 mJ/cm² dose:
   - Does CD_final ≈ 15 nm (Stage 1) match the measured value?
   - Is LER ≈ 2.5–2.7 nm a reasonable absolute number?
   - Does the dependence on PEB time / temperature match plan §4?

2. Process-window shape:
   - Does the Stage 5 status map agree with a published high-NA process window?
   - Does the small-pitch closure happen at the right pitch?

3. Standing-wave amplitude reduction:
   - Does the Stage 6 thin-vs-thick trend match measured top-coat / barrier
     experiments?
```

### Where each calibration knob would land

```text
- kdep, Hmax: absolute scaling of H and P. Adjust by a multiplicative
  factor against measured LER.
- dose definition: confirm that dose_norm's 40 mJ/cm² reference matches
  the EUV reference dose used in measurements.
- DH (acid diffusion): compare temperature-dependent DH against Arrhenius
  data.
- quencher kq: validate that Q's reaction rate sits in a reasonable
  range for the resist family.
```

### Priority

```text
1. CD calibration  →  dose / Hmax adjustment (factor 1±0.X)
2. LER calibration →  initial roughness amp + correlation length
3. Process window  →  kdep / quencher scaling
4. Standing wave   →  absorption_length_nm (currently 30 nm, picked by hand)
```

Only after this calibration loop closes can we move to:

```text
- Stage 6B (full 3D x-y-z)
- Stage 3B (σ=5, 8 budget search expanded)
- Stage 5C (σ=0 small-pitch process window)
- Stage 1A.3 (kdep, dose extension to retry σ=5/8 budget)
- New chemistry (e.g., PAG profile, full Dill model)
```

---

## Artefact index

```text
configs/                                # 8 stage / phase configs
  v2_stage1_clean_geometry.yaml          (Stage 1 baseline)
  v2_baseline_lspace.yaml                (Stage 1B over-budget reference)
  v2_stage2_dh_time.yaml                 (Stage 2)
  v2_stage3_electron_blur.yaml           (Stage 3)
  v2_stage4_weak_quencher.yaml           (Stage 4)
  v2_stage4b_cd_locked.yaml              (Stage 4B)
  v2_stage5_pitch_dose.yaml              (Stage 5)
  v2_stage6_xz_standing_wave.yaml        (Stage 6)
  xz_companions.yaml                     (single-line x-z renders)

src/
  geometry.py, roughness.py              (line/space + edge roughness)
  electron_blur.py                       (2D Gaussian blur)
  exposure_high_na.py                    (Dill + 1D builders + x-z exposure)
  fd_solver_2d.py                        (x-y solver)
  fd_solver_xz.py                        (x-z solver, Neumann-z mirror)
  metrics_edge.py                        (LER / LWR / CD + CD-lock + PSD bands)
  visualization.py                       (plot helpers)

experiments/
  run_sigma_sweep_helpers.py             (CD-locked + PSD bands integration)
  01_lspace_baseline/                    (Stage 1, 1A, 1B)
  02_dh_time_sweep/                      (Stage 2)
  03_electron_blur/                      (Stage 3)
  04_weak_quencher/                      (Stage 4)
  04b_cd_locked/                         (Stage 4B + Stage 5B mini-sweep)
  05_pitch_dose/                         (Stage 5)
  06_xz_standing_wave/                   (Stage 6)
  render_xz_companions/                  (xz cross-section presentation)

calibration/
  configs/cal01_hmax_kdep_dh.yaml        (Phase 1)
  configs/cal02a_sensitivity.yaml        (Phase 2A)
  configs/cal03_atlas_xy.yaml            (Phase 2B Part A)
  configs/cal04_atlas_xz.yaml            (Phase 2B Part B)
  configs/cal05_smallpitch.yaml          (Phase 2B Part C)
  experiments/cal01_*..cal05_*           (Phase 1 / 2A / 2B runners)
  calibration_targets.yaml               (frozen OP, internal targets, status flags)
  calibration_plan.md                    (per-phase log + decisions)
  README.md                              (calibration intent + how to run)

tests/                                   (32 / 32 passing)

outputs/
  figures/                               (~600 figures across 8 stages + 3 phases + xz companions)
  logs/                                  (CSV + JSON summaries per sweep)
  fields/                                (Stage 1 npz field snapshots)

study_notes/
  README.md                              (index)
  01_stage1_clean_geometry.md
  02_stage2_dh_time_sweep.md
  03_stage3_electron_blur.md
  04_stage4_weak_quencher.md
  05_stage5_pitch_dose.md
  06_stage4B_cd_locked.md
  07_stage6_xz_standing_wave.md

EXPERIMENT_PLAN.md                       (status header + per-stage spec & results)
RESULTS_INDEX.md                         (per-stage table → folder / CSV / figure dir / 1-line conclusion)
FUTURE_WORK.md                           (gated future work)
README.md                                (one-page entry)
STUDY_SUMMARY.md                         (this file)
```

---

## v2 first-pass closeout

Stages 1, 1A, 1B, 2, 3, 4, 4B, 5, 6 are all complete; every stage has a study note and a merged PR (#29 – #41); the robust OP is identified and frozen.

## Calibration policy (frozen 2026-04-30)

```text
External reference data is not loaded.
The v2 OP is frozen as the internally-consistent nominal OP.
calibration_status         = internal-consistency only
published_data_loaded      = false
v2_OP_frozen               = true

Every later sweep / experiment must be labelled as one of:
  - sensitivity study
  - controllability study
  - hypothesis test

The labels "calibration" and "calibrated to real" are only allowed once
external measurement / literature data has been added to
calibration_targets.yaml (i.e. published_data_loaded=true). Until then,
all work is internal exploration.
```

Details: `calibration/calibration_plan.md` and `calibration/calibration_targets.yaml`.

---

## v3 (screening) overlay — first-pass closed (2026-04-30)

A separate submodule, [`reaction_diffusion_peb_v3_screening/`](../reaction_diffusion_peb_v3_screening/), sits **on top of** this frozen v2 nominal model. v3 is candidate screening + 6-class defect classification on the v2 nominal physics — **not** external calibration. v3 never modifies the v2 OP and `published_data_loaded` stays `false`.

The v3 first-pass loop ran Stages 01 → 04D and is closed:

```text
01 label-schema validation      PASS  (all 6 labels reachable)
02 Monte-Carlo dataset          PASS  (1 000-row Sobol seed)
03 surrogate baseline           PASS  (RF classifier + regressor)
04 active-learning iteration    PASS  (16 → 186 defects, 1 iter)
04B failure-seeking expansion   PASS  (defects 186 → 1 928)
04C roughness expansion         PASS  (roughness 3 → 321)
04D operational-zone closeout   PASS  (5/5 hard gates)
```

Stage 04D operational-zone hard gates (held-out 80/20, seed=13):

```text
CD_locked operational MAE  ≤ 0.15 nm   PASS  (0.0696)
LER_CD_locked operational  ≤ 0.03 nm   PASS  (0.0232)
P_line_margin operational  ≤ 0.03      PASS  (0.0166)
balanced accuracy          ≥ 0.93      PASS  (0.934)
macro F1                   ≥ 0.93      PASS  (0.949)

false_robust_valid_rate (informational)  0.020
false_defect_rate       (informational)  0.000
v2_OP_frozen / published_data_loaded     unchanged (true / false)
```

Pointers: [`reaction_diffusion_peb_v3_screening/README.md`](../reaction_diffusion_peb_v3_screening/README.md) for the full pipeline + per-stage results, and [`reaction_diffusion_peb_v3_screening/study_notes/03_v3_stage04d.md`](../reaction_diffusion_peb_v3_screening/study_notes/03_v3_stage04d.md) for the closeout narrative. Stage 05 (autoencoder / inverse fit) remains optional future work.
