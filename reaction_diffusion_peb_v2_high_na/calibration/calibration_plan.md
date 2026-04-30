# PEB v2 calibration plan

## v2 OP freeze (2026-04-30)

```text
External reference data is not available.
Freeze the v2 first-pass OP as the internal-consistent nominal OP:
  pitch=24, dose=40, sigma=2, DH=0.5, time=30,
  kdep=0.5, Hmax=0.2, kloss=0.005, Q0=0.02, kq=1.0, DQ=0.0

calibration_status: internal-consistency only.
published_data_loaded: false.
v2_OP_frozen: true.

All later runs must be labelled as one of:
  - sensitivity study
  - controllability study
  - hypothesis test
The expressions "calibration" or "calibrated to real" are forbidden until published_data_loaded=true.
```

## Strategy

```text
With no external reference, pre-quantify the dynamic range and
sensitivity of the chemistry knobs so that, when external data does
arrive, we already know which knob can correct what and to what extent.
```

`calibration_targets.yaml` holds the quantitative targets and the frozen OP, while this document accumulates the per-phase progress, decisions, and results.

---

## Phase 1 — chemistry core-variable sweep

### 1.1 Purpose

```text
How close is the v2 OP's (CD, LER) to the target (15 nm / 2.6 nm)?
If it is not close, which of Hmax / kdep / DH is responsible?
```

### 1.2 Sweep

```yaml
fixed:
  pitch_nm:        24
  line_cd_nm:      12.5
  dose_mJ_cm2:     40
  sigma_nm:        2
  time_s:          30
  Q0_mol_dm3:      0.02
  kq_s_inv:        1.0
  DQ_nm2_s:        0.0
  kloss_s_inv:     0.005

sweep:
  Hmax_mol_dm3: [0.15, 0.18, 0.20, 0.22]
  kdep_s_inv:   [0.35, 0.5, 0.65]
  DH_nm2_s:     [0.3, 0.5, 0.8]
```

Total 4 × 3 × 3 = 36 runs.

### 1.3 Measurements (per cell)

```text
CD_locked_nm
LER_CD_locked_nm
P_threshold_locked
P_line_margin
area_frac
psd_mid_band (mid-band power after PEB)
status (Stage-5 classification — robust_valid / valid / under_exposed / merged / unstable)
distance_to_target = sqrt( ((CD_locked-15)/15)^2 + ((LER_locked-2.6)/2.6)^2 )
```

### 1.4 Gate / decision

```text
PASS:
  at least one cell with distance_to_target < 0.10 (only cells with status robust_valid or valid count)
  → adopt that cell's (Hmax, kdep, DH) as the new OP candidate
  → proceed to Phase 2

PASS-marginal:
  some cell has distance < 0.20 but none below 0.10
  → adopt the best cell and proceed to Phase 2 (with a finer grid)

FAIL:
  minimum distance > 0.20
  → Phase 2A: extended sweep that also covers dose, sigma, abs_len
```

### 1.5 Phase 1 results (executed)

#### Measurement-methodology fix

The first version of the distance function used `CD_locked`, but by definition the CD-locked algorithm forces the contour to the design CD (~12.5 nm), so every row gives CD ≈ 12.5 and the distance ends up at a systematic offset of ~0.17. **The CD term in the distance is therefore changed to use the fixed-threshold CD (`CD_final_nm`)** — the same metric as the "developed pattern CD" reported in published / measured data.

```python
distance_to_target = sqrt( ((CD_fixed - 15)/15)^2 + ((LER_locked - 2.6)/2.6)^2 )
```

The LER term keeps the CD-locked LER as is (intrinsic roughness, with displacement bias removed).

#### Sweep results (36 runs)

```text
gate                           : PASS
best score                     : 0.0054   (target: < 0.10)
n_selectable                   : 24 / 36   (robust_valid or valid)
v2 OP score (Hmax/kdep/DH=
            0.20/0.50/0.50)    : 0.0588   (PASS)
```

#### Top 5 candidates (CD = fixed-threshold; LER = CD-locked)

| Hmax | kdep | DH  | status        | CD_fix | CD_lock | LER  | score  | margin | area  | psd_mid |
|------|------|-----|---------------|--------|---------|------|--------|--------|-------|---------|
| 0.20 | 0.50 | 0.80 | robust_valid | **15.05** | 12.30 | **2.59** | **0.0054** | +0.068 | 0.627 | 0.787 |
| 0.15 | 0.65 | 0.80 | robust_valid | 14.46 | 12.24 | 2.58 | 0.0365 | +0.051 | 0.603 | 0.797 |
| 0.22 | 0.50 | 0.30 | robust_valid | 15.00 | 12.53 | 2.46 | 0.0550 | +0.151 | 0.625 | 1.378 |
| 0.22 | 0.50 | 0.50 | robust_valid | 15.46 | 12.42 | 2.47 | 0.0578 | +0.131 | 0.644 | 1.030 |
| 0.18 | 0.65 | 0.30 | robust_valid | 15.31 | 12.49 | 2.46 | 0.0586 | +0.165 | 0.638 | 1.385 |

#### Offset classification

- **All kdep=0.35 rows have score > 0.20** — under_exposed dominates. The issue is not that acid generation is weak, but that the reaction rate is insufficient, so the lines do not deprotect enough.
- **The standalone effect of Hmax is small** — at fixed (kdep, DH), changing Hmax from 0.15 to 0.22 only varies CD by ~1-3 nm (it is not the primary variable).
- **DH has the most direct effect on LER** — DH 0.3→0.8 moves LER from 2.46 to 2.59 (converging exactly onto the target of 2.6).
- **CD depends roughly on the product of (kdep, DH)** — the minimum of the distance heatmap sits at (kdep=0.5, DH=0.8).

```text
offset diagnosis:
  acid generation (Hmax)   : small effect — Hmax 0.20 is reasonable
  reaction rate (kdep)     : kdep=0.35 insufficient, 0.5-0.65 appropriate
  acid diffusion (DH)      : raising DH from 0.5 to 0.8 brings both CD and LER closer to target
  electron blur, dose, abs_len : need additional verification in Phase 2A (fixed in this sweep)
```

#### Decision

**The v2 OP itself (Hmax=0.20, kdep=0.50, DH=0.50) gives score 0.0588 < 0.10 → it already matches the internal target**.
The top candidate (DH=0.80) fits even better, but considering the trade-off against the Stage 2 robust window (DH=0.5/t=30), keeping the v2 OP is recommended from a production-readiness standpoint.

```text
Phase 1 finding summary:
  v2 OP matches the internal calibration target (score 0.0588).
  CD is marginally low (14.53 vs 15.0) — correctable by changing DH from 0.5 to 0.8.
  However, the internal target itself was derived from v2 first-pass observations —
  a true calibration still requires an external reference.
```

#### Caveat — risk of recursive calibration

```text
The calibration target (CD=15, LER=2.6) was derived from the Stage 1 values of
the v2 first-pass.
In other words, this Phase 1 only verified "does v2 reproduce its own first-pass
result".
A real calibration requires external published / measured data.
→ Phase 3 (external-reference comparison) is the essential next step.
→ Before that, it is fine to run a short Phase 2 (process-window re-verification)
   on the new OP candidate (DH=0.8).
```

#### Next-step decision (resolved)

```text
[2026-04-30 closed]
Phase 1 closed as an internal-consistency check.
The recommended v2 OP (Hmax=0.20, kdep=0.50, DH=0.50) is unchanged.
The DH=0.80 candidate is recorded only in
calibration_targets.yaml > internal_best_score_candidate.
Any "calibrated to real" declaration is deferred until external reference data
arrives with published_data_loaded=true.

Next → Phase 2A (sensitivity / controllability study, NOT true calibration).
```

---

## Phase 2A — sensitivity / controllability study

### 2A.1 Purpose

```text
When each variable is varied one at a time (OAT) around the v2 OP anchor,
how much do the metrics move?
- Pre-determine, before any external reference arrives, which knob can correct
  what and to what extent.
- Quantify each metric's per-variable dynamic range and sensitivity coefficient.
- Make the intent explicit as a "controllability map", not "calibrated to real".
```

### 2A.2 Anchor

```text
x-y anchor (region around Stage 5):
  pitch=24, line_cd=12.5, dose=40, σ=2, time=30,
  Hmax=0.20, kdep=0.50, DH=0.50, kloss=0.005,
  Q0=0.02, kq=1.0, DQ=0

x-z anchor (region around Stage 6):
  film_thickness=20 nm, amplitude=0.10, period=6.75, abs_len=30
  remaining chemistry identical to the x-y anchor
```

### 2A.3 Sweep (OAT)

```yaml
# Only one variable departs from the anchor at a time.
xy_sweeps:
  dose_mJ_cm2:        [21, 28.4, 40, 44.2, 59, 60]   # same as Stage 5
  electron_blur_sigma_nm: [0, 1, 2, 3]                  # Stage 1A compatible range
  DH_nm2_s:           [0.3, 0.5, 0.8]                # includes the Phase 1 best candidate

xz_sweeps:
  dose_mJ_cm2:           [21, 28.4, 40, 44.2, 59, 60]
  electron_blur_sigma_nm:    [0, 1, 2, 3]
  absorption_length_nm:  [15, 20, 30, 50, 100]      # only meaningful for x-z
  DH_nm2_s:              [0.3, 0.5, 0.8]
```

x-y 13 runs + x-z 18 runs = 31 runs total.

### 2A.4 Measurements (per row)

```text
x-y row:
  CD_final_nm (fixed), CD_locked_nm
  LER_CD_locked_nm, LER_after_PEB_P_nm
  P_line_margin, area_frac, contrast
  status (Stage-5 classification)
  distance_to_target

x-z row:
  H0_z_modulation_pct, H0_z_modulation_sw_only_pct
  P_final_z_modulation_pct, modulation_reduction_pct
  top_bottom_asymmetry
  LER_fixed_threshold_nm (z-track), LER_CD_locked_nm
  psd_locked_mid
  bounds (H_min, P_min, P_max)
```

### 2A.5 Sensitivity report

per (variable, metric) pair:

```text
relative_span_pct = 100 * (metric_max - metric_min) / metric_anchor
local_slope       = rate of change of d(metric)/d(variable) (finite difference near the anchor)
status_changes    = number of status transitions during the variable sweep
```

### 2A.6 Gates

No gates — this phase is for building the controllability map. However, the report must include:

```text
- ranking of which variable has the largest effect on which metric
- indication of which variable's sensitive zone the v2 OP sits in
- indication of which knob, when external reference arrives, can correct what and to what extent
```

### 2A.7 Phase 2A results (executed)

#### Anchor measurements

```text
xy anchor (v2 OP, pitch=24): 
  CD_fix=14.527, LER_lock=2.471, margin=+0.096, area=0.606, status=robust_valid

xz anchor (thick=20, A=0.10, abs_len=30):
  H0_zmod=52.05%, P_zmod=20.34%, mod_red=60.93%, asym=0.185, 
  LER_lock(z)=2.353, PSD_mid=1.454
```

#### Sensitivity (relative span pct of metric over its sweep range vs anchor)

`relative_span_pct = 100 × (max - min) / |anchor|`. This quantifies the user-spec phrasing of "how much does each variable move each metric".

**x-y mode** (`pitch=24`, anchor at v2 OP):

| variable | CD_fixed | LER_locked | P_line_margin | area_frac | contrast |
|---|---|---|---|---|---|
| dose_mJ_cm2 (21–60)        | **62.6%** | 0.78% | **270%** | **62.4%** | 17.3% |
| sigma_nm (0–3)             | 18.2%  | 2.15% | 17.8% | 18.1% | **37.1%** |
| DH_nm2_s (0.3–0.8)         | 5.5%   | 5.3%  | 50.8% | 5.4%  | **40.5%** |

**x-z mode** (`thick=20, A=0.10, abs_len=30 anchor`):

| variable | H0_zmod | P_zmod | mod_red | top/bot | LER_z_locked | PSD_mid |
|---|---|---|---|---|---|---|
| dose_mJ_cm2          | 43.7% | 86.5% | 25.9% | 76.4% | 17.4% | 35.0% |
| sigma_nm             | 1.4%  | 8.3%  | 4.4%  | 7.5%  | **69.3%** | **129.1%** |
| absorption_length_nm | **151.6%** | **220.7%** | 43.0% | **190.8%** | **185.0%** | **484.2%** |
| DH_nm2_s             | 0.0%  | 8.1%  | 5.2%  | 7.3%  | 26.6% | 43.5% |

#### Key findings (controllability map)

```text
1. Primary knobs for CD / process window:
   - dose (CD 62.6%, area 62.4%, margin 270%) — main dimension of the Stage 5 process window
   - sigma is secondary (CD 18%, contrast 37%)
   - DH fine-tunes LER but has only a small effect on CD (5.5%)

2. Primary knobs for z-modulation / side-wall:
   - absorption_length (overwhelming across every z-metric; PSD_mid 484%)
   - dose weakens z-modulation through H0 saturation (P_zmod 86%, top/bot 76%)
   - sigma has almost no effect on z-modulation itself (H0_zmod 1.4%) but strongly tunes side-wall LER (69%)
   - DH is mild in the z-direction

3. Where the v2 OP sits inside each sensitive zone:
   - dose=40 sits stably at the centre of the process window (matches Stage 5)
   - sigma=2 is in the middle of the robust region σ ∈ [0,3]
   - DH=0.5 is a reasonable trade-off between LER and margin (margin is 0.028 larger than the DH=0.8 candidate)
   - abs_len=30 sits at the logarithmic middle of the z-mod knob (logarithmic mean of 15..100 ≈ 39)

4. When external reference arrives, which knob can correct what and how far:
   - CD offset > 5%: dose correction. Current dose 21–60 gives CD 8.1–17.2 (range 9.1 nm)
   - LER offset (intrinsic): DH correction. DH 0.3–0.8 gives LER 2.46–2.59 (range 0.13)
                              or sigma correction (LER varies only ~2.15% over sigma 0–3)
   - z-modulation offset: abs_len correction. abs_len 15–100 gives P_zmod 5.2–50.1 (range 9.6×)
   - top/bottom asymmetry offset: abs_len correction (0.40→0.05 over 15→100)

5. Incompatible regions:
   - dose=21 → under_exposed (every sigma, DH)
   - sigma=3, DH=0.8, dose=40 → still robust but with margin loss
   - abs_len=15 → mod_red 50%, LER 5 nm large (not the desired regime, but still a valid region)
```

#### Conclusion

```text
Controllability map complete. When external reference data arrives:
- CD bias within 0~3 nm: dose correction alone is enough
- LER bias within 0~0.2 nm: DH correction is enough
- z-mod / sidewall bias: abs_len is the dominant knob
- sigma is a secondary tuner for the process-window position

Still internal-only. external published_data_loaded=false.
Recommended next phase:

(α) Acquire external reference data → set published_data_loaded=true in
    calibration_targets.yaml + update targets/values → re-run Phase 1 + Phase 2A
    → Phase 3 (real calibration)

(β) Proceed with deferred Stages (Stage 3B / 5C / 6B or new chemistry) — calibration deferred

Current step: (α) is a prerequisite. While waiting for external data, choose
between (β) or another task.
```

---

## Phase 2B — sensitivity atlas (sensitivity / controllability / hypothesis study)

NOT a calibration. v2 OP frozen. Atlas = mapping the dynamic range / sensitivity around the nominal OP.

### Part A — x-y atlas

```yaml
anchor: frozen v2 OP (pitch=24, dose=40, sigma=2, DH=0.5, time=30,
                       Q0=0.02, kq=1.0, Hmax=0.2, kdep=0.5)

OAT sweeps:
  dose_mJ_cm2: [21, 28.4, 40, 44.2, 59, 60]
  sigma_nm:    [0, 1, 2, 3]
  DH_nm2_s:    [0.3, 0.5, 0.8]
  time_s:      [20, 30, 45]
  Q0_mol_dm3:  [0.0, 0.005, 0.01, 0.02, 0.03]

pair sweeps:
  dose × sigma : 6 × 4 = 24
  DH   × time  : 3 × 3 = 9
  sigma × Q0   : 4 × 5 = 20
  pitch × dose : 6 × 6 = 36   (pitch ∈ {16,18,20,24,28,32})
  pitch × Q0   : 6 × 5 = 30
```

### Part B — x-z atlas (4D grid)

```yaml
chemistry frozen at v2 OP.
period_nm = 6.75 (Stage 6).

grid:
  film_thickness_nm:    [15, 20, 30]
  standing_wave_amplitude: [0.0, 0.05, 0.10, 0.20]
  absorption_length_nm: [15, 30, 60, 100]
  DH_nm2_s:             [0.3, 0.5, 0.8]
=> 3 × 4 × 4 × 3 = 144 runs.
```

### Part C — Stage 5C small-pitch follow-up

Hypothesis: does weakening the quencher or lowering sigma restore the process window for pitch ∈ {18, 20}? (Follow-up to Stage 5 + Stage 4B.)

```yaml
pitch_nm:    [18, 20]
dose_mJ_cm2: [21, 28.4, 40]
sigma_nm:    [0, 1, 2]
quencher:    {off, weak (Q0=0.01, kq=1.0)}
DH_nm2_s:    [0.3, 0.5]
time_s:      [20, 30]
=> 2 × 3 × 3 × 2 × 2 × 2 = 144 runs.
```

Stage 6B (full 3D) is deferred until external data or a concrete interaction hypothesis appears.

### 2B.X Results (executed)

#### Part A — x-y atlas (141 rows total)

OAT and the 5 pair sweeps were all executed. The automatic domain alignment (pitch×5) worked correctly, and every row was saved to CSV / heatmap.

**OAT key trends (anchor: CD_fix=14.53, LER_lock=2.47, margin=0.10, robust_valid)**:

```text
dose 21..60        : CD_fix 8.1 → 17.2 (CD is the most dose-sensitive)
                     LER variation ≤ 1 % (matches Phase 2A exactly)
                     dose=21 only → under_exposed
sigma 0..3         : CD 13.0 → 15.7, margin 0.10 ± 0.02 (stays in the robust region)
DH 0.3..0.8        : LER 2.46 → 2.59 (DH is the fine knob for LER)
time 20..45        : LER 2.40 → 2.55 (matches Stage 2)
                     from time=45 onwards, merging risk in the dose=40, σ=2 region
Q0 0..0.03         : margin +0.142 → +0.061 (Q0 ↑ → margin compressed)
                     reconfirms Stage 4's monotone-decreasing dCD_shift effect
```

**Pair sweep highlights (status-heatmap-focused summary)**:

```text
dose × sigma  : robust_valid dominates the dose↑, σ↓ region. The whole dose=21 column is under_exposed.
DH × time     : DH=0.3, t=20 → under_exposed; DH=0.8, t=45 → merged.
                The middle region (DH=0.5, t=30) is robust_valid — exactly the anchor location.
sigma × Q0    : margin decreases monotonically with Q0; margin < 0.05 limit reached from σ=3, Q0=0.03.
pitch × dose  : Reproduces the Stage 5 result. pitch=16 closed, pitch ≥ 24 wide window.
pitch × Q0   : at pitch=18, 20 the quencher narrows the robust_valid region (matches the Stage 5 control).
```

#### Part B — x-z atlas (144 rows, full 4D grid)

bounds_ok 144/144. anchor (thick=20, A=0.10, abs_len=30, DH=0.5).

**Sensitivity (full-grid relative span vs anchor metric)**:

```text
metric                            thickness   amplitude   abs_len   DH
H0_z_modulation_sw_only_pct       (0 by definition)  monotone↑   strong   ~0
P_final_z_modulation_pct          large       monotone↑   strong   small
modulation_reduction_pct          thin>thick  ↓ at A↑     mid     small
top_bottom_asymmetry              ↑ thick     small       strong↓  small
sidewall_x_displacement_std       ↑ thick     small       strong↓  ↑ DH
psd_mid_band_locked               ↑ thick     small       strong↓  ↑ DH
```

The full 144-cell sweep reconfirms — consistent with the Phase 2A OAT results — that abs_len is the dominant knob across every z-metric.

#### Part C — Stage 5C small-pitch hypothesis (144 runs)

**Hypothesis**: does lowering σ or weakening the quencher restore the process window for pitch ∈ {18, 20}?

**Status counts per (pitch, σ, quencher)** (each cell = 12 runs over dose × DH × time):

```text
  pitch  σ   q_mode    robust  valid  under  merged
    18   0   off          3      3      5      1
    18   0   weak         2      3      6      1
    18   1   off          1      3      5      3
    18   1   weak         1      3      6      2
    18   2   off          0      3      5      4
    18   2   weak         1      2      6      3
    20   0   off          4      3      5      0
    20   0   weak         3      3      6      0
    20   1   off          4      3      5      0
    20   1   weak         3      3      6      0
    20   2   off          2      3      5      2
    20   2   weak         2      3      6      1
```

**Key findings**:

```text
Lowering σ is the dominant knob for restoring the small-pitch process window.
  pitch=18, σ=0: 3 robust (vs σ=2: 0-1 robust)  → ★ changing σ is decisive
  pitch=20, σ=0,1: 4 robust   (vs σ=2: 2 robust)

Weakening the quencher has a small or even negative effect.
  In most (pitch, σ) combinations, weak quencher → -1 robust count
  (reconfirming the Stage 5 finding that the quencher narrows small-pitch tolerance).

Best per pitch:
  pitch=18: dose=28.4, σ=0, DH=0.3, t=30, weak  → robust_valid, CD_shift=+0.25, LER_lock=3.05
  pitch=20: dose=28.4, σ=0, DH=0.3, t=30, weak  → robust_valid, CD_shift=-0.23, LER_lock=2.69
  → both at σ=0, dose=28.4, DH=0.3 (light chemistry, OP different from large-pitch)
```

#### 2B Overall conclusion

```text
1. The recommended v2 OP (pitch=24, dose=40, σ=2, DH=0.5, time=30, Q0=0.02, kq=1.0)
   is internal-consistent and is suitable as the anchor of the Phase 2B atlas.
   Still frozen. External reference is not available.

2. Quantitative mapping of the knobs available for correction once external data arrives:
   - x-y CD bias       → dose (primary)
   - x-y LER bias      → DH (fine), σ (secondary)
   - x-y margin / area → dose, Q0
   - x-z standing wave → absorption_length (overwhelmingly dominant)
   - x-z top/bot asym  → absorption_length

3. Stage 5C hypothesis confirmed:
   A separate OP exists for pitch ∈ {18, 20} (σ=0, dose=28.4, DH=0.3, t=30,
   quencher weak). Different chemistry / domain from the v2 OP. If subsequent
   work depends on small-pitch, this OP can be used as a hypothesis-verified
   candidate. Still internal-only.

4. Stage 6B (full 3D x-y-z) remains deferred. It will only start once external
   reference arrives or once a concrete y-roughness × z-modulation interaction
   hypothesis emerges.
```

#### Next-step decision

```text
RECOMMEND: wait for external reference data.

Options (in the meantime):
  - additional hypothesis sweep (detailed analysis of a specific pair / triple)
  - keep Stage 3B / 5C / 6B deferred
  - close the freeze in its current state and move on to another task / a v3 plan

calibration_status remains internal-consistency only.
```

---

## Phase 2 — process-window re-verification

Condition: only run if a new OP candidate was adopted in Phase 1.

### 2.1 Purpose

Verify that the new (Hmax, kdep, DH) reproduces the Stage 5 process-window shape (pitch=16 closed, pitch ≥ 24 wide window) as is.

### 2.2 Sweep

Apply the same pitch × dose grid as in Stage 5 (6 × 6 = 36 runs) to the new OP.

### 2.3 Gates

```text
PASS:
  the robust_valid region at pitch=24 is robust_valid across the entire dose=28.4 ~ 60 range (or wider)
  the pitch=16 process window is still closed (or the cliff has moved to ≤ 18)
  → proceed to Phase 3

FAIL:
  the process window has narrowed or the cliff position is outside a reasonable range
  → discard the Phase 1 result, fall back to Phase 2A
```

### 2.4 Phase 2 results

(not executed)

---

## Phase 3 — external-reference comparison

Condition: Phase 2 passed.

### 3.1 Purpose

Quantitative comparison against published or measured high-NA EUV PEB data.

### 3.2 Comparison items

```text
- absolute CD, LER values at 24 nm pitch / 12.5 nm CD / 40 mJ/cm²
- LER reduction curves vs DH × time or vs PEB temperature
- pitch dependence (Stage 5 status-map shape)
- standing-wave amplitude reduction (Stage 6 trend)
```

### 3.3 Calibration-knob mapping

```text
- absolute CD offset → correct via dose / Hmax / kdep
- absolute LER offset → correct initial roughness amp or correlation length
- process-window-shape offset → kdep / quencher scaling
- standing-wave reduction offset → correct abs_len
```

### 3.4 Phase 3 results

(not executed)

---

## Phase 4 — start of deferred stages

Condition: Phase 3 passed.

```text
Stage 3B (σ=5/8 compatible budget search) — extending kdep / dose / Hmax
Stage 5C (σ=0 small-pitch process window)
Stage 6B (full 3D x-y-z)
or new chemistry (Dill ABC model, PAG profile, etc.)
```

The starting order is decided based on the Phase 3 results.

---

## Decision tree

```text
Phase 1 sweep
  │
  ├── PASS (best score < 0.10)  →  Phase 2 (process window)
  │                                  │
  │                                  ├── PASS  →  Phase 3 (external reference)
  │                                  │             │
  │                                  │             ├── PASS  →  Phase 4 (deferred stages)
  │                                  │             └── FAIL  →  re-tune chemistry, loop back to Phase 1
  │                                  │
  │                                  └── FAIL  →  Phase 2A (extended sweep) → re-run Phase 1
  │
  ├── PASS-marginal (best score 0.10-0.20)  →  Phase 2 with finer grid
  │
  └── FAIL (best score > 0.20)  →  Phase 2A (extending dose/σ/abs_len)
                                     →  re-run Phase 1 with new variables
```
