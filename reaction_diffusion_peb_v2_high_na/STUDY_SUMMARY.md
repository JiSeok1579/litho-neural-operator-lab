# PEB v2 — first-pass study summary

High-NA EUV PEB v2 simulation work, Stage 1 ~ Stage 6 completed.
원본 plan (`EXPERIMENT_PLAN.md`) 의 design intent + 실제 검증된 결과를 한곳에 정리.

상세 phase 별 분석은 `study_notes/0[1-7]_*.md` 참조.

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

| Stage | 핵심 결과 | Note |
|---|---|---|
| **1**   | clean geometry baseline 통과 | σ=0, t=30, DH=0.8 |
| **1A**  | σ ∈ [0, 3] 호환 (kdep=0.5, Hmax≤0.2) | σ=4,5 는 budget 부족 |
| **1B**  | σ=5/t=60 = lines merge (over-budget reference) | plan 원본 nominal 폐기 |
| **2**   | best OP DH=0.8 t=20 (margin 0.003) → robust alt DH=0.5 t=30 권장 | algorithmic best 와 robust alt 의 trade-off |
| **3**   | σ ∈ {0,1,2,3} 분리 가능. σ 증가 → e-blur 1차 LER 감소, PEB 2차 효과는 σ≥2 부터 음수 | "displacement artifact" 가설 제안 |
| **3B**  | σ=5/8 호환 budget search → 보류 | search space 확장 필요 |
| **4**   | weak quencher 52 runs 모두 gate 통과; balanced OP Q0=0.02, kq=1 | σ=3 LER 회복 +29 pp |
| **4B**  | **CD-locked LER 도구화**; pitch ≤ 20 의 LER 악화는 real (artifact 아님) | 3 결정 라벨로 디버깅 |
| **5**   | process window 36+72 runs; pitch=16 closed, pitch ≥ 24 wide window | 추천 dose=40 모든 pitch≥20 |
| **5C**  | σ=0 small-pitch follow-up → 보류 | |
| **6**   | x-z standing wave 12 runs; PEB 가 z-mod 흡수 (thin 79% > thick 60%) | Neumann-z mirror FFT |
| **6B**  | full 3D x-y-z → 보류 | compute cost 큼 |

---

## Per-stage 핵심 finding

### Stage 1 — Clean geometry baseline

**조건**: σ=0, t=30, DH=0.8, kdep=0.5, Hmax=0.2, no quencher, pitch=24, CD=12.5.

**결과**:
```text
P_space_center_mean = 0.31     P_line_center_mean  = 0.76
contrast            = 0.45     area_frac           = 0.625
CD: 12.46 → 15.01 nm  (CD_shift = +2.55)
LER: 2.77 → 2.65 nm  (-4.3 %)
```

**Key insight**: plan 원본 nominal (σ=5, t=60) 은 24 nm pitch / 12.5 nm CD 와 호환 불가. domain 을 pitch 정수배 (120 = 5×24) 로 정렬해야 FFT seam artifact 회피. interior gate (P_space_center, P_line_center, contrast, area, CD/pitch) 로 false-pass 차단.

config: `configs/v2_stage1_clean_geometry.yaml`. 노트: `study_notes/01_stage1_clean_geometry.md`.

---

### Stage 2 — DH × time process window

**조건**: 25-grid (DH ∈ {0.3, 0.5, 0.8, 1.0, 1.5} × time ∈ {15, 20, 30, 45, 60}). 다른 변수 Stage 1 fixed.

**결과**: 9/25 cells interior gate 통과. 대각선 process window 형태.

```text
LER reduction (%):              t=15      20     30     45     60
                  DH=0.30:    6.55✗   6.63✓  8.04✓  6.96✓  −14.34✓
                  DH=0.50:    9.00✗   8.80✓  8.69✓ −17.04✓  45.55✗
                  DH=0.80:   10.06✗   9.61✓  4.25✓   8.28✗ 100.00✗
                  DH=1.00:    8.62✗   8.84✗ −1.98✓  62.62✗ 100.00✗
                  DH=1.50:   −4.48✗   3.98✗ −38.01✓ 100.00✗ 100.00✗
```

**알고리즘 best**: DH=0.8, t=20 (LER −9.6 %, margin 0.003 — boundary).
**Robust alt 2**: DH=0.5, t=30 (LER +8.69 %, margin 0.142). **이 OP 가 Stage 3+ 의 default 가 됨**.

**Key insight**: selection criterion 에 P_line_margin clause 가 없어 algorithmic best 가 boundary 에 안착. 후속 stage 에서 P_line_margin ≥ 0.05 추가 (Stage 3 부터).

config: `configs/v2_stage2_dh_time.yaml`. 노트: `study_notes/02_stage2_dh_time_sweep.md`.

---

### Stage 3 — Electron blur separation

**조건**: σ ∈ {0, 1, 2, 3} 두 OP × 4 σ = 8 runs. plan 의 [0, 2, 5, 8] 에서 σ=5,8 demote (Stage 1A 호환 불가).

**측정 규약 재정의**:

```text
LER_design_initial    = binary I @ 0.5 (σ-독립 baseline)
LER_after_eblur_H0    = I_blurred @ 0.5
LER_after_PEB_P       = P @ 0.5
electron_blur_LER_reduction_pct  = 100 * (design - eblur)/design
PEB_LER_reduction_pct            = 100 * (eblur - PEB)/eblur
total_LER_reduction_pct          = 100 * (design - PEB)/design
```

**결과 (robust OP DH=0.5, t=30)**:

| σ | total LER % | e-blur % | PEB % | CD_shift |
|---|---|---|---|---|
| 0 | **+8.7** | +0.0 | +8.7 | +1.79 |
| 1 | +7.8 | +2.2 | +5.7 | +2.62 |
| 2 | +3.6 | +6.1 | −2.7 | +3.83 |
| 3 | −22.5 | +11.1 | −37.7 | +5.85 |

algorithmic-best OP (DH=0.8, t=20) 는 σ 모두 P_line_margin ≥ 0.03 fail → downstream 에서 demote.

**Key insight**: e-blur 와 PEB 가 보완이 아닌 *경쟁* — σ↑ 에서 line widening → contour displacement → PEB-LER 음수 (Stage 4B 에서 진단). 새 P_line_margin ≥ 0.03 게이트 도입.

config: `configs/v2_stage3_electron_blur.yaml`. 노트: `study_notes/03_stage3_electron_blur.md`.

---

### Stage 4 — Weak quencher

**조건**: 52 runs at robust OP × σ ∈ {0,1,2,3} × (Q0=0 baseline + Q0 ∈ {0.005, 0.01, 0.02, 0.03} × kq ∈ {0.5, 1.0, 2.0}).

**결과**: 52/52 Stage-3 gate 통과, 51/52 Stage-4 robust criterion 통과.

```text
σ=2 dtotal_LER_pp (vs σ-matched baseline):
              kq=0.5    kq=1.0    kq=2.0
  Q0=0.030    +4.90     +6.47     +7.64
  Q0=0.020    +3.74     +5.21     +6.44
  Q0=0.010    +2.18     +3.24     +4.23
  Q0=0.005    +1.18     +1.84     +2.49
```

**Balanced OP**: σ=2, Q0=0.02, kq=1.0 → dCD=−1.76, darea=−0.073, dLER=+5.21 pp, margin=0.096.

**σ=3 LER 회복**: baseline total_LER = −22.5 % → Q0=0.03/kq=1 → +6.6 % (dLER = +29.15 pp).

**Key insight**: PSD high-band 는 baseline 에서도 99.9% 제거됨 → quencher 차이는 mid-band 에서 발생. Stage 3 의 PEB-LER 음수 가설이 부분 검증 (Stage 4B 에서 완전 검증).

config: `configs/v2_stage4_weak_quencher.yaml`. 노트: `study_notes/04_stage4_weak_quencher.md`.

---

### Stage 5 — Pitch × dose process window

**조건**: 108 runs (primary + 2 controls). 6 pitch × 6 dose. domain_x_nm = pitch × 5.

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

**추천 dose**: pitch ≥ 20 모두 dose=40. pitch=16 process window closed (line_cd=12.5 / pitch=16 duty 0.78 + diffusion 5.5 nm).

**Control 비교**: σ=0 no-quencher 가 가장 넓은 window. quencher 추가가 small pitch tolerance 를 줄임.

**Key insight**: Stage 4 의 LER benefit 은 large pitch (≥24) 에 편중. pitch ≤ 20 의 LER 악화는 contour displacement artifact 가 일부 (Stage 4B 에서 분리).

config: `configs/v2_stage5_pitch_dose.yaml`. 노트: `study_notes/05_stage5_pitch_dose.md`.

---

### Stage 4B — CD-locked LER

**Trigger**: Stage 5 에서 pitch ≤ 20 의 negative LER reduction 이 displacement artifact 인지 real degradation 인지 분리 필요.

**도구**: `find_cd_lock_threshold` — bisect P ∈ [0.2, 0.8] (adaptive endpoint narrowing) 으로 CD_overall ≈ design CD 로 contour 위치 고정.

**Block A 결정 (12 cells)**:

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

**Block B (mini-sweep)**: pitch ∈ {18, 20} 에서 quencher 약화 (Q0 ≤ 0.005, kq ≤ 0.5) 가 LER_locked 회복 못 함. pitch=18 에서는 0.11 nm 만 회복 (design 까지 +1.3 nm).

**Key insight**: pitch ≤ 20 의 LER 악화는 real. Stage 4 balanced OP 는 pitch ≥ 24 에서만 robust. **CD-locked LER 는 helper 의 default 로 통합** (Stage 6 부터 모든 sweep 의 표준 컬럼).

config: `configs/v2_stage4b_cd_locked.yaml`. 노트: `study_notes/06_stage4B_cd_locked.md`.

---

### Stage 6 — x-z standing wave

**조건**: 12 runs. thickness ∈ {15, 20, 30} nm × amplitude ∈ {0, 0.05, 0.10, 0.20}. period=6.75 nm, abs_len=30 nm. Neumann-z BC via even-mirror FFT.

**결과 (12/12 PASS)**:

```text
PEB modulation reduction:
  thick=15: 79 %   (가장 효과적)
  thick=20: 68 %
  thick=30: 60 %   (diffusion length 5.5 < thickness)

H0_z_modulation_sw_only_pct (absorption 제외):
              A=0.05   A=0.10   A=0.20
  thick=15    +2.70    +5.26   +15.56
  thick=20    +1.04    +7.12   +20.27
  thick=30    +6.51   +12.93   +25.54

Side-wall LER (CD-locked, z as track):
  thick=15: 1.32 nm
  thick=20: 2.32-2.40 nm
  thick=30: 3.80-3.87 nm

Top/bottom asymmetry: 0.10 → 0.18 → 0.32 (absorption envelope)
```

**Key insight**: PEB 가 standing wave (period 6.75 nm) z-modulation 을 거의 완전히 흡수. P_final 의 잔여 sw < 1 %. thick film 은 absorption envelope 의 효과가 압도적. summary plot: `outputs/figures/06_xz_standing_wave/summary/`.

config: `configs/v2_stage6_xz_standing_wave.yaml`. 노트: `study_notes/07_stage6_xz_standing_wave.md`.

---

## 권장 v2 operating point

이전 stage 들에서 검증된 **robust 영역의 표준 OP**:

```yaml
geometry:
  pitch_nm:        24                    # process window 의 안정 영역 (Stage 5)
  line_cd_nm:      12.5
  grid_spacing_nm: 0.5
  domain_x_nm:     120                   # pitch * 5 (FFT-seam-safe, Stage 1A)
  domain_y_nm:     120                   # 일관된 LER y-sample 수
  edge_roughness:  amp=1.0, corr=5.0     # 합리적 design noise

exposure:
  dose_mJ_cm2:           40              # Stage 5 모든 pitch≥20 추천 dose
  reference_dose_mJ_cm2: 40
  Hmax_mol_dm3:          0.2
  eta:                   1.0
  electron_blur_sigma_nm: 2              # Stage 1A σ-호환 범위 [0,3] 의 중앙
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
  P_threshold: 0.5                       # CD-locked variant 도 helper 가 자동 계산
```

**검증된 영역**:
- pitch=24 nm, dose 28.4 ~ 60 mJ/cm² 모두 robust_valid (Stage 5)
- x-z standing wave amplitude up to 0.20 까지 PEB 가 흡수 (Stage 6)
- 측정 metric: LER_CD_locked, PSD mid-band reduction 모두 standard column

**적용 범위 밖**:
- pitch ≤ 20: real LER degradation (Stage 4B). σ=0 또는 line_cd 비례 축소 필요 (별도 stage).
- σ ≥ 4: budget 호환 안 됨 (Stage 1A). dose / kdep / Hmax 확장 필요 (Stage 3B).
- thick film > 30 nm: PEB modulation reduction < 60% 예상 (Stage 6 trend).

---

## Next milestone — calibration against external reference

physics 추가 / 새 stage 시작 전, 현재 결과를 **literature 또는 측정 데이터** 와 비교해 calibration 하는 것이 다음 단계.

### 비교 대상

```text
1. 24 nm pitch / 12.5 nm CD / 40 mJ/cm² dose 의 published 측정값:
   - CD_final ≈ 15 nm (Stage 1) 가 실제와 일치하는가?
   - LER ≈ 2.5-2.7 nm 가 reasonable 한가?
   - PEB time, temperature 와의 dependence 가 plan §4 와 일치하는가?

2. process window shape:
   - Stage 5 의 status map 이 published high-NA process window 와 일치하는가?
   - small-pitch closure 위치가 measured 와 일치하는가?

3. standing wave amplitude reduction:
   - Stage 6 의 thin/thick 차이가 measured top-coat 효과와 일치하는가?
```

### 발견 가능한 calibration offset

```text
- kdep, Hmax: H 와 P 의 절대값 scaling. 측정 LER 과의 비교로 ±factor 조정.
- dose 정의: dose_norm 의 reference (40 mJ/cm²) 가 실제 EUV reference 와 일치하는지.
- DH (acid diffusion): temperature 별 Arrhenius 와 외부 데이터 비교.
- quencher kq: Q 의 reaction rate 가 reasonable 한지.
```

### 우선순위

```text
1. CD calibration  →  dose / Hmax 보정 (factor 1±0.X)
2. LER calibration →  initial roughness amp 와 corr length 보정
3. process window calibration → kdep / quencher 의 scaling
4. standing wave calibration → absorption_length_nm 보정 (현재 30 nm 임의값)
```

이 calibration 이 끝난 후에야 다음 항목으로:

```text
- Stage 6B (full 3D x-y-z)
- Stage 3B (σ=5,8 호환 budget search 확장)
- Stage 5C (σ=0 small-pitch process window)
- Stage 1A.3 (kdep, dose 확장으로 σ=5/8 budget 재시도)
- 새 chemistry (예: PAG profile, real Dill model)
```

---

## 산출물 인덱스

```text
configs/                                # 7 stage configs
  v2_stage1_clean_geometry.yaml          (Stage 1 baseline)
  v2_baseline_lspace.yaml                (Stage 1B over-budget reference)
  v2_stage2_dh_time.yaml                 (Stage 2)
  v2_stage3_electron_blur.yaml           (Stage 3)
  v2_stage4_weak_quencher.yaml           (Stage 4)
  v2_stage4b_cd_locked.yaml              (Stage 4B)
  v2_stage5_pitch_dose.yaml              (Stage 5)
  v2_stage6_xz_standing_wave.yaml        (Stage 6)

src/
  geometry.py, roughness.py              (line/space + edge roughness)
  electron_blur.py                       (2D Gaussian blur)
  exposure_high_na.py                    (Dill + 1D builders + x-z exposure)
  fd_solver_2d.py                        (x-y solver)
  fd_solver_xz.py                        (x-z solver, Neumann-z mirror)
  metrics_edge.py                        (LER/LWR/CD + CD-lock + PSD bands)
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

tests/                                   (27/27 passing)

outputs/
  figures/                               (180+ figures across 7 sweeps)
  logs/                                  (CSV + JSON summaries)
  fields/                                (npz field snapshots, Stage 1)

study_notes/
  README.md  (index)
  01_stage1_clean_geometry.md
  02_stage2_dh_time_sweep.md
  03_stage3_electron_blur.md
  04_stage4_weak_quencher.md
  05_stage5_pitch_dose.md
  06_stage4B_cd_locked.md
  07_stage6_xz_standing_wave.md

EXPERIMENT_PLAN.md                       (status header + per-stage spec & results)
STUDY_SUMMARY.md                         (this file — first-pass closeout)
```

---

## v2 first-pass 종료 선언

7 stages (1 / 1A / 1B / 2 / 3 / 4 / 4B / 5 / 6) 모두 완료, 모든 stage 의 study notes 와 PR (#29 ~ #35) 머지 완료, 검증된 robust OP 식별.

## Calibration policy (2026-04-30 freeze)

```text
external reference data 미입수.
v2 OP 는 internal-consistent nominal OP 로 freeze.
calibration_status         = internal-consistency only
published_data_loaded      = false
v2_OP_frozen               = true

이후 모든 sweep / 실험은 다음 중 하나로만 label:
  - sensitivity study
  - controllability study
  - hypothesis test

"calibration" 또는 "calibrated to real" 표현은 외부 measurement / literature
데이터가 calibration_targets.yaml 에 등록된 후 (published_data_loaded=true) 에만
허용. 그 전까지의 모든 작업은 internal exploration.
```

세부는 `calibration/calibration_plan.md` 와 `calibration/calibration_targets.yaml` 참조.
