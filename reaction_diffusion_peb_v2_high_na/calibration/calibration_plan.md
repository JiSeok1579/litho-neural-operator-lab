# PEB v2 calibration plan

## v2 OP freeze (2026-04-30)

```text
external reference data 미입수.
v2 first-pass OP 를 internal-consistent nominal OP 로 freeze:
  pitch=24, dose=40, sigma=2, DH=0.5, time=30,
  kdep=0.5, Hmax=0.2, kloss=0.005, Q0=0.02, kq=1.0, DQ=0.0

calibration_status: internal-consistency only.
published_data_loaded: false.
v2_OP_frozen: true.

이후 모든 runs 는 다음 중 하나로 label:
  - sensitivity study
  - controllability study
  - hypothesis test
"calibration" 또는 "calibrated to real" 표현은 published_data_loaded=true 까지 금지.
```

## Strategy

```text
external reference 가 없는 상태에서 chemistry knob 의 dynamic range 와
sensitivity 를 미리 정량화해, 외부 데이터가 도입되었을 때 어느 knob 으로
어디까지 보정 가능한지 사전 매핑.
```

`calibration_targets.yaml` 가 정량적 목표 + frozen OP 를, 본 문서가 phase 별 진행 / 결정 / 결과를 누적 기록한다.

---

## Phase 1 — chemistry 핵심 변수 sweep

### 1.1 목적

```text
v2 OP 의 (CD, LER) 가 target (15 nm / 2.6 nm) 에 얼마나 가까운가?
가깝지 않다면 Hmax / kdep / DH 중 어느 변수가 설명하는가?
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

총 4 × 3 × 3 = 36 runs.

### 1.3 측정 (per cell)

```text
CD_locked_nm
LER_CD_locked_nm
P_threshold_locked
P_line_margin
area_frac
psd_mid_band (PEB 후 mid band power)
status (Stage-5 분류 — robust_valid / valid / under_exposed / merged / unstable)
distance_to_target = sqrt( ((CD_locked-15)/15)^2 + ((LER_locked-2.6)/2.6)^2 )
```

### 1.4 게이트 / 결정

```text
PASS:
  cell 중 distance_to_target < 0.10 인 cell 이 1개 이상 (robust_valid 또는 valid 인 cell 만 선정)
  → 이 cell 의 (Hmax, kdep, DH) 를 새 OP 후보로 채택
  → Phase 2 진행

PASS-marginal:
  distance < 0.20 인 cell 이 있으나 0.10 미만 없음
  → best cell 채택 후 Phase 2 진행 (범위 더 미세하게)

FAIL:
  최소 distance > 0.20
  → Phase 2A: dose, sigma, abs_len 까지 확장 sweep
```

### 1.5 Phase 1 결과 (executed)

#### 측정 방법론 수정

처음 작성한 distance 함수가 `CD_locked` 을 사용했는데, CD-locked 알고리즘은 정의상 contour 를 design CD (~12.5 nm) 로 강제 이동시키므로 모든 row 에서 CD ≈ 12.5 → distance 가 systematic offset 으로 ~0.17 만 나옴. **distance 의 CD 항은 fixed-threshold CD (`CD_final_nm`) 로 변경** — published / measured 의 "developed pattern CD" 와 같은 metric.

```python
distance_to_target = sqrt( ((CD_fixed - 15)/15)^2 + ((LER_locked - 2.6)/2.6)^2 )
```

LER 항은 CD-locked LER 그대로 (intrinsic roughness, displacement bias 제거).

#### Sweep 결과 (36 runs)

```text
gate                           : PASS
best score                     : 0.0054   (target: < 0.10)
n_selectable                   : 24 / 36   (robust_valid 또는 valid)
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

#### offset 분류

- **kdep=0.35 row 들 모두 score > 0.20** — under_exposed 우세. acid generation 이 약한 게 아니라 reaction rate 가 부족해 lines 가 충분히 deprotect 되지 않음.
- **Hmax 단독 영향은 작음** — 같은 (kdep, DH) 에서 Hmax 0.15→0.22 변화 시 CD 가 ~1-3 nm 변동 (1순위 변수 아님).
- **DH 가 LER 에 가장 직접적인 영향** — DH 0.3→0.8 에서 LER 2.46→2.59 (정확히 target 2.6 으로 수렴).
- **CD 는 (kdep, DH) 의 곱에 가까운 의존성** — distance heatmap 의 minimum 이 (kdep=0.5, DH=0.8) 에 위치.

```text
offset 진단: 
  acid generation (Hmax)   : 영향 작음 — Hmax 0.20 이 합리적
  reaction rate (kdep)     : kdep=0.35 부족, 0.5-0.65 적절
  acid diffusion (DH)      : DH=0.5 → 0.8 로 올리면 CD 와 LER 둘 다 target 에 더 가까워짐
  electron blur, dose, abs_len : Phase 2A 에서 추가 검증 필요 (현재 sweep 에서는 fixed)
```

#### 결정

**v2 OP (Hmax=0.20, kdep=0.50, DH=0.50) 자체가 score 0.0588 < 0.10 → 이미 internal target 과 일치**.
top candidate (DH=0.80) 가 더 잘 맞으나, Stage 2 의 robust window (DH=0.5/t=30) trade-off 와 비교해 production-ready 한 측면에서 v2 OP 유지 권장.

```text
Phase 1 finding 요약:
  v2 OP 는 internal calibration target 과 일치 (score 0.0588).
  CD 가 marginal 하게 작음 (14.53 vs 15.0) — DH=0.5 → 0.8 변경 시 보정 가능.
  그러나 internal target 은 v2 first-pass 관측값에서 도출 — 진짜 calibration 은 외부 reference 필요.
```

#### Caveat — recursive calibration 위험

```text
calibration target (CD=15, LER=2.6) 은 v2 first-pass 의 Stage 1 값에서 도출.
즉 본 Phase 1 은 "v2 가 자신의 first-pass 결과를 재현하는가" 를 검증한 것.
실제 calibration 은 외부 published / measured 데이터가 필요.
→ Phase 3 (외부 reference 비교) 가 본질적 다음 단계.
→ 그 전에 Phase 2 (process window 재검증) 를 새 OP 후보 (DH=0.8) 로 짧게 돌려도 됨.
```

#### 다음 단계 결정 (resolved)

```text
[2026-04-30 closed]
Phase 1 = internal-consistency check 로 종결.
v2 권장 OP (Hmax=0.20, kdep=0.50, DH=0.50) 변경 없음.
DH=0.80 candidate 는 calibration_targets.yaml > internal_best_score_candidate 에 기록만.
"calibrated to real" 선언은 외부 reference 데이터가 published_data_loaded=true 로
들어온 후로 보류.

다음 → Phase 2A (sensitivity / controllability study, NOT true calibration).
```

---

## Phase 2A — sensitivity / controllability study

### 2A.1 목적

```text
v2 OP anchor 주변에서 변수 한 개씩 (OAT) 변동시킬 때 metric 들이 얼마나 움직이는가?
- 외부 reference 가 도입되었을 때 어떤 knob 으로 어디까지 조정 가능한지 미리 파악
- 각 metric 의 변수별 dynamic range 와 sensitivity coefficient 정량화
- "calibrated to real" 가 아닌 "controllability map" 으로 의도 명시
```

### 2A.2 Anchor

```text
x-y anchor (Stage 5 주변 영역):
  pitch=24, line_cd=12.5, dose=40, σ=2, time=30,
  Hmax=0.20, kdep=0.50, DH=0.50, kloss=0.005,
  Q0=0.02, kq=1.0, DQ=0

x-z anchor (Stage 6 주변 영역):
  film_thickness=20 nm, amplitude=0.10, period=6.75, abs_len=30
  나머지 chemistry 는 x-y anchor 와 동일
```

### 2A.3 Sweep (OAT)

```yaml
# 한 번에 한 변수만 anchor 에서 벗어남.
xy_sweeps:
  dose_mJ_cm2:        [21, 28.4, 40, 44.2, 59, 60]   # Stage 5 와 동일
  electron_blur_sigma_nm: [0, 1, 2, 3]                  # Stage 1A 호환 범위
  DH_nm2_s:           [0.3, 0.5, 0.8]                # Phase 1 best 후보 포함

xz_sweeps:
  dose_mJ_cm2:           [21, 28.4, 40, 44.2, 59, 60]
  electron_blur_sigma_nm:    [0, 1, 2, 3]
  absorption_length_nm:  [15, 20, 30, 50, 100]      # x-z 만 의미
  DH_nm2_s:              [0.3, 0.5, 0.8]
```

x-y 13 runs + x-z 18 runs = 31 runs total.

### 2A.4 측정 (per row)

```text
x-y row:
  CD_final_nm (fixed), CD_locked_nm
  LER_CD_locked_nm, LER_after_PEB_P_nm
  P_line_margin, area_frac, contrast
  status (Stage-5 분류)
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
local_slope       = d(metric)/d(variable) 의 변화율 (anchor 근처 finite difference)
status_changes    = 변수 sweep 동안 발생한 status transition 수
```

### 2A.6 게이트

게이트 없음 — 본 phase 는 controllability map 작성용. 단 다음을 reporting 에 포함:

```text
- 어느 변수가 어느 metric 에 가장 큰 영향을 주는지 ranked
- v2 OP 가 어느 변수의 sensitive zone 에 위치하는지 표시
- 외부 reference 가 들어오면 어느 knob 으로 어디까지 조정 가능한지 표시
```

### 2A.7 Phase 2A 결과 (executed)

#### Anchor 측정값

```text
xy anchor (v2 OP, pitch=24): 
  CD_fix=14.527, LER_lock=2.471, margin=+0.096, area=0.606, status=robust_valid

xz anchor (thick=20, A=0.10, abs_len=30):
  H0_zmod=52.05%, P_zmod=20.34%, mod_red=60.93%, asym=0.185, 
  LER_lock(z)=2.353, PSD_mid=1.454
```

#### Sensitivity (relative span pct of metric over its sweep range vs anchor)

`relative_span_pct = 100 × (max - min) / |anchor|`. 사용자 spec 의 "어느 변수가 어느 metric 을 얼마나 움직이는가" 를 정량화.

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

#### 핵심 finding (controllability map)

```text
1. CD / process window 의 주요 knob:
   - dose (CD 62.6%, area 62.4%, margin 270%) — Stage 5 process window 의 주된 dimension
   - sigma 는 보조적 (CD 18%, contrast 37%)
   - DH 는 LER 를 미세 조절하나 CD 에 작은 영향 (5.5%)

2. z-modulation / side-wall 의 주요 knob:
   - absorption_length (모든 z-metric 에서 압도적; PSD_mid 484%)
   - dose 는 H0 saturation 통해 z-mod 약화 (P_zmod 86%, top/bot 76%)
   - sigma 는 z-mod 자체에는 거의 영향 없음 (H0_zmod 1.4%) 하나 side-wall LER 강하게 조절 (69%)
   - DH 는 z-direction 에서 mild

3. v2 OP 가 sensitive zone 어디에 위치하는지:
   - dose=40 은 process window 중심부에 안정 (Stage 5 와 일치)
   - sigma=2 는 σ ∈ [0,3] 의 robust 영역 중간
   - DH=0.5 는 LER vs margin 의 합리적 절충 (DH=0.8 candidate 보다 margin 0.028 더 큼)
   - abs_len=30 은 z-mod knob 의 logarithmic 중간 위치 (15..100 의 logarithmic 평균 ~ 39)

4. 외부 reference 가 들어오면 어느 knob 으로 어디까지 조정 가능한가:
   - CD offset > 5%: dose 보정. 현재 dose 21–60 에서 CD 8.1–17.2 (range 9.1 nm)
   - LER offset (intrinsic): DH 보정. DH 0.3–0.8 에서 LER 2.46–2.59 (range 0.13)
                              또는 sigma 보정 (sigma 0–3 에서 LER 변동 작음 ~ 2.15%)
   - z-modulation offset: abs_len 보정. abs_len 15–100 에서 P_zmod 5.2–50.1 (range 9.6×)
   - top/bottom asymmetry offset: abs_len 보정 (15→100 에서 0.40→0.05)

5. 비호환 영역:
   - dose=21 → under_exposed (모든 sigma, DH)
   - sigma=3, DH=0.8, dose=40 → 여전히 robust 하지만 margin 손실
   - abs_len=15 → mod_red 50%, LER 5 nm 큼 (원하는 곳 아님 but 가능 영역)
```

#### 결론

```text
controllability map 작성 완료. 외부 reference 데이터가 들어오면:
- CD bias 0~3 nm 이내: dose 단독 보정으로 충분
- LER bias 0~0.2 nm 이내: DH 보정 충분
- z-mod / sidewall bias: abs_len 이 dominant knob
- sigma 는 process window 위치의 secondary tuner

여전히 internal-only. external published_data_loaded=false.
다음 phase 진행 권장:

(α) 외부 reference 데이터 입수 → calibration_targets.yaml 의 published_data_loaded=true
    + targets/values 갱신 → Phase 1 + Phase 2A 재실행 → Phase 3 (real calibration)

(β) deferred Stage 진행 (Stage 3B / 5C / 6B 또는 새 chemistry) — calibration 보류

지금 단계: (α) 가 prerequisite. 외부 데이터 대기 중에는 (β) 또는 다른 task 결정.
```

---

## Phase 2B — sensitivity atlas (sensitivity / controllability / hypothesis study)

NOT a calibration. v2 OP frozen. Atlas = nominal OP 주변 dynamic range / sensitivity 매핑.

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

Hypothesis: weak quencher 약화 또는 sigma↓ 가 pitch ∈ {18, 20} 의 process window 를 회복시키는가? (Stage 5 + Stage 4B 의 follow-up)

```yaml
pitch_nm:    [18, 20]
dose_mJ_cm2: [21, 28.4, 40]
sigma_nm:    [0, 1, 2]
quencher:    {off, weak (Q0=0.01, kq=1.0)}
DH_nm2_s:    [0.3, 0.5]
time_s:      [20, 30]
=> 2 × 3 × 3 × 2 × 2 × 2 = 144 runs.
```

Stage 6B (full 3D) 는 외부 데이터 또는 구체적 interaction hypothesis 가 등장하기 전까지 deferred.

### 2B.X 결과 (executed)

#### Part A — x-y atlas (141 rows total)

OAT 와 5 pair sweeps 모두 실행. 도메인 자동 정렬 (pitch×5) 정상 작동, 모든 row CSV / heatmap 저장.

**OAT 핵심 trend (anchor: CD_fix=14.53, LER_lock=2.47, margin=0.10, robust_valid)**:

```text
dose 21..60        : CD_fix 8.1 → 17.2 (CD 가 가장 dose-sensitive)
                     LER 변동 ≤ 1 % (정확히 Phase 2A 와 일치)
                     dose=21 only → under_exposed
sigma 0..3         : CD 13.0 → 15.7, margin 0.10 ± 0.02 (robust 영역 유지)
DH 0.3..0.8        : LER 2.46 → 2.59 (DH 가 LER 의 fine knob)
time 20..45        : LER 2.40 → 2.55 (Stage 2 와 일치)
                     time=45 부터 dose=40, σ=2 영역에서 merging 위험
Q0 0..0.03         : margin +0.142 → +0.061 (Q0 증가 → margin 압박)
                     Stage 4 의 dCD_shift 단조 감소 효과 재확인
```

**Pair sweep 핵심 (status heatmap 위주 요약)**:

```text
dose × sigma  : dose↑, σ↓ 영역 robust_valid 우세. dose=21 column 전체 under_exposed.
DH × time     : DH=0.3, t=20 → under_exposed; DH=0.8, t=45 → merged.
                중간 영역 (DH=0.5, t=30) robust_valid 정확히 anchor 위치.
sigma × Q0    : Q0 증가 시 margin 단조 감소; σ=3, Q0=0.03 부터 margin < 0.05 한계.
pitch × dose  : Stage 5 결과 재현. pitch=16 closed, pitch ≥ 24 wide window.
pitch × Q0    : pitch=18, 20 에서 quencher 가 robust_valid 영역을 좁힘 (Stage 5 control 과 일치).
```

#### Part B — x-z atlas (144 rows, 4D grid 전수)

bounds_ok 144/144. anchor (thick=20, A=0.10, abs_len=30, DH=0.5).

**Sensitivity (full-grid relative span vs anchor metric)**:

```text
metric                            thickness   amplitude   abs_len   DH
H0_z_modulation_sw_only_pct       (정의상 0)   monotone↑   strong   ~0
P_final_z_modulation_pct          large       monotone↑   strong   small
modulation_reduction_pct          thin>thick  ↓ at A↑     mid     small
top_bottom_asymmetry              ↑ thick     small       strong↓  small
sidewall_x_displacement_std       ↑ thick     small       strong↓  ↑ DH
psd_mid_band_locked               ↑ thick     small       strong↓  ↑ DH
```

abs_len 이 모든 z-metric 에서 dominant knob 임을 144-cell 전수 데이터로 재확인 (Phase 2A 의 OAT 와 일치).

#### Part C — Stage 5C small-pitch hypothesis (144 runs)

**Hypothesis**: σ↓ 또는 quencher 약화 가 pitch ∈ {18, 20} 의 process window 를 회복?

**Status counts per (pitch, σ, quencher)** (각 cell = 12 runs over dose × DH × time):

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

**핵심 finding**:

```text
σ↓ 가 small-pitch process window 회복의 dominant knob.
  pitch=18, σ=0: 3 robust (vs σ=2: 0-1 robust)  → ★ σ 변경이 결정적
  pitch=20, σ=0,1: 4 robust   (vs σ=2: 2 robust)

quencher 약화 효과는 작거나 부정적.
  대부분 (pitch, σ) 조합에서 weak quencher → robust 수 -1 (Stage 5 의 quencher 가
  small-pitch tolerance 를 좁힌다는 finding 재확인).

Best per pitch:
  pitch=18: dose=28.4, σ=0, DH=0.3, t=30, weak  → robust_valid, CD_shift=+0.25, LER_lock=3.05
  pitch=20: dose=28.4, σ=0, DH=0.3, t=30, weak  → robust_valid, CD_shift=-0.23, LER_lock=2.69
  → 둘 다 σ=0, dose=28.4, DH=0.3 (소량의 chemistry, large-pitch 와 다른 OP)
```

#### 2B 종합 결론

```text
1. v2 권장 OP (pitch=24, dose=40, σ=2, DH=0.5, time=30, Q0=0.02, kq=1.0) 는
   internal-consistent 하며 Phase 2B atlas 의 anchor 로 적합.
   여전히 freeze 상태. external reference 미입수.

2. 외부 데이터가 들어오면 보정 가능한 knob 정량적 매핑 완료:
   - x-y CD bias       → dose (1차)
   - x-y LER bias      → DH (fine), σ (secondary)
   - x-y margin / area → dose, Q0
   - x-z standing wave → absorption_length (압도적 dominant)
   - x-z top/bot asym  → absorption_length

3. Stage 5C hypothesis 확인:
   pitch ∈ {18, 20} 에 적용할 별도 OP 가 존재 (σ=0, dose=28.4, DH=0.3, t=30,
   quencher weak). v2 OP 와는 다른 chemistry / domain. 다음 작업이 small-pitch
   에 의존한다면 이 OP 를 hypothesis-verified candidate 로 사용 가능.
   여전히 internal-only.

4. Stage 6B (full 3D x-y-z) 는 보류 그대로. external reference 또는 구체적
   y-roughness × z-modulation interaction hypothesis 등장 후에만 시작.
```

#### 다음 단계 결정

```text
RECOMMEND: 외부 reference 데이터 입수 대기.

옵션 (그동안):
  - 추가 hypothesis sweep (특정 pair / triple 의 detailed 분석)
  - Stage 3B / 5C / 6B 보류 유지
  - 현 상태로 freeze 종결 후 다른 task / 다른 v3 plan 으로 이동

calibration_status 는 internal-consistency only 그대로.
```

---

## Phase 2 — process window 재검증

조건: Phase 1 에서 새 OP 후보가 채택되었을 때만 실행.

### 2.1 목적

새 (Hmax, kdep, DH) 가 Stage 5 의 process window shape (pitch=16 closed, pitch ≥ 24 wide window) 를 그대로 재현하는지 확인.

### 2.2 Sweep

Stage 5 와 동일한 pitch × dose grid (6 × 6 = 36 runs) 를 새 OP 에 적용.

### 2.3 게이트

```text
PASS:
  pitch=24 의 robust_valid 영역이 dose=28.4 ~ 60 범위 내 모두 robust_valid (또는 더 넓음)
  pitch=16 process window 가 여전히 closed (또는 cliff 가 18 이하로 이동)
  → Phase 3 진행

FAIL:
  process window 이 좁아지거나 cliff 위치가 합리적 범위 밖
  → Phase 1 결과 폐기, Phase 2A 로 fallback
```

### 2.4 Phase 2 결과

(미실행)

---

## Phase 3 — 외부 reference 비교

조건: Phase 2 통과.

### 3.1 목적

published 또는 measured high-NA EUV PEB 데이터와 quantitative 비교.

### 3.2 비교 항목

```text
- 24 nm pitch / 12.5 nm CD / 40 mJ/cm² 의 CD, LER 절대값
- DH × time 또는 PEB temperature 별 LER 감소 곡선
- pitch dependence (Stage 5 status map shape)
- standing wave amplitude reduction (Stage 6 trend)
```

### 3.3 Calibration knob 매핑

```text
- CD 절대값 offset → dose / Hmax / kdep 보정
- LER 절대값 offset → initial roughness amp 또는 corr length 보정
- process window shape offset → kdep / quencher scaling
- standing wave reduction offset → abs_len 보정
```

### 3.4 Phase 3 결과

(미실행)

---

## Phase 4 — deferred stages 시작

조건: Phase 3 통과.

```text
Stage 3B (σ=5/8 호환 budget search) — kdep / dose / Hmax 확장
Stage 5C (σ=0 small-pitch process window)
Stage 6B (full 3D x-y-z)
또는 new chemistry (Dill ABC 모델, PAG profile 등)
```

Phase 3 결과에 따라 시작 순서 결정.

---

## 결정 트리

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
  │                                  └── FAIL  →  Phase 2A (확장 sweep) → Phase 1 재실행
  │
  ├── PASS-marginal (best score 0.10-0.20)  →  Phase 2 with finer grid
  │
  └── FAIL (best score > 0.20)  →  Phase 2A (dose/σ/abs_len 확장)
                                     →  Phase 1 재실행 with new variables
```
