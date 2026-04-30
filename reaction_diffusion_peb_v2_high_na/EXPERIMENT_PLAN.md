# High-NA EUV PEB v2 실험 지시서

## 0. 목적

현재 `reaction_diffusion_peb/` 폴더는 그대로 유지한다.  
새로운 두 번째 실험 폴더를 만들어, 기존 toy/sandbox 성격의 PEB 모델을 **High-NA EUV line/space 기반 공정 지향 실험**으로 재설정한다.

새 폴더의 권장 이름:

```text
reaction_diffusion_peb_v2_high_na/
```

v2의 목표는 다음이다.

```text
High-NA EUV 조건에서
line/space pitch, CD, edge roughness, electron blur, film thickness, z 방향 standing wave,
weak quencher 조건을 단계적으로 추가하여
PEB가 LER/LWR/CD shift에 미치는 영향을 정량화한다.
```

핵심 질문은 다음이다.

```text
1. 초기 line edge roughness가 PEB diffusion 후 얼마나 줄어드는가?
2. roughness smoothing과 CD shift 사이의 trade-off는 어느 정도인가?
3. electron blur와 PEB acid diffusion blur를 분리해서 볼 수 있는가?
4. quencher를 약하게 넣었을 때 acid tail과 CD shift가 줄어드는가?
5. z 방향 film thickness / standing wave modulation이 PEB 후 얼마나 완화되는가?
```

---

## 1. 기존 v1 폴더의 한계

기존 `reaction_diffusion_peb/`는 PEB physics sandbox로서는 성공적이다. 그러나 실제 High-NA EUV 공정 실험으로 보기에는 다음 한계가 있다.

### 1.1 Geometry가 약함

기존은 주로 synthetic Gaussian spot 또는 단순 line-space exposure map을 사용한다.

```text
Gaussian / simple synthetic I(x,y)
→ H0(x,y)
→ diffusion / reaction
```

하지만 실제로 필요한 것은 다음이다.

```text
pitch
line CD
half pitch
edge roughness amplitude
edge roughness correlation length
LWR / LER metric
CD shift metric
```

즉 v1은 “field가 잘 확산되는가”를 보기에는 좋지만, “High-NA line/space pattern이 PEB 후 어떻게 변하는가”를 보기에는 geometry 정보가 부족하다.

### 1.2 z 방향 정보가 없음

v1의 core model은 기본적으로 2D이다.

```text
H(x,y,t)
Q(x,y,t)
P(x,y,t)
```

따라서 다음 현상을 직접 볼 수 없다.

```text
resist film thickness effect
standing wave along z
top / bottom boundary effect
x-z cross-section striation
```

v2에서는 처음부터 full 3D로 가지 말고, 먼저 `x-z` 단면 실험을 별도 stage로 둔다.

### 1.3 Exposure가 normalized toy dose임

v1의 exposure는 대략 다음 형태다.

```text
H0 = Hmax * (1 - exp(-eta * dose * I))
```

여기서 `dose=1.0`은 normalized dose이지, 실제 `mJ/cm^2` 단위의 dose가 아니다.  
v2에서는 실제 dose 값과 normalized dose를 분리한다.

```text
dose_mJ_cm2
reference_dose_mJ_cm2
dose_norm = dose_mJ_cm2 / reference_dose_mJ_cm2
```

### 1.4 Quencher parameter가 너무 강한 regime으로 들어감

v1에서 문제가 된 핵심 조합은 다음이다.

```text
H0_peak ≈ 0.126
Q0 = 0.1
kq >= 5 s^-1
kdep = 0.05 또는 0.5 s^-1
P_threshold = 0.5
```

이 조합에서는 acid가 polymer를 충분히 deprotect하기 전에 quencher에게 너무 빨리 잡힌다.  
그 결과 `P > 0.5` contour가 사라진다.

v2에서는 반드시 다음 순서로 진행한다.

```text
1. quencher off baseline
2. weak quencher only
3. medium quencher
4. stiff quencher는 별도 stress-test
```

### 1.5 PINN/FNO는 기본 solver로 부적절

v1에서 PINN은 diffusion-only 및 deprotection에서 정량 정확도가 낮았다.  
특히 Phase 5 PINN에서는 `P_min < 0` 문제가 있었다.

v2에서는 초기 실험의 truth solver를 다음으로 고정한다.

```text
primary solver: FD / FFT
secondary analysis: optional PINN/FNO only after FD baseline is verified
```

PINN/FNO는 v2 초기에 사용하지 않는다.

---

## 2. v2 실험의 방향성

v2는 “phase를 많이 늘리는 것”이 목적이 아니다.  
목표는 **geometry와 공정 조건을 현실화하고, 각 물리항의 효과를 분리해서 보는 것**이다.

v2의 기본 전략:

```text
Step A: geometry를 실제 line/space scale로 바꾼다.
Step B: electron blur와 initial roughness를 넣는다.
Step C: quencher 없이 PEB baseline을 확인한다.
Step D: weak quencher를 추가한다.
Step E: x-z standing wave 실험을 추가한다.
Step F: pitch / dose / DH / time sweep을 수행한다.
```

v2에서 처음부터 넣지 말아야 할 것:

```text
strong kq = 100~1000 s^-1
large Q0 = 0.1
full 3D solver
PINN / FNO surrogate
complex development rate model
```

---

## 3. 새 폴더 구조

권장 구조:

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

## 4. v2 입력 파라미터

아래 값들은 업로드된 High-NA EUV PEB 정량 파라미터 표를 v2 실험용으로 정리한 것이다.

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

초기 baseline은 `pitch=24 nm`, `line_cd=12.5 nm`로 시작한다.  
바로 16 nm pitch로 가지 않는다. 작은 pitch는 process window가 좁아서 원인 분리가 어렵다.

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

주의:

```text
dose_mJ_cm2는 실제 단위값이고,
dose_norm은 acid generation 식에 넣는 normalized scale이다.
```

초기에는:

```text
dose_norm = dose_mJ_cm2 / 40.0
```

으로 둔다. 이후 실험값에 맞춰 `eta` 또는 `Hmax`를 calibration한다.

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

초기 baseline에서는 `kdep=0.5`를 사용한다.  
`kdep=0.05`는 v1 결과상 threshold contour를 만들기엔 너무 약하므로 v2 기본값으로 쓰지 않는다.

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

중요:

```text
Q0=0.1과 kq>=5를 baseline으로 쓰지 않는다.
강한 quencher는 P contour를 없애므로 별도 stress-test로 분리한다.
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

초기에는 full dissolution model을 넣지 않는다.  
`P > P_threshold` contour로 developable region을 정의한다.

### 4.6 z direction / standing wave parameters

z 방향은 v2 후반에 별도 stage로 넣는다.

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

x-z 초기 exposure model:

```text
I(x,z) = I_xy(x) * [1 + A*cos(2*pi*z/period + phase)] * exp(-z/absorption_length)
```

---

## 5. 실험 단계

## Stage 1 — 2D line/space baseline, no quencher (CLEAN GEOMETRY)

### 목적

기존 Gaussian toy map 대신 실제 line/space pitch/CD 기반 H0를 만들고, quencher 없이 PEB smoothing이 정상적으로 나오는지 확인한다.

### 중요 — nominal 변경 사유

§4.2 / §4.3 의 원본 nominal `electron_blur_sigma_nm=5, time_s=60` 조합은 24 nm pitch / 12.5 nm CD 에서 **interior gate 통과 불가**임이 calibration 결과로 확인됨 (see Stage 1A 아래).
- σ=5, t=60 → P_space_center_mean=0.83, area_frac=1.0, CD_final≈pitch (line 완전 merge).
- σ=5 의 경우 t={10,15,20,30} × DH={0.3,0.8} × kdep=0.5 × Hmax={0.1,0.15,0.2} grid 전수 실패.
- 따라서 σ=5/t=60 은 Stage 1 baseline이 **아님**. §6.4 over-budget 스트레스 케이스로 분리.

Stage 1 의 baseline 은 electron blur 효과가 제거된 **clean geometry** 조건으로 재정의한다.
Electron blur 효과는 Stage 3 에서 σ-호환 budget 과 함께 별도 평가한다.

### 설정 (clean geometry baseline)

```yaml
geometry.pitch_nm: 24
geometry.line_cd_nm: 12.5
geometry.domain_x_nm: 120          # 5 * pitch — pitch-aligned (FFT seam artifact 방지)
geometry.domain_y_nm: 120
exposure.dose_mJ_cm2: 40
exposure.electron_blur_sigma_nm: 0  # NO e-blur for Stage 1
peb.DH_nm2_s: 0.8
peb.time_s: 30                      # half of original nominal — 60s에서는 모든 σ가 merge
peb.kdep_s_inv: 0.5
peb.kloss_s_inv: 0.005
quencher.enabled: false
development.P_threshold: 0.5
```

config: `configs/v2_stage1_clean_geometry.yaml`

### 출력

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

### 성공 기준 — interior gate (seam-artifact-resistant)

```text
H >= 0
0 <= P <= 1
interior P_space_center_mean < 0.50    # line 사이 strip 평균이 임계 미만
interior P_line_center_mean  > 0.65    # line 중심 strip 평균이 임계 이상
contrast = P_line_mean - P_space_mean > 0.15
area_frac (P>=threshold) < 0.90        # 전 영역 over-deprotect 방지
CD_final / pitch < 0.85                # line 들이 merge 하지 않음
CD shift 와 LER 가 측정 가능
```

global P_min 은 사용하지 않는다. 도메인이 pitch 의 정수배가 아니면 FFT seam 의 wider-space artifact 가 P_min 을 인위적으로 낮추기 때문.

### 검증된 결과 (σ=0, t=30)

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

### 목적

§4.2 의 sweep `electron_blur_sigma_nm: [0,2,5,8]` 를 의미 있게 수행하려면, 각 σ 에 대해 line-space 구분이 살아있는 (t, DH, Hmax, kdep) budget 이 필요하다.
원본 `kdep=0.5, time=60, DH=0.8, Hmax=0.2` 는 24 nm pitch 에서 σ≥4 부터 lines 가 merge 하므로 σ-스윕이 불가능.
Stage 1A 는 각 σ 의 호환 budget 을 찾는다.

### 절차

```text
Stage 1A.1  σ sweep at fixed t=30, DH=0.8, kdep=0.5, Hmax=0.2:
              σ ∈ {0,1,2,3,4,5}
              gate = interior gate 위 그대로
              결과: σ ∈ {0,1,2,3} pass, σ ∈ {4,5} fail.

Stage 1A.2  σ=5 budget search (기존 spec 범위):
              time × DH grid: time ∈ {10,15,20,30}, DH ∈ {0.3,0.8}, kdep=0.5
              필요 시 Hmax sweep ∈ {0.1,0.15,0.2} at the highest-contrast (t,DH)
              결과: 전수 실패 (σ=5,t=20,DH=0.3,Hmax=0.2 가 cond_line 0.630<0.65 로 가장 근접).
              결론: 위 search 범위 내에서 σ=5 호환 budget 없음.
              해석: σ=5/24 nm pitch 에서 I_blurred contrast 가 너무 약하다.

Stage 1A.3  필요 시 search space 확장:
              kdep ∈ {0.5, 1.0}                 # P_line lift
              dose_mJ_cm2 ∈ {40, 50, 60}        # H0 contrast lift
              또는 σ 의 effective upper bound 를 σ_max=3 으로 확정.
```

### Stage 1A 관측

| σ (nm) | t (s) | DH | Hmax | P_space | P_line | contrast | area_frac | CD/p | passed |
|--------|-------|-----|------|---------|--------|----------|-----------|------|--------|
| 0      | 30    | 0.8 | 0.2  | 0.31    | 0.76   | 0.45     | 0.625     | 0.63 | ✅ |
| 1      | 30    | 0.8 | 0.2  | 0.34    | 0.77   | 0.42     | 0.667     | 0.67 | ✅ |
| 2      | 30    | 0.8 | 0.2  | 0.40    | 0.76   | 0.37     | 0.739     | 0.74 | ✅ |
| 3      | 30    | 0.8 | 0.2  | 0.46    | 0.76   | 0.30     | 0.852     | 0.85 | ✅ (limit) |
| 4      | 30    | 0.8 | 0.2  | 0.53    | 0.75   | 0.22     | 0.961     | 0.94 | ❌ |
| 5      | 30    | 0.8 | 0.2  | 0.58    | 0.73   | 0.16     | 1.000     | 0.98 | ❌ |
| 5      | 20    | 0.3 | 0.2  | 0.38    | 0.63   | 0.25     | 0.598     | 0.60 | ❌ (P_line 0.02 low) |

### Stage 1A 결정

```text
24 nm pitch 에서 σ-호환 운용 범위 = σ ∈ [0, 3] nm (kdep=0.5, Hmax=0.2 spec 내)
σ ≥ 4 는 budget search space 확장 후 재검토 (Stage 1A.3).
σ=5 와 σ=5/t=60 는 §6.4 over-budget 스트레스 케이스로 분리.
```

---

## Stage 1B — over-budget stress reference (σ=5, t=60)

### 목적

Plan §4 의 원본 nominal 이 line merge 를 일으킨다는 것을 명시적으로 기록.
공정 윈도우 평가에서 "이 조합은 비정상 영역" 의 reference 로 사용.

### 설정

config: `configs/v2_baseline_lspace.yaml` (헤더에 OVER-BUDGET 경고 명시)

### 관측 결과

```text
P_space_center_mean = 0.83
P_line_center_mean  = 0.90
contrast            = 0.07
area_frac           = 1.000
CD_final / pitch    = 0.98
모든 interior gate fail. lines 가 슬랩으로 합쳐짐.
```

이 결과는 Stage 5 process-window 분석에서 "high-σ × long-t × strong-kdep 조합은 lines collapse" 경향의 데이터 포인트로 사용한다.

---

## Stage 2 — DH / PEB time smoothing sweep

### 목적

PEB diffusion length가 roughness와 CD shift에 주는 영향을 분리한다.

### Sweep

```yaml
DH_sweep_nm2_s: [0.3, 0.5, 0.8, 1.0, 1.5]    # 실제 수행: 5점 (원안의 3점 + 0.5/1.0 추가)
time_sweep:     [15, 20, 30, 45, 60]          # 실제 수행: lower 절반 추가
```

### 봐야 할 경향

```text
DH 증가 → LER 감소
DH 증가 → CD shift 증가
PEB time 증가 → LER 감소
PEB time 증가 → CD shift 증가
너무 큰 DH/time → line edge blur 과도
```

### 검증된 결과

config: `configs/v2_stage2_dh_time.yaml` (Stage 1 baseline 그대로 + sweep 변수만 override)

```text
LER reduction (%) — DH (rows) × time (cols), ✓ = interior gate pass

                15         20         30         45         60
  DH=0.30:    6.55✗     6.63✓     8.04✓     6.96✓   -14.34✓
  DH=0.50:    9.00✗     8.80✓     8.69✓   -17.04✓    45.55✗
  DH=0.80:   10.06✗     9.61✓     4.25✓     8.28✗   100.00✗
  DH=1.00:    8.62✗     8.84✗    -1.98✓    62.62✗   100.00✗
  DH=1.50:   -4.48✗     3.98✗   -38.01✓   100.00✗   100.00✗

✗ 영역의 LER% 는 신뢰 불가 (lines merged → edge extract NaN/0).
✗에서의 100% 는 artifact, ✓ 영역에서만 의미 있음.
```

알고리즘 best (max LER% s.t. CD_shift ≤ 3, CD/p < 0.85, area_frac < 0.9):
**DH=0.8 nm²/s, t=20 s** — LER 9.61% 감소, CD_shift = −1.18 nm, P_line=0.65 (margin 0.003).

실용 권장 (P_line margin 큰 후보): **DH=0.5, t=20** (LER 8.80%, P_line=0.68) 또는 **DH=0.5, t=30** (LER 8.69%, P_line=0.79).

세부 분석은 `study_notes/02_stage2_dh_time_sweep.md` 참조.

---

## Stage 3 — electron blur 분리 실험

### 목적

EUV 2차 전자 blur 와 PEB acid diffusion blur 를 분리한다.

### Sweep (revised)

```yaml
electron_blur_sigma_nm: [0, 1, 2, 3]      # 24 nm pitch / kdep=0.5 / Hmax<=0.2 호환 범위
DH_nm2_s, time_s: Stage 2 의 두 operating point
                  - robust OP             : DH=0.5, t=30
                  - algorithmic-best OP   : DH=0.8, t=20
```

원안 `[0, 2, 5, 8]` 에서 σ=5, σ=8 은 24 nm pitch / kdep=0.5 / Hmax≤0.2 budget 과 호환되지 않음 (Stage 1A 에서 search space 전수 fail). 호환 budget 을 찾으려면 dose / kdep / Hmax 를 늘려야 하며 이는 별도 stage (Stage 3B) 로 분리.

### 측정 규약 (재정의)

Stage 1/2 의 "initial edge" 정의는 σ-의존성 (`I_blurred` 가 σ 따라 매끈해짐) 이 있어 σ-스윕에서 불공정. Stage 3 부터는 LER 를 세 단계로 분리해서 측정.

```text
LER_design_initial    : binary I (blur 전) 의 threshold 0.5 contour
LER_after_eblur_H0    : I_blurred (blur 후, PEB 전) 의 threshold 0.5 contour
LER_after_PEB_P       : P (PEB 후) 의 threshold 0.5 contour

electron_blur_LER_reduction_pct = 100 * (LER_design_initial - LER_after_eblur_H0) / LER_design_initial
PEB_LER_reduction_pct           = 100 * (LER_after_eblur_H0 - LER_after_PEB_P)   / LER_after_eblur_H0
total_LER_reduction_pct         = 100 * (LER_design_initial - LER_after_PEB_P)   / LER_design_initial
```

`LER_design_initial` 은 σ 독립이므로 σ-스윕의 baseline 으로 일관된다.

### Gate (Stage 3, 강화)

Stage 1/2 interior gate + `P_line_margin >= 0.03` 추가.

```text
P_space_center_mean < 0.50
P_line_center_mean  > 0.65
P_line_margin = P_line_center_mean - 0.65 >= 0.03   # 새로 추가
contrast > 0.15
area_frac < 0.90
CD_final / pitch < 0.85
CD_final, LER_after_PEB_P 가 finite
```

P_line_margin 게이트는 Stage 2 에서 algorithmic best (DH=0.8, t=20) 가 P_line=0.6534 로 boundary 에 안착했던 문제를 방지한다.

### 분석

```text
electron_blur 단독 효과 (PEB 전): I_blurred → LER 감소 = electron_blur_LER_reduction_pct
PEB 단독 효과 (electron_blur 후): I_blurred → P    → LER 감소 = PEB_LER_reduction_pct
두 효과 합산                   : binary I  → P    → LER 감소 = total_LER_reduction_pct
```

### 성공 기준

```text
σ 증가 → LER_after_eblur_H0 감소 (electron blur 의 1차 smoothing)
σ 증가 → LER_after_PEB_P 감소 또는 유사 (PEB 가 추가 smoothing)
두 효과를 별도 컬럼으로 분리 가능
gate 통과 σ 가 OP 별로 1 개 이상 존재
```

---

## Stage 3B — σ=5/8 호환 budget search (future / optional)

### 목적

Stage 1A 에서 `σ=5` 의 호환 budget 이 spec 범위 (`kdep=0.5`, `Hmax∈{0.1,0.15,0.2}`) 안에 없음을 확인. Stage 3 결과로 σ ∈ {0,1,2,3} 의 분리 분석은 가능하지만, 실제 High-NA EUV 의 e-beam blur 는 σ ≈ 5 nm 가 reference. 이 σ 값을 살리려면 search space 확장이 필요.

### 진행은 Stage 3 결과를 본 후 결정

trigger 조건 (다음 중 하나가 만족되면 Stage 3B 를 시작):

```text
- Stage 3 의 electron_blur_LER_reduction_pct 가 σ=3 까지의 trend 로 σ=5/8 의 효과 추정 불가
- PSD 분해에서 high-frequency cutoff 를 σ=3 까지로는 정량 못함
- 외부 reference (literature / 실측) 와의 비교가 σ=5 에서 필수
```

### Search space (Stage 3B)

```text
변수 1: dose_mJ_cm2          ∈ {40, 50, 60}    # H0 contrast 강화
변수 2: kdep_s_inv           ∈ {0.5, 1.0}      # P_line lift
변수 3: Hmax_mol_dm3         ∈ {0.1, 0.15, 0.2}
변수 4: σ                    ∈ {5, 8}
변수 5: time_s               ∈ {15, 20, 30}
변수 6: DH_nm2_s             ∈ {0.3, 0.5, 0.8}
```

전수 grid (3×2×3×2×3×3 = 324) 는 너무 크므로, factorial design 또는 Latin-Hypercube 등 sampling 으로 축소.

### 성공 기준 (Stage 3B)

Stage 3 의 interior gate (P_line_margin 포함) 를 σ=5 와 σ=8 각각에서 **하나 이상** 의 (dose, kdep, Hmax, t, DH) 조합이 통과.

---

## Stage 4 — weak quencher 실험

### 목적

v1 에서 너무 강했던 quencher 를 약하게 재도입한다. acid tail 감소와 CD shift 억제, 그리고 Stage 3 에서 발견한 σ-증가 시 PEB LER 악화 (high σ 에서 line widening → contour 가 design edge 에서 멀어져 LER 증가) 를 완화하는지 확인.

### Operating point

Stage 2 의 robust OP **만** 사용한다. Stage 3 에서 algorithmic-best OP 가 P_line_margin gate 를 모든 σ 에서 fail 하므로 downstream 에서 채택하지 않는다.

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

총 4 × (1 + 4×3) = 52 runs.

### Gate (Stage 3 그대로)

```text
P_space_center_mean < 0.50
P_line_center_mean  > 0.65
P_line_margin       >= 0.03
contrast > 0.15
area_frac < 0.90
CD_final / pitch < 0.85
CD_final, LER_after_PEB_P 가 finite
```

### Stage 4 비교 기준 (vs same-σ no-quencher baseline)

```text
dCD_shift_nm      < 0     # quencher 가 line widening 을 줄임
darea_frac        < 0     # quencher 가 over-deprotect area 를 줄임
dtotal_LER_pp     >= -1.0 # total LER reduction 이 1pp 이상 떨어지면 안 됨
P_line_margin     >= 0.05 # robust 후보의 추가 안전 margin
```

위 조건을 모두 만족하는 row 를 robust candidate 로 표시한다.

### 측정 (Stage 3 measurement convention 그대로 + PSD)

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

### 검증된 결과

config: `configs/v2_stage4_weak_quencher.yaml`

**52 runs 모두 Stage-3 gate 통과.** σ=3, Q0=0.03, kq=2 한 row 만 robust margin 미달 (P_line_margin = 0.039 < 0.05).

quencher 도입 효과 (vs σ-matched baseline):

```text
모든 σ 에서:
  dCD_shift   < 0   (line widening 감소)
  darea_frac  < 0   (over-deprotect 감소)
  dtotal_LER ≥ 0   (LER reduction 가 baseline 보다 증가 또는 동등)

특히 σ=3 (Stage 3 에서 PEB 가 LER 를 -22.5% 로 악화시켰던 case):
  Q0=0.03, kq=1.0  →  total LER reduction = +6.64% (baseline -22.5%, dLER=+29.15pp)
  Q0=0.02, kq=2.0  →  total LER reduction = +5.95% (dLER=+28.47pp)
  → quencher 가 line widening 을 막아 contour 가 design edge 가까이 유지되며
    LER 측정이 정상화됨.

σ=2 (primary analysis) 의 dtotal_LER_pp heatmap (Q0 행, kq 열):
              kq=0.5  kq=1.0  kq=2.0
  Q0=0.030    +4.90   +6.47   +7.64
  Q0=0.020    +3.74   +5.21   +6.44
  Q0=0.010    +2.18   +3.24   +4.23
  Q0=0.005    +1.18   +1.84   +2.49
```

PSD 분석:
- 모든 row 에서 `psd_high_band_reduction_pct ≈ 99–100%` — high-frequency edge noise 는 baseline 에서도 PEB 로 거의 완전히 제거됨.
- LER 변화의 차이는 low/mid band (long-range wiggle, main corr regime) 에서 발생.
- 즉 quencher 의 효과는 high-freq 노이즈가 아닌 mid-freq 의 line wiggle 을 줄이는 데서 온다.

추천 σ=2 robust candidate:

```text
Q0=0.02, kq=1.0  (균형형)   : dCD=-1.76, darea=-0.073, dLER=+5.21pp, margin=0.096
Q0=0.03, kq=2.0  (max LER 감소): dCD=-3.54, darea=-0.147, dLER=+7.64pp, margin=0.053
```

세부 분석은 `study_notes/04_stage4_weak_quencher.md` 참조.

### 실패 기준 (원래 plan 의 정의 유지)

```text
Q0=0.01, kq=1 부터 P>0.5 contour 가 사라지면
Hmax/dose/kdep 중 하나가 너무 약하거나 Q0 가 여전히 강한 것
→ 본 sweep 에서는 발생하지 않음 (모든 row P contour 정상).
```

### Stage 4B (deferred — CD-locked LER)

Stage 3/4 의 LER 비교는 fixed P_threshold=0.5 contour 에서 측정하므로, σ 또는 quencher 가 CD 를 바꾸면 contour 위치가 함께 이동해 LER 측정 위치가 달라진다. 이를 보정하기 위한 **CD-locked LER** (각 row 의 P_threshold 를 자동 조정해 CD_final = CD_initial 로 맞추고 LER 비교) 는 Stage 4 (fixed-threshold) 완료 후 별도 stage 로 진행.

trigger: Stage 5 (process window) 또는 외부 reference 비교에서 σ/quencher-induced CD shift 가 LER 비교를 오염시키는 경우.

---

## Stage 5 — pitch / dose process window

### 목적

Stage 4 의 balanced operating point 가 다양한 pitch / dose 에서 유지되는지 확인.
pitch 별로 robust_valid 한 dose window 크기와 추천 dose 를 정량화.

### Operating point (primary)

```yaml
sigma_nm    : 2.0
DH_nm2_s    : 0.5
time_s      : 30
kdep_s_inv  : 0.5
kloss_s_inv : 0.005
Hmax_mol_dm3: 0.2
quencher    : Q0=0.02, kq=1.0, DQ=0    # Stage 4 balanced
line_cd_nm  : 12.5                       # pitch 별 고정
```

### Sweep

```yaml
pitch_sweep_nm:    [16, 18, 20, 24, 28, 32]
dose_sweep_mJ_cm2: [21, 28.4, 40, 44.2, 59, 60]
domain_x_nm  = pitch_nm * 5     # n_periods_x = 5
domain_y_nm  = 120              # 고정 (LER y-sample 수 동일)
```

총 36 runs (primary). 추가 controls 36 + 36 = 108 runs.

### Optional controls

```text
control_sigma0_no_q : sigma=0, quencher disabled  → 측정 규약 σ-독립 reference
control_sigma2_no_q : sigma=2, quencher disabled  → quencher 만의 효과 분리
```

### Per-run 분류 (precedence 순)

```text
unstable      : NaN / Inf / bounds violation / 등고선 추출 실패
merged        : P_space_mean >= 0.50  OR  area_frac >= 0.90  OR  CD/pitch >= 0.85
under_exposed : P_line_mean  < 0.65
low_contrast  : contrast      <= 0.15  (드물게 등장. 본 sweep 에서는 0)
valid         : interior gate 통과, margin 부족
robust_valid  : interior gate 통과 AND P_line_margin >= 0.05
```

### 추천 dose 선택 알고리즘

각 pitch 에서:

```text
1. robust_valid pool 우선
2. 없으면 valid pool
3. pool 안에서 |CD_shift_nm| 최소화
4. 동률 시 total_LER_reduction_pct 최대화
5. 동률 시 P_line_margin 최대화
```

### 검증된 결과 (primary OP)

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

→ pitch=16 에서는 dose 어디에서도 valid 영역 없음. process window 닫힘.
→ pitch ≥ 24 에서는 4 dose 점이 robust_valid (workable window).
```

추천 dose per pitch:

```text
pitch=16: 추천 없음 (process window closed)
pitch=18: dose=28.4 (valid only, margin 0.012, CD_shift +1.99, LER -34.18%)
pitch=20: dose=40   (robust_valid, margin 0.098, CD_shift +4.10, LER -26.22%)
pitch=24: dose=40   (robust_valid, margin 0.096, CD_shift +2.07, LER  +8.77%)
pitch=28: dose=40   (robust_valid, margin 0.095, CD_shift +1.81, LER +14.33%)
pitch=32: dose=40   (robust_valid, margin 0.095, CD_shift +1.78, LER +15.03%)
```

pitch ≤ 20 에서 음수 LER reduction 은 Stage 3 / 4 에서 진단된 contour-displacement artifact (line widening 으로 contour 가 design edge 에서 벗어남). Stage 4B (CD-locked LER) 에서 수정될 예정이지만 robust_valid 분류 자체는 정상.

### Control 비교 (primary 대비)

```text
control_sigma0_no_q:  pitch=16 만 process window 닫힘. pitch ≥ 18 부터 robust_valid 대부분
control_sigma2_no_q:  pitch=20 까지는 process window 좁음, pitch ≥ 24 에서 robust_valid 대부분
primary (sigma=2 + Q0=0.02, kq=1):  pitch=20 에서 dose=40 단 한 점만 robust_valid
```

흥미로운 발견: **quencher 추가가 small pitch process window 를 좁힌다.** 이는 Stage 4 의 LER 개선 효과와 trade-off (line widening 감소 = contour 가 더 가까워져 small pitch 에서 merge 임계 빨리 도달).

### 성공 기준

```text
✓ 20~24 nm pitch 에서 stable contour 형성 → primary 에서 pitch=20 (dose=40) / pitch=24 (4 dose) robust_valid
✓ 16~18 nm pitch 에서 process window 가 좁아지는 경향 → pitch=16 closed, pitch=18 valid only at one dose
✓ high dose 에서 CD widening / over-deprotection → pitch ≤ 20, dose ≥ 44 에서 merged
✓ low dose 에서 under-deprotection → 모든 pitch, dose=21 에서 under_exposed
```

세부 분석은 `study_notes/05_stage5_pitch_dose.md` 참조.

---

## Stage 6 — x-z standing wave 실험

### 목적

z 방향 film thickness와 standing wave modulation이 PEB 후 완화되는지 확인한다.

### 시작 설정

```yaml
film.enabled_z: true
film.film_thickness_nm: 20
film.dz_nm: 0.5
standing_wave.enabled: true
standing_wave.period_nm: 6.75
standing_wave.amplitude: 0.1
```

### Model

```text
I(x,z) = I_xy(x) * [1 + A*cos(2*pi*z/6.75 + phase)] * exp(-z/absorption_length)
H0(x,z) = Hmax * (1 - exp(-eta*dose_norm*I(x,z)))
```

### 출력

```text
I(x,z)
H0(x,z)
H_final(x,z)
P_final(x,z)
z profile at line center
standing wave contrast before / after PEB
```

### 성공 기준

```text
A=0이면 z modulation 없음
A>0이면 H0/P에 z 방향 층상 modulation 발생
PEB 후 modulation amplitude 감소
film thickness 15/20/30 nm에 따라 z modulation 영향 차이 확인
```

---

## 6. 구현 세부 지시

### 6.1 기존 폴더 보존

절대 기존 `reaction_diffusion_peb/` 파일을 수정하지 않는다.

```text
Do not edit:
reaction_diffusion_peb/
```

새 실험은 모두 아래에서 진행한다.

```text
reaction_diffusion_peb_v2_high_na/
```

### 6.2 v1 코드 재사용 원칙

v1의 검증된 solver는 복사해서 사용하되, import coupling은 피한다.

권장:

```text
copy selected FD/FFT utilities into v2 src/
```

비권장:

```text
from reaction_diffusion_peb.src... import ...
```

이유는 v2가 독립 재현 가능한 실험이어야 하기 때문이다.

### 6.3 첫 구현 대상

가장 먼저 구현할 파일:

```text
src/geometry.py
src/roughness.py
src/electron_blur.py
src/metrics_edge.py
experiments/01_lspace_baseline/run_baseline_no_quencher.py
```

### 6.4 Edge metric 구현

`P_threshold` contour에서 edge 위치를 추출한다.

권장 방식:

```text
각 y row에 대해 P(x,y)=P_threshold가 되는 x 위치를 interpolation으로 찾음
edge_x(y)를 얻음
LER = 3*sigma(edge_x - mean(edge_x)) 또는 RMS 기준 병기
CD(y) = right_edge_x(y) - left_edge_x(y)
LWR = 3*sigma(CD(y))
```

PSD metric:

```text
edge_residual(y) = edge_x(y) - smooth_mean_edge
FFT(edge_residual)
PSD_before / PSD_after 비교
```

### 6.5 물리 bounds test

모든 stage에서 다음을 test로 강제한다.

```text
H >= -1e-8
Q >= -1e-8, if Q enabled
0 <= P <= 1
no NaN
no Inf
mass budget reasonable
```

---

## 7. 산출물 규칙

각 실험은 figure와 CSV를 반드시 같이 저장한다.

```text
outputs/figures/*.png
outputs/logs/*.csv
outputs/fields/*.npz
```

권장 figure:

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

권장 CSV columns:

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

## 8. 판단 기준

### 정상적인 물리 경향

```text
DH 증가 → H peak 감소
DH 증가 → LER 감소
DH 증가 → CD shift 증가
PEB time 증가 → Pmax 증가
PEB time 증가 → LER 감소
temperature 증가 → reaction rate 증가
weak quencher 증가 → acid tail 감소
strong quencher 증가 → Pmax 감소
standing wave amplitude 증가 → z modulation 증가
PEB diffusion → z modulation 감소
```

### 즉시 중단해야 하는 비정상 경향

```text
H가 큰 음수가 됨
P가 0 미만 또는 1 초과
Q가 음수가 됨
DH 증가했는데 peak가 증가
time 증가했는데 Pmax가 감소, 단 강한 acid loss 예외는 별도 확인
Q0=0.01, kq=1에서 contour가 완전히 사라짐
모든 dose에서 P>threshold area가 0
모든 dose에서 전 영역 P>threshold
```

---

## 9. v2 첫 번째 목표 결과 (Stage 1 종료 기준)

v2의 첫 성공 목표는 거창한 full High-NA simulation이 아니다.  
아래 하나만 먼저 성공하면 된다.

```text
24 nm pitch, 12.5 nm CD line-space (도메인 = 5 * pitch = 120 nm) 에서
초기 edge roughness 가 있는 H0를 만들고,
quencher 없이 PEB 후 P>0.5 contour 가 line-space 분리를 유지하며 (interior gate),
LER_before > LER_after 이면서 CD_shift 를 정량화한다.
```

**이 목표는 σ=0, t=30, DH=0.8, kdep=0.5, Hmax=0.2 조건에서 달성됨** (§Stage 1 검증 결과 표 참조).
σ=5/t=60 nominal 은 §Stage 1B 의 over-budget 스트레스 케이스로 분리됨.

이 결과가 나오기 전까지는 다음을 하지 않는다.

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

## 11. 추천 weak quencher config: `configs/v2_weak_quencher.yaml`

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

## 12. 추천 x-z config: `configs/v2_xz_standing_wave.yaml`

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

## 13. 최종 정리

v2는 v1을 대체하지 않는다. v1은 물리항별 solver 검증용으로 유지한다.  
v2는 High-NA EUV 조건에서 geometry, roughness, electron blur, film thickness, weak quencher를 추가한 공정 지향 실험이다.

가장 중요한 원칙:

```text
한 번에 모든 물리를 켜지 않는다.
geometry → blur → diffusion/deprotection → weak quencher → pitch/dose sweep → x-z standing wave 순서로 간다.
```

v2의 첫 번째 완료 기준:

```text
24 nm pitch / 12.5 nm CD / clean geometry (σ=0) / no quencher 조건에서
P>0.5 contour 가 line-space 분리를 유지하며 (interior gate),
LER 감소와 CD shift 가 동시에 정량화되는 것.
```

**Stage 1 status (calibration 후 확정):**

```text
✅ 달성 — σ=0, t=30, DH=0.8, kdep=0.5, Hmax=0.2 (configs/v2_stage1_clean_geometry.yaml)
   P_space_mean=0.31, P_line_mean=0.76, contrast=0.45
   CD: 12.46 → 15.01 nm (+2.55 nm)
   LER: 2.77 → 2.65 nm
   모든 interior gate PASS

⚠ Calibration finding — σ=5/t=60 nominal 은 24 nm pitch 에서 over-budget.
   이 nominal 은 Stage 1B 스트레스 케이스로 분리.
   σ-호환 운용 범위 (kdep=0.5, Hmax=0.2 spec 내): σ ∈ [0, 3] nm.
   σ ≥ 4 호환 budget 은 search space 확장 후 재검토 (Stage 1A.3).
```
