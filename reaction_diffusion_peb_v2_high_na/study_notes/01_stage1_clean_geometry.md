# Phase 1 — Stage 1 + Stage 1A: clean geometry baseline 과 σ-호환 budget

## 0. 한 줄 요약

EXPERIMENT_PLAN §4 의 원본 nominal (`σ=5 nm, t=60 s`) 은 24 nm pitch / 12.5 nm CD 에서 line/space 를 만들지 못하고 전 영역이 over-deprotect 된다. Stage 1 baseline 을 **clean geometry 조건 (σ=0, t=30)** 으로 재정의하고, σ ≠ 0 이 가능한 운용 범위를 calibration 으로 찾았다 (σ ∈ [0, 3] nm at kdep=0.5, Hmax=0.2).

---

## 1. 목적

### Stage 1 의 plan-stated 목적

```text
24 nm pitch / 12.5 nm CD line-space 에서
초기 edge roughness 가 있는 H0 를 만들고,
quencher 없이 PEB 후 P>0.5 contour 를 얻고,
LER_before > LER_after 이면서 CD_shift 를 정량화한다.
```

### 이 phase 에서 추가로 답해야 했던 것

- plan §4 의 nominal 파라미터가 실제로 line-space 패턴을 만드는가?
- 만들지 못한다면 어떤 파라미터가 호환되는가?
- electron blur 효과를 stage 별로 분리하려면 σ 의 운용 범위는?

---

## 2. 진행 단계

1. v2 폴더 스캐폴딩: `configs/`, `src/`, `experiments/01_lspace_baseline/`, `tests/`, `outputs/`.
2. 핵심 모듈 구현
   - `src/geometry.py`: line-space mask + edge roughness
   - `src/roughness.py`: 1D Gaussian-correlated edge perturbation (FFT 기반)
   - `src/electron_blur.py`: 2D periodic Gaussian blur
   - `src/exposure_high_na.py`: Dill-style `H0 = Hmax (1 - exp(-η · dose_norm · I))`
   - `src/fd_solver_2d.py`: 연산자 분할 (operator splitting). H, Q 는 spectral exact step (`exp(-(D|k|² + k_decay) dt)`), 반응항은 explicit Euler. P 는 explicit `P += dt · k_dep · H · (1-P)`.
   - `src/metrics_edge.py`: line 별 left/right edge interpolation, LER/LWR/CD
   - `src/visualization.py`: field/contour overlay
3. 테스트 5종 작성 후 14/14 통과.
4. plan §10 의 nominal 로 첫 baseline 실행 (`σ=5, t=60, DH=0.8, kdep=0.5, Hmax=0.2`).
5. 결과 비정상 → 게이트 정의 / 스윕 / calibration 으로 진행.

---

## 3. 발생한 문제와 해결 방법

### 문제 1 — 출력 경로 버그 (`parents[2]` 오용)

**증상**: 첫 baseline 실행 직후 metrics 는 stdout 에 찍혔는데 `outputs/` 안에 실제 파일이 없었다. `find` 결과 엉뚱한 위치에 중첩된 `reaction_diffusion_peb_v2_high_na/reaction_diffusion_peb_v2_high_na/outputs/...` 가 생성됨.

**원인**: 실행 스크립트에서

```python
V2_DIR = Path(__file__).resolve().parents[2] / "reaction_diffusion_peb_v2_high_na"
```

로 작성. `parents[2]` 는 이미 `reaction_diffusion_peb_v2_high_na` 디렉터리이므로, 거기에 다시 `/ "reaction_diffusion_peb_v2_high_na"` 를 붙이면 한 단계 깊은 가짜 디렉터리가 만들어진다.

**해결**:

```python
V2_DIR = Path(__file__).resolve().parents[2]
DEFAULT_OUT = V2_DIR / "outputs"
```

로 수정하고 잘못 만들어진 디렉터리는 `rm -rf` 로 제거.

**교훈**: `parents[N]` 카운팅은 항상 `__file__` 부터 한 단계씩 손가락으로 짚어 확인할 것. 특히 디렉터리 이름과 패키지 이름이 같을 때 헷갈린다.

---

### 문제 2 — plan nominal 에서 line-space 가 합쳐짐

**증상**: `σ=5, t=60, DH=0.8, kdep=0.5, Hmax=0.2` 로 실행하면

```text
P_min = 0.60   (전 영역 P > 0.5)
area(P>0.5) = 16384 nm² = 도메인 전체
CD_initial = 12.5 nm  →  CD_final = 23.5 nm  (pitch=24 의 거의 전체)
LER_final = 0.0  ← edge 가 도메인 가장자리로만 빠진 거짓-감소
```

plan §8 의 즉시 중단 기준 "모든 dose 에서 전 영역 P > threshold" 에 정확히 해당.

**원인 분석 (4 가지가 동시에 작용)**:

1. σ=5 nm 가 24 nm pitch 에 비해 큼.
   pitch 의 fundamental 푸리에 모드 감쇠 = `exp(-(2π/p)²·σ²/2) = exp(-1.71) ≈ 0.18`.
   duty=12.5/24=0.52 이므로 blur 후 `I_blurred ∈ [0.34, 0.70]` — space 도 0 이 아님.
2. `Hmax · η · dose = 0.2` 로 약한 I 도 H ≈ 0.058 까지 변환.
3. `DH · t = 0.8 × 60 = 48 nm²` → diffusion length √(2·48) ≈ 9.8 nm 로 half-pitch (12 nm) 와 거의 같다. 공간 평균화.
4. `kdep · H_min · t ≈ 0.5 × 0.038 × 60 = 1.14` → space 의 P_min ≈ 1 - e^-1.14 ≈ 0.68.

즉 노광/확산/반응 budget 이 line-space 분리 능력 (= I_blurred contrast) 보다 크다.

**해결**: phase 후속 단계에서 σ-스윕을 통해 budget 을 줄이는 방향으로 진행 (문제 4 까지 누적).

---

### 문제 3 — `P_min` 게이트가 FFT seam artifact 로 false-pass

**증상**: σ ∈ {0,1,2,3,4,5} sweep at t=60 을 돌렸을 때

```text
σ=1, t=60 →  P_min=0.384 (< 0.45 통과)  passed=True
            그러나 contour 그림은 lines 가 합쳐져 있음
```

P_min 게이트는 통과했지만 그림에서 line/space 가 보이지 않음.

**원인**: 도메인 `domain_x_nm = 128`, `pitch = 24`. line 5 개가 x = 12, 36, 60, 84, 108 에 위치하고 다음 line 위치는 132 로 도메인 밖. FFT 의 periodic BC 때문에 x=128 ↔ x=0 이 연결되어 그 이음매 (seam) 에 effective space 가 19.5 nm 로 interior space (11.5 nm) 보다 넓어진다. 그곳만 P 가 0.5 미만으로 떨어져 global P_min 이 낮게 나와 게이트를 통과한 것. **interior space 의 P 는 모두 0.5 이상**이라 line 들은 사실 합쳐져 있다.

**해결**: 두 단계로 처리.

1. **도메인을 pitch 의 정수배로 정렬**: `domain_x_nm = 5 × 24 = 120 nm`. seam 이 정확히 한 주기와 일치.
   helper `ensure_pitch_aligned_domain()` 추가하여 자동으로 가장 가까운 정수배로 스냅.
2. **게이트를 interior 측정으로 교체**: global `P_min` 대신 line 사이 mid-strip 에서의 평균/최소 (`P_space_center_mean`) 와 line 중심 strip 평균 (`P_line_center_mean`) 을 쓴다. 추가로:
   - `P_space_center_mean < 0.50`
   - `P_line_center_mean > 0.65`
   - `contrast = P_line_mean - P_space_mean > 0.15`
   - `area_frac < 0.90`
   - `CD_final / pitch < 0.85` (lines 가 merge 안 함)

이 두 가지를 같이 적용하지 않으면 안 된다. 도메인만 정렬해도 면적 게이트가 흐리고, 게이트만 바꿔도 boundary 효과가 LER 측정을 오염시킨다.

**교훈**: FFT-기반 spectral solver 에서는 **도메인이 패턴 주기의 정수배여야 한다**. 그렇지 않으면 측정값 어딘가에 seam artifact 가 숨어 있다.

---

### 문제 4 — t=60 에서는 모든 σ 가 fail

**증상**: 도메인 정렬 + interior 게이트로 다시 σ 스윕 (t=60):

```text
σ=0 → P_space=0.65, area_frac=1.00, CD/pitch=0.98  (fail)
σ=5 → P_space=0.83, area_frac=1.00, CD/pitch=0.98  (fail)
모든 σ 에서 lines merge
```

t=60 자체가 too long.

**해결**: fallback 으로 sigma=best (highest contrast) 에서 time sweep `[30, 45, 60]`.

```text
σ=0, t=30  →  P_space=0.31, P_line=0.76, contrast=0.45, CD/pitch=0.625, passed=True
σ=0, t=45  →  P_space=0.51, area_frac=0.94, CD/pitch=0.93, passed=False
σ=0, t=60  →  P_space=0.65, area_frac=1.00, CD/pitch=0.98, passed=False
```

t=30 이 첫 통과 지점.

---

### 문제 5 — clean geometry 만 통과 (σ=0). σ ≠ 0 이면 계속 fail

**증상**: 위 결과 (σ=0, t=30) 는 통과하지만, 사용자 의도는 electron blur 효과 (σ ≠ 0) 도 다룰 수 있어야 한다는 것.

**해결**: option C 로 분리.

- Stage 1 = clean geometry baseline (σ=0, t=30) 으로 재정의.
- Stage 1A 신설: σ-호환 budget calibration.
- Stage 1B 신설: σ=5/t=60 nominal 을 over-budget 스트레스 케이스로 보존.

σ 스윕을 t=30 으로 고정해 다시 돌렸다.

```text
σ=0, t=30 → passed (margin: P_space 0.31, contrast 0.45, area 0.625, CD/p 0.625)
σ=1, t=30 → passed (P_space 0.34, contrast 0.42, area 0.667, CD/p 0.667)
σ=2, t=30 → passed (P_space 0.40, contrast 0.37, area 0.739, CD/p 0.74)
σ=3, t=30 → passed but margin tight (P_space 0.46, area 0.852, CD/p 0.85 ← 한계 직전)
σ=4, t=30 → fail (area_frac 0.961)
σ=5, t=30 → fail (contrast 0.16, area 1.00)
```

---

### 문제 6 — σ=5 는 사용자-지정 search space 안에서 통과 budget 없음

**증상**: σ=5 budget search

```text
Stage A: σ=5 × time ∈ {10,15,20,30} × DH ∈ {0.3,0.8} × kdep=0.5 × Hmax=0.2  → 8 / 8 fail
Stage B: σ=5 × time=20  × DH=0.3      × Hmax ∈ {0.1, 0.15, 0.2}             → 3 / 3 fail
가장 근접: σ=5, t=20, DH=0.3, Hmax=0.2 → P_line = 0.630 (게이트 0.65 에 0.02 부족)
```

**원인**: σ=5 / 24 nm pitch 에서

- I_blurred 진폭이 너무 작음 (interior I 범위 약 0.34–0.70)
- Hmax=0.2 의 saturated H0 진폭도 작음
- kdep=0.5 의 budget 을 더 줄이면 (P_line) 도 같이 떨어져서 line 게이트 fail
- kdep 을 키우거나 dose 를 키우면 P_line 을 올릴 수 있지만 search space 밖

**결론**: 이 search space 안에서는 σ=5 호환 불가. σ=5 호환 budget 을 정말 찾으려면 다음 중 하나 필요.

```text
- kdep ∈ {0.5, 1.0} 로 확장
- dose_mJ_cm2 ∈ {40, 50, 60} 으로 contrast 강화
- 또는 σ_max = 3 으로 effective upper bound 인정
```

이 결정은 Stage 3 (electron blur 분리 실험) 직전에 다시 다룬다.

---

### 부수 문제 — σ 가 커지면 LER_initial 이 작아짐

**증상**: σ=3 의 결과에서 `LER_initial=2.47, LER_final=3.67` — diffusion 후 LER 가 늘어났다는 비물리적 결과.

**원인**: 측정 규약 문제. "initial" edge 를 `I_blurred` (=blur 가 이미 적용된 intensity) 의 threshold contour 로 정의했기 때문에, σ 가 커질수록 I_blurred 자체가 매끈해져 LER_initial 이 작게 잡힌다. PEB 후의 P contour 에서는 acid diffusion 까지 고려된 총 blur 가 들어가므로, σ 의 영향이 LER 값으로는 제대로 드러나지 않는다.

**임시 처리**: 게이트는 LER 절대값에 의존하지 않고 `P_space`/`P_line`/`contrast`/`area`/`CD/pitch` 로만 판단. LER reduction 은 reporting 메트릭으로만 사용.

**TODO (Stage 2 또는 Stage 3 직전에)**: "initial edge" 정의를 binary I (blur 전) 의 threshold contour 로 바꾸거나, σ-별 "no-PEB" reference run 을 따로 측정해서 PEB 효과만 분리. plan §6.4 의 PSD 분석에서 이 문제를 다시 다룰 것.

---

## 4. 의사결정 로그

| 결정 | 채택 옵션 | 이유 |
|---|---|---|
| 도메인 크기 | 128 → 120 nm | pitch 의 정수배 → FFT seam artifact 제거. `ensure_pitch_aligned_domain()` 으로 자동 스냅. |
| 게이트 정의 | `P_min < 0.45` (글로벌) → interior gate (P_space/P_line/contrast/area/CD/pitch) | 글로벌 P_min 은 boundary effect 로 false-pass 가능. interior 측정만 line-space 분리를 보증. |
| Stage 1 nominal | (σ=5, t=60) → (σ=0, t=30) | (σ=5, t=60) 이 24 nm pitch 와 호환 안 됨. σ=0/t=30 이 interior gate margin 가장 큼. |
| (σ=5, t=60) 처리 | 폐기 X / over-budget 스트레스 케이스로 보존 ✓ | "이 조합은 비정상 영역" 의 reference 데이터 포인트로 가치 있음. Stage 5 process window 분석에서 활용. |
| σ=5 호환 budget 탐색 종료 시점 | search space 확장 안 함 | 사용자 spec ("kdep=0.5 first, then Hmax sweep") 안에서 fail 확정. 확장은 Stage 3 직전에 다시 결정. |
| σ 운용 범위 | σ ∈ [0, 3] nm at (t=30, DH=0.8, kdep=0.5, Hmax=0.2) | σ=4 부터 area_frac > 0.95 로 lines merge. σ=3 도 CD/pitch=0.85 한계라 실용 상한 = 3. |
| measurement convention | σ 의존성 알면서 `I_blurred` initial contour 사용 유지 | Stage 2/3 에서 정식으로 재정의 예정. 지금 바꾸면 게이트와의 정합성이 깨짐. |

---

## 5. 검증된 결과

### Stage 1 baseline (`configs/v2_stage1_clean_geometry.yaml`)

```text
σ = 0 nm
t = 30 s
DH = 0.8 nm²/s
kdep = 0.5 s⁻¹
kloss = 0.005 s⁻¹
Hmax = 0.2 mol/dm³
quencher = off
domain = 120 × 120 nm  (pitch 5 개)

Outcome:
  H_peak = 0.077, H_min = 0.034
  P_max  = 0.787, P_min = 0.219
  P_space_center_mean = 0.311
  P_line_center_mean  = 0.763
  contrast            = 0.452
  area_frac           = 0.625
  CD_initial = 12.46 nm  →  CD_final = 15.01 nm  (CD_shift = +2.55 nm)
  LER_initial = 2.77 nm  →  LER_final = 2.65 nm  (-4.3 %)
  모든 interior gate PASS, no NaN, bounds OK
```

### σ-호환 운용 범위 (Stage 1A)

```text
fix:  t=30, DH=0.8, kdep=0.5, Hmax=0.2
σ ∈ [0, 3] nm  → interior gate PASS
σ ≥ 4 nm       → fail (area_frac 또는 contrast)
```

### over-budget 스트레스 (Stage 1B)

```text
σ=5, t=60 (plan §4 원본 nominal):
  P_space_mean=0.83, contrast=0.07, area_frac=1.00, CD/pitch=0.98
  → lines collapse into a slab
```

---

## 6. 후속 작업

- **Stage 2** (DH × time sweep): σ=0 baseline 에서 출발. plan §5.2 그대로 진행 가능.
- **Stage 3 직전** (electron blur 분리 실험): σ ∈ {0, 2, 5, 8} 의도를 살리려면 σ=5/8 의 호환 budget 이 필요. Stage 1A.3 (kdep, dose 확장) 을 먼저 돌릴지, 아니면 σ_max=3 으로 plan 자체를 수정할지 결정 필요.
- **measurement convention**: "initial" edge 의 σ 의존성 제거. 후보 두 가지 — (a) binary I 의 threshold contour 사용, (b) σ-별 no-PEB reference. Stage 2 시작 전에 결정.
- **PSD-based edge metric**: plan §6.4 의 edge PSD 미구현. Stage 2 의 LER smoothing 분석에서 필요.

---

## 7. 산출물 목록

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
  run_sigma_sweep_helpers.py         # 공유 helper (interior gate 포함)
  01_lspace_baseline/
    run_baseline_no_quencher.py      # 단일 baseline run
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
  §5  Stage 1 재정의 (clean geometry)
  §5  Stage 1A 신설 (σ-호환 budget calibration)
  §5  Stage 1B 신설 (over-budget reference)
  §9  Stage 1 종료 기준 갱신
  §10 Stage 1 baseline config 갱신
  §13 최종 정리에 calibration finding 반영
```
