# Phase 7 — Stage 6: x-z standing wave + helper 통합 (CD-locked LER, PSD mid-band)

## 0. 한 줄 요약

`solve_peb_xz` (FFT mirror-extension on z) 와 1D / x-z exposure builder 를 추가, 12 runs (thickness × amplitude) 모두 gate 통과. **standing wave 의 H0 z-modulation 은 A 와 thickness 에 단조 증가 (sw_only +1 ~ +26 %)**, **PEB 가 thin film (15 nm) 에서 79 %, thick film (30 nm) 에서 60 % 감쇠**. PEB 후의 잔여 z-modulation 은 거의 absorption envelope 의 효과 — standing wave 단독으로는 0.14-0.85 % 만 남음. helper 에는 CD-locked LER 와 PSD mid-band reduction 을 standard column 으로 통합 (~50 ms / run cost).

---

## 1. 목적

```text
1. PEB 가 standing wave (period 6.75 nm) 의 z-modulation 을 얼마나 줄이는가?
2. film thickness (15 / 20 / 30 nm) 에 따라 PEB 효과 차이?
3. top/bottom asymmetry 가 absorption envelope 으로 얼마나 발생?
4. CD-locked LER 측정이 (x, z) 단면에서 사이드월 변동을 잡아낼 수 있나?
```

또한 user request 로 helper 의 default LER 비교를 CD-locked 으로 통합하고 PSD mid-band reduction 을 standard 컬럼으로 추가.

---

## 2. 진행 단계

1. **helper 통합** (`run_sigma_sweep_helpers.py`):
   - 매 run 후 `find_cd_lock_threshold(P, ..., cd_target=CD_initial)` 호출.
   - 새 컬럼: `LER_CD_locked_nm`, `CD_locked_nm`, `P_threshold_locked`, `cd_lock_status`, `total_LER_reduction_locked_pct`, `psd_locked_low/mid/high`, `psd_mid_band_reduction_pct`, `psd_mid_band_reduction_locked_pct`.
   - 기존 fixed-threshold 컬럼은 유지 (Stage 1-4 backward compat).
   - cost: ≈ 50 ms / run, 기존 sweep 들 영향 없음 (CSV columns 추가).

2. **1D exposure + x-z exposure builders** (`src/exposure_high_na.py`):
   - `line_space_intensity_1d(domain_x, dx, pitch, line_cd) → I_x(x), x, line_centers`
   - `gaussian_blur_1d(field, dx, sigma) → blurred I`
   - `build_xz_intensity(I_x, z, period, A, phase, abs_len) → I(z, x)`

3. **x-z PEB solver** (`src/fd_solver_xz.py`):
   - `_even_mirror_extend_z`: 도메인을 length 2*nz-2 로 확장해 Neumann BC 자동 만족.
   - `_spectral_diffusion_decay_xz`: extended field 에 2D FFT, decay multiply, iFFT, crop.
   - `solve_peb_xz`: 연산자 분할. spectral diffusion+kloss → quencher reaction → P explicit.
   - bounds clipping 매 step.

4. **단위 시험 6 추가** (총 27/27 통과):
   - mirror extension 정확성
   - trapezoidal z-integral 보존 (mirror trick 의 invariant)
   - naive sum 의 boundary 보정 차이가 small 임 확인
   - solver bounds (no_quencher path)
   - exposure A=0 separable
   - exposure A>0 has z-modulation

5. **Stage 6 sweep**: 3 thickness × 4 amplitude = 12 runs.

---

## 3. 발생한 문제와 해결 방법

### 문제 1 — Mirror trick 으로 naive H sum 이 정확히 보존되지 않음

**증상**: 첫 conservation test (`abs(H_after.sum() - H.sum()) < 1e-7 * H.sum()`) fail. 0.10 % 차이.

**원인**: even-mirror extension 으로 만든 periodic 함수의 DC 모드는 **trapezoidal sum** 에 대응 (`H[1:-1].sum() + 0.5*(H[0] + H[-1])`). naive sum 은 이 invariant 와 boundary 가중치 차이만큼 어긋남.

**해결**: 

- conservation test 를 **trapezoidal invariant** 기준으로 변경 (`< 1e-7 * |inv_before|`). 통과.
- naive sum 변화가 boundary-row sum 보다 작은지 보조 test 로 확인.

**교훈**: spectral solver 에서 BC 가 Neumann 이면 mirror-extension 이 만드는 invariant 는 trapezoidal rule. 보존성 시험은 그 invariant 로 해야 한다.

---

### 문제 2 — A=0 임에도 H0 의 z-modulation 이 0 이 아님

**증상**: 첫 row gate 가 A=0 row 들에서 fail. `thick=20, A=0` 의 H0_z_modulation_pct = 44.93 % (정상치는 ~0 여야).

**원인**: `I(x,z) = I_x(x) * (1 + A*cos(...)) * exp(-z/abs_len)`. A=0 면 cos 항은 사라지지만 `exp(-z/30)` 는 그대로 있어 z 방향으로 단조 감쇠. 이는 절대적 z-modulation 이 아니라 **absorption envelope 의 자연 감쇠** 라 standing wave 와 무관.

**해결**: 

- 새 컬럼 `H0_z_modulation_sw_only_pct = total - same-thickness A=0 baseline`. A=0 에서는 정의에 의해 0, A>0 에서는 standing wave 단독 효과.
- gate 도 sw_only 로 변경: `A=0 → sw_only ≈ 0, A>0 → sw_only > 0 monotone`.

**교훈**: 다중 z 의존 항 (standing wave + absorption) 이 있는 모델에서는 측정 지표를 항별로 분리해야 한다. "A=0 baseline 차감" 은 가장 간단한 분리법.

---

### 문제 3 — mass_budget_drift_pct 가 -35 % 로 큼

**증상**: 12 runs 모두 mass_budget_drift ≈ -35 %. 솔버 버그 우려.

**원인 분석**:

```text
H 소모 경로 (no quencher 가정 시):
  kloss = 0.005 / s, t = 30 s  →  exp(-0.005*30) - 1 ≈ -14 %
quencher 추가 시:
  ∫(kq * H * Q) dt 가 H 적분에서 빠짐. Q0 = 0.02, kq = 1.0 으로
  Q 가 H 의 ~ 1/5 이라 추가 ~ 20 % 손실.
total expected: -14 % + (-20 %) = -34 % ~ -35 %  ✓
```

물리적으로 정상.

**해결**: study note 에 명시. column 이름은 그대로 유지 (drift = "총 H 적분의 H0 → H_final 상대 변화"). Stage 6 outputs 에 정상 기대값 ≈ -35 % 라는 주석 추가.

**교훈**: mass budget 측정은 "보존" 이 아니라 "예상되는 소모량" 을 sanity check 하는 데 쓴다. 실제 PEB 에서 H 는 kloss + quencher 로 줄어든다.

---

### 문제 4 — Stage 6 의 LER 정의 모호성

**증상**: user spec 의 `CD_locked_LER on z-averaged P(x,z)` 해석이 모호. z-averaging 으로 1D 가 되면 LER 측정 불가.

**해결 (이 stage 의 처리)**:

- `extract_edges` 를 `P(x, z)` 에 대해 호출. 함수의 "y-axis" 인자에 z 를 넣음. 결과: 각 z 에서 line 의 left/right edge 가 추출되어 (n_lines × n_z) 의 edge track 들. **물리적 의미**: 사이드월의 x-위치가 z 따라 변하는 정도 → "side-wall LER".
- LER = 3 * std(edge_x) across z. CD-locked 도 동일하게 적용.
- z-averaged P(x) 로부터 단순 CD 만 별도 계산 (LER 정의 불가).

이 해석은 study note 에 명시. user 가 다른 의미로 의도했을 가능성 인지하면서 가장 reasonable 한 단일 정의 채택.

**교훈**: spec 에 모호함이 있으면 (a) 가장 reasonable 한 단일 정의 채택 (b) 결과 보고서에서 정의 명시 (c) user feedback 으로 조정 — 의 순서가 가장 효율적.

---

## 4. 의사결정 로그

| 결정 | 채택 | 이유 |
|---|---|---|
| z 방향 BC | Neumann (no-flux) at top/bottom | spec. 실제 PEB 의 photoresist film 경계 조건과 가장 가까움. |
| FFT 처리 방법 | even-mirror extension | DCT-II 와 동등하면서 plain np.fft 로 구현 가능. 외부 lib (scipy.fft) 의존 회피. |
| 도메인 y 차원 | omit (x-z 만) | spec 의 field variables H(x,z,t) 에 명시. 3D 통합은 별도 stage. |
| line_cd_nm | 12.5 (Stage 1-5 와 동일) | spec. 다른 옵션은 변수가 너무 많아짐. |
| OP 선택 | pitch=24 robust OP only | Stage 4B 가 pitch≤20 부적합 결론. spec 도 pitch=24 만 명시. |
| LER 정의 | extract_edges treating z as y-axis (= side-wall x-displacement std) | spec 모호 → 가장 reasonable 한 단일 정의. study note 에 explicit. |
| H0 z-modulation 정의 | (max - min) / mean × 100% | peak-to-peak 가 사용자 직관에 가깝다. std/mean 도 비교용으로 검토했지만 less intuitive. |
| sw_only 정의 | total - A=0 baseline at same thickness | absorption envelope 분리. 가장 단순한 detrending. Fourier-mode 추출 더 정확하지만 overengineering. |
| `mass_budget_drift_pct` 의미 | "총 H 적분의 상대 변화" (예상치 -35%) | "drift" 가 error 를 시사하지만 column rename 시 backward compat 깨짐. comment 로 보강. |
| helper integration scope | CD-locked + PSD mid-band 만 | spec 그대로. 추가 metric 은 Stage 별 runner 가 자체적으로 계산. |
| 단위 시험 추가 | 6개 (mirror, trapezoidal, naive sum bound, solver bounds, exposure separable, exposure modulation) | core invariants 모두 커버. 더 추가하면 over-test. |

---

## 5. 검증된 결과

### Helper 통합

```text
run_sigma_sweep_helpers.py 에 추가된 column (모든 stage 공유):
  P_threshold_locked, cd_lock_status, CD_locked_nm
  LER_CD_locked_nm, total_LER_reduction_locked_pct
  psd_locked_low/mid/high
  psd_mid_band_reduction_pct, psd_mid_band_reduction_locked_pct

cost overhead: ≈ 50 ms / run (extract_edges × ~10 회 bisection iterations).
backward compat: 기존 fixed 컬럼 유지. Stage 1-4 sweep 들 동일하게 작동.
```

### Stage 6 결과 표

| thickness (nm) | A | sw_only_pct | H0_zmod_pct | Pf_zmod_pct | mod_red_pct | top/bot asym | LER_lock (nm) | bounds |
|---|---|---|---|---|---|---|---|---|
| 15 | 0.00 | 0.00 | 32.75 | 9.86 | 70.0 | 0.10 | 1.32 | ok |
| 15 | 0.05 | +2.70 | 35.45 | 10.00 | 71.8 | 0.10 | 1.32 | ok |
| 15 | 0.10 | +5.26 | 38.00 | 10.06 | 73.5 | 0.10 | 1.32 | ok |
| 15 | 0.20 | +15.56 | 48.31 | 10.20 | 78.9 | 0.10 | 1.32 | ok |
| 20 | 0.00 | 0.00 | 44.93 | 20.05 | 55.4 | 0.18 | 2.32 | ok |
| 20 | 0.05 | +1.04 | 45.96 | 20.19 | 56.1 | 0.18 | 2.33 | ok |
| 20 | 0.10 | +7.12 | 52.05 | 20.34 | 60.9 | 0.19 | 2.35 | ok |
| 20 | 0.20 | +20.27 | 65.20 | 20.71 | 68.2 | 0.19 | 2.40 | ok |
| 30 | 0.00 | 0.00 | 70.51 | 38.17 | 45.9 | 0.32 | 3.87 | ok |
| 30 | 0.05 | +6.51 | 77.03 | 38.24 | 50.4 | 0.32 | 3.85 | ok |
| 30 | 0.10 | +12.93 | 83.44 | 38.31 | 54.1 | 0.32 | 3.83 | ok |
| 30 | 0.20 | +25.54 | 96.06 | 38.42 | 60.0 | 0.32 | 3.80 | ok |

### 정성적 결론

1. **PEB 가 standing-wave-induced z-modulation 을 매우 효과적으로 흡수**.
   thin film (15 nm) 에서는 H0 의 sw 변동이 P_final 에서 거의 완전히 사라지고, P_final 의 z-mod (10 %) 는 거의 다 absorption envelope 의 결정성.
2. **Thick film (30 nm) 에서 PEB smoothing 이 약함**.
   diffusion length √(2·DH·t) = 5.5 nm 가 30 nm thickness 보다 짧아 z-direction homogenization 이 부분적. modulation_reduction 가 thin (79 %) > thick (60 %).
3. **Top/bottom asymmetry 는 thickness 에 단조 증가** (0.10 → 0.32). absorption length 30 nm 가 thick=30 까지 가면 P 가 절반 가까이 떨어진다.
4. **사이드월 LER (CD-locked) 가 thick film 에서 큼** (1.32 → 3.87 nm). thick film 의 sidewall 이 z-방향으로 더 흔들림.
5. **standing wave amplitude 가 LER 에 미치는 영향은 작음** (< 0.1 nm 변화). 이는 P_final 의 sw 잔여가 < 1 % 라서 이미 LER noise floor 이하.

---

## 6. 후속 작업

- **Stage 6B (full 3D x-y-z)**: y-direction roughness 와 z-modulation 의 결합 효과. compute cost 크다 (3D FFT 매 step). 의의 있는 후속이지만 다른 stage 우선.
- **Stage 6 의 figure 확장**: 현재 thick=20 만 figure 저장. thick=15, 30 도 그릴지 결정 가능.
- **Stage 4B σ=0 follow-up at small pitch**: 보류.
- **Stage 3B σ=5/8 호환 budget**: 보류.
- **plot pipeline 정리**: 모든 stage 의 heatmap utility 가 분산되어 있음. shared utility 모듈로 모을지 결정.

---

## 7. 산출물

```text
src/
  exposure_high_na.py
    + line_space_intensity_1d, gaussian_blur_1d, build_xz_intensity
  fd_solver_xz.py                   # new — Neumann-z spectral solver
  metrics_edge.py                   # 변동 없음 (Stage 4B 그대로)

experiments/
  run_sigma_sweep_helpers.py        # CD-locked + PSD mid-band integration
  06_xz_standing_wave/
    __init__.py
    run_xz_sweep.py                 # 12 runs + 4 figures + CSV/JSON

configs/
  v2_stage6_xz_standing_wave.yaml

tests/
  test_solver_xz.py                 # 6 new tests (총 27/27 passing)

outputs/
  figures/06_xz_standing_wave/      # 4 thickness=20 panels: I, H0, P (+ A 별 subset)
  logs/06_xz_standing_wave_summary.csv
  logs/06_xz_standing_wave_summary.json

EXPERIMENT_PLAN.md
  §Stage 6 promoted to "executed" 결과 표 + 성공 기준 매핑

study_notes/
  07_stage6_xz_standing_wave.md  (this file)
  README.md  index 업데이트
```
