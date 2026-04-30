# PEB v2 calibration plan

## Strategy

```text
v2 first-pass OP 가 만들어내는 절대값 (CD, LER, process window) 이
external reference 또는 published high-NA EUV 와 일치하는지 검증하고,
일치하지 않으면 어느 chemistry knob (Hmax / kdep / DH / σ / dose / abs_len)
에서 offset 이 발생하는지 분류하여 보정한다.
```

`calibration_targets.yaml` 가 정량적 목표를, 본 문서가 phase 별 진행 / 결정 / 결과를 누적 기록한다.

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

#### 다음 단계 결정

```text
RECOMMEND: Phase 1 finding 이 internal-consistency check 임을 인지하고, 
          외부 reference 데이터를 입수하기 전까지는 다음 우선순위:

(a) 외부 reference 데이터 입수 →  calibration_targets.yaml 갱신 →  Phase 1 재실행
    (가장 본질적, but 데이터 입수가 prerequisite)

(b) Phase 2A — dose / σ / abs_len 까지 확장한 broader sweep
    (우리가 controllable 하지 않은 변수 추가 검증)

(c) DH=0.8 로 OP 살짝 조정해 Phase 2 (process window) 짧게 재검증
    (low-cost sanity check, but recursive 위험 있음)

지금 단계: (a) 가 prerequisite 이므로 데이터 입수 대기. 그 동안 (b) 또는 (c) 는 user 결정.
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
