# Phase 2 — Stage 2: DH × time sweep

## 0. 한 줄 요약

clean geometry baseline (σ=0) 위에서 DH × time 25-grid 를 돌렸다. 알고리즘 선택은 **DH=0.8, t=20** (LER 9.61% 감소, CD_shift −1.18 nm) 이지만 P_line 게이트 margin 이 매우 좁아, 실용적으로는 **DH=0.5, t=20** (LER 8.80%, CD_shift −0.93, P_line=0.68) 또는 **DH=0.5, t=30** (LER 8.69%, CD_shift +1.79, P_line=0.79) 을 권장.

---

## 1. 목적

- Plan §5 Stage 2: PEB diffusion length 가 roughness smoothing 과 CD shift 에 미치는 영향을 분리.
- 정량 질문 두 개를 답한다.
  - DH 와 time 중 어떤 것이 LER 감소에 더 기여하는가?
  - LER 감소를 얻기 위해 감수해야 하는 CD shift 는 얼마인가?

---

## 2. 진행 단계

1. Stage 2 base config `configs/v2_stage2_dh_time.yaml` 작성. geometry/exposure/chemistry 는 Stage 1 baseline 과 동일, `time_s` 와 `DH_nm2_s` 는 sweep 스크립트가 override.
2. `experiments/02_dh_time_sweep/run_dh_time_sweep.py` 작성. grid:
   - DH ∈ {0.3, 0.5, 0.8, 1.0, 1.5} nm²/s
   - time ∈ {15, 20, 30, 45, 60} s
   - 합 25 runs at σ=0, kdep=0.5, kloss=0.005, Hmax=0.2, quencher off.
3. 두 단계 평가
   - **Interior gate** (Stage 1A 와 동일): P_space_mean<0.5, P_line_mean>0.65, contrast>0.15, area_frac<0.90, CD/pitch<0.85, finite metrics.
   - **Selection bound**: 통과한 run 중에서 추가 조건 (CD_shift ≤ 3.0, CD/pitch < 0.85, area_frac < 0.90) 을 만족하는 후보 안에서 LER_reduction_pct 최대화.
4. CSV / figures / best.json 저장. fail/selection 거절 사유를 cell-by-cell 로 기록.

---

## 3. 발생한 문제와 해결

### 문제 1 — `LER_final = 0` 이 100% reduction 으로 잡힘 (merged-line artifact)

**증상**: 게이트 fail 인 cells (예: DH=1.0, t=60) 의 `LER_reduction_pct = 100.00%`. 그런데 실제로는 lines 가 합쳐져 edge 가 추출 불가.

**원인**: line 들이 merge 되면 edge_extract 가 NaN 또는 0 으로 떨어져 final LER 가 0 이 되고, 비율은 100% 로 계산된다. 이는 metric 계산 코드의 정상 동작이지만 의미가 없는 값.

**해결**: selection 알고리즘은 interior gate 를 통과한 run 에서만 최댓값을 찾으므로 이 artifact 가 selection 에 영향을 주지 않는다. 다만 보고용 표에 fail_reason 을 같이 출력해서 100% 가 artifact 임을 분명히 했다.

**미해결 / Stage 2 의 한계**: LER metric 자체는 contour 기반이므로, 게이트 fail 영역에서는 신뢰할 수 없다. Stage 2 는 게이트 통과 영역만 분석한다는 전제로 진행.

---

### 문제 2 — 알고리즘 best 의 P_line margin 이 너무 좁다

**증상**: DH=0.8, t=20 (algorithmic best) 의 `P_line_center_mean = 0.6534` 로 게이트 (>0.65) 에 0.003 만 남는다. seed 한 번 바꾸면 fail 할 수도 있는 결과.

**원인**: 사용자-지정 selection criterion 이 LER_reduction_pct 만 최대화하고 P_line margin / contrast margin 을 고려하지 않는다.

**해결**: 알고리즘 best 는 spec 대로 보고하되, 안전 margin 이 큰 alternates 를 study note 에 같이 기록.

```text
algorithmic best : DH=0.8, t=20 → LER 9.61%, CD_shift -1.18, P_line 0.65 (margin 0.003)
robust alt 1     : DH=0.5, t=20 → LER 8.80%, CD_shift -0.93, P_line 0.68 (margin 0.03)
robust alt 2     : DH=0.5, t=30 → LER 8.69%, CD_shift +1.79, P_line 0.79 (margin 0.14)
robust alt 3     : DH=0.3, t=30 → LER 8.04%, CD_shift +1.32, P_line 0.81 (margin 0.16)
```

향후 Stage 2 selection 을 강화하려면 P_line margin ≥ 0.05 등의 추가 제약을 거는 것이 합리적.

---

### 문제 3 — 작은 (DH, t) 영역에서 line CD 가 줄어듦

**증상**: DH=0.8, t=20 / DH=0.5, t=15 등에서 `CD_shift` 가 음수 (예: -1.18, -3.29). 일반적으로 PEB diffusion 은 line 을 넓힌다고 알려져 있는데?

**원인**: 측정 규약 + 짧은 PEB 시간의 상호작용.
- `CD_initial` 은 binary mask `I_blurred` (σ=0 에서는 단순 binary) 의 threshold 0.5 contour → line 폭 = 12.5 nm.
- `CD_final` 은 PEB 후 `P` 의 threshold 0.5 contour → P=0.5 가 되는 위치.
- t=20 / DH=0.8 에서 diffusion length ≈ √(2·0.8·20) ≈ 5.7 nm. 이는 line 폭 (12.5 nm) 의 절반에 가깝다. acid 가 line 밖으로 흘러 H 가 평균화되면 P_line_center_mean 도 그만큼 낮아지고 (0.65), P=0.5 contour 는 line 안쪽 으로 들어온다.
- 즉 짧은 t 에서는 line 중심도 P=0.5 를 겨우 넘는 수준이라, CD 가 binary 폭보다 작아질 수 있다.

**해결**: 이는 물리적으로 정상적인 결과 (under-developed line) 이지만 사용자가 흔히 기대하는 "PEB 는 항상 line 을 넓힌다" 와는 어긋난다. 이를 study note 에 명시하고, 실용적으로는 P_line_center_mean ≥ 0.75 정도 영역 (예: DH=0.5, t=30) 을 권장.

**교훈**: CD_shift 의 부호 자체로 정상/비정상을 판단하면 안 된다. P_line 의 절대값과 같이 봐야 한다. under-developed (P_line~0.65) 영역에서는 CD shift 가 음수가 정상.

---

## 4. 의사결정 로그

| 결정 | 채택 | 이유 |
|---|---|---|
| Stage 2 grid 크기 | DH 5 × time 5 = 25 runs | 사용자-지정. 더 작게 하면 process window 의 boundary 가 흐려진다. |
| measurement convention | Stage 1 그대로 (LER initial = `I_blurred` threshold) | σ=0 에서는 `I_blurred` = binary mask 와 같아 σ-의존성 문제 없음. Stage 3 직전에 재정의 예정 (Stage 1 note 참조). |
| selection criterion | 사용자 spec (max LER% s.t. CD_shift≤3, CD/p<0.85, area<0.9) | 그대로 적용. 알고리즘 best 와 safe alternates 를 함께 보고. |
| best 의 marginal P_line 처리 | 알고리즘 best 그대로 + alternates 별도 기록 | spec 변경 없이 사용자가 안전성 vs LER 우선 균형 직접 결정 가능. |
| 음의 CD_shift 의 해석 | 정상 결과로 분류 | under-developed line 의 contour position 차이. selection 에 별도 페널티 없음. |

---

## 5. 검증된 결과

### 25-run grid (LER reduction %, ✓=interior gate pass)

```
                15         20         30         45         60
  DH=0.30:    6.55✗     6.63✓     8.04✓     6.96✓   -14.34✓
  DH=0.50:    9.00✗     8.80✓     8.69✓   -17.04✓    45.55✗
  DH=0.80:   10.06✗     9.61✓     4.25✓     8.28✗   100.00✗
  DH=1.00:    8.62✗     8.84✗    -1.98✓    62.62✗   100.00✗
  DH=1.50:   -4.48✗     3.98✗   -38.01✓   100.00✗   100.00✗
```

- diagonal 형태의 process window. 좌하단 = under-developed (P_line<0.65), 우상단 = merged (P_space>0.5).
- pass 영역의 LER 감소는 4–10% 범위.
- ✗ 영역의 LER% 는 신뢰 불가 (특히 100% 는 artifact).

### 알고리즘 best (max LER% s.t. CD_shift≤3, CD/p<0.85, area<0.9)

```text
DH = 0.8 nm²/s
t  = 20 s
P_space_center_mean = 0.162
P_line_center_mean  = 0.653  ← gate margin 0.003 (tight)
contrast            = 0.491
area_frac           = 0.471
CD: 12.46 → 11.28 nm  (CD_shift = -1.18 nm)
LER: 2.77 → 2.51 nm   (-9.61 %)
```

### 실용 권장 (margin 큰 alternates)

| label | DH | t | LER% | CD_shift | P_line | P_line margin | 비고 |
|---|---|---|---|---|---|---|---|
| algorithmic best | 0.80 | 20 | **+9.61** | −1.18 | 0.653 | 0.003 | tight; under-developed 영역 |
| robust alt 1 | 0.50 | 20 | +8.80 | −0.93 | 0.678 | 0.028 | LER 손실 0.8 pp 로 margin 9× 확보 |
| robust alt 2 | 0.50 | 30 | +8.69 | +1.79 | 0.792 | 0.142 | CD shift 양의 부호, P_line 충분 |
| robust alt 3 | 0.30 | 30 | +8.04 | +1.32 | 0.814 | 0.164 | 가장 healthy |

### 정성적 경향 (plan §8 정상 경향과의 비교)

| plan §8 expected | observed |
|---|---|
| DH 증가 → LER 감소 | partial yes — DH=0.3→0.5→0.8 에서 LER% 10–11% 까지 증가, 이후 DH≥1.0 에서 다시 감소 (gate fail 우세) |
| DH 증가 → CD shift 증가 | yes — 같은 t 에서 DH 가 클수록 CD_shift 더 양으로 |
| time 증가 → LER 감소 | partial yes — t=15→20→30 에서 LER% 증가, t=45,60 에서 lines 가 merge 하며 신뢰 불가 |
| time 증가 → CD shift 증가 | yes |
| 너무 큰 DH/t → line edge blur 과도 | yes — 우상단 영역 = lines merged into a slab |

---

## 6. 후속 작업

- **measurement convention 재정의** (Stage 3 직전): "initial" edge 의 σ-의존성 제거. binary I 또는 σ-별 no-PEB reference 도입.
- **Stage 3 (electron blur 분리)**: σ=0 에서 σ ∈ {0,2,3} 으로 확장. σ=5,8 은 Stage 1A.3 (kdep, dose 확장) 이 선행 필요.
- **selection 강화**: 다음 stage 부터 selection 에 P_line margin (예: ≥ 0.05) 또는 contrast margin (예: ≥ 0.20) 을 추가하면 알고리즘 best 가 항상 robust 한 영역에서 선택된다.
- **edge PSD metric** (plan §6.4): LER frequency 분해. Stage 3 와 함께.
- **process window plot**: DH × t 평면에 pass/fail + LER% iso-line 을 그린 single figure. Stage 5 process-window 분석 prework.

---

## 7. 산출물

```text
configs/v2_stage2_dh_time.yaml
experiments/02_dh_time_sweep/
  __init__.py
  run_dh_time_sweep.py
outputs/
  figures/02_dh_time_sweep/                 # 25 P maps + 25 contour overlays
  logs/02_dh_time_sweep.csv                 # 전체 metric (full)
  logs/02_dh_time_sweep_summary.csv         # core columns + fail/selection reason
  logs/02_dh_time_sweep_best.json           # algorithmic best
study_notes/02_stage2_dh_time_sweep.md      # 본 노트
```
