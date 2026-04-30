# Phase 5 — Stage 5: pitch × dose process window

## 0. 한 줄 요약

Stage-4 balanced OP (σ=2, Q0=0.02, kq=1.0, DH=0.5, t=30) 에서 6 pitch × 6 dose = 36 runs primary + 72 runs control = **108 runs**. **pitch=16 은 process window 닫힘**, pitch=18 은 valid-only 1점, pitch ≥ 24 는 robust_valid window 4 dose 폭. 흥미로운 발견: **quencher 가 small pitch 에서는 window 를 좁힌다** (Stage 4 의 LER 개선 효과와 small-pitch tolerance 사이 trade-off). 추천 dose 는 pitch ≥ 20 모두 dose=40 (Stage 4 와 일치).

---

## 1. 목적

- Plan §5 Stage 5: process window shape 와 pitch-dependent operating point.
- 추가로 다음을 답한다.
  - small pitch (16, 18) 에서 chemistry 가 무너지는 경계?
  - quencher 의 small-pitch tolerance 영향?
  - σ 단독, quencher 단독, 둘 다 의 process window 차이?

---

## 2. 진행 단계

1. **Stage 5 base config** (`configs/v2_stage5_pitch_dose.yaml`) 작성. Stage-4 balanced OP 그대로.
2. **Stage 5 sweep script** (`experiments/05_pitch_dose/run_pitch_dose_sweep.py`):
   - 6 pitch × 6 dose × 3 blocks (primary + 2 controls) = 108 runs.
   - per pitch: domain_x_nm = pitch * 5 (`n_periods_x=5`), domain_y_nm = 120 nm (LER y-sample 일관성).
   - per run: cfg deepcopy → pitch / domain_x / dose 갱신 → `run_one_with_overrides`.
   - 분류: unstable / merged / under_exposed / low_contrast / valid / robust_valid (precedence 순).
3. **Heatmap figures** (per block × 6 metrics): status, CD_shift, total_LER%, P_line_margin, area_frac, contrast.
4. **Primary contour overlays** 36 개 저장 (control 은 heatmap 만).
5. **Recommendation** per pitch: robust_valid first → min |CD_shift| → max LER% → max margin.

---

## 3. 발생한 문제와 해결

### 문제 1 — pitch=16 에서 모든 dose 가 merged 또는 under_exposed

**증상**: primary OP, pitch=16, dose ∈ {21..60} → dose=21 만 under_exposed, 나머지 5 dose 는 모두 merged. valid 영역이 단 한 점도 없음.

**원인 분석**:

- pitch=16 / line_cd=12.5 → space=3.5 nm.
- diffusion length √(2·DH·t) = √(2·0.5·30) ≈ 5.5 nm > space.
- 즉 line center 의 acid 가 space 를 가득 채울 만큼 멀리 퍼지는 게 normal.
- 추가로 σ=2 nm 의 electron blur 가 space 를 더 매끈하게 만들어 P_space 를 더 올린다.
- 결과: dose 를 낮추면 lines 자체가 P_line < 0.65 (under_exposed), dose 를 올리면 space 가 P > 0.5 (merged). 중간 영역 없음.

**해석**: 24 nm pitch 에서 발견한 "DH·t = 15 nm² = (line_cd/2)²" budget 이 16 nm pitch 에서는 line_cd / pitch 듀티가 0.78 로 너무 높고 space 가 너무 좁다. process window 진짜 닫힘.

**해결 (이 stage 의 처리)**: 

- pitch=16 은 추천 없음으로 표기. plan 의 sweep range 를 인정하면서도 chemistry 한계를 명시.
- 향후 pitch=16 까지 살리려면 (a) line_cd 를 pitch 의 약 50% 로 비례 축소, 또는 (b) DH·t budget 축소 (예: t=15 또는 DH=0.3) 가 필요. Stage 5 에서는 보존, Stage 6 / 7 의 follow-up 으로 분리.

---

### 문제 2 — pitch ≤ 20 에서 negative total_LER_reduction_pct

**증상**: pitch=20, dose=40 → robust_valid (모든 gate 통과) 인데 total_LER% = -26.22%. pitch=18, dose=28.4 → valid 인데 -34.18%.

**원인**: Stage 3 / 4 에서 진단된 contour-displacement artifact 의 small-pitch 버전. small pitch 에서는 line widening 비율 (CD_shift / pitch) 이 더 크게 보여 contour 가 design edge 로부터 더 멀리 벗어난다.

```text
pitch=20, dose=40: CD_shift = +4.10 nm  →  CD_final/pitch = 0.83
                     contour 가 design edge 위치에서 4.1 nm 떨어져 있음
                     LER 측정이 design edge 와 무관한 위치에서 이뤄짐
```

**해결**: 

- robust_valid 분류 자체는 정상 — gate 들은 모두 만족 (contour, area, P 값).
- LER reduction 의 절대값 비교는 small pitch 에서 misleading.
- 절대값 대신 **same-pitch** 의 baseline 과 비교한 delta 를 사용해야 의미 있음. 본 stage 는 absolute value 만 계산해 study note 에 caveat 명시.
- Stage 4B (CD-locked LER) 가 이 문제를 정조준할 예정.

---

### 문제 3 — quencher 가 small-pitch process window 를 좁힌다

**증상**: control_sigma2_no_q (quencher off) vs primary (Q0=0.02, kq=1.0) 비교:

| pitch | control_σ2_no_q robust_valid 수 | primary robust_valid 수 |
|---|---|---|
| 16 | 0 | 0 |
| 18 | 0 | 0 |
| 20 | 1 (dose=28.4) | 1 (dose=40) |
| 24 | 5 | 4 |
| 28 | 5 | 5 |
| 32 | 5 | 5 |

quencher 추가로 robust window 가 비슷하거나 약간 줄어듦. 특히 pitch=18 에서 control_sigma0_no_q (σ=0, quencher off) 는 dose=28.4 robust_valid 인데, primary 와 control_sigma2_no_q 는 valid only.

**원인 가설**:

- Stage 4 에서 quencher 의 효과는 line widening 감소 → contour 가 더 가까이 머묾. 24 nm pitch 에서는 LER 개선으로 작동.
- 그러나 small pitch 에서는 line 이 이미 좁고 contour 가 가깝다 (CD/p 큼). quencher 가 H 를 추가로 잡아먹으면 P_line 이 떨어지고 P_line_margin 이 더 줄어든다 (`pitch=18 dose=28.4`: control_σ0_noq margin=+0.070 → primary margin=+0.012).
- 즉 **quencher 의 LER benefit 은 large pitch 에 편중, small pitch 에서는 P_line 손실로 robust window 축소**.

**해석**: Stage 4 balanced OP (Q0=0.02, kq=1.0) 를 small pitch 에 그대로 적용하면 안 된다. pitch ≤ 20 에서는 (a) quencher 약화 (Q0 ≤ 0.005 또는 kq ≤ 0.5), 또는 (b) dose 보정이 필요.

**해결 (이 stage 의 처리)**: study note 에 명시. 향후 stage 또는 적용 단계에서 pitch-dependent 한 quencher 조정이 필요하다는 lesson.

---

### 문제 4 — heatmap 의 100% LER reduction 가 merged artifact

**증상**: pitch=16 / 18 의 merged cells 에서 `total_LER_reduction_pct = +100.00%`.

**원인**: 이미 Stage 2 에서 진단된 `LER_after_PEB_P = 0` (lines merged → edges extract 가 NaN/0) 의 100% reduction artifact. merged 으로 분류되었으므로 selection 알고리즘이 이를 채택하지 않는다.

**해결**: heatmap title 과 study note 에 "merged 셀 LER 값은 artifact" 명시. heatmap 의 vmin/vmax 를 [-30, +15] 로 잡아 100% 셀은 색상이 saturated red 로 표시되어 시각적으로 구분.

---

## 4. 의사결정 로그

| 결정 | 채택 | 이유 |
|---|---|---|
| OP 선택 | Stage-4 balanced OP only | 사용자 지정. algorithmic-best 는 Stage 3 부터 demote. |
| line_cd 처리 | 12.5 nm 고정 | 사용자 지정. plan §4.1 그대로. duty 가 pitch 마다 변하는 것 알면서 진행. |
| domain_x_nm | pitch * 5 | 사용자 지정 n_periods_x=5. FFT seam artifact 회피. |
| domain_y_nm | 120 nm 고정 | LER y-sample 수 일관 유지. 다른 옵션은 pitch 따라 ny 가 바뀌어 LER 측정의 noise floor 가 달라짐. |
| controls 실행 | 두 control block 모두 실행 (skip 가능 옵션 유지) | quencher 와 e-blur 의 단독 효과 분리. |
| 분류 precedence | unstable → merged → under_exposed → low_contrast → valid → robust_valid | 사용자 지정. low_contrast 는 사용자 spec 에 없지만 contrast 게이트 fail 의 fallback 으로 추가. |
| 추천 알고리즘 | robust_valid first → min |CD_shift| → max LER% → max margin | 사용자 지정. |
| LER 의 음수 / 100% artifact 처리 | reporting 만, 분류는 그대로 | gate 와 분류는 contour/area 기반이라 영향 없음. |
| pitch=16 추천 | "추천 없음" 으로 명시 | 사용자가 원래 제안한 process window 가 chemistry 한계 외부에 있음을 정직히 보고. |
| Stage 4B / 3B | 그대로 보류 | trigger 미발생. |

---

## 5. 검증된 결과

### Primary OP (108 runs 중 36 primary)

```text
Status counts (primary):
  unstable      :  0
  merged        : 17  (pitch 16/18 거의 전부, pitch 20 의 dose ≥ 44.2, pitch 16 의 dose 28.4)
  under_exposed :  6  (모든 pitch, dose=21)
  low_contrast  :  0
  valid         :  5  (pitch 18~24 의 dose=28.4)
  robust_valid  : 14
```

### 추천 dose per pitch (primary)

| pitch | rec dose | status | margin | CD_shift | LER% | comment |
|---|---|---|---|---|---|---|
| 16 | — | none | — | — | — | process window closed |
| 18 | 28.4 | valid | 0.012 | +1.99 | -34.18 | margin tight; LER artifact |
| 20 | 40 | robust_valid | 0.098 | +4.10 | -26.22 | LER artifact (CD widening) |
| 24 | 40 | robust_valid | 0.096 | +2.07 | +8.77 | Stage 4 와 일치 |
| 28 | 40 | robust_valid | 0.095 | +1.81 | +14.33 | comfortable |
| 32 | 40 | robust_valid | 0.095 | +1.78 | +15.03 | comfortable |

**모든 pitch ≥ 20 에서 dose=40 으로 동일**. dose 변화에 둔감. 이는 Stage 4 의 quencher 가 acid budget 을 안정화한 결과로 해석.

### Control 비교 (robust_valid 셀 수)

```text
                     pitch=16  18  20  24  28  32
  primary               0      0   1   4   5   5
  control_σ0_no_q       0      1   3   5   5   5
  control_σ2_no_q       0      0   1   5   5   5
```

- σ=0 no-quencher 가 가장 넓은 window. e-blur 와 quencher 모두 small pitch tolerance 를 줄이는 방향.
- e-blur 추가 (control_σ0_no_q → control_σ2_no_q): pitch=18 에서 robust_valid 1점 손실, pitch=20 에서 2점 손실, pitch=24+ 에서는 같음.
- quencher 추가 (control_σ2_no_q → primary): pitch=24 에서 1점 손실, 나머지 동일.

### 정성적 경향 (plan §5 expected vs observed)

| plan §5 expected | observed |
|---|---|
| 20–24 nm pitch 에서 stable contour | yes — pitch=20 1점, pitch=24 4점 robust_valid |
| 16–18 nm pitch 에서 process window 좁아짐 | yes — pitch=16 closed, pitch=18 valid only 1점 |
| high dose 에서 CD widening / over-deprotection | yes — pitch ≤ 20, dose ≥ 44 모두 merged |
| low dose 에서 under-deprotection | yes — 모든 pitch, dose=21 under_exposed |

---

## 6. 후속 작업

- **Stage 6 (x-z standing wave)**: 다음 메인 stage. plan §5 Stage 6 그대로.
- **Stage 4B (CD-locked LER)**: pitch ≤ 20 의 LER artifact 가 의사결정에 영향 → trigger 발생. 그러나 Stage 6 가 우선. Stage 6 후 또는 Stage 5 follow-up 에서 진행.
- **pitch-dependent quencher** (가설 검증): pitch ≤ 20 에서 quencher 약화 (Q0=0.005 등) 가 robust window 회복하는지 mini-sweep. 본 stage 의 lesson 으로 발견된 follow-up 이라 plan 에 추가 안 함.
- **line_cd scaling** (대안 1): pitch=16 까지 process window 살리려면 line_cd ≈ pitch/2 로 비례 축소. 별도 stage 로 분리 가능.
- **DH·t budget reduction at small pitch** (대안 2): pitch=16 에서 DH=0.3 또는 t=15 로 budget 축소. 본 stage 의 dose 와 별도 차원의 sweep.
- **Stage 3B** (σ=5/8 호환 budget): 여전히 보류.

---

## 7. 산출물

```text
configs/v2_stage5_pitch_dose.yaml
experiments/05_pitch_dose/
  __init__.py
  run_pitch_dose_sweep.py

outputs/
  figures/05_pitch_dose/
    primary/                              # 6 heatmaps
    control_sigma0_no_q/                  # 6 heatmaps
    control_sigma2_no_q/                  # 6 heatmaps
    primary_contours/                     # 36 contour overlays
  logs/05_pitch_dose_summary.csv          # 108 rows full metric
  logs/05_pitch_dose_summary.json
  logs/05_pitch_dose_recommendation.json  # per-pitch rec

EXPERIMENT_PLAN.md
  §Stage 5 갱신 (OP, sweep, classification, recommendation algorithm,
  검증 결과, control 비교, plan §5 success 기준 매핑)

study_notes/
  05_stage5_pitch_dose.md  (this file)
  README.md  index 업데이트
```
