# Phase 4 — Stage 4: weak quencher + PSD band metric + Stage 4B 보류

## 0. 한 줄 요약

robust OP (DH=0.5, t=30) 에서 σ × Q0 × kq 52-run sweep 을 돌렸다. **52 runs 전수 Stage-3 gate 통과**. 51/52 robust candidate. 가장 큰 발견은 **σ=3 에서 Stage 3 의 PEB-LER 악화 (-22.5%) 가 weak quencher 로 +6.6% 까지 회복**되는 것 — quencher 가 line widening 을 막아 contour 가 design edge 근처에 머물게 해 LER 측정이 정상화. PSD high-band 는 baseline 에서도 99% 이상 제거되어 quencher 차이는 low/mid band 에서만 나타남.

---

## 1. 목적

Plan §5 Stage 4 의 명목 목적 + Stage 3 에서 발견한 후속 질문.

```text
1. weak quencher (Q0 ≤ 0.03, kq ≤ 2) 가 acid tail 과 CD shift 를 줄이는가?
2. quencher 가 σ-증가 시의 PEB LER 악화 (line widening 으로 contour 가 design edge
   를 벗어나는 현상) 를 완화하는가?
3. quencher 의 효과가 어느 spatial frequency band 에서 나타나는가?
   (PSD low / mid / high 분해)
```

---

## 2. 진행 단계

1. **edge PSD metric 추가** (`src/metrics_edge.py`):
   - `compute_edge_band_powers(edges, dy_nm, bands)` — 각 edge track 의 residual rfft → band power 합산 → 모든 track 평균.
   - default bands: low [0, 0.05), mid [0.05, 0.20), high [0.20, ∞) nm⁻¹ (≈ λ > 20 nm / 5–20 nm / < 5 nm).
   - `stack_lr_edges` helper — left+right edges 를 (2*n_lines, ny) 로 합치는 단순 concat.
   - 단위 시험 3개 추가 (sinusoid concentrate / zero / shape).

2. **helper 확장** (`experiments/run_sigma_sweep_helpers.py`):
   - quencher overrides (`quencher_enabled`, `Q0_mol_dm3`, `DQ_nm2_s`, `kq_s_inv`) 를 `run_one_with_overrides` 에 추가.
   - PSD band 계산을 design / e-blur / PEB 세 stage 모두에 자동 수행.
   - `psd_high_band_reduction_pct = 100*(design_high - PEB_high)/design_high` 단일 reduction 값도 출력.

3. **Stage 4 sweep 실행** (`experiments/04_weak_quencher/run_quencher_sweep.py`):
   - robust OP 고정 (DH=0.5, t=30, kdep=0.5, Hmax=0.2, kloss=0.005, pitch=24, CD=12.5).
   - σ ∈ {0,1,2,3} × (1 baseline + 4 Q0 × 3 kq) = 52 runs.
   - 각 row 를 same-σ no-quencher baseline 과 비교 (`dCD_shift_nm`, `darea_frac`, `dtotal_LER_pp`).
   - Stage-3 gate + Stage-4 robust criteria (`P_line_margin ≥ 0.05`, `dCD_shift < 0`, `darea_frac < 0`, `dtotal_LER_pp ≥ -1.0`).
   - σ=2 contour overlay 13 개 + Q0×kq metric heatmap 4 개 저장.

4. **plan §Stage 4 갱신**, 결과 표 + Stage 4B (CD-locked LER) deferral 명시.

---

## 3. 발생한 문제와 해결

### 문제 1 — Stage 3 의 σ-증가 시 PEB LER 악화의 정체

**증상 (Stage 3 회상)**: σ ∈ {0,1,2,3} 에서 `PEB_LER_reduction_pct` 가 +8.7% → -37.7% 로 단조 악화.

**Stage 4 의 검증**: σ=3 baseline 에서 `total_LER_reduction = -22.5%` (LER 가 늘어남). 거기에 weak quencher (예: Q0=0.03, kq=1) 를 추가하면 `+6.6%` 로 회복 (dLER = +29.15 pp).

```text
dCD_shift = -3.79 nm   ← line 이 baseline 대비 3.8 nm 더 좁아짐
darea_frac = -0.159    ← over-deprotect area 16% 감소
dtotal_LER_pp = +29.15 ← LER reduction 이 -22.5% → +6.6% 로 회복
```

**해석**: 

- Stage 3 에서 추정한 가설 ("PEB 가 LER 를 늘리는 게 아니라 line widening 이 contour 를 design edge 에서 멀어지게 해 LER 측정이 부정확해지는 것") 이 검증됨.
- quencher 가 acid tail 을 잡아주면 line 이 좁아지고 contour 가 design edge 가까이 유지 → LER 측정이 정상화.
- 즉 Stage 3 의 음수 PEB_LER_reduction 은 PEB 의 물리 자체의 문제가 아니라 "contour 위치 ↔ design edge 위치" 의 상대적 미스매치 문제였음.

**향후 처리**: Stage 4B 의 CD-locked LER (P_threshold 자동 조정) 으로 contour 위치를 design CD 에 맞추면 PEB-only smoothing 의 진짜 효과를 분리 가능. 지금은 fixed-threshold 측정만 보유.

---

### 문제 2 — PSD high-band 가 baseline 에서도 ~100% reduction

**증상**: `psd_high_band_reduction_pct` 가 σ=0,1,2 의 모든 row 에서 99.9% 이상. quencher 의 효과 차이가 high-band 에서 거의 보이지 않음.

**원인**: high-band (f ≥ 0.20 nm⁻¹, λ < 5 nm) 의 design noise 는 edge correlation length (5 nm) 보다 짧은 white-noise tail 이라, PEB diffusion (scale ~ √(2DH·t) = 5.5 nm at robust OP) 만으로 거의 완전히 평균화. quencher 는 추가로 줄일 여지가 없음.

**해석**: high-band 는 quencher 효과 분석에 부적합. 비교는 mid-band (5–20 nm) 또는 total LER 로 해야 한다.

**해결**: study note 와 plan 에 명시. 향후 stage 에서 mid-band reduction 을 메인 PSD metric 으로 사용.

---

### 문제 3 — fixed-threshold 비교가 σ/quencher 의 CD 변화에 오염됨

**증상**: σ 가 다르거나 quencher 가 들어가면 line CD 가 달라져 (예: σ=3 baseline CD/p = 0.76, σ=3 + Q0=0.03 + kq=2 → CD/p = 0.55), P_threshold=0.5 contour 가 다른 위치에서 측정됨. LER 절대값 비교가 misleading 가능.

**해결 (이번 stage 의 처리)**: 

- 절대값 비교 (`LER_after_PEB_P_nm`) 대신 **same-σ baseline 대비 delta** 사용 (`dtotal_LER_pp`). σ 는 row 마다 fix 하므로 σ 의 영향은 빠지고 quencher 의 영향만 남는다.
- σ 간 비교 (예: σ=0 vs σ=3) 는 LER 절대값으로 하지 않고 Stage 5 의 process-window 또는 Stage 4B 의 CD-locked 측정으로 미룸.

**Stage 4B trigger**: Stage 5 또는 외부 reference 비교에서 CD shift 의 LER 오염이 결정에 영향을 미치는 경우.

---

## 4. 의사결정 로그

| 결정 | 채택 | 이유 |
|---|---|---|
| OP 선택 | robust OP (DH=0.5, t=30) only | Stage 3 에서 algorithmic-best OP 가 P_line_margin gate 모두 fail. Stage 4 부터 algorithmic-best 인용 안 함. |
| σ 범위 | {0, 1, 2, 3} | Stage 3 결정과 일치. σ=5/8 은 plan §Stage 3B 보류 그대로. |
| Q0 범위 | {0, 0.005, 0.01, 0.02, 0.03} | 사용자 지정. Plan 원안의 `[0,0.01,0.02,0.03,0.05]` 보다 lower-end 더 조밀. 0.05 는 v1 issue 영역이므로 제외. |
| kq 범위 | {0.5, 1.0, 2.0} | 사용자 지정. plan §4.4 의 `kq_sweep_safe` 의 lower 3점. |
| DQ | 0 고정 | quencher diffusion 까지 켜면 변수 증가. Stage 4 에서는 quencher 가 spatially uniform 으로 시작해 H 와의 reaction 으로만 변하는 단순화. |
| baseline 정의 | per-σ no-quencher (Q0=0) | σ 의 영향과 quencher 의 영향 분리. |
| kq 무관성 (Q0=0) | baseline 1 row only | Q0=0 일 때 kq 가 결과에 영향 없으므로 row 수 절감 (4 × (1 + 12) = 52). |
| 비교 metric | dCD_shift, darea_frac, dtotal_LER_pp | 사용자 지정. PSD high-band 는 baseline 에서도 saturate 하므로 보조. |
| robust criterion | P_line_margin ≥ 0.05 + dCD<0 + darea<0 + dLER≥-1pp | 사용자 지정. dLER ≥ -1pp 는 "materially worsen" 의 정량 정의. |
| dLER threshold | -1.0 pp | "materially" 의 임의 기준. 1pp 는 stage 1 baseline 의 LER reduction (8.7%) 의 ~12% 수준이라 noise 와 구분 가능한 최소 변화. |
| σ=2 가 primary | 사용자 지정 | σ=0 은 e-blur 효과 없음, σ=3 은 baseline 자체가 비정상 (LER 음수). σ=2 가 e-blur + PEB + quencher 셋이 모두 의미 있는 영역. |
| Stage 4B (CD-locked) deferral | trigger 조건 명시 후 보류 | fixed-threshold Stage 4 가 robust 결과 충분히 만들어 지금 당장 필요 X. |

---

## 5. 검증된 결과

### Gate / robust 통계

```text
52 runs 전체:
  Stage-3 strengthened gate pass : 52 / 52
  Stage-4 robust candidate       : 51 / 52   (σ=3, Q0=0.03, kq=2 만 margin=0.039 < 0.05)
```

### σ=2 (primary): dtotal_LER_pp heatmap

```text
                kq=0.5    kq=1.0    kq=2.0
  Q0=0.030      +4.90     +6.47     +7.64
  Q0=0.020      +3.74     +5.21     +6.44
  Q0=0.010      +2.18     +3.24     +4.23
  Q0=0.005      +1.18     +1.84     +2.49

baseline (Q0=0): total_LER_reduction = +3.56%
quencher 추가 시 +1.18pp ~ +7.64pp 추가 감소.
```

### σ=2 추천 candidates

| label | Q0 | kq | dCD | darea | dLER pp | margin | total LER% (final) |
|---|---|---|---|---|---|---|---|
| balanced | 0.020 | 1.0 | -1.76 | -0.073 | +5.21 | 0.096 | +8.77 |
| max-LER | 0.030 | 2.0 | -3.54 | -0.147 | +7.64 | 0.053 | +11.19 |
| max-margin | 0.005 | 0.5 | -0.29 | -0.012 | +1.18 | 0.132 | +4.74 |

### σ=3 의 LER 회복 (Stage 3 발견의 검증)

```text
σ=3 baseline (no quencher):
  total_LER_reduction = -22.51%   (PEB 후 LER 가 늘어남)
  CD/p = 0.76, area_frac = 0.77
σ=3 + Q0=0.03, kq=1.0:
  total_LER_reduction = +6.64%    (LER 가 정상적으로 줄어듦)
  CD/p = 0.61, area_frac = 0.61
  dLER = +29.15 pp, dCD = -3.79 nm, darea = -0.159
```

quencher 가 line widening 을 막아 contour 위치가 design edge 가까이 머물게 해 LER 측정이 정상화됨.

### PSD 분석

```text
psd_high_band_reduction_pct ≈ 99.7% – 100%  (모든 row, baseline 포함)
→ high-band noise (λ < 5 nm) 는 PEB diffusion length (~5 nm) 가 이미 거의 완전히 평균화.
→ quencher 의 LER 차이는 mid-band 에서 발생 (low/mid power 추적이 추후 분석에 더 유용).
```

PSD low/mid 는 study note 부속물로 기록되어 있고 (`outputs/logs/04_weak_quencher_summary.csv`), Stage 4B 또는 PSD-focused 분석에서 다시 다룰 예정.

### 정성적 경향 (plan §8 expected vs observed)

| plan §8 expected | observed |
|---|---|
| Q0 또는 kq 증가 → acid tail 감소 | yes — `dCD_shift` 가 단조 감소 |
| 적당한 quencher → CD shift 감소 | yes — 모든 quencher row 에서 dCD<0 |
| 너무 강한 quencher → Pmax / area 감소 | partial — Q0=0.03, kq=2.0 에서 area 가 0.49–0.62 까지 떨어지지만 P_line_margin 은 0.04–0.05 로 robust 한계. 본 sweep 범위에서는 "너무 강한" regime 에 도달 직전. |
| Q0=0.01, kq=1 에서 P contour 유지 | yes — 모든 σ 에서 (margin 0.077 – 0.144) |

---

## 6. 후속 작업

- **Stage 5 (pitch / dose process window)**: σ=2 + Q0=0.02, kq=1.0 (balanced) 또는 σ=0 baseline 으로 시작. pitch ∈ {16,18,20,24,28,32} 에서 process window shape 변화 측정.
- **Stage 4B (CD-locked LER)**: trigger 발생 시 진행. Stage 5 의 결과를 본 후 결정.
- **PSD mid-band 분석**: low/mid band 의 quencher 의존성을 별도 plot 으로. 지금 column 은 CSV 에만 있고 figure 없음.
- **plan §Stage 3B (σ=5,8 호환 budget)**: 여전히 보류. trigger 미충족.

---

## 7. 산출물

```text
src/metrics_edge.py
  + DEFAULT_PSD_BANDS, compute_edge_band_powers, stack_lr_edges

experiments/run_sigma_sweep_helpers.py
  + quencher overrides, PSD band columns

configs/v2_stage4_weak_quencher.yaml
experiments/04_weak_quencher/
  __init__.py
  run_quencher_sweep.py

outputs/
  figures/04_weak_quencher_sigma2/        # 13 contour overlays for sigma=2
  figures/04_weak_quencher_summary/       # 4 Q0×kq heatmaps for sigma=2
  logs/04_weak_quencher_summary.csv       # 52 rows, full metric set
  logs/04_weak_quencher_summary.json

tests/
  test_edge_metrics.py + 3 PSD tests   (17/17 passing)

EXPERIMENT_PLAN.md
  §5 Stage 4 spec 갱신 (sweep, gate, comparison criteria, results)
  §5 Stage 4B (CD-locked LER) deferral 신설

study_notes/
  04_stage4_weak_quencher.md  (this file)
  README.md  index 업데이트
```
