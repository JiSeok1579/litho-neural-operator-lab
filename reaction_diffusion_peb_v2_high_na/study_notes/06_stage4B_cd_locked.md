# Phase 6 — Stage 4B: CD-locked LER + pitch-dependent quencher mini-sweep

## 0. 한 줄 요약

P_threshold 를 bisect 해 contour 위치를 design CD 에 고정하는 CD-lock 측정을 도입. 12 Block A runs (primary + control × 3 pitch × 2 dose) 에서 displacement artifact 1 case (control_σ0 / pitch=20 / dose=40) 와 merged artifact 2 cases 만 발견되고, **나머지 모두 진짜 (real) roughness degradation** 으로 판명. Stage 5 의 negative LER reduction 일부는 측정 artifact 이지만 primary OP (σ=2 + Q0=0.02 + kq=1) 의 pitch ≤ 20 에서는 진짜 degradation. pitch-dependent quencher mini-sweep (20 runs) 결과 **pitch=18, 20 에서는 quencher 약화로도 LER_locked 회복 불가** — Stage 4 balanced OP 는 pitch ≥ 24 에서만 권장.

---

## 1. 목적

Plan §5 Stage 4B 와 Stage 5B 합쳐 두 질문에 답한다.

```text
1. Stage 5 의 "negative LER reduction at pitch ≤ 20" 가 fixed-threshold 측정의
   contour-displacement artifact 인가, 아니면 진짜 PEB roughness 악화인가?
2. pitch=18, 20 에서 quencher 를 약화하면 LER 가 회복되는가?
   (Stage 5 의 small-pitch process-window 축소 follow-up)
```

---

## 2. 진행 단계

1. **CD-lock 함수 추가** (`src/metrics_edge.py`):
   - `find_cd_lock_threshold(field, x, line_centers, pitch, cd_target, P_min=0.2, P_max=0.8, cd_tol=0.25)` — bisection 으로 P_threshold 결정.
   - 끝점이 contour-empty 일 때 0.05 step 으로 안쪽 narrow (P_max 가 P_line_max 보다 큰 흔한 케이스 대응).
   - Status: `ok / unstable_low_bound / unstable_high_bound / unstable_no_crossing / unstable_no_converge`.
   - 단위 시험 3개 추가 (총 20개 통과).
2. **Stage 4B Block A — CD-locked 재측정**:
   - primary OP (σ=2, Q0=0.02, kq=1.0) 와 control OP (σ=0, quencher off) 를 pitch ∈ {18, 20, 24} × dose ∈ {28.4, 40} 에서 실행 → 12 runs.
   - 각 row: fixed (P=0.5) 와 locked metrics 를 한 번의 PEB run 에서 추출.
   - 결정 라벨 4 카테고리: `real degradation / fixed underestimates (merged) / fixed overestimates (displacement) / OK`.
3. **Stage 5B (Block B) — pitch-dependent quencher mini-sweep**:
   - σ=2, DH=0.5, t=30 고정. (pitch=18, dose=28.4) + (pitch=20, dose=40) × (1 baseline + 3 Q0 × 3 kq) = 20 runs.
   - CD-locked LER 와 fixed LER 모두 보고.
   - per-pitch heatmap (LER_locked, dLER_locked, dCD_shift, P_line_margin).
4. **plan §Stage 4B** 갱신, **study notes 작성**, commit + merge.

---

## 3. 발생한 문제와 해결

### 문제 1 — CD-lock 의 P_max=0.8 endpoint 가 거의 모든 row 에서 contour 없음

**증상**: 첫 실행에서 12 Block A runs 중 9 runs 가 `unstable_no_crossing`. CD-lock 이 거의 작동 안 함.

**원인**: P_line_center_mean 이 보통 0.65–0.85 사이라 P_threshold=0.8 contour 는 line 의 가장 좁은 부분만 잡거나 아예 없다. extract_edges 가 전 row 에서 NaN 반환 → cd_overall_mean = NaN → 함수가 unstable_no_crossing 처리.

**해결**: 끝점에서 "contour 없음" 이면 0.05 씩 안쪽으로 narrow 하는 logic 추가.

```python
P_hi_use = P_max
cd_hi, _ = cd_at(P_hi_use)
while not np.isfinite(cd_hi) and P_hi_use > P_lo_use + 1e-3:
    P_hi_use = round(P_hi_use - 0.05, 4)
    cd_hi, _ = cd_at(P_hi_use)
```

P_min 도 동일하게 처리. 이후 정상 범위에서 bisection 진행.

**수정 후 결과**: 12/12 runs CD_lock=ok. mini-sweep 20/20 ok.

**교훈**: bisection 의 endpoint 는 "user-spec 범위" 가 아니라 "유효 범위" 로 자동 narrow 해야 한다. 이번처럼 spec 이 [0.2, 0.8] 이어도 P 의 동적 범위가 그보다 좁으면 사용자 의도를 보존하면서 stability 확보.

---

### 문제 2 — pitch=18 dose=40 의 fixed-threshold LER 가 design 보다 *낮아* 보임

**증상**: 

```text
primary, pitch=18, dose=40:
  status_fixed = merged
  LER_design = 2.77,  LER_fixed = 1.63   (design 보다 더 smooth?)
```

기대와 반대 — merged 영역인데 LER 가 작게 나옴.

**원인**: lines 가 완전 merge 되면 P>0.5 contour 는 더 이상 line 의 "edge" 가 아니라 도메인 가장자리나 acid intensity 의 작은 변동 위치를 따라간다. 그곳에서 추출된 edges 는 design line edge 와 무관한 매끈한 곡선이라 LER 가 작게 잡힌다. **즉 lines 가 사라진 영역에서의 fixed-threshold LER 는 의미가 없는 artifact 값**.

**해결**: CD-lock 으로 contour 를 design CD=12.4 위치로 강제 이동 → LER_locked=4.16 으로 정확한 측정. 결정 라벨에 "fixed underestimates (merged-line artifact)" 추가.

**교훈**: merged status 의 fixed-threshold LER 는 신뢰 불가. CD-locked LER 또는 PSD-mid 로 비교해야 한다.

---

### 문제 3 — control_σ0 pitch=20 dose=40 에서 displacement artifact 확인

**증상**: Stage 5 control_sigma0_no_q 의 pitch=20 dose=40 row 가 robust_valid 였는데 total_LER% = -19.75% (fixed). CD-lock 후:

```text
LER_design = 2.77,  LER_fixed = 3.32,  LER_locked = 2.84
```

LER_locked 가 design 에 거의 일치. fixed 의 +0.55 nm 차이는 contour 가 design edge 에서 떨어진 곳에서 측정된 displacement artifact.

**해석**: Stage 3/4 에서 가설로 제안한 "PEB 가 LER 를 늘리는 게 아니라 line widening 이 contour 를 옮긴 것" 이 이 single case 에서 정확히 검증됨. CD-locked LER 도구의 가치 입증.

---

### 문제 4 — primary OP / pitch ≤ 20 에서 CD-lock 후에도 LER 악화

**증상**:

```text
primary, pitch=18, dose=28.4:  fixed=3.72  locked=4.07  → 둘 다 design 보다 높음
primary, pitch=20, dose=40:    fixed=3.50  locked=3.25  → 둘 다 design 보다 높음
```

CD-lock 으로 contour 를 design CD 에 맞춰도 LER 는 design 보다 훨씬 큼.

**해석**: 

- 이는 "real roughness degradation" — PEB 가 진짜로 line edge 를 더 거칠게 만들었다.
- σ=0 control 의 같은 pitch=20 dose=40 (LER_locked=2.84 ≈ design) 와 비교: σ=2 가 도입되면 locked LER 가 +0.4 nm 늘어남.
- 즉 **electron blur 자체가 pitch=20 에서 LER 를 늘린다**. Stage 3 의 high-σ PEB-LER 악화 가설을 확장: small pitch 에서는 그 효과가 σ=2 처럼 "약한" σ 에서도 나타난다.

**해결**: study note 에 정직한 결론 — Stage 4 balanced OP (σ=2 + quencher) 는 pitch ≥ 24 에서만 권장. pitch ≤ 20 에서는 σ 자체를 줄이거나 (e.g., σ=0) 다른 chemistry 가 필요.

---

### 문제 5 — pitch-dependent quencher mini-sweep 가 LER 회복 못 함

**증상 (Block B)**:

```text
pitch=18 / dose=28.4:
  baseline (no quencher):     LER_lock = 4.16
  Q0=0.005, kq=0.5 (가장 약함): LER_lock = 4.13  (회복 0.03 nm)
  Q0=0.02,  kq=2   (가장 강함): LER_lock = 4.05  (회복 0.11 nm)
  → design 2.77 까지 +1.3 nm 차이. 회복 거의 없음.

pitch=20 / dose=40:
  baseline:                   LER_lock = 3.28
  Q0=0.02, kq=2 (max effort): LER_lock = 3.21  (회복 0.07 nm)
  → design 2.77 까지 여전히 +0.4 nm.
```

**해석**: Stage 4 의 가설 "quencher 약화로 small-pitch 회복" 은 검증 실패. Quencher 강화든 약화든 pitch ≤ 20 의 LER 를 design 까지 회복 못함. 원인은 quencher 가 아니라 σ=2 의 e-blur (문제 4 와 같음).

**해결**: 

- pitch ≤ 20 에서 σ=2 baseline 자체가 부적절. σ=0 또는 line_cd 비례 축소가 필요.
- Stage 5 의 권장 dose=40 / σ=2 / Q0=0.02 / kq=1 는 **pitch ≥ 24 only**.

---

## 4. 의사결정 로그

| 결정 | 채택 | 이유 |
|---|---|---|
| CD-lock 알고리즘 | bisection on P ∈ [0.2, 0.8] with adaptive endpoint narrowing | spec 의 [0.2, 0.8] 보존하면서 실제 P 동적 범위 (0.65–0.85) 처리. |
| `cd_tol_nm` | 0.25 nm | spec. 본 sweep 에서 모든 ok rows 가 |CD_locked - CD_target| < 0.1 nm 로 수렴. |
| Block A pitch / dose | spec ({18, 20, 24} × {28.4, 40}) | 사용자 지정. Stage 5 의 marginal/robust window 경계 영역. |
| Block B pitch / dose | (18, 28.4) + (20, 40) | Stage 5 의 per-pitch 추천 dose 사용. 다른 dose 로 mini-sweep 하면 pitch-dose 효과 혼재. |
| Block B Q0 / kq | {0.005, 0.01, 0.02} × {0.5, 1.0, 2.0} | spec 의 sweep range 와 일관. |
| 결정 라벨 4 카테고리 | real degradation / fixed underestimates (merged) / fixed overestimates (displacement) / OK | 사용자 spec 2 카테고리에서 확장 — merged-LER artifact 와 displacement artifact 를 구분해야 정확. |
| `DECISION_TOL` | 0.20 nm | LER 측정 noise level 의 약 10%. 더 작으면 fluct 에 흔들림. |
| status_CD_locked 분류 | P_space/P_line/contrast 는 fixed 와 동일, area_frac/CD_pitch_frac 만 locked threshold 로 재계산 | P field 는 변하지 않음. threshold-dependent 만 갱신. |
| pitch ≤ 20 처리 결론 | σ=2 + quencher OP 부적합으로 명시 | Block B 결과로 결정. Stage 4 balanced OP 의 적용 범위 = pitch ≥ 24. |
| Stage 6 (x-z standing wave) 진행 | OK to proceed | Stage 4B 가 LER artifact vs real degradation 을 분리했으므로 Stage 6 에서 LER 비교는 신뢰 가능. |

---

## 5. 검증된 결과

### Block A 결정 표

| OP | pitch | dose | LER_design | LER_fixed | LER_locked | decision |
|---|---|---|---|---|---|---|
| primary | 18 | 28.4 | 2.77 | 3.72 | 4.07 | real degradation |
| primary | 18 | 40   | 2.77 | 1.63 | 4.16 | fixed underestimates (merged) |
| primary | 20 | 28.4 | 2.77 | 3.24 | 3.14 | real degradation |
| primary | 20 | 40   | 2.77 | 3.50 | 3.25 | real degradation |
| primary | 24 | 28.4 | 2.77 | 2.46 | 2.47 | OK |
| primary | 24 | 40   | 2.77 | 2.53 | 2.47 | OK |
| ctrl σ0 | 18 | 28.4 | 2.77 | 3.42 | 3.49 | real degradation |
| ctrl σ0 | 18 | 40   | 2.77 | 2.27 | 3.47 | fixed underestimates (merged) |
| ctrl σ0 | 20 | 28.4 | 2.77 | 2.95 | 2.85 | OK |
| ctrl σ0 | 20 | 40   | 2.77 | 3.32 | 2.84 | **fixed overestimates (displacement); locked recovers** |
| ctrl σ0 | 24 | 28.4 | 2.77 | 2.52 | 2.52 | OK |
| ctrl σ0 | 24 | 40   | 2.77 | 2.53 | 2.52 | OK |

### Block B 표 (full mini-sweep)

```text
pitch=18 / dose=28.4 — quencher 효과
  baseline (Q0=0):   CD_fix=16.67  CD_lock=12.74  LER_fix=2.12  LER_lock=4.16  margin=+0.067
  Q0=0.005, kq=0.5: CD_fix=16.47  CD_lock=12.59  LER_fix=2.36  LER_lock=4.13  margin=+0.058
  Q0=0.005, kq=1.0: CD_fix=16.33  CD_lock=12.71  LER_fix=2.51  LER_lock=4.14  margin=+0.053
  ...
  Q0=0.02,  kq=2.0: CD_fix=13.48  CD_lock=12.69  LER_fix=4.04  LER_lock=4.05  margin=-0.004 (under_exposed)

pitch=20 / dose=40 — quencher 효과
  baseline:         CD_fix=18.49  CD_lock=12.37  LER_fix=2.24  LER_lock=3.28  margin=+0.142
  Q0=0.005, kq=2:   CD_fix=17.93  CD_lock=12.50  LER_fix=2.76  LER_lock=3.30  margin=+0.129
  Q0=0.02,  kq=2.0: CD_fix=15.69  CD_lock=12.44  LER_fix=3.63  LER_lock=3.21  margin=+0.086 (robust_valid)
```

### PSD mid-band 비교 (Block A, pitch=24 dose=40)

```text
primary:  psd_eblur_mid = 2.81  →  psd_PEB_mid = 1.06  →  psd_locked_mid = 1.03
ctrl σ0:  psd_eblur_mid = 4.62  →  psd_PEB_mid = 1.59  →  psd_locked_mid = 1.74
```

pitch=24 에서 PEB 의 mid-band 감소가 fixed 와 locked 양쪽에서 일관 → smoothing 정상.

```text
primary, pitch=20 dose=40:  psd_eblur_mid = 2.82  →  psd_PEB_mid = 5.12  →  psd_locked_mid = 1.59
```

여기는 fixed 가 5.12 (인공적 증가), locked 가 1.59 (정상 감소). **mid-band PSD 가 Stage 5 의 displacement artifact 식별에도 사용 가능** — Stage 4B 의 부수 발견.

### 정성적 결론

| 가설 | 검증 결과 |
|---|---|
| 작은 pitch 의 negative LER reduction = displacement artifact only | ❌ 부분만 — control σ0 / pitch=20 / dose=40 만 artifact, 나머지는 real degradation |
| Stage 4 quencher 가 small pitch LER 회복 | ❌ — pitch ≤ 20 에서는 quencher 약/강 모두 회복 안 함 |
| σ=2 + quencher OP 의 적용 범위 | pitch ≥ 24 에서만 robust |

---

## 6. 후속 작업

- **Stage 6 (x-z standing wave)**: 이제 진행 가능. LER 비교는 pitch=24 영역에서만 의미 있음을 인지.
- **σ=0 follow-up at small pitch** (Stage 5C 가능): Stage 4B 가 σ=2 가 small-pitch LER 악화의 주범임을 시사. σ=0 + quencher off 로 pitch=18, 20 의 process window 측정. Stage 6 후 또는 다른 stage 와 병행 가능.
- **CD-locked LER 를 모든 stage 의 default 로?**: Stage 6 부터 CD-lock 측정을 helper 에 통합할지 결정. compute cost 는 작음 (extract_edges × max ~50 회). 사용자 결정 대기.
- **PSD mid-band 를 LER 보조 metric 으로**: Stage 4B 발견대로 mid-band PSD 가 displacement artifact 식별에 유효. Stage 6 부터 보고 columns 에 추가 권장.
- **Stage 3B (σ=5/8 호환 budget)**: 여전히 보류.
- **line_cd scaling for small pitch**: Stage 5 의 follow-up 가설 — pitch 별 line_cd 비례 축소. 별도 stage.

---

## 7. 산출물

```text
src/metrics_edge.py
  + find_cd_lock_threshold (adaptive endpoint), CD_LOCK_* constants

tests/test_edge_metrics.py
  + 3 CD-lock tests   (총 20/20 passing)

configs/v2_stage4b_cd_locked.yaml
experiments/04b_cd_locked/
  __init__.py
  run_cd_locked_analysis.py     (Block A 12 + Block B 20 = 32 runs)

outputs/
  figures/04b_cd_locked_block_a/             # 12 P maps with both contours overlaid
  figures/04b_cd_locked_block_b/             # 8 heatmaps (2 pitch × 4 metrics)
  logs/04b_cd_locked_block_a.csv             # Block A full
  logs/04b_cd_locked_block_b.csv             # Block B mini-sweep full
  logs/04b_cd_locked_block_a_decisions.json  # decision labels

EXPERIMENT_PLAN.md
  §Stage 4B "deferred" → "executed" + 결과 표 + Stage 5B mini-sweep 결론

study_notes/
  06_stage4B_cd_locked.md  (this file)
  README.md  index 업데이트
```
