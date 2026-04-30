# Phase 3 — Stage 3: electron blur 분리 + 측정 규약 재정의

## 0. 한 줄 요약

LER 측정을 design / e-blur / PEB 세 단계로 분리하고, σ ∈ {0,1,2,3} 을 두 Stage-2 OP 에서 돌렸다. **robust OP (DH=0.5, t=30)** 가 Stage-3 강화 게이트를 4개 σ 전부 통과, **algorithmic-best OP (DH=0.8, t=20)** 는 4개 σ 전부 실패 (P_line_margin < 0.03). σ 가 커질수록 electron blur 의 LER 감소는 단조 증가 (+0% → +11%) 하지만 PEB 의 LER 감소는 단조 감소하다 음으로 (+8.7% → −37.7%) 떨어져 total 도 σ=0 에서 최대 (+8.7%). σ=5/8 는 plan §Stage 3B 로 분리.

---

## 1. 목적

- Plan §5 Stage 3: electron blur smoothing 효과를 PEB acid diffusion smoothing 효과와 정량적으로 분리.
- 두 가지 정량 질문에 답한다.
  - electron blur σ 가 커지면 H0 단계에서 LER 가 얼마나 줄어드는가?
  - 그 위에 PEB 가 얹히면 추가 LER 감소가 얼마인가?
- 동시에 Stage 1/2 에서 발견한 measurement convention 의 σ-의존성 문제를 정리.

---

## 2. 진행 단계

1. **plan revision** (option b): plan §5 Stage 3 sweep 을 `[0,2,5,8]` → `[0,1,2,3]` 으로 축소. σ=5,8 은 24 nm pitch / kdep=0.5 / Hmax≤0.2 spec 과 호환 budget 없음을 Stage 1A 에서 확인했기 때문. 호환 budget 탐색은 plan §Stage 3B (future) 로 분리.
2. **measurement convention 재정의**: `run_sigma_sweep_helpers.py` 의 `run_one_with_overrides` 에 세 단계 LER 측정 추가.
   - `LER_design_initial_nm`     ← `extract_edges(I, threshold=0.5)`      (σ 독립)
   - `LER_after_eblur_H0_nm`     ← `extract_edges(I_blurred, 0.5)`
   - `LER_after_PEB_P_nm`        ← `extract_edges(P, 0.5)`
   - 세 reduction percentage 도 함께 출력.
3. **gate 강화**: `P_line_margin = P_line_center_mean − 0.65` 항상 출력. Stage 3 sweep script 는 기존 interior gate 에 더해 `P_line_margin >= 0.03` 을 요구 (sweep script 안에서만 적용, helper 의 `passed` 는 그대로).
4. **sweep**: 두 OP × σ 4 점 = 8 runs.
   - robust OP             : DH=0.5, t=30
   - algorithmic-best OP   : DH=0.8, t=20
5. **결과 분석 + figures + CSV/JSON 저장**.

---

## 3. 발생한 문제와 해결 방법

### 문제 1 — `_stage3_passed` 컬럼명 / `passed` 컬럼명 불일치 → CSV writer 오류

**증상**: 첫 실행에서

```text
ValueError: dict contains fields not in fieldnames: 'passed'
```

**원인**: CSV `fieldnames` 에 `_stage3_passed` 를 넣었지만 row dict 에서 `pop("_stage3_passed")` 후 `row["passed"]` 로 다시 넣어서 키 이름이 어긋남.

**해결**: `src_keys` (read 용) 와 `out_keys` (write 용 = `src_keys + ["passed", "fail_reason"]`) 를 분리하고, row 빌드 시 명시적으로 `row["passed"] = r["_stage3_passed"]` 로 대입. JSON dump 도 같은 방식으로 정리.

**교훈**: dict→csv 매핑은 build 시점에 바로 final-name 으로 채우는 게 안전. pop+rename 은 fieldnames 와 동기화해야 함.

---

### 문제 2 — algorithmic-best OP 가 σ=0 에서 이미 P_line_margin 게이트 fail

**증상**: DH=0.8, t=20 의 σ=0 에서 `P_line=0.6534, margin=0.003 < 0.03` → fail. σ=1,2,3 모두 fail.

**원인**: Stage 2 의 algorithmic best 자체가 P_line 게이트 boundary (0.65) 에 0.003 margin 으로 안착했기 때문 (Stage 2 선정 criterion 에 margin 조건이 없었음). σ 가 커지면 line center 평균이 약간씩 더 떨어지는 경향이 있어 margin 이 더 줄거나 음으로 됨 (σ=3 에서 P_line=0.641 < 0.65).

**해결**: 이는 Stage-3 gate 강화의 의도된 결과. 사용자가 명시한 `P_line_margin >= 0.03` 조건이 Stage-2 boundary OP 를 정확히 걸러낸다.

**의사결정**: algorithmic-best OP 는 Stage 3 의 σ 스윕에 부적합하다고 결론. 이후 stage 부터 Stage 2 result 를 인용할 때는 robust OP 를 default 로 사용한다.

---

### 문제 3 — σ 가 커지면 PEB 의 LER 감소가 음수로 떨어진다

**증상**: robust OP 에서

```text
σ=0: PEB_LER_reduction = +8.7%
σ=1: PEB_LER_reduction = +5.7%
σ=2: PEB_LER_reduction = -2.7%
σ=3: PEB_LER_reduction = -37.7%
```

PEB 가 *LER 를 늘리는* 결과. plan §8 의 정상 경향 ("PEB time 증가 → LER 감소") 과 어긋남.

**원인 분석**:

- electron blur 가 커지면 `I_blurred` edge 가 매끈해지고 `LER_after_eblur_H0` 가 작게 잡힘 (σ=0:2.77 → σ=3:2.47).
- PEB 후에는 acid 가 line 밖으로 더 퍼져 line 이 widening 되며 contour 위치가 design edge 로부터 더 멀어진다 (σ=3 에서 CD_shift=+5.85, CD/p=0.76).
- 새 contour 위치는 acid 가 모인 ridge 가 아니라, 광범위 diffusion 의 외곽이라 local 노이즈가 상대적으로 크다.
- 그 결과 PEB 후 contour 의 std (= LER) 가 e-blur 후 보다 커진다.

즉 σ 가 큰 영역에서는 PEB 가 "acid diffusion" 보다는 "line widening" 을 주로 하고 있어 LER 측정 위치가 바뀐 것.

**해결**: 데이터 자체는 정상. 정성적 해석을 study note 에 명시하고, total LER reduction (design → PEB) 을 비교의 reference 로 사용. total 은 σ=0 에서 최대 (+8.7%).

**교훈**: LER 절대값 비교는 contour 가 같은 위치에 있을 때만 의미가 있다. σ 가 다르면 contour 위치 (CD) 가 다르므로 `LER_after_PEB_P` 단독 비교는 misleading. 다음 stage 부터는 같은 CD 에서 비교하거나 PSD-domain 으로 분석하는 것이 옳다.

---

### 부수 문제 — σ=2,3 에서 `electron_blur_LER_reduction_pct` 가 σ=1 대비 비선형으로 큼

**증상**:

```text
σ=1: e-blur reduction = +2.2%
σ=2: e-blur reduction = +6.1%
σ=3: e-blur reduction = +11.1%
```

monotonic 증가는 맞지만 σ 와 LER 감소가 정비례하지 않음.

**원인**: edge roughness 의 PSD 가 σ 의 Gaussian 필터에 다르게 반응. 짧은 correlation length (5 nm) noise 는 σ ≥ 5 nm 에서 거의 다 사라지지만 σ=1,2 에서는 일부만 감쇠. PSD 분석을 하면 명확해질 것.

**향후 작업**: plan §6.4 의 edge PSD metric 구현 후 σ-dependent 감쇠 곡선 그리기.

---

## 4. 의사결정 로그

| 결정 | 채택 | 이유 |
|---|---|---|
| Stage 3 σ 범위 | [0,2,5,8] → [0,1,2,3] | σ=5,8 호환 budget 없음 (Stage 1A). σ=1 추가로 0–3 범위 더 조밀하게. |
| σ=5/8 처리 | plan §Stage 3B (future / optional) 로 분리 | dose, kdep, Hmax 확장 search 가 필요. 지금 시작하면 24 runs+. trigger 조건 명시 후 보류. |
| measurement convention | 3-stage LER 측정 (design / e-blur / PEB) | σ 의존성 제거. 3-stage 분리 표기로 효과 분해 가능. |
| `LER_design_initial` 의 threshold | binary I @ 0.5 | binary 에서는 0.5 가 자연스러운 edge. σ 와 무관. |
| Stage-3 gate strengthening 적용 범위 | sweep script 안에서만 (helper 의 `passed` 는 변경 X) | Stage 1/2 backward compatibility. helper 는 reusable component 로 유지. |
| algorithmic-best OP 의 처리 | Stage 3 부적합으로 결론, 이후 stage 부터 robust OP 를 default | 모든 σ 에서 P_line_margin fail. 향후 stage 의 OP 로 쓰지 않음. |
| total_LER_reduction 의 reference baseline | `LER_design_initial` (σ-독립) | σ 별로 비교 가능한 유일한 base. |

---

## 5. 검증된 결과

### Robust OP (DH=0.5, t=30) — 모든 σ Stage-3 gate 통과

| σ (nm) | P_space | P_line | margin | CD_shift | LER_design | LER_eblur | LER_PEB | e-blur% | PEB% | total% | passed |
|--------|---------|--------|--------|----------|------------|-----------|---------|---------|------|--------|--------|
| 0      | 0.210   | 0.792  | 0.142  | +1.79    | 2.772      | 2.772     | 2.531   | +0.0    | +8.7 | **+8.7** | ✓ |
| 1      | 0.245   | 0.794  | 0.144  | +2.62    | 2.772      | 2.711     | 2.556   | +2.2    | +5.7 | +7.8     | ✓ |
| 2      | 0.311   | 0.790  | 0.140  | +3.83    | 2.772      | 2.604     | 2.673   | +6.1    | -2.7 | +3.6     | ✓ |
| 3      | 0.398   | 0.780  | 0.130  | +5.85    | 2.772      | 2.465     | 3.396   | +11.1   | -37.7 | -22.5   | ✓ |

### Algorithmic-best OP (DH=0.8, t=20) — 모든 σ fail (Stage-3 strengthened gate)

| σ (nm) | P_line | margin | passed | reason |
|--------|--------|--------|--------|--------|
| 0      | 0.653  | 0.003  | ✗      | margin < 0.03 |
| 1      | 0.656  | 0.006  | ✗      | margin < 0.03 |
| 2      | 0.651  | 0.001  | ✗      | margin < 0.03 |
| 3      | 0.641  | -0.009 | ✗      | P_line < 0.65 |

### 정성적 경향 (plan §8 와의 비교)

| plan §8 expected | observed (robust OP) |
|---|---|
| σ 증가 → I_blurred edge 더 smooth | yes — `LER_after_eblur_H0` 단조 감소 (2.77 → 2.47) |
| σ 증가 → PEB 후 추가 smoothing | **no** — `LER_after_PEB_P` 가 σ ≥ 2 부터 증가. 이유는 "PEB 가 line widening 위주로 작용하고 contour 가 design edge 에서 멀어짐". |
| total LER 은 σ=0 에서 최대 | yes — total +8.7% 이 가장 큼 |
| σ 증가 → CD shift 증가 | yes — +1.79 → +5.85 |

### 핵심 finding

**σ=0 의 PEB-only smoothing 이 σ=3 의 e-blur+PEB 합계보다 효과 더 크다.** 즉 24 nm pitch 와 같은 좁은 pitch 에서는 electron blur 가 클수록 PEB 가 LER 를 추가로 줄이기 어렵다. 이는 e-blur 와 PEB 가 *서로 보완하지 않고* 일부 *경쟁* 한다는 것을 시사.

---

## 6. 후속 작업

- **Stage 4 (weak quencher)**: robust OP (DH=0.5, t=30, σ=0) 를 default 로 시작. quencher 가 acid tail 을 줄여 σ-스윕에서 PEB 의 LER 증가 문제를 완화할 수 있는지 확인.
- **edge PSD** (plan §6.4): σ 별 attenuation curve 을 frequency 영역에서 그려, σ 가 어떤 spatial frequency 의 noise 를 제거하는지 정량.
- **Stage 3B** (optional): trigger 조건이 만족되면 `(dose × kdep × Hmax × σ ∈ {5,8} × t × DH)` factorial sampling. 지금 당장은 보류.
- **CD-locked LER 비교** (Stage 4 또는 Stage 5 에서): σ 별로 contour 가 다른 CD 위치에 있는 문제 해결을 위해 `P_threshold` 를 CD-equalize 하도록 자동 조정하는 옵션 도입.
- **algorithmic-best OP 의 처리**: 이후 stage 인용 시 robust OP 를 default 로. algorithmic-best 는 "P_line margin 부족" 이라는 lesson 으로만 인용.

---

## 7. 산출물

```text
configs/v2_stage3_electron_blur.yaml
experiments/03_electron_blur/
  __init__.py
  run_eblur_sweep.py
experiments/run_sigma_sweep_helpers.py     # 3-stage LER 측정 추가
outputs/
  figures/03_electron_blur/                # 8 P maps + 8 contour overlays
  logs/03_electron_blur.csv                # full metric rows
  logs/03_electron_blur_summary.csv        # core columns + reasons
  logs/03_electron_blur_summary.json       # JSON twin
study_notes/03_stage3_electron_blur.md     # 본 노트
EXPERIMENT_PLAN.md
  §5 Stage 3 sweep [0,2,5,8] → [0,1,2,3]
  §5 Stage 3 measurement convention 재정의 (3-stage LER)
  §5 Stage 3 gate 강화 (P_line_margin >= 0.03)
  §5 Stage 3B (future) 신설
```
