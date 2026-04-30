# PEB v2 calibration

v2 first-pass closeout 후 외부 reference / 측정 데이터 와의 calibration 단계.
chemistry / sweep 추가 전에 v2 가 합리적인 절대값을 만들어내는지 검증하고, 발견된 offset 을 분류해 적절한 parameter (Hmax, kdep, DH, σ, abs_len, dose) 를 보정한다.

## 목적 (Phase 1)

```text
1. v2 권장 OP (pitch=24, dose=40, σ=2, t=30, DH=0.5, kdep=0.5, Hmax=0.2,
   Q0=0.02, kq=1.0) 가 다음을 만족하는가?
     CD ≈ 15 nm
     LER ≈ 2.5 ~ 2.7 nm
2. Stage 5 의 process window shape 가 reference 와 일치하는가?
3. offset 이 있다면 어느 메커니즘에서 발생하는가?
   - acid generation (Hmax)
   - reaction rate (kdep)
   - acid diffusion (DH)
   - electron blur (σ)
   - z absorption (abs_len)
```

## 구조

```text
calibration/
  README.md                         # 이 파일
  calibration_targets.yaml          # CD / LER / process-window 목표치 + tolerance + 출처
  calibration_plan.md               # Phase 1-4 plan + 결정 트리
  configs/
    cal01_hmax_kdep_dh.yaml         # Phase 1 base config
  experiments/
    cal01_hmax_kdep_dh/
      run_cal01.py                  # Phase 1 sweep runner

  # outputs 는 v2 의 outputs/ 아래 cal01_* prefix 로 저장
  outputs/figures/cal01_*           (in: reaction_diffusion_peb_v2_high_na/outputs/figures/)
  outputs/logs/cal01_*              (in: reaction_diffusion_peb_v2_high_na/outputs/logs/)
```

## 실행

```bash
python -m reaction_diffusion_peb_v2_high_na.calibration.experiments.cal01_hmax_kdep_dh.run_cal01 \
    --config reaction_diffusion_peb_v2_high_na/calibration/configs/cal01_hmax_kdep_dh.yaml
```

## Phase 별 게이트

```text
Phase 1 (Hmax × kdep × DH):
  통과 조건 = "best score < 0.1" 인 cell 이 1개 이상
  실패 시 → Phase 2A (dose / σ / abs_len 확장)

Phase 2 (process window 재검증):
  Phase 1 의 best cell 로 Stage 5 의 pitch × dose grid 재실행
  통과 조건 = pitch=20-32 의 robust_valid 영역이 일치 또는 더 넓어짐

Phase 3 (외부 reference 비교):
  Phase 2 까지의 OP 와 published / measured CD / LER / process-window 비교
  통과 조건 = systematic offset < tolerance

Phase 4 (deferred stages 진행):
  Stage 3B, 5C, 6B 또는 새 chemistry. Phase 3 통과 후에만 시작.
```

## 산출물

각 Phase 의 결과는 `calibration_plan.md` 에 누적 기록 (one-document policy: phase 별 history + decision).
