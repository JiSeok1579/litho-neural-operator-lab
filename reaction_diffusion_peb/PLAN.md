# Reaction-Diffusion / PEB 국소 모듈 최종 구축 지시안

## 0. 문서 목적

이 문서는 기존 GitHub 프로젝트 `litho-neural-operator-lab`의 전체 방향은 그대로 유지하면서, 그 안에 **Reaction-Diffusion / PEB(Post-Exposure Bake)** 부분만 독립적으로 실험할 수 있는 국소 모듈을 구축하기 위한 최종 지시안이다.

이번 최종안의 핵심 수정점은 다음이다.

```text
데이터가 아직 없는 상태에서는 DeepONet/FNO보다 PINN을 먼저 사용한다.
단, PINN만 단독으로 쓰지 않고 FD/FFT physics solver와 함께 비교한다.
FD/FFT/PINN 결과가 쌓인 뒤 DeepONet/FNO를 operator surrogate로 후속 도입한다.
```

즉 전체 순서는 다음과 같다.

```text
Physics equation
  ↓
FD / FFT baseline solver
  ↓
PINN comparison
  ↓
Simulation dataset accumulation
  ↓
DeepONet / FNO surrogate
```

---

## 1. 핵심 원칙

```text
1. 기존 프로젝트는 건드리지 않는다.
2. reaction_diffusion_peb/ 폴더를 새로 만들어 국소적으로 운용한다.
3. Mask / optics / inverse design / FNO 코드는 초기에는 섞지 않는다.
4. 초기 입력은 synthetic aerial image를 사용한다.
5. Diffusion-only부터 시작한다.
6. FD/FFT baseline을 먼저 또는 동시에 만든다.
7. PINN은 DeepONet/FNO보다 먼저 사용한다.
8. DeepONet/FNO는 데이터셋 생성 이후 후속 surrogate로 둔다.
9. stiff reaction인 kq는 작은 값부터 시작한다.
10. 모든 파라미터는 단계별로 활성화한다.
```

---

## 2. 기존 전체 프로젝트와의 관계

기존 전체 프로젝트의 큰 흐름은 다음과 같다.

```text
Mask
  ↓
Diffraction spectrum
  ↓
Pupil / NA filtering
  ↓
Aerial image
  ↓
Inverse aerial optimization
  ↓
Exposure / acid generation
  ↓
PEB diffusion
  ↓
Reaction-diffusion
  ↓
Resist threshold / contour
  ↓
PINN / FNO / DeepONet experiments
```

이번 국소 모듈은 아래 부분만 다룬다.

```text
Aerial image or synthetic exposure map
  ↓
Exposure / acid generation
  ↓
PEB diffusion
  ↓
Acid loss
  ↓
Quencher neutralization
  ↓
Deprotection reaction
  ↓
Temperature-dependent reaction rate
  ↓
Soft threshold / latent resist image
```

즉, 이번 작업은 **mask optics가 아니라 resist 내부 PEB 물리**에 집중한다.

---

## 3. 새로 만들 폴더

기존 프로젝트 루트 아래에 다음 폴더를 새로 만든다.

```text
litho-neural-operator-lab/
  reaction_diffusion_peb/
```

권장 구조:

```text
litho-neural-operator-lab/
  README.md
  docs/
  src/
  experiments/
  configs/

  reaction_diffusion_peb/
    README.md
    PLAN.md
    PARAMETER_SCOPE.md
    requirements.txt

    configs/
      minimal_diffusion.yaml
      acid_loss.yaml
      quencher_reaction.yaml
      deprotection.yaml
      temperature_peb.yaml
      full_reaction_diffusion.yaml
      pinn_diffusion.yaml
      pinn_reaction_diffusion.yaml
      parameter_sweep.yaml

    src/
      synthetic_aerial.py
      exposure.py
      diffusion_fd.py
      diffusion_fft.py
      reaction_diffusion.py
      deprotection.py
      arrhenius.py
      threshold.py
      metrics.py
      visualization.py
      io.py

      pinn_base.py
      pinn_diffusion.py
      pinn_reaction_diffusion.py

      dataset_builder.py

    experiments/
      01_synthetic_aerial/
        run_gaussian_spot.py
        run_line_space.py

      02_diffusion_baseline/
        run_diffusion_fd.py
        run_diffusion_fft.py
        compare_fd_fft.py

      03_pinn_diffusion/
        run_pinn_diffusion.py
        compare_fd_fft_pinn.py

      04_acid_loss/
        run_acid_loss_fd.py
        run_acid_loss_pinn.py
        compare_fd_pinn.py

      05_deprotection/
        run_deprotection_fd.py
        run_deprotection_pinn.py

      06_temperature_peb/
        run_temperature_sweep.py
        run_time_sweep.py

      07_quencher_reaction/
        run_quencher_reaction_safe.py
        run_quencher_reaction_stiff.py

      08_full_reaction_diffusion/
        run_full_model.py

      09_dataset_generation/
        generate_fd_dataset.py
        generate_pinn_dataset.py
        validate_dataset.py

      10_operator_learning_optional/
        train_deeponet.py
        train_fno.py
        evaluate_operator_surrogate.py

      11_advanced_stochastic/
        run_temperature_uniformity.py
        run_molecular_blur.py
        run_ensemble.py

    outputs/
      figures/
      logs/
      arrays/
      metrics/
      datasets/
      checkpoints/
```

---

## 4. 기존 코드와 분리하는 원칙

초기에는 `reaction_diffusion_peb/` 내부 코드가 자기 폴더 안에서 완결되도록 한다.

초기에는 아래 모듈을 import하지 않는다.

```text
root src/
mask/
optics/
inverse/
neural_operator/
closed_loop/
```

초기 입력은 synthetic aerial image를 사용한다.

나중에 전체 프로젝트와 연결할 때는 파일 기반 인터페이스를 사용한다.

```text
main optics module
  outputs/aerial_image.npy
      ↓
reaction_diffusion_peb input
      ↓
PEB / reaction-diffusion simulation
      ↓
latent resist image
```

권장 입력/출력:

```text
input:
  aerial_image.npy
  metadata.json

output:
  acid_after_peb.npy
  quencher_after_peb.npy
  deprotected_fraction.npy
  resist_latent.npy
  metrics.csv
```

---

## 5. 왜 PINN을 DeepONet/FNO보다 먼저 쓰는가?

현재 상태는 다음과 같다.

```text
데이터는 아직 없음
물리 PDE는 있음
파라미터 범위는 있음
```

이때 DeepONet/FNO는 바로 쓰기 어렵다.  
DeepONet/FNO는 operator learning 모델이므로 다음과 같은 데이터셋이 필요하다.

```text
input:
  I(x,y), H0(x,y), DH0, kdep, kq, kloss, T, PEB time

output:
  H(x,y,t_final), Q(x,y,t_final), P(x,y,t_final), R(x,y)
```

아직 이런 입력-출력 쌍이 없으므로, 먼저 physics solver와 PINN으로 데이터를 만들고 검증해야 한다.

PINN은 다음 이유로 먼저 사용할 수 있다.

```text
1. PDE residual을 loss로 사용한다.
2. 데이터가 적어도 initial condition과 boundary condition으로 학습 가능하다.
3. diffusion / reaction-diffusion 물리 이해에 적합하다.
4. DeepONet/FNO용 데이터셋 생성 전 단계로 좋다.
```

하지만 PINN만 단독으로 쓰면 검증이 어렵다.  
따라서 다음 비교 구조가 필수다.

```text
FD solver
  vs
FFT heat-kernel solver
  vs
PINN solver
```

특히 diffusion-only 단계에서는 FFT heat-kernel solution이 좋은 기준 역할을 한다.

---

## 6. 전체 추천 진행 순서

기존 계획에서 PINN을 뒤쪽 optional로 두지 않고, 앞쪽 비교 모듈로 올린다.

최종 추천 순서:

```text
Phase 1:
  Synthetic aerial image 생성
  Exposure → initial acid H0 생성

Phase 2:
  Diffusion-only FD solver
  Diffusion-only FFT solver
  FD/FFT 비교

Phase 3:
  PINN diffusion solver
  FD/FFT/PINN 비교

Phase 4:
  Acid loss 추가
  FD/PINN 비교

Phase 5:
  Deprotection 추가
  H → P 모델 구현
  FD/PINN 비교 가능

Phase 6:
  Temperature / Arrhenius 추가

Phase 7:
  Quencher reaction 추가
  작은 kq부터 시작
  stiff handling 필요 시 별도 구현

Phase 8:
  Full reaction-diffusion 통합

Phase 9:
  FD/PINN simulation 결과를 dataset으로 저장

Phase 10:
  DeepONet/FNO operator surrogate 학습

Phase 11:
  Advanced stochastic / Petersen / z-axis 확장
```

---

## 7. 입력과 출력

## 7.1 입력

초기에는 synthetic aerial image를 사용한다.

입력 후보:

```text
1. Gaussian spot
2. Line-space sinusoidal exposure
3. Contact-hole-like 2D Gaussian array
4. Two-spot interference-like exposure
5. User-provided .npy aerial image
```

입력 변수:

```text
I(x, y): normalized aerial / exposure intensity map
dose: normalized exposure dose
grid_spacing_nm: spatial grid size
```

## 7.2 출력

각 실험은 다음을 출력한다.

```text
H0(x,y): initial acid concentration
H(x,y,t): acid concentration after PEB
Q(x,y,t): quencher concentration, if enabled
P(x,y,t): deprotected fraction, if enabled
R(x,y): soft-thresholded resist latent image
metrics.csv: concentration / contour / error metrics
figures: before-after visualization
```

PINN 비교 실험은 추가로 다음을 출력한다.

```text
pinn_loss_curve.csv
pde_residual_map.npy
fd_vs_pinn_error.npy
fd_fft_pinn_comparison.png
```

Dataset generation 단계에서는 다음을 출력한다.

```text
outputs/datasets/
  diffusion_dataset.npz
  acid_loss_dataset.npz
  reaction_diffusion_dataset.npz
  metadata.json
```

---

## 8. 변수 정의

| Symbol | Code name | Meaning |
|---|---|---|
| `I(x,y)` | `I` | normalized aerial / exposure intensity |
| `H(x,y,t)` | `H` | acid concentration |
| `H0(x,y)` | `H0` | initial acid concentration after exposure |
| `Q(x,y,t)` | `Q` | quencher concentration |
| `P(x,y,t)` | `P` | deprotected fraction |
| `R(x,y)` | `R` | soft threshold resist latent image |
| `D_H` | `DH` | acid diffusion coefficient |
| `D_Q` | `DQ` | quencher diffusion coefficient |
| `k_q` | `kq` | acid-quencher neutralization rate |
| `k_loss` | `kloss` | acid loss rate |
| `k_dep` | `kdep` | deprotection rate |
| `T` | `temperature_c` | PEB temperature in Celsius |
| `t_PEB` | `peb_time_s` | PEB time |
| `Lz` | `film_thickness_nm` | resist film thickness |
| `Ea` | `activation_energy_kj_mol` | activation energy |

주의:

```text
D라는 한 글자는 사용하지 않는다.
Dose와 diffusion coefficient가 헷갈리기 때문이다.

Dose는 dose.
Acid diffusion coefficient는 DH.
Quencher diffusion coefficient는 DQ.
Deprotected fraction은 P.
```

---

## 9. 대상 파라미터 범위

| 분류 | 파라미터 | 정량적 범위 / 수치 | 단위 | 역할 |
|---|---|---:|---|---|
| 공정 조건 | PEB 온도 `temperature_c` | 80 ~ 120, MOR: 125 | °C | Arrhenius 계수로 reaction rate 보정 |
| 공정 조건 | PEB 시간 `peb_time_s` | 60 ~ 90 | s | PEB simulation time, PINN time domain upper bound |
| 공정 조건 | 온도 균일도 `temperature_uniformity_c` | ±0.02 | °C | stochastic / CDU proxy |
| 확산 모델 | 초기 acid 확산 계수 `DH0_nm2_s` | 0.3 ~ 1.5 | nm²/s | acid diffusion strength |
| 확산 모델 | Petersen 확산 가속 계수 `petersen_alpha` | 0.5 ~ 3.0 | - | nonlinear diffusion modulation |
| 확산 모델 | Quencher 확산 `DQ_nm2_s` | ≤ 0.1 × DH0 | nm²/s | quencher diffusion |
| 반응 속도 | 탈보호 속도 `kdep_s_inv` | 0.01 ~ 0.5 | s⁻¹ | deprotected fraction 생성 |
| 반응 속도 | 중화 속도 `kq_s_inv` | 100 ~ 1000 | s⁻¹ | acid-quencher neutralization, stiff term |
| 반응 속도 | 산 손실 속도 `kloss_s_inv` | 0.001 ~ 0.05 | s⁻¹ | acid loss / trap / decay |
| 반응 속도 | 활성화 에너지 `activation_energy_kj_mol` | 약 100 | kJ/mol | temperature-dependent rate correction |
| 레지스트 물성 | 필름 두께 `film_thickness_nm` | < 30 | nm | optional z-axis domain |
| 레지스트 물성 | 분자/입자 크기 `particle_size_nm` | 0.5 MOR ~ 1.0 CAR | nm | molecular blur / resolution scale |
| 레지스트 물성 | 초기 acid 농도 `Hmax_mol_dm3` | 0.1 ~ 0.3 | mol/dm³ | exposure-linked acid generation |
| 수치 설정 | 격자 `grid_spacing_nm` | 0.5 ~ 1.0 | nm | spatial discretization |
| 수치 설정 | 앙상블 반복 `ensemble_runs` | ≥ 10 ~ 20 | runs | stochastic variation analysis |

---

## 10. 파라미터별 실험 가능 여부

| 파라미터 그룹 | 가능 여부 | 단계 |
|---|---:|---|
| PEB 온도 / 시간 | 가능 | Phase 2, 6 |
| 온도 균일도 | 후반 가능 | Phase 11 |
| DH0 acid diffusion | 가능 | Phase 2 |
| Petersen alpha | 후반 가능 | Phase 11 |
| Quencher diffusion DQ | 가능 | Phase 7 |
| kdep | 가능 | Phase 5 |
| kq | 가능하지만 주의 | Phase 7 |
| kloss | 가능 | Phase 4 |
| Ea | 가능 | Phase 6 |
| 필름 두께 Lz | 후반 가능 | Phase 11 / optional z-axis PINN |
| 분자/입자 크기 | 후반 가능 | Phase 11 |
| 초기 acid 농도 H0/Hmax | 가능 | Phase 1 |
| 격자 0.5~1.0 nm | 가능 | all PDE phases |
| ensemble 10~20회 | 후반 가능 | Phase 11 |

요약:

```text
바로 가능:
PEB time, DH0, H0/Hmax, grid, kloss, kdep, DQ, temperature, Ea

주의해서 가능:
kq = 100~1000 s^-1
→ stiff하므로 작은 값부터 시작

후반 확장:
Petersen alpha, temperature uniformity, molecular size, ensemble, Lz/z-axis

DeepONet/FNO:
FD/PINN 데이터셋 생성 후 가능
```

---

# Phase 1. Synthetic Aerial + Exposure

## 목적

Optics / mask simulation 없이 Reaction-diffusion / PEB만 공부하기 위해 synthetic exposure map을 만들고, 이를 초기 acid concentration으로 변환한다.

## 구현 파일

```text
reaction_diffusion_peb/src/synthetic_aerial.py
reaction_diffusion_peb/src/exposure.py

reaction_diffusion_peb/experiments/01_synthetic_aerial/run_gaussian_spot.py
reaction_diffusion_peb/experiments/01_synthetic_aerial/run_line_space.py
```

## synthetic aerial 함수

```python
def gaussian_spot(grid_size, sigma_px, center=None):
    ...

def line_space(grid_size, pitch_px, duty=0.5, contrast=1.0):
    ...

def contact_array(grid_size, pitch_px, sigma_px):
    ...

def normalize_intensity(I):
    ...
```

## Exposure 모델

```math
H_0(x,y)
=
H_{max}
\left(
1 - \exp(-\eta \cdot dose \cdot I(x,y))
\right)
```

## 구현 함수

```python
def acid_generation(I, dose=1.0, eta=1.0, Hmax=0.2):
    # I: normalized aerial intensity, range [0, 1]
    # dose: normalized exposure dose
    # eta: acid generation efficiency
    # Hmax: maximum acid concentration [mol/dm^3]
    ...
```

## 실험 파라미터

```yaml
exposure:
  dose: 1.0
  eta: 1.0
  Hmax_mol_dm3: 0.2
```

Sweep:

```yaml
sweep:
  Hmax_mol_dm3: [0.1, 0.2, 0.3]
  dose: [0.5, 1.0, 1.5, 2.0]
  eta: [0.5, 1.0, 2.0]
```

## 확인할 것

```text
I(x,y)가 0~1 범위로 normalize되는지
dose가 증가하면 H0가 증가하는지
H0가 Hmax를 넘지 않는지
I가 0이면 H0도 0에 가까운지
```

---

# Phase 2. Diffusion-only FD / FFT Baseline

## 목적

PEB 동안 acid가 확산되는 기본 물리를 FD와 FFT로 구현한다.

이 단계는 PINN 검증을 위한 baseline이다.

## PDE

```math
\frac{\partial H}{\partial t}
=
D_H \nabla^2 H
```

## 구현 파일

```text
reaction_diffusion_peb/src/diffusion_fd.py
reaction_diffusion_peb/src/diffusion_fft.py

reaction_diffusion_peb/experiments/02_diffusion_baseline/run_diffusion_fd.py
reaction_diffusion_peb/experiments/02_diffusion_baseline/run_diffusion_fft.py
reaction_diffusion_peb/experiments/02_diffusion_baseline/compare_fd_fft.py
```

## FD update

```math
H^{n+1}
=
H^n
+
\Delta t D_H \nabla^2 H^n
```

안정 조건:

```math
\Delta t
\leq
\frac{\Delta x^2}{4D_H}
```

## FFT heat-kernel solution

```math
\hat{H}(f_x,f_y,t)
=
\hat{H}_0(f_x,f_y)
\exp
\left[
-4\pi^2D_H(f_x^2+f_y^2)t
\right]
```

## 실험 파라미터

```yaml
peb:
  time_s: 60.0

grid:
  grid_spacing_nm: 1.0

diffusion:
  DH0_nm2_s: 0.8
```

Sweep:

```yaml
sweep:
  peb_time_s: [60, 75, 90]
  DH0_nm2_s: [0.3, 0.8, 1.5]
  grid_spacing_nm: [0.5, 1.0]
```

## 확인할 것

```text
시간이 지날수록 H가 smooth해지는지
DH가 커질수록 blur가 커지는지
FD와 FFT 결과가 비슷한지
loss term이 없을 때 total acid mass가 보존되는지
```

---

# Phase 3. PINN Diffusion Baseline

## 목적

데이터가 없는 현재 상태에서 DeepONet/FNO보다 먼저 PINN을 사용해 diffusion PDE를 학습한다.

단, PINN 결과는 반드시 FD/FFT와 비교한다.

## 대상 PDE

```math
\frac{\partial H}{\partial t}
-
D_H \nabla^2 H
=
0
```

## PINN 입력/출력

입력:

```text
x, y, t
```

출력:

```text
H(x,y,t)
```

## Loss

```math
L
=
L_{PDE}
+
L_{IC}
+
L_{BC}
```

PDE residual:

```math
r(x,y,t)
=
\frac{\partial H}{\partial t}
-
D_H
\left(
\frac{\partial^2 H}{\partial x^2}
+
\frac{\partial^2 H}{\partial y^2}
\right)
```

```math
L_{PDE} = \|r(x,y,t)\|^2
```

Initial condition:

```math
L_{IC}
=
\|H(x,y,0)-H_0(x,y)\|^2
```

Boundary condition:

```math
L_{BC}
=
\|\nabla H \cdot n\|^2
```

또는 periodic boundary condition을 사용한다.

## 구현 파일

```text
reaction_diffusion_peb/src/pinn_base.py
reaction_diffusion_peb/src/pinn_diffusion.py

reaction_diffusion_peb/experiments/03_pinn_diffusion/run_pinn_diffusion.py
reaction_diffusion_peb/experiments/03_pinn_diffusion/compare_fd_fft_pinn.py
```

## 실험 파라미터

```yaml
pinn:
  hidden_layers: 4
  hidden_width: 64
  activation: tanh
  collocation_points: 20000
  ic_points: 4096
  bc_points: 2048
  epochs: 5000
  learning_rate: 0.001
```

## 확인할 것

```text
PINN이 diffusion-only를 FD/FFT와 비슷하게 재현하는지
sharp initial condition에서 PINN이 어려워하는지
PDE residual이 줄어드는지
IC error와 PDE residual이 동시에 줄어드는지
PINN이 빠른 solver가 아니라 PDE 학습 도구임을 이해했는지
```

---

# Phase 4. Acid Loss + PINN 비교

## 목적

Acid 자연 소멸, trap, 비활성화 효과를 추가한다.

## PDE

```math
\frac{\partial H}{\partial t}
=
D_H \nabla^2 H
-
k_{loss}H
```

PINN residual:

```math
r(x,y,t)
=
\frac{\partial H}{\partial t}
-
D_H \nabla^2H
+
k_{loss}H
```

## 구현 파일

```text
reaction_diffusion_peb/src/reaction_diffusion.py
reaction_diffusion_peb/src/pinn_reaction_diffusion.py

reaction_diffusion_peb/experiments/04_acid_loss/run_acid_loss_fd.py
reaction_diffusion_peb/experiments/04_acid_loss/run_acid_loss_pinn.py
reaction_diffusion_peb/experiments/04_acid_loss/compare_fd_pinn.py
```

## 실험 파라미터

```yaml
reaction:
  kloss_s_inv: 0.005
```

Sweep:

```yaml
sweep:
  kloss_s_inv: [0.001, 0.005, 0.01, 0.05]
```

## 확인할 것

```text
kloss가 0이면 diffusion-only와 같아지는지
kloss가 커질수록 전체 acid mass가 감소하는지
FD와 PINN의 H(x,y,t_final)이 비슷한지
```

---

# Phase 5. Deprotection

## 목적

Acid가 resin deprotection을 촉진하는 과정을 추가한다.

## 모델

Deprotected fraction `P`를 사용한다.

```math
\frac{\partial P}{\partial t}
=
k_{dep}H(1-P)
```

여기서:

```text
P = 0: 아직 deprotected되지 않음
P = 1: 완전히 deprotected됨
```

## 구현 파일

```text
reaction_diffusion_peb/src/deprotection.py

reaction_diffusion_peb/experiments/05_deprotection/run_deprotection_fd.py
reaction_diffusion_peb/experiments/05_deprotection/run_deprotection_pinn.py
```

## 실험 파라미터

```yaml
reaction:
  kdep_s_inv: 0.05
```

Sweep:

```yaml
sweep:
  kdep_s_inv: [0.01, 0.05, 0.1, 0.5]
```

## 확인할 것

```text
H가 높은 영역에서 P가 빠르게 증가하는지
PEB 시간이 길수록 P가 커지는지
kdep가 커질수록 threshold contour가 넓어지는지
P는 0~1 범위를 유지하는지
```

---

# Phase 6. Temperature / Arrhenius PEB

## 목적

PEB 온도에 따라 reaction rate가 달라지는 효과를 추가한다.

## Arrhenius correction

기준 온도 `T_ref`에서의 rate `k_ref`가 있을 때:

```math
k(T)
=
k_{ref}
\exp
\left[
-\frac{E_a}{R}
\left(
\frac{1}{T_K}
-
\frac{1}{T_{ref,K}}
\right)
\right]
```

여기서:

```text
T_K = temperature_c + 273.15
R = 8.314 J/(mol K)
Ea = activation energy [J/mol]
```

## 구현 파일

```text
reaction_diffusion_peb/src/arrhenius.py

reaction_diffusion_peb/experiments/06_temperature_peb/run_temperature_sweep.py
reaction_diffusion_peb/experiments/06_temperature_peb/run_time_sweep.py
```

## 실험 파라미터

```yaml
peb:
  temperature_c: 100.0
  temperature_ref_c: 100.0
  time_s: 60.0

arrhenius:
  activation_energy_kj_mol: 100.0
```

Sweep:

```yaml
sweep:
  peb_temperature_c: [80, 90, 100, 110, 120, 125]
  peb_time_s: [60, 75, 90]
  activation_energy_kj_mol: [100]
```

## 확인할 것

```text
T가 올라가면 kdep, kq, kloss가 증가하는지
T sweep이 P profile과 final resist threshold에 영향을 주는지
온도 변화가 작은 경우에도 reaction rate가 민감하게 변하는지
MOR 125°C preset이 별도로 동작하는지
```

---

# Phase 7. Quencher Reaction

## 목적

Acid와 quencher의 neutralization을 추가한다.

이 단계부터 stiff reaction-diffusion 가능성이 커진다.

## PDE

```math
\frac{\partial H}{\partial t}
=
D_H \nabla^2 H
-
k_qHQ
-
k_{loss}H
```

```math
\frac{\partial Q}{\partial t}
=
D_Q \nabla^2 Q
-
k_qHQ
```

## PINN residual

```math
r_H
=
\frac{\partial H}{\partial t}
-
D_H \nabla^2H
+
k_qHQ
+
k_{loss}H
```

```math
r_Q
=
\frac{\partial Q}{\partial t}
-
D_Q \nabla^2Q
+
k_qHQ
```

## 구현 파일

```text
reaction_diffusion_peb/src/reaction_diffusion.py
reaction_diffusion_peb/src/pinn_reaction_diffusion.py

reaction_diffusion_peb/experiments/07_quencher_reaction/run_quencher_reaction_safe.py
reaction_diffusion_peb/experiments/07_quencher_reaction/run_quencher_reaction_stiff.py
```

## 실험 파라미터

처음에는 안정성을 위해 작은 값으로 시작한다.

```yaml
diffusion:
  DH0_nm2_s: 0.8
  DQ_ratio: 0.1

reaction:
  kq_s_inv: 1.0
  kloss_s_inv: 0.005
  Q0_mol_dm3: 0.1
```

Sweep:

```yaml
sweep:
  DQ_ratio: [0.05, 0.1]
  kq_s_inv_safe: [1, 5, 10]
  kq_s_inv_target: [100, 300, 1000]
```

주의:

```text
kq = 100 ~ 1000 s^-1는 stiff하다.
처음부터 이 범위를 쓰지 않는다.
먼저 kq = 1, 5, 10 s^-1에서 동작 확인 후,
time step을 줄이거나 semi-implicit / operator splitting을 도입한 뒤 큰 kq를 테스트한다.
```

큰 `kq`를 다루려면 다음 중 하나가 필요하다.

```text
1. 매우 작은 dt 사용
2. operator splitting
3. semi-implicit method
4. implicit reaction update
5. adaptive time step
```

PINN도 stiff reaction에서는 어려울 수 있다.  
따라서 stiff kq 범위에서는 PINN을 먼저 성공시키려고 하지 말고, FD/semi-implicit baseline을 먼저 안정화한다.

## 확인할 것

```text
H가 높은 영역에서 Q가 빠르게 소모되는지
Q가 충분히 많으면 H가 강하게 억제되는지
DQ가 작으면 Q profile이 acid보다 덜 퍼지는지
kq가 커지면 수치 안정성이 나빠지는지
PINN residual 학습이 kq 증가에 따라 어려워지는지
```

---

# Phase 8. Full Reaction-Diffusion 통합

## 목적

이전 단계들을 하나의 모델로 통합한다.

## 최종 PDE

```math
\frac{\partial H}{\partial t}
=
\nabla \cdot (D_H \nabla H)
-
k_qHQ
-
k_{loss}H
```

```math
\frac{\partial Q}{\partial t}
=
\nabla \cdot (D_Q \nabla Q)
-
k_qHQ
```

```math
\frac{\partial P}{\partial t}
=
k_{dep}H(1-P)
```

## 구현 파일

```text
reaction_diffusion_peb/src/reaction_diffusion.py

reaction_diffusion_peb/experiments/08_full_reaction_diffusion/run_full_model.py
```

## 실험 파라미터

```yaml
peb:
  time_s: 60.0
  temperature_c: 100.0
  temperature_ref_c: 100.0

grid:
  grid_spacing_nm: 1.0

exposure:
  dose: 1.0
  eta: 1.0
  Hmax_mol_dm3: 0.2

diffusion:
  model: constant
  DH0_nm2_s: 0.8
  DQ_ratio: 0.1

reaction:
  kdep_ref_s_inv: 0.05
  kq_ref_s_inv: 1.0
  kloss_ref_s_inv: 0.005
  Q0_mol_dm3: 0.1

arrhenius:
  enabled: false
  activation_energy_kj_mol: 100.0

threshold:
  P_threshold: 0.5
  beta: 20.0
```

## 확인할 것

```text
각 항을 끄면 이전 단계 결과와 일치하는지
diffusion만 켜면 Phase 2와 일치하는지
kloss만 추가하면 Phase 4와 일치하는지
deprotection을 켜면 Phase 5와 일치하는지
temperature를 켜면 Phase 6과 일치하는지
Q reaction을 켜면 Phase 7과 일치하는지
```

---

# Phase 9. Dataset Generation

## 목적

FD/FFT/PINN 결과를 축적하여 DeepONet/FNO 학습용 데이터셋을 만든다.

이 단계가 있어야 DeepONet/FNO로 넘어갈 수 있다.

## Dataset input

```text
I(x,y)
H0(x,y)
DH0_nm2_s
DQ_ratio
kdep_s_inv
kq_s_inv
kloss_s_inv
temperature_c
peb_time_s
dose
eta
```

## Dataset output

```text
H(x,y,t_final)
Q(x,y,t_final)
P(x,y,t_final)
R(x,y)
```

## 구현 파일

```text
reaction_diffusion_peb/src/dataset_builder.py

reaction_diffusion_peb/experiments/09_dataset_generation/generate_fd_dataset.py
reaction_diffusion_peb/experiments/09_dataset_generation/generate_pinn_dataset.py
reaction_diffusion_peb/experiments/09_dataset_generation/validate_dataset.py
```

## 저장 포맷

```text
outputs/datasets/
  diffusion_dataset.npz
  acid_loss_dataset.npz
  reaction_diffusion_dataset.npz
  metadata.json
```

metadata 예시:

```json
{
  "grid_size": 128,
  "grid_spacing_nm": 1.0,
  "samples": 1000,
  "input_fields": ["I", "H0"],
  "input_parameters": ["DH0_nm2_s", "kdep_s_inv", "kloss_s_inv", "temperature_c", "peb_time_s"],
  "output_fields": ["H_final", "P_final", "R"],
  "solver": "fd"
}
```

## 확인할 것

```text
각 sample의 input/output shape이 일관적인지
parameter range가 metadata에 저장되는지
train/validation/test split이 가능한지
FD-generated data와 PINN-generated data를 구분하는지
stiff kq 데이터는 별도 subset으로 분리하는지
```

---

# Phase 10. DeepONet / FNO Optional Surrogate

## 목적

데이터셋이 쌓인 뒤, PEB reaction-diffusion operator를 학습한다.

이 단계는 초기 필수가 아니다.  
DeepONet/FNO는 PINN과 FD/FFT 결과가 쌓인 후 진행한다.

## 학습할 operator

```text
(I(x,y), parameters)
  → H(x,y,t_final), P(x,y,t_final), R(x,y)
```

또는:

```text
H0(x,y), DH0, kdep, kloss, T, t_PEB
  → P(x,y,t_final)
```

## DeepONet 사용 위치

DeepONet은 다음처럼 쓰기 좋다.

```text
Branch input:
  H0 field encoding
  process parameters

Trunk input:
  coordinate x, y, optionally t

Output:
  H(x,y,t), P(x,y,t)
```

## FNO 사용 위치

FNO는 다음처럼 쓰기 좋다.

```text
Input channels:
  H0(x,y)
  parameter maps: DH0, kdep, kloss, T, t_PEB

Output channels:
  H_final(x,y)
  P_final(x,y)
  R(x,y)
```

## 구현 파일

```text
reaction_diffusion_peb/experiments/10_operator_learning_optional/train_deeponet.py
reaction_diffusion_peb/experiments/10_operator_learning_optional/train_fno.py
reaction_diffusion_peb/experiments/10_operator_learning_optional/evaluate_operator_surrogate.py
```

## 확인할 것

```text
DeepONet/FNO는 데이터가 쌓인 뒤에만 시작한다.
FD/PINN 대비 surrogate error를 측정한다.
parameter extrapolation에서는 성능이 떨어질 수 있음을 확인한다.
stiff kq case는 별도 평가한다.
```

---

# Phase 11. Advanced Stochastic / Petersen / z-axis

## 목적

초기 모델이 안정화된 뒤 고급 파라미터를 추가한다.

## Petersen nonlinear diffusion

권장 표기:

```math
D_H
=
D_{H0}
\exp(\alpha P)
```

또는:

```math
D_H
=
D_{H0}
\exp(\alpha \cdot dose\_field)
```

주의:

```text
기존 표의 DH = DH0 exp(αD)에서 D가 무엇인지 명확히 해야 한다.
Dose, diffusion coefficient, deprotected fraction을 모두 D라고 쓰면 안 된다.
```

권장:

```text
P = deprotected fraction
dose_field = normalized local exposure dose
DH = acid diffusion coefficient
```

## Temperature uniformity

```yaml
stochastic:
  temperature_uniformity_c: 0.02
```

각 ensemble run에서 다음처럼 perturbation을 준다.

```text
temperature_c_run = temperature_c + Normal(0, temperature_uniformity_c)
```

## Molecular blur / particle size

```yaml
resist:
  particle_size_nm: 1.0
```

사용 방식:

```text
1. final P field에 Gaussian blur 적용
2. stochastic noise scale로 사용
3. minimum meaningful grid scale로 사용
```

## z-axis / film thickness

초기 모델:

```text
H(x, y, t)
Q(x, y, t)
P(x, y, t)
```

후반 z-axis 모델:

```text
H(x, y, z, t)
Q(x, y, z, t)
P(x, y, z, t)
```

PINN에서 z-axis를 넣는 경우:

```text
input: x, y, z, t
output: H, Q, P
domain z: 0 ~ film_thickness_nm
```

## advanced sweep

```yaml
advanced:
  petersen_alpha: [0.5, 1.0, 2.0, 3.0]
  temperature_uniformity_c: 0.02
  molecular_blur_nm: [0.5, 1.0]
  film_thickness_nm: [20, 30]
  ensemble_runs: [10, 20]
```

---

## 11. 추천 config 파일

## 11.1 minimal_diffusion.yaml

```yaml
grid:
  grid_size: 128
  grid_spacing_nm: 1.0

peb:
  time_s: 60.0

exposure:
  dose: 1.0
  eta: 1.0
  Hmax_mol_dm3: 0.2

diffusion:
  DH0_nm2_s: 0.8
```

## 11.2 pinn_diffusion.yaml

```yaml
grid:
  grid_size: 128
  grid_spacing_nm: 1.0

peb:
  time_s: 60.0

diffusion:
  DH0_nm2_s: 0.8

pinn:
  hidden_layers: 4
  hidden_width: 64
  activation: tanh
  collocation_points: 20000
  ic_points: 4096
  bc_points: 2048
  epochs: 5000
  learning_rate: 0.001
```

## 11.3 acid_loss.yaml

```yaml
grid:
  grid_size: 128
  grid_spacing_nm: 1.0

peb:
  time_s: 60.0

exposure:
  dose: 1.0
  eta: 1.0
  Hmax_mol_dm3: 0.2

diffusion:
  DH0_nm2_s: 0.8

reaction:
  kloss_s_inv: 0.005
```

## 11.4 quencher_reaction.yaml

```yaml
grid:
  grid_size: 128
  grid_spacing_nm: 1.0

peb:
  time_s: 60.0

exposure:
  dose: 1.0
  eta: 1.0
  Hmax_mol_dm3: 0.2

diffusion:
  DH0_nm2_s: 0.8
  DQ_ratio: 0.1

reaction:
  kq_s_inv: 1.0
  kloss_s_inv: 0.005
  Q0_mol_dm3: 0.1
```

## 11.5 deprotection.yaml

```yaml
grid:
  grid_size: 128
  grid_spacing_nm: 1.0

peb:
  time_s: 60.0

exposure:
  dose: 1.0
  eta: 1.0
  Hmax_mol_dm3: 0.2

diffusion:
  DH0_nm2_s: 0.8
  DQ_ratio: 0.1

reaction:
  kq_s_inv: 1.0
  kloss_s_inv: 0.005
  kdep_s_inv: 0.05
  Q0_mol_dm3: 0.1
```

## 11.6 temperature_peb.yaml

```yaml
grid:
  grid_size: 128
  grid_spacing_nm: 1.0

peb:
  time_s: 60.0
  temperature_c: 100.0
  temperature_ref_c: 100.0

arrhenius:
  enabled: true
  activation_energy_kj_mol: 100.0

exposure:
  dose: 1.0
  eta: 1.0
  Hmax_mol_dm3: 0.2

diffusion:
  DH0_nm2_s: 0.8
  DQ_ratio: 0.1

reaction:
  kq_ref_s_inv: 1.0
  kloss_ref_s_inv: 0.005
  kdep_ref_s_inv: 0.05
  Q0_mol_dm3: 0.1
```

## 11.7 parameter_sweep.yaml

```yaml
sweep:
  peb_temperature_c: [80, 90, 100, 110, 120, 125]
  peb_time_s: [60, 75, 90]

  Hmax_mol_dm3: [0.1, 0.2, 0.3]

  DH0_nm2_s: [0.3, 0.8, 1.5]
  DQ_ratio: [0.05, 0.1]

  kloss_s_inv: [0.001, 0.005, 0.01, 0.05]
  kdep_s_inv: [0.01, 0.05, 0.1, 0.5]

  kq_s_inv_safe: [1, 5, 10]
  kq_s_inv_target: [100, 300, 1000]

  activation_energy_kj_mol: [100]

  grid_spacing_nm: [0.5, 1.0]

advanced:
  petersen_alpha: [0.5, 1.0, 2.0, 3.0]
  temperature_uniformity_c: 0.02
  molecular_blur_nm: [0.5, 1.0]
  film_thickness_nm: [20, 30]
  ensemble_runs: [10, 20]
```

---

## 12. GitHub 작업 방식

## 기존 프로젝트는 그대로 둔다

금지:

```text
기존 README 전체 갈아엎기
기존 src/ 구조에 바로 섞기
FNO / optics 모듈과 직접 결합하기
DeepONet/FNO부터 시작하기
```

허용:

```text
reaction_diffusion_peb/ 폴더 추가
reaction_diffusion_peb/README.md 추가
reaction_diffusion_peb/PLAN.md 추가
reaction_diffusion_peb/PARAMETER_SCOPE.md 추가
reaction_diffusion_peb/configs/ 추가
reaction_diffusion_peb/src/ 추가
reaction_diffusion_peb/experiments/ 추가
```

## 추천 브랜치명

```text
feature/reaction-diffusion-peb
```

또는:

```text
peb-pinn-reaction-module
```

## 추천 커밋 순서

```text
Commit 1:
  Add reaction_diffusion_peb workspace skeleton

Commit 2:
  Add parameter scope and sweep configs

Commit 3:
  Add synthetic aerial generation and exposure model

Commit 4:
  Add diffusion-only FD/FFT baseline solvers

Commit 5:
  Add PINN diffusion baseline and FD/FFT/PINN comparison

Commit 6:
  Add acid loss model with FD/PINN comparison

Commit 7:
  Add deprotection model

Commit 8:
  Add Arrhenius temperature-dependent PEB

Commit 9:
  Add quencher reaction model with safe kq range

Commit 10:
  Add full reaction-diffusion demo

Commit 11:
  Add dataset generation from FD/PINN simulations

Commit 12:
  Add optional DeepONet/FNO surrogate plan or prototype

Commit 13:
  Add advanced stochastic/Petersen/z-axis plan
```

## 추천 PR 제목

```text
Add local Reaction-Diffusion / PEB PINN-first study module
```

## 추천 PR 설명

```markdown
## Summary

This PR adds a local `reaction_diffusion_peb/` workspace for studying post-exposure bake resist physics independently from the main lithography/neural-operator pipeline.

Because no simulation dataset exists yet, the module follows a physics-first and PINN-first workflow:

1. Build FD/FFT baselines for diffusion.
2. Train PINNs against the same PDEs and compare with FD/FFT.
3. Add acid loss, deprotection, temperature dependence, and quencher reaction incrementally.
4. Generate reusable simulation datasets.
5. Keep DeepONet/FNO as optional downstream operator surrogates after data accumulation.

## Scope

- Keep existing project structure unchanged
- Add local Reaction-Diffusion / PEB workspace
- Use synthetic exposure maps first
- Build FD/FFT physics baselines
- Add PINN diffusion and reaction-diffusion experiments before DeepONet/FNO
- Track process, diffusion, reaction, resist, and numerical parameter ranges
- Separate safe kq values from stiff target kq values

## Out of scope

- Mask simulation
- Fourier optics
- Full lithography pipeline integration
- Production-grade stiff PDE solver
- Real calibrated CAR/MOR fitting
- DeepONet/FNO as the first modeling step
```

---

## 13. 실행 예시

## 환경 생성

```bash
cd litho-neural-operator-lab/reaction_diffusion_peb
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Synthetic aerial 생성

```bash
python experiments/01_synthetic_aerial/run_gaussian_spot.py
```

## Diffusion baseline 실행

```bash
python experiments/02_diffusion_baseline/run_diffusion_fd.py
python experiments/02_diffusion_baseline/run_diffusion_fft.py
python experiments/02_diffusion_baseline/compare_fd_fft.py
```

## PINN diffusion 실행

```bash
python experiments/03_pinn_diffusion/run_pinn_diffusion.py
python experiments/03_pinn_diffusion/compare_fd_fft_pinn.py
```

## Acid loss 실행

```bash
python experiments/04_acid_loss/run_acid_loss_fd.py
python experiments/04_acid_loss/run_acid_loss_pinn.py
python experiments/04_acid_loss/compare_fd_pinn.py
```

## Deprotection 실행

```bash
python experiments/05_deprotection/run_deprotection_fd.py
```

## Temperature sweep 실행

```bash
python experiments/06_temperature_peb/run_temperature_sweep.py
```

## Quencher reaction 실행

```bash
python experiments/07_quencher_reaction/run_quencher_reaction_safe.py
```

## Dataset 생성

```bash
python experiments/09_dataset_generation/generate_fd_dataset.py
python experiments/09_dataset_generation/validate_dataset.py
```

---

## 14. 최소 성공 기준

이번 국소 모듈의 최소 성공 기준은 다음이다.

```text
1. Synthetic exposure map을 만들 수 있다.
2. Exposure map에서 initial acid H0를 만들 수 있다.
3. FD diffusion-only PEB에서 acid blur가 시간과 DH에 따라 증가한다.
4. FFT diffusion과 FD diffusion 결과가 비슷하다.
5. PINN diffusion이 FD/FFT baseline을 일정 수준 재현한다.
6. PINN의 PDE residual, IC loss, BC loss를 추적할 수 있다.
7. Acid loss를 켜면 total acid mass가 감소한다.
8. Acid-loss PINN이 FD와 비교 가능하다.
9. Deprotection P가 H가 높은 영역에서 증가한다.
10. Temperature를 올리면 Arrhenius correction에 의해 reaction rate가 증가한다.
11. Quencher reaction은 safe kq range에서 먼저 안정적으로 동작한다.
12. kq target range는 stiff handling 전까지 별도 분리한다.
13. Soft threshold를 통해 latent resist image를 만들 수 있다.
14. FD/PINN 결과를 dataset으로 저장할 수 있다.
15. DeepONet/FNO는 dataset 생성 이후 optional로 시작한다.
```

---

## 15. 초기에는 하지 않을 것

초기 Reaction-Diffusion / PEB 모듈에서는 아래는 하지 않는다.

```text
1. 3D Maxwell / RCWA / FDTD
2. Mask 3D effect
3. Inverse mask optimization
4. DeepONet/FNO를 첫 모델로 사용
5. 실제 calibrated CAR/MOR fitting
6. full 3D z-axis resist simulation
7. production-grade stochastic LER/LWR model
8. OPC / ILT integration
```

후속으로 가능한 항목:

```text
1. FD/PINN dataset 기반 DeepONet/FNO
2. Petersen nonlinear diffusion
3. temperature uniformity ensemble
4. molecular blur
5. z-axis PINN
6. main optics pipeline과 파일 기반 연결
```

---

## 16. 최종 요약

이번 작업의 최종 방향은 다음이다.

```text
기존 GitHub 프로젝트는 그대로 둔다.
reaction_diffusion_peb/ 폴더를 새로 만든다.
그 안에서 Reaction-diffusion / PEB만 독립적으로 운용한다.
데이터가 없으므로 DeepONet/FNO보다 PINN을 먼저 사용한다.
하지만 PINN만 단독으로 믿지 않고 FD/FFT baseline과 반드시 비교한다.
Diffusion-only → PINN diffusion → acid loss → deprotection → temperature → quencher → full model 순서로 확장한다.
FD/PINN 결과를 dataset으로 저장한 뒤 DeepONet/FNO를 optional surrogate로 진행한다.
```

가장 먼저 만들 파일은 다음이다.

```text
reaction_diffusion_peb/
  README.md
  PLAN.md
  PARAMETER_SCOPE.md
  requirements.txt

  configs/
    minimal_diffusion.yaml
    pinn_diffusion.yaml
    parameter_sweep.yaml

  src/
    synthetic_aerial.py
    exposure.py
    diffusion_fd.py
    diffusion_fft.py
    pinn_base.py
    pinn_diffusion.py

  experiments/
    01_synthetic_aerial/run_gaussian_spot.py
    02_diffusion_baseline/run_diffusion_fd.py
    02_diffusion_baseline/run_diffusion_fft.py
    03_pinn_diffusion/run_pinn_diffusion.py
    03_pinn_diffusion/compare_fd_fft_pinn.py
```

첫 목표는 이것이다.

```text
Gaussian synthetic aerial image
  → initial acid H0
  → 60s PEB diffusion using FD/FFT
  → PINN diffusion 학습
  → FD/FFT/PINN before-after 비교
  → figure, metrics, loss curve 저장
```

이 첫 목표가 안정적으로 동작하면 그다음에 acid loss, deprotection, temperature, quencher reaction, dataset generation, DeepONet/FNO surrogate를 순서대로 추가한다.
