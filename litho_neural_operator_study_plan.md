# Neural Operator / PINN 기반 Lithography 공부용 시뮬레이터 실험 계획서

## 0. 목표 정의

이 프로젝트의 목표는 논문 제출이나 SOTA 성능 검증이 아니라, **물리 법칙을 직접 구현하고 이해한 뒤 neural operator와 PINN을 어디에 붙일 수 있는지 학습하는 것**이다.

핵심 목표는 다음과 같다.

```text
Mask geometry
    ↓
Diffraction spectrum
    ↓
Pupil / NA filtering
    ↓
Aerial image
    ↓
Exposure / acid generation
    ↓
Reaction-diffusion / PEB
    ↓
Resist latent image / threshold contour
    ↓
Inverse optimization
```

최종적으로는 다음 질문에 답할 수 있어야 한다.

1. Mask pattern이 Fourier domain에서 어떤 diffraction spectrum을 만드는가?
2. Projection pupil과 NA 제한이 aerial image를 어떻게 바꾸는가?
3. 원하는 위치의 intensity를 키우고 주변 leakage를 줄이는 mask optimization이 가능한가?
4. 왜 background aerial을 완전히 0으로 만들 수 없는가?
5. Resist diffusion과 threshold가 aerial image를 어떻게 변형하는가?
6. FNO / DeepONet은 어떤 operator를 학습하는가?
7. PINN은 reaction-diffusion PDE 이해에 어떤 도움을 주는가?
8. Neural surrogate가 물리 solver를 대체할 때 어떤 위험이 있는가?

---

## 1. 전체 개발 환경

### 1.1 권장 언어

```text
Python 3.10+
```

### 1.2 핵심 라이브러리

| 목적 | 툴 / 라이브러리 |
|---|---|
| 수치 계산 | NumPy |
| 시각화 | Matplotlib |
| 자동미분 / 최적화 | PyTorch |
| FFT / Fourier optics | NumPy FFT 또는 PyTorch FFT |
| PDE finite difference | NumPy / PyTorch |
| PINN 구현 | PyTorch |
| FNO / DeepONet 구현 | PyTorch |
| 실험 로그 | YAML / JSON / CSV |
| 결과 저장 | PNG / NPZ / HDF5 |
| 문서화 | Markdown |

### 1.3 선택 라이브러리

| 목적 | 선택 툴 |
|---|---|
| 실험 설정 관리 | Hydra 또는 OmegaConf |
| 진행 상황 표시 | tqdm |
| 데이터 저장 | h5py |
| Interactive notebook | Jupyter |
| Unit test | pytest |
| GPU 사용 | CUDA-enabled PyTorch |

---

## 2. 프로젝트 폴더 구조

```text
litho_neural_operator_study/
  README.md
  requirements.txt

  configs/
    scalar_imaging.yaml
    inverse_aerial.yaml
    partial_coherence.yaml
    resist.yaml
    pinn.yaml
    fno.yaml

  src/
    common/
      grid.py
      fft_utils.py
      visualization.py
      metrics.py
      io.py

    mask/
      patterns.py
      transmission.py
      constraints.py

    optics/
      scalar_diffraction.py
      pupil.py
      coherent_imaging.py
      partial_coherence.py
      defocus.py

    inverse/
      losses.py
      optimize_mask.py
      regularizers.py

    resist/
      exposure.py
      diffusion_fd.py
      diffusion_fft.py
      reaction_diffusion.py
      threshold.py

    pinn/
      pinn_base.py
      pinn_diffusion.py
      pinn_reaction_diffusion.py

    neural_operator/
      fno2d.py
      deeponet.py
      datasets.py
      train_fno.py
      train_deeponet.py

    closed_loop/
      surrogate_optimizer.py
      active_learning.py

  experiments/
    01_scalar_diffraction/
    02_inverse_aerial/
    03_partial_coherence/
    04_resist_diffusion/
    05_pinn_diffusion/
    06_fno_correction/
    07_closed_loop_inverse/

  outputs/
    figures/
    checkpoints/
    logs/
    datasets/
```

---

## 3. 실험 단계 개요

| Phase | 실험명 | 핵심 물리 | 주요 툴 | 산출물 |
|---|---|---|---|---|
| 1 | Scalar diffraction | Fourier optics | NumPy / PyTorch FFT | diffraction spectrum, aerial image |
| 2 | Coherent aerial imaging | Pupil filtering | FFT, Matplotlib | mask vs image 비교 |
| 3 | Inverse aerial optimization | Differentiable optics | PyTorch autograd | optimized mask |
| 4 | Partial coherence | Source integration | PyTorch FFT | illumination별 aerial image |
| 5 | Resist exposure + diffusion | Reaction-diffusion | FD / FFT PDE solver | latent image, resist contour |
| 6 | PINN diffusion | PDE residual learning | PyTorch PINN | PINN vs FD 비교 |
| 7 | Synthetic 3D mask correction | 3D effect approximation | PyTorch / FNO | correction operator dataset |
| 8 | FNO / DeepONet surrogate | Operator learning | PyTorch | surrogate model |
| 9 | Closed-loop inverse design | Surrogate-assisted optimization | PyTorch | target enhancement + leakage suppression |

---

# Phase 1. Scalar Diffraction 직접 구현

## 1.1 목표

가장 먼저 2D scalar Fourier optics를 직접 구현한다.

이 단계에서는 mask를 얇은 복소 transmission function으로 가정한다.

```math
t(x,y) = A(x,y)e^{i\phi(x,y)}
```

Binary mask의 경우:

```math
t(x,y) =
\begin{cases}
1, & \text{open} \\
0, & \text{blocked}
\end{cases}
```

Diffraction spectrum:

```math
T(f_x,f_y) = \mathcal{F}\{t(x,y)\}
```

## 1.2 사용할 툴

```text
NumPy
NumPy FFT
Matplotlib
```

또는 이후 autograd 연결을 고려하면:

```text
PyTorch
torch.fft
Matplotlib
```

## 1.3 구현 파일

```text
src/mask/patterns.py
src/mask/transmission.py
src/optics/scalar_diffraction.py
experiments/01_scalar_diffraction/demo_line_space.py
experiments/01_scalar_diffraction/demo_contact_hole.py
```

## 1.4 구현 내용

### patterns.py

구현할 기본 mask pattern:

```text
1. line-space pattern
2. contact hole
3. isolated line
4. two-bar pattern
5. elbow pattern
6. random binary mask
```

### transmission.py

```python
def binary_transmission(mask):
    return mask.astype(np.complex64)

def attenuated_phase_shift(mask, attenuation=0.06, phase=np.pi):
    clear = 1.0 + 0j
    absorber = np.sqrt(attenuation) * np.exp(1j * phase)
    return np.where(mask > 0.5, clear, absorber)
```

### scalar_diffraction.py

```python
def diffraction_spectrum(t):
    return fftshift(fft2(ifftshift(t)))
```

## 1.5 확인할 결과

각 mask마다 다음을 출력한다.

```text
1. mask image
2. diffraction amplitude spectrum
3. diffraction phase spectrum
4. log amplitude spectrum
```

## 1.6 배울 점

이 단계에서 확인해야 할 물리적 포인트:

```text
sharp edge → high frequency spectrum 증가
line-space pitch 감소 → diffraction order 간격 증가
contact hole → 2D sinc-like spectrum
phase mask → intensity는 같아도 phase spectrum이 달라짐
```

---

# Phase 2. Coherent Aerial Image 구현

## 2.1 목표

Projection lens의 NA 제한을 pupil filter로 구현하고, mask spectrum이 pupil을 통과한 뒤 wafer plane aerial image를 만드는 과정을 확인한다.

Pupil:

```math
P(f_x,f_y) =
\begin{cases}
1, & \sqrt{f_x^2+f_y^2} \leq \frac{NA}{\lambda} \\
0, & \text{otherwise}
\end{cases}
```

Wafer field:

```math
E(x,y) = \mathcal{F}^{-1}\{T(f_x,f_y)P(f_x,f_y)\}
```

Aerial image:

```math
I(x,y) = |E(x,y)|^2
```

## 2.2 사용할 툴

```text
PyTorch FFT
Matplotlib
```

PyTorch를 권장하는 이유는 Phase 3에서 mask optimization에 autograd를 바로 연결하기 위해서이다.

## 2.3 구현 파일

```text
src/optics/pupil.py
src/optics/coherent_imaging.py
experiments/01_scalar_diffraction/demo_coherent_aerial.py
```

## 2.4 구현 내용

### pupil.py

```python
def circular_pupil(fx, fy, wavelength, NA):
    cutoff = NA / wavelength
    return ((fx**2 + fy**2) <= cutoff**2).float()
```

### coherent_imaging.py

```python
def coherent_aerial_image(mask_t, pupil):
    T = torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(mask_t)))
    E = torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(T * pupil)))
    I = torch.abs(E) ** 2
    return I / (I.max() + 1e-12)
```

## 2.5 실험 조건

```text
grid size: 128 x 128 또는 256 x 256
wavelength: normalized 1.0
NA: 0.2, 0.4, 0.6, 0.8
mask: line-space, contact hole
```

## 2.6 확인할 결과

```text
NA가 낮을수록 edge blur 증가
NA가 높을수록 high frequency 보존
contact hole 주변 ringing 확인
isolated feature의 side-lobe 확인
```

## 2.7 산출물

```text
outputs/figures/phase2_mask.png
outputs/figures/phase2_spectrum.png
outputs/figures/phase2_pupil.png
outputs/figures/phase2_aerial_NA_sweep.png
```

---

# Phase 3. Inverse Aerial Optimization

## 3.1 목표

원하는 wafer target pattern을 정의하고, mask를 gradient descent로 업데이트해서 target region은 밝게 하고 background leakage는 줄인다.

이 단계가 사용자가 말한:

```text
aerial을 최소화하면서 회절을 반복해 원하는 곳에 효과를 주는 실험
```

에 해당한다.

## 3.2 최적화 변수

처음에는 mask를 continuous variable로 둔다.

```math
m(x,y) \in [0,1]
```

Transmission:

```math
t(x,y) = \sigma(\alpha m(x,y))
```

여기서 `alpha`는 binarization strength이다.

## 3.3 Loss 정의

Target region intensity를 키우고 forbidden region intensity를 낮춘다.

```math
L =
\lambda_T \|I_T - I_{\text{target}}\|^2
+
\lambda_B \|I_B\|^2
+
\lambda_{TV} TV(m)
+
\lambda_{bin} m(1-m)
```

각 항의 의미:

| Loss | 의미 |
|---|---|
| target loss | 원하는 영역의 aerial image를 target에 맞춤 |
| background leakage loss | 노출되면 안 되는 영역의 intensity 감소 |
| TV regularization | mask가 지나치게 noisy해지는 것 방지 |
| binarization loss | mask가 0 또는 1에 가까워지도록 유도 |

## 3.4 사용할 툴

```text
PyTorch
torch.fft
torch.optim.Adam
Matplotlib
```

## 3.5 구현 파일

```text
src/inverse/losses.py
src/inverse/regularizers.py
src/inverse/optimize_mask.py
experiments/02_inverse_aerial/demo_target_spot.py
experiments/02_inverse_aerial/demo_forbidden_region.py
```

## 3.6 실험 3-A: Target spot 만들기

### 조건

```text
grid: 128 x 128
target: 중앙 작은 원형 spot
forbidden: target 제외 전체 영역
NA: 0.4
optimizer: Adam
iterations: 1000
learning rate: 1e-2
```

### 산출물

```text
initial mask
optimized mask
initial aerial
optimized aerial
loss curve
background leakage curve
```

### 성공 기준

```text
target region 평균 intensity 증가
background 평균 intensity 감소
peak side-lobe 감소
```

## 3.7 실험 3-B: Forbidden region 추가

특정 영역에 intensity가 생기면 안 되도록 forbidden mask를 둔다.

```math
L_B = \sum_{(x,y) \in B} I(x,y)^2
```

### 확인할 점

```text
target을 키우면 forbidden region leakage가 증가할 수 있음
loss weight에 따라 target fidelity와 leakage suppression 사이 trade-off 발생
```

## 3.8 실험 3-C: Binary mask annealing

처음에는 continuous mask로 시작하고, 후반에 binarization을 강화한다.

```text
iteration 0-300: alpha = 1
iteration 300-700: alpha = 4
iteration 700-1000: alpha = 12
```

### 확인할 점

```text
continuous mask는 성능이 좋지만 물리적으로 만들기 어려움
binary mask로 갈수록 aerial quality가 낮아질 수 있음
```

---

# Phase 4. Partial Coherence / Source Integration

## 4.1 목표

실제 lithography에 더 가까운 partial coherent imaging을 구현한다.

여러 source point에 대해 aerial image를 계산하고 incoherent sum을 취한다.

```math
I(x,y) = \sum_s w_s |E_s(x,y)|^2
```

## 4.2 사용할 툴

```text
PyTorch FFT
Matplotlib
```

## 4.3 구현 파일

```text
src/optics/partial_coherence.py
src/optics/source.py
experiments/03_partial_coherence/demo_source_shapes.py
```

## 4.4 구현할 source shape

```text
1. on-axis coherent source
2. annular source
3. dipole source
4. quadrupole source
5. random source points
```

## 4.5 방법론

각 source point는 pupil shift 또는 illumination angle shift로 처리한다.

간단한 공부용 구현에서는 source point마다 spectrum 또는 pupil을 shift한다.

```python
for sx, sy, w in source_points:
    shifted_pupil = shift_pupil(pupil, sx, sy)
    I += w * coherent_aerial_image(mask_t, shifted_pupil)
```

## 4.6 실험

### 실험 4-A: Line-space source sweep

```text
mask: vertical line-space
source: coherent, annular, dipole-x, dipole-y
metric: image contrast, NILS
```

### 실험 4-B: Contact hole source sweep

```text
mask: contact hole
source: coherent, annular, quadrupole
metric: peak intensity, side-lobe, background leakage
```

## 4.7 배울 점

```text
illumination shape에 따라 같은 mask라도 aerial image가 달라짐
dipole illumination은 특정 방향 line-space에 유리함
partial coherence는 side-lobe와 contrast를 동시에 바꿈
source optimization이 mask optimization만큼 중요함
```

---

# Phase 5. Resist Exposure + Diffusion

## 5.1 목표

Aerial image를 resist latent image로 변환한다.

처음에는 단순화된 chemically amplified resist 모델을 사용한다.

## 5.2 Exposure model

Aerial intensity가 acid를 생성한다고 둔다.

```math
A_0(x,y) = 1 - \exp(-\eta D I(x,y))
```

변수:

| 변수 | 의미 |
|---|---|
| A0 | 초기 acid concentration |
| eta | acid generation coefficient |
| D | exposure dose |
| I | aerial image |

## 5.3 Diffusion model

PEB 중 acid가 diffusion한다고 둔다.

```math
\frac{\partial A}{\partial t}
=
D_A \nabla^2 A
```

## 5.4 Reaction-diffusion 확장

나중에는 quencher 또는 deprotection reaction을 넣는다.

```math
\frac{\partial A}{\partial t}
=
D_A \nabla^2 A - k A Q
```

```math
\frac{\partial Q}{\partial t}
=
- k A Q
```

## 5.5 사용할 툴

```text
NumPy 또는 PyTorch
Finite difference method
FFT heat kernel convolution
Matplotlib
```

## 5.6 구현 파일

```text
src/resist/exposure.py
src/resist/diffusion_fd.py
src/resist/diffusion_fft.py
src/resist/reaction_diffusion.py
src/resist/threshold.py
experiments/04_resist_diffusion/demo_dose_sweep.py
experiments/04_resist_diffusion/demo_diffusion_length.py
```

## 5.7 방법 A: Finite Difference

2D Laplacian:

```math
\nabla^2 A_{i,j}
=
\frac{
A_{i+1,j} + A_{i-1,j} + A_{i,j+1} + A_{i,j-1} - 4A_{i,j}
}{\Delta x^2}
```

Time stepping:

```math
A^{n+1} = A^n + \Delta t D_A \nabla^2 A^n
```

안정 조건:

```math
\Delta t \leq \frac{\Delta x^2}{4D_A}
```

## 5.8 방법 B: FFT heat kernel

Diffusion equation의 해는 Gaussian convolution이다.

Fourier domain:

```math
\hat{A}(f_x,f_y,t)
=
\hat{A}_0(f_x,f_y)
\exp(-4\pi^2 D_A (f_x^2+f_y^2)t)
```

이 방법은 diffusion만 있을 때 빠르고 정확하다.

## 5.9 Threshold model

```math
R(x,y)
=
\sigma(\beta(A(x,y) - A_{th}))
```

여기서 beta가 클수록 hard threshold에 가까워진다.

## 5.10 실험

### 실험 5-A: Dose sweep

```text
dose: 0.5, 1.0, 1.5, 2.0
확인: CD-like width 변화
```

### 실험 5-B: Diffusion length sweep

```text
diffusion length: 0, 1, 2, 4, 8 pixels
확인: edge blur, leakage smoothing
```

### 실험 5-C: Aerial vs resist 비교

```text
aerial image leakage가 threshold 아래면 resist에서는 사라짐
threshold 근처 leakage는 resist error로 변환됨
```

---

# Phase 6. PINN으로 Diffusion / Reaction-Diffusion 풀기

## 6.1 목표

PINN을 이용해 diffusion equation을 풀어보고, finite difference 및 FFT solution과 비교한다.

중요한 점은 PINN을 빠른 solver로 맹신하는 것이 아니라, **PDE residual과 boundary condition을 loss로 넣는다는 개념을 이해하는 것**이다.

## 6.2 대상 PDE

기본 diffusion:

```math
\frac{\partial A}{\partial t} - D_A \nabla^2 A = 0
```

Reaction-diffusion:

```math
\frac{\partial A}{\partial t} - D_A \nabla^2 A + kAQ = 0
```

## 6.3 사용할 툴

```text
PyTorch
Autograd
Matplotlib
```

## 6.4 구현 파일

```text
src/pinn/pinn_base.py
src/pinn/pinn_diffusion.py
src/pinn/pinn_reaction_diffusion.py
experiments/05_pinn_diffusion/compare_fd_pinn.py
```

## 6.5 PINN 입력과 출력

입력:

```text
x, y, t
```

출력:

```text
A(x,y,t)
```

네트워크:

```text
MLP
SIREN optional
Fourier feature encoding optional
```

## 6.6 Loss 구성

```math
L =
L_{\text{PDE}}
+
L_{\text{IC}}
+
L_{\text{BC}}
```

PDE residual:

```math
r(x,y,t)
=
\frac{\partial A}{\partial t}
-
D_A
\left(
\frac{\partial^2 A}{\partial x^2}
+
\frac{\partial^2 A}{\partial y^2}
\right)
```

```math
L_{\text{PDE}} = \|r(x,y,t)\|^2
```

Initial condition:

```math
L_{\text{IC}} = \|A(x,y,0) - A_0(x,y)\|^2
```

Boundary condition:

```math
L_{\text{BC}} = \|\nabla A \cdot n\|^2
```

또는 periodic boundary condition을 사용한다.

## 6.7 비교 기준

```text
PINN solution vs finite difference solution
PINN solution vs FFT heat kernel solution
MSE
edge error
training time
inference time
```

## 6.8 배울 점

```text
PINN은 PDE residual을 직접 줄이는 방식임
sharp initial condition에서는 학습이 어려울 수 있음
단순 diffusion에서는 FFT/FD가 더 빠르고 정확할 수 있음
PINN은 inverse parameter estimation이나 복잡한 boundary에서 의미가 커짐
```

---

# Phase 7. Synthetic 3D Mask Correction 만들기

## 7.1 목표

진짜 3D RCWA/FDTD를 바로 구현하지 않고, 공부용 synthetic 3D correction을 만든다.

Thin-mask spectrum:

```math
T_{\text{thin}}(f_x,f_y)
```

Synthetic 3D correction:

```math
T_{\text{3D}}(f_x,f_y)
=
T_{\text{thin}}(f_x,f_y) C(f_x,f_y;\theta)
```

여기서 `C`는 absorber height, sidewall angle, incidence angle, polarization에 따른 correction처럼 가정한다.

## 7.2 사용할 툴

```text
PyTorch
NumPy
Matplotlib
```

## 7.3 구현 파일

```text
src/neural_operator/synthetic_3d_correction_data.py
experiments/06_fno_correction/generate_synthetic_dataset.py
```

## 7.4 Correction 모델 예시

```math
C(f_x,f_y;\theta)
=
a(f_x,f_y;\theta)
e^{i\phi(f_x,f_y;\theta)}
```

Amplitude correction:

```math
a(f_x,f_y)
=
\exp(-\gamma (f_x^2+f_y^2))
```

Phase correction:

```math
\phi(f_x,f_y)
=
\alpha f_x + \beta f_y + \delta(f_x^2 - f_y^2)
```

비대칭 shadowing:

```math
C_{\text{asym}} =
1 + s \tanh(c f_x)
```

## 7.5 데이터셋 구성

입력:

```text
mask pattern
thin diffraction spectrum real part
thin diffraction spectrum imaginary part
parameters theta
```

출력:

```text
corrected spectrum real part
corrected spectrum imaginary part
or correction delta spectrum
```

권장 형태:

```text
input channels:
  mask
  Re(T_thin)
  Im(T_thin)
  theta maps

output channels:
  Re(Delta T)
  Im(Delta T)
```

## 7.6 배울 점

```text
3D mask effect는 thin-mask diffraction에 대한 complex correction으로 볼 수 있음
amplitude correction과 phase correction은 aerial image에 다르게 작용함
phase error가 intensity error로 뒤늦게 나타날 수 있음
```

---

# Phase 8. FNO / DeepONet Surrogate 학습

## 8.1 목표

Phase 7에서 만든 synthetic 3D correction operator를 FNO 또는 DeepONet으로 학습한다.

이 단계의 핵심은 neural network가 단순 image-to-image가 아니라 **operator mapping**을 학습한다는 점을 이해하는 것이다.

---

## 8.2 FNO 실험

### 대상 mapping

```math
\{m(x,y), T_{\text{thin}}(f_x,f_y), \theta\}
\rightarrow
\Delta T(f_x,f_y)
```

또는:

```math
\{m(x,y), \theta\}
\rightarrow
T_{\text{3D}}(f_x,f_y)
```

### 사용할 툴

```text
PyTorch
torch.fft
FNO2D implementation
```

### 구현 파일

```text
src/neural_operator/fno2d.py
src/neural_operator/datasets.py
src/neural_operator/train_fno.py
experiments/06_fno_correction/train_fno_correction.py
```

### FNO 구조

```text
input channels:
  mask
  Re(T_thin)
  Im(T_thin)
  theta_1
  theta_2
  theta_3

hidden width:
  32 or 64

Fourier modes:
  12 or 16

layers:
  4

output channels:
  Re(Delta T)
  Im(Delta T)
```

### Loss

```math
L =
\| \Delta T_{\text{pred}} - \Delta T_{\text{true}} \|^2
+
\lambda_I
\| I_{\text{pred}} - I_{\text{true}} \|^2
```

여기서 두 번째 항은 corrected spectrum으로 aerial image를 만든 뒤 intensity 차이를 계산하는 physics-aware loss이다.

---

## 8.3 DeepONet 실험

### 대상 mapping

DeepONet은 branch net과 trunk net으로 나눈다.

Branch input:

```text
mask representation
theta parameters
```

Trunk input:

```text
frequency coordinate fx, fy
```

Output:

```text
Re(Delta T(fx, fy)), Im(Delta T(fx, fy))
```

### 사용할 툴

```text
PyTorch
MLP
Fourier feature encoding
```

### 구현 파일

```text
src/neural_operator/deeponet.py
src/neural_operator/train_deeponet.py
experiments/06_fno_correction/train_deeponet_correction.py
```

### 장점

```text
원하는 frequency coordinate에서만 query 가능
grid resolution이 달라도 확장 가능
operator learning 개념을 공부하기 좋음
```

### 단점

```text
FNO보다 구현과 학습이 조금 더 번거로움
2D dense spectrum 전체를 예측할 때는 FNO가 더 간단함
```

---

## 8.4 평가 metric

```text
spectrum MSE
complex amplitude relative error
phase error
aerial image MSE
target intensity error
background leakage error
optimization 결과 차이
```

## 8.5 배울 점

```text
spectrum error가 작아도 aerial image error가 클 수 있음
phase error는 intensity 결과에 중요함
neural surrogate는 training distribution 밖에서 위험함
physics-aware loss가 pure spectrum MSE보다 유리할 수 있음
```

---

# Phase 9. Closed-loop Surrogate-assisted Inverse Design

## 9.1 목표

FNO / DeepONet surrogate를 inverse mask optimization loop에 넣는다.

즉:

```text
mask variable
  ↓
thin diffraction
  ↓
neural 3D correction surrogate
  ↓
corrected spectrum
  ↓
aerial image
  ↓
loss
  ↓
gradient update
  ↓
mask update
```

## 9.2 사용할 툴

```text
PyTorch
Trained FNO / DeepONet
Adam optimizer
Matplotlib
```

## 9.3 구현 파일

```text
src/closed_loop/surrogate_optimizer.py
src/closed_loop/active_learning.py
experiments/07_closed_loop_inverse/optimize_with_fno.py
```

## 9.4 Loss

```math
L =
\lambda_T L_{\text{target}}
+
\lambda_B L_{\text{background}}
+
\lambda_R L_{\text{regularization}}
+
\lambda_M L_{\text{manufacturability}}
+
\lambda_S L_{\text{surrogate_safety}}
```

각 항:

| Loss | 의미 |
|---|---|
| target | 원하는 위치 intensity 확보 |
| background | forbidden region leakage 억제 |
| regularization | mask smoothness |
| manufacturability | 너무 작은 feature 방지 |
| surrogate safety | training distribution 밖으로 벗어나는 mask 방지 |

## 9.5 실험

### 실험 9-A: Physics-only optimization

```text
thin-mask scalar optics만 사용
```

### 실험 9-B: Synthetic 3D correction 포함

```text
true synthetic correction 사용
```

### 실험 9-C: FNO surrogate correction 사용

```text
FNO가 예측한 correction 사용
```

### 실험 9-D: 결과 비교

비교 항목:

```text
optimized mask
aerial image
target intensity
background leakage
loss curve
spectrum difference
```

## 9.6 중요한 검증

Surrogate optimization 결과를 반드시 true synthetic correction으로 다시 검증한다.

```text
optimized mask
  ↓
FNO predicted correction으로 얻은 aerial
  ↓
true synthetic correction으로 얻은 aerial
  ↓
차이 확인
```

이 비교를 통해 neural surrogate가 optimization 중에 속고 있는지 확인한다.

## 9.7 배울 점

```text
optimizer는 surrogate의 약점을 이용할 수 있음
surrogate 결과가 좋아 보여도 true physics에서 깨질 수 있음
fallback validation이 필요함
```

---

# Phase 10. Optional: Active Learning Loop

## 10.1 목표

Optimization 중 surrogate가 자신 없어하는 mask를 찾아 training set에 추가한다.

공부용에서는 uncertainty를 간단하게 구현한다.

## 10.2 방법론

### 방법 A: Ensemble uncertainty

FNO를 여러 개 학습한다.

```text
FNO_1, FNO_2, FNO_3, FNO_4, FNO_5
```

예측 분산:

```math
U = Var(\Delta T_{\text{pred}})
```

### 방법 B: Distance-to-training-distribution

Mask feature vector를 만들고 training data와의 거리를 계산한다.

```text
feature:
  area fraction
  perimeter
  spectrum energy distribution
  minimum feature size
```

### 방법 C: True correction validation

Synthetic true correction을 oracle로 쓰고, error 큰 sample을 dataset에 추가한다.

## 10.3 구현 파일

```text
src/closed_loop/active_learning.py
experiments/07_closed_loop_inverse/active_learning_demo.py
```

## 10.4 배울 점

```text
surrogate는 학습 영역 안에서만 믿을 수 있음
inverse optimization은 distribution shift를 쉽게 발생시킴
active learning은 solver 데이터 비용을 줄이는 핵심 전략임
```

---

# 전체 실험 순서

## 권장 진행 순서

```text
Week 1:
  Phase 1 - Scalar diffraction
  Phase 2 - Coherent aerial image

Week 2:
  Phase 3 - Inverse aerial optimization
  Phase 4 - Partial coherence

Week 3:
  Phase 5 - Resist exposure + diffusion
  Phase 6 - PINN diffusion 비교

Week 4:
  Phase 7 - Synthetic 3D correction
  Phase 8 - FNO correction surrogate

Week 5:
  Phase 9 - Closed-loop surrogate inverse design
  Phase 10 - Active learning optional
```

논문용이 아니므로 기간은 자유롭게 늘려도 된다. 중요한 것은 각 단계에서 그림과 물리적 해석을 남기는 것이다.

---

# 최소 구현 버전

시간이 부족하면 아래만 구현해도 충분히 의미 있다.

```text
1. binary mask 생성
2. FFT로 diffraction spectrum 계산
3. circular pupil 적용
4. inverse FFT로 aerial image 계산
5. target / background loss 정의
6. PyTorch autograd로 mask optimization
7. exposure + diffusion + threshold 구현
8. FNO로 synthetic correction 학습
9. surrogate optimization 결과를 true correction과 비교
```

---

# 최소 성공 기준

공부용 프로젝트의 성공 기준은 성능 수치가 아니라 다음을 직접 확인하는 것이다.

```text
1. mask edge가 diffraction spectrum의 high frequency를 만든다.
2. NA cutoff가 aerial image를 blur시킨다.
3. target intensity를 높이면 side-lobe나 leakage가 생길 수 있다.
4. forbidden region loss를 넣으면 leakage가 줄지만 target quality와 trade-off가 생긴다.
5. diffusion은 resist edge를 blur시킨다.
6. threshold 아래 leakage는 resist에서 사라질 수 있다.
7. phase correction은 aerial image에 큰 영향을 줄 수 있다.
8. FNO는 synthetic 3D correction operator를 근사할 수 있다.
9. surrogate로 최적화한 mask는 true correction에서 반드시 재검증해야 한다.
```

---

# 주요 Metric 정의

## Aerial image metric

### Target mean intensity

```math
M_T = \frac{1}{|T|}\sum_{(x,y)\in T} I(x,y)
```

### Background leakage

```math
M_B = \frac{1}{|B|}\sum_{(x,y)\in B} I(x,y)
```

### Peak side-lobe ratio

```math
PSLR = \frac{\max_{(x,y)\in B} I(x,y)}{\max_{(x,y)\in T} I(x,y)}
```

### Contrast

```math
C = \frac{I_{\max} - I_{\min}}{I_{\max} + I_{\min}}
```

### NILS-like metric

```math
NILS = w \frac{|\nabla I|}{I}
```

정확한 lithography NILS와는 차이가 있을 수 있지만, 공부용으로 edge slope를 이해하는 데 충분하다.

---

## Resist metric

### Thresholded area

```math
A_R = \sum_{x,y} \mathbf{1}[A(x,y) > A_{th}]
```

### Soft contour error

```math
L_R =
\| \sigma(\beta(A-A_{th})) - R_{\text{target}} \|^2
```

### CD-like width

line-space 또는 contact hole에서 threshold contour 폭을 측정한다.

---

# 각 방법론을 왜 쓰는가

## Fourier optics

마스크가 만드는 회절과 projection lens의 filtering을 가장 직접적으로 이해할 수 있다.

## PyTorch autograd

Aerial image 계산 전체를 differentiable하게 만들어 mask를 gradient descent로 직접 업데이트할 수 있다.

## Partial coherence source integration

실제 lithography에서 illumination이 aerial image에 미치는 영향을 이해할 수 있다.

## Finite difference diffusion

Reaction-diffusion PDE의 기본 구조와 안정 조건을 이해할 수 있다.

## FFT heat kernel diffusion

Diffusion equation의 analytic Fourier-domain solution을 이해할 수 있다.

## PINN

PDE residual, initial condition, boundary condition을 loss로 학습하는 개념을 이해할 수 있다.

## FNO

Grid-to-grid operator mapping, Fourier mode truncation, long-range interaction approximation을 이해할 수 있다.

## DeepONet

입력 함수에서 출력 함수로 가는 operator learning 개념을 이해할 수 있다.

## Closed-loop inverse design

Neural surrogate가 단순 예측 모델이 아니라 optimization loop 안에서 어떻게 사용되는지 이해할 수 있다.

---

# 최종 산출물

최종적으로 다음 파일과 그림을 남긴다.

```text
outputs/figures/
  phase1_mask_spectrum.png
  phase2_aerial_NA_sweep.png
  phase3_inverse_before_after.png
  phase4_source_shape_comparison.png
  phase5_resist_diffusion_sweep.png
  phase6_pinn_vs_fd.png
  phase8_fno_prediction.png
  phase9_surrogate_inverse_validation.png

outputs/logs/
  inverse_aerial_loss.csv
  fno_training_log.csv
  surrogate_optimization_log.csv

outputs/datasets/
  synthetic_3d_correction_train.npz
  synthetic_3d_correction_test.npz

outputs/checkpoints/
  fno_correction.pt
  deeponet_correction.pt
```

---

# 최종 데모 구성

최종 데모는 하나의 notebook 또는 script로 구성한다.

```text
demo_full_pipeline.ipynb
```

구성:

```text
1. target pattern 정의
2. initial mask 생성
3. physics-only aerial image 계산
4. inverse optimization 수행
5. optimized mask 확인
6. resist diffusion / threshold 적용
7. synthetic 3D correction 적용
8. FNO surrogate correction 적용
9. true correction vs surrogate correction 비교
10. 결과 metric table 출력
```

---

# 권장 학습 순서

이 프로젝트를 공부용으로 제대로 이해하려면 다음 순서로 개념을 정리한다.

```text
1. Fourier transform과 diffraction
2. Pupil function과 NA
3. Coherent imaging
4. Partial coherent imaging
5. Inverse problem과 gradient descent
6. Total variation regularization
7. Reaction-diffusion equation
8. Finite difference stability
9. PINN residual loss
10. Neural operator
11. FNO
12. DeepONet
13. Surrogate-assisted optimization
14. Active learning
```

---

# 중요한 주의점

## 1. 이것은 실제 lithography simulator가 아니다

초기 구현은 scalar diffraction 기반의 공부용 모델이다. 실제 EUV/DUV lithography는 vector Maxwell, mask topography, polarization, multilayer, flare, aberration, resist chemistry 등 훨씬 복잡하다.

## 2. Neural operator를 먼저 믿지 않는다

항상 다음 순서를 지킨다.

```text
physics 구현
  ↓
physics 결과 이해
  ↓
neural surrogate 학습
  ↓
surrogate 결과 검증
  ↓
optimization에 투입
```

## 3. 완벽한 localization은 불가능하다

NA 제한과 diffraction 때문에 target만 밝히고 주변을 완전히 0으로 만드는 것은 일반적으로 불가능하다. 목표는 다음과 같이 잡는다.

```text
target region: threshold above
background region: threshold below
edge region: high slope
process variation: robust
```

## 4. 최적화 결과는 항상 물리적으로 해석한다

Optimized mask가 이상한 pattern을 만들면 실패가 아니라 학습 기회다.

확인할 질문:

```text
왜 이런 diffraction order가 생겼는가?
왜 side-lobe가 생겼는가?
왜 forbidden region leakage가 줄었는가?
왜 mask가 noisy해졌는가?
왜 binary mask가 continuous mask보다 성능이 낮은가?
```

---

# 최종 요약

이 프로젝트의 핵심은 다음 한 문장이다.

```text
Fourier optics와 reaction-diffusion을 직접 구현해서 lithography 물리를 이해하고,
그 위에 FNO / DeepONet / PINN을 보정 또는 가속 블록으로 붙여 neural surrogate의 역할과 한계를 공부한다.
```

가장 중요한 구현 순서는 다음이다.

```text
1. mask → diffraction spectrum
2. spectrum → pupil filtering → aerial image
3. aerial loss → inverse mask optimization
4. aerial → exposure → diffusion → resist
5. diffusion PDE → PINN 비교
6. thin-mask → synthetic 3D correction
7. correction operator → FNO / DeepONet
8. surrogate-assisted inverse design
9. true physics validation
```
