# Reaction-Diffusion / PEB submodule — implementation plan

## 0. Purpose

This document is the implementation plan for an isolated submodule that
studies **Reaction-Diffusion / Post-Exposure Bake (PEB)** physics
inside the existing `litho-neural-operator-lab` repository, without
disturbing the main pipeline.

The single most important methodological choice is:

```text
With no simulation dataset on hand yet, use PINN before DeepONet / FNO.
But never trust the PINN alone — always benchmark it against an FD or
FFT physics solver. Once enough FD / FFT / PINN results have been
accumulated as paired data, only then introduce DeepONet / FNO as
operator surrogates.
```

The end-to-end ordering is therefore:

```text
Physics equation
  ↓
FD / FFT baseline solver
  ↓
PINN comparison
  ↓
Simulation dataset accumulation
  ↓
DeepONet / FNO surrogate (downstream / optional)
```

---

## 1. Core principles

```text
1.  Do not modify the existing main project.
2.  Operate entirely inside reaction_diffusion_peb/.
3.  Do not couple to the main repo's mask / optics / inverse / FNO modules
    in the early phases.
4.  Use synthetic aerial images as inputs at first.
5.  Start with diffusion-only.
6.  Build the FD / FFT baselines first (or alongside) the PINN.
7.  Use PINN before DeepONet / FNO.
8.  Treat DeepONet / FNO as a downstream surrogate after data exists.
9.  For the stiff acid–quencher reaction (`kq`), start at small values.
10. Activate every parameter incrementally; never enable everything at once.
```

---

## 2. Relationship to the main project

The full main-project pipeline is:

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

This submodule covers only the resist-internal part:

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

In short, the focus here is **resist chemistry inside the PEB**, not
mask optics. The main project already touches the same equations at a
study-grade level in its Phase 5; this submodule goes deeper:
physically-meaningful units (mol/dm³, nm²/s, Celsius), Arrhenius
temperature dependence, deprotection kinetics, stiff quencher reaction,
and PINN-first methodology.

---

## 3. Folder layout

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

## 4. Isolation from the rest of the codebase

Early on, every module under `reaction_diffusion_peb/` should be
self-contained. Do **not** import from the main repo's:

```text
src/mask/
src/optics/
src/inverse/
src/neural_operator/
src/closed_loop/
```

Initial inputs are synthetic exposure maps generated inside the
submodule. When the time comes to plug back into the main pipeline,
do it through file-based interfaces:

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

Recommended I/O:

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

## 5. Why PINN before DeepONet / FNO

The current state is:

```text
- No simulation dataset.
- A well-defined family of PDEs.
- A reasonable parameter range.
```

DeepONet / FNO cannot be trained immediately. Both are **operator-
learning** models that need an existing supervised dataset:

```text
input:   I(x, y), H0(x, y), DH0, kdep, kq, kloss, T, t_PEB
output:  H(x, y, t_final), Q(x, y, t_final), P(x, y, t_final), R(x, y)
```

We do not have these (input, output) pairs yet, so the only path is to
build a physics solver and a PINN first, accumulate paired data, and
only then move to operator learning.

PINN is usable right away because:

```text
1. The training signal is the PDE residual (no data needed).
2. Initial- and boundary-condition penalties supplement it.
3. It is well-suited to learning diffusion / reaction-diffusion physics.
4. PINN solutions become a natural seed for DeepONet / FNO datasets.
```

But PINN alone is hard to validate. Hence the mandatory baseline:

```text
FD solver
  vs
FFT heat-kernel solver
  vs
PINN solver
```

For diffusion-only the FFT heat-kernel solution is essentially exact
and serves as the truth reference.

---

## 6. Recommended phase ordering

```text
Phase 1:  Synthetic aerial image and exposure → initial acid H0
Phase 2:  Diffusion-only FD and FFT baselines, FD vs FFT comparison
Phase 3:  PINN diffusion solver, FD / FFT / PINN three-way comparison
Phase 4:  Add acid loss, FD vs PINN comparison
Phase 5:  Add deprotection, build the H → P kinetic model
Phase 6:  Add temperature dependence (Arrhenius)
Phase 7:  Add quencher reaction, starting at small kq
Phase 8:  Integrate everything into one full reaction-diffusion model
Phase 9:  Save FD / PINN simulation results as datasets
Phase 10: Train DeepONet / FNO operator surrogates on those datasets
Phase 11: Advanced stochastic / Petersen / z-axis extensions
```

---

## 7. Inputs and outputs

### 7.1 Inputs

Initially, generate synthetic aerial images inside the submodule.
Candidates:

```text
1. Gaussian spot
2. Line-space sinusoidal exposure
3. Contact-hole-like 2D Gaussian array
4. Two-spot interference-like exposure
5. User-provided .npy aerial image
```

Input variables:

```text
I(x, y)         : normalized aerial / exposure intensity map
dose            : normalized exposure dose
grid_spacing_nm : spatial grid spacing
```

### 7.2 Outputs

Each experiment writes:

```text
H0(x, y)            : initial acid concentration
H(x, y, t)          : acid concentration after PEB
Q(x, y, t)          : quencher concentration (when enabled)
P(x, y, t)          : deprotected fraction (when enabled)
R(x, y)             : soft-thresholded resist latent image
metrics.csv         : concentration, contour, and error metrics
figures             : before / after visualization
```

PINN comparison experiments additionally write:

```text
pinn_loss_curve.csv
pde_residual_map.npy
fd_vs_pinn_error.npy
fd_fft_pinn_comparison.png
```

The dataset-generation phase writes:

```text
outputs/datasets/
  diffusion_dataset.npz
  acid_loss_dataset.npz
  reaction_diffusion_dataset.npz
  metadata.json
```

---

## 8. Variable nomenclature

| Symbol | Code name | Meaning |
|---|---|---|
| `I(x, y)` | `I` | normalized aerial / exposure intensity |
| `H(x, y, t)` | `H` | acid concentration |
| `H0(x, y)` | `H0` | initial acid concentration after exposure |
| `Q(x, y, t)` | `Q` | quencher concentration |
| `P(x, y, t)` | `P` | deprotected fraction |
| `R(x, y)` | `R` | soft-thresholded resist latent image |
| `D_H` | `DH` | acid diffusion coefficient |
| `D_Q` | `DQ` | quencher diffusion coefficient |
| `k_q` | `kq` | acid–quencher neutralization rate |
| `k_loss` | `kloss` | acid loss rate |
| `k_dep` | `kdep` | deprotection rate |
| `T` | `temperature_c` | PEB temperature (Celsius) |
| `t_PEB` | `peb_time_s` | PEB time |
| `Lz` | `film_thickness_nm` | resist film thickness |
| `Ea` | `activation_energy_kj_mol` | activation energy |

Naming caveat:

```text
Never use the bare letter D — it is ambiguous between "dose" and
"diffusion coefficient".

  dose                            -> dose
  acid diffusion coefficient      -> DH
  quencher diffusion coefficient  -> DQ
  deprotected fraction            -> P
```

---

## 9. Parameter scope

| Group | Parameter | Range | Unit | Role |
|---|---|---:|---|---|
| Process | PEB temperature `temperature_c` | 80 – 120 (MOR preset 125) | °C | Arrhenius-correct reaction rates |
| Process | PEB time `peb_time_s` | 60 – 90 | s | Simulation horizon and PINN time-domain upper bound |
| Process | Temperature uniformity `temperature_uniformity_c` | ±0.02 | °C | Stochastic / CDU proxy |
| Diffusion | Initial acid diffusion `DH0_nm2_s` | 0.3 – 1.5 | nm²/s | Acid diffusion strength |
| Diffusion | Petersen acceleration `petersen_alpha` | 0.5 – 3.0 | – | Nonlinear diffusion modulation |
| Diffusion | Quencher diffusion `DQ_nm2_s` | ≤ 0.1 × `DH0` | nm²/s | Quencher diffusion |
| Reaction | Deprotection rate `kdep_s_inv` | 0.01 – 0.5 | s⁻¹ | Generates the deprotected fraction `P` |
| Reaction | Neutralization rate `kq_s_inv` | 100 – 1000 | s⁻¹ | Acid–quencher reaction (stiff) |
| Reaction | Acid loss rate `kloss_s_inv` | 0.001 – 0.05 | s⁻¹ | Acid trap / decay |
| Reaction | Activation energy `activation_energy_kj_mol` | ≈ 100 | kJ/mol | Temperature-dependent rate correction |
| Resist | Film thickness `film_thickness_nm` | < 30 | nm | Optional z-axis domain |
| Resist | Particle size `particle_size_nm` | 0.5 (MOR) – 1.0 (CAR) | nm | Molecular blur / resolution scale |
| Resist | Initial acid `Hmax_mol_dm3` | 0.1 – 0.3 | mol/dm³ | Exposure-linked acid generation |
| Numerical | Grid `grid_spacing_nm` | 0.5 – 1.0 | nm | Spatial discretization |
| Numerical | Ensemble `ensemble_runs` | 10 – 20 | runs | Stochastic variation analysis |

### 9.1 Per-parameter readiness

| Parameter group | Status | Phase |
|---|---|---|
| PEB temperature / time | available | Phase 2, 6 |
| Temperature uniformity | later | Phase 11 |
| `DH0` acid diffusion | available | Phase 2 |
| Petersen `alpha` | later | Phase 11 |
| Quencher diffusion `DQ` | available | Phase 7 |
| `kdep` | available | Phase 5 |
| `kq` | available with care | Phase 7 |
| `kloss` | available | Phase 4 |
| `Ea` | available | Phase 6 |
| Film thickness `Lz` | later | Phase 11 / optional z-axis PINN |
| Molecular size | later | Phase 11 |
| Initial acid `H0` / `Hmax` | available | Phase 1 |
| 0.5 – 1.0 nm grid | available | all PDE phases |
| 10 – 20 ensemble runs | later | Phase 11 |

### 9.2 Summary

```text
Available immediately:
  PEB time, DH0, H0/Hmax, grid, kloss, kdep, DQ, temperature, Ea

Available with care:
  kq = 100 – 1000 s^-1
    -> Stiff. Start at small values.

Defer to later phases:
  Petersen alpha, temperature uniformity, molecular size,
  ensemble runs, Lz / z-axis

DeepONet / FNO:
  Available only after FD / PINN datasets exist.
```

---

# Phase 1. Synthetic aerial + exposure

## Goal

Build the resist-side study without depending on optics or mask
simulation. Generate a synthetic exposure map and convert it into an
initial acid concentration field.

## Files

```text
reaction_diffusion_peb/src/synthetic_aerial.py
reaction_diffusion_peb/src/exposure.py

reaction_diffusion_peb/experiments/01_synthetic_aerial/run_gaussian_spot.py
reaction_diffusion_peb/experiments/01_synthetic_aerial/run_line_space.py
```

## Synthetic aerial helpers

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

## Exposure model

```math
H_0(x, y) = H_{max} \left( 1 - \exp(-\eta \cdot \text{dose} \cdot I(x, y)) \right)
```

```python
def acid_generation(I, dose=1.0, eta=1.0, Hmax=0.2):
    # I    : normalized aerial intensity, range [0, 1]
    # dose : normalized exposure dose
    # eta  : acid generation efficiency
    # Hmax : maximum acid concentration [mol/dm^3]
    ...
```

## Parameters

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

## Things to verify

```text
- I(x, y) is normalized to [0, 1].
- Increasing dose increases H0 monotonically.
- H0 never exceeds Hmax.
- I = 0 implies H0 ≈ 0.
```

---

# Phase 2. Diffusion-only FD / FFT baseline

## Goal

Implement the basic PEB acid diffusion in two ways. Both serve as the
reference solution against which the Phase-3 PINN will be evaluated.

## PDE

```math
\frac{\partial H}{\partial t} = D_H \nabla^2 H
```

## Files

```text
reaction_diffusion_peb/src/diffusion_fd.py
reaction_diffusion_peb/src/diffusion_fft.py

reaction_diffusion_peb/experiments/02_diffusion_baseline/run_diffusion_fd.py
reaction_diffusion_peb/experiments/02_diffusion_baseline/run_diffusion_fft.py
reaction_diffusion_peb/experiments/02_diffusion_baseline/compare_fd_fft.py
```

## FD update

```math
H^{n+1} = H^n + \Delta t\, D_H \nabla^2 H^n
```

CFL stability:

```math
\Delta t \leq \frac{\Delta x^2}{4 D_H}
```

## FFT heat-kernel solution

```math
\hat{H}(f_x, f_y, t) = \hat{H}_0(f_x, f_y) \exp\left[ -4 \pi^2 D_H (f_x^2 + f_y^2) t \right]
```

## Parameters

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

## Things to verify

```text
- H smooths with time.
- Larger DH gives more blur.
- FD and FFT agree on the same initial condition.
- Without loss terms, total acid mass is conserved.
```

---

# Phase 3. PINN diffusion baseline

## Goal

Train a PINN against the same diffusion PDE before introducing
DeepONet / FNO. Always cross-check PINN against FD and FFT.

## Target PDE

```math
\frac{\partial H}{\partial t} - D_H \nabla^2 H = 0
```

## PINN I/O

Input:

```text
x, y, t
```

Output:

```text
H(x, y, t)
```

## Loss

```math
L = L_{PDE} + L_{IC} + L_{BC}
```

PDE residual:

```math
r(x, y, t) = \frac{\partial H}{\partial t} - D_H \left( \frac{\partial^2 H}{\partial x^2} + \frac{\partial^2 H}{\partial y^2} \right)
```

```math
L_{PDE} = \|r(x, y, t)\|^2
```

Initial condition:

```math
L_{IC} = \|H(x, y, 0) - H_0(x, y)\|^2
```

Boundary condition (or periodic):

```math
L_{BC} = \|\nabla H \cdot n\|^2
```

## Files

```text
reaction_diffusion_peb/src/pinn_base.py
reaction_diffusion_peb/src/pinn_diffusion.py

reaction_diffusion_peb/experiments/03_pinn_diffusion/run_pinn_diffusion.py
reaction_diffusion_peb/experiments/03_pinn_diffusion/compare_fd_fft_pinn.py
```

## Parameters

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

## Things to verify

```text
- PINN reproduces the FD / FFT diffusion-only result.
- Sharp initial conditions are difficult for the PINN.
- PDE residual decreases over training.
- IC error and PDE residual decrease together.
- The PINN is a PDE-fitting tool, not a fast solver.
```

---

# Phase 4. Acid loss + PINN comparison

## Goal

Add the natural decay / trapping term to the diffusion equation.

## PDE

```math
\frac{\partial H}{\partial t} = D_H \nabla^2 H - k_{loss} H
```

PINN residual:

```math
r(x, y, t) = \frac{\partial H}{\partial t} - D_H \nabla^2 H + k_{loss} H
```

## Files

```text
reaction_diffusion_peb/src/reaction_diffusion.py
reaction_diffusion_peb/src/pinn_reaction_diffusion.py

reaction_diffusion_peb/experiments/04_acid_loss/run_acid_loss_fd.py
reaction_diffusion_peb/experiments/04_acid_loss/run_acid_loss_pinn.py
reaction_diffusion_peb/experiments/04_acid_loss/compare_fd_pinn.py
```

## Parameters

```yaml
reaction:
  kloss_s_inv: 0.005
```

Sweep:

```yaml
sweep:
  kloss_s_inv: [0.001, 0.005, 0.01, 0.05]
```

## Things to verify

```text
- kloss = 0 reproduces diffusion-only.
- Larger kloss reduces total acid mass.
- FD and PINN agree on H(x, y, t_final).
```

---

# Phase 5. Deprotection

## Goal

Add the acid-driven deprotection step.

## Model

The deprotected fraction `P` evolves as:

```math
\frac{\partial P}{\partial t} = k_{dep} H (1 - P)
```

with:

```text
P = 0  : fully protected
P = 1  : fully deprotected
```

## Files

```text
reaction_diffusion_peb/src/deprotection.py

reaction_diffusion_peb/experiments/05_deprotection/run_deprotection_fd.py
reaction_diffusion_peb/experiments/05_deprotection/run_deprotection_pinn.py
```

## Parameters

```yaml
reaction:
  kdep_s_inv: 0.05
```

Sweep:

```yaml
sweep:
  kdep_s_inv: [0.01, 0.05, 0.1, 0.5]
```

## Things to verify

```text
- High-H regions accumulate P faster.
- Longer PEB increases P.
- Larger kdep widens the post-threshold contour.
- P stays in [0, 1].
```

---

# Phase 6. Temperature / Arrhenius PEB

## Goal

Make reaction rates temperature-dependent.

## Arrhenius correction

For a reference rate `k_ref` defined at `T_ref`:

```math
k(T) = k_{ref} \exp\left[ -\frac{E_a}{R} \left( \frac{1}{T_K} - \frac{1}{T_{ref,K}} \right) \right]
```

where:

```text
T_K = temperature_c + 273.15
R   = 8.314 J / (mol K)
Ea  = activation energy [J / mol]
```

## Files

```text
reaction_diffusion_peb/src/arrhenius.py

reaction_diffusion_peb/experiments/06_temperature_peb/run_temperature_sweep.py
reaction_diffusion_peb/experiments/06_temperature_peb/run_time_sweep.py
```

## Parameters

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

## Things to verify

```text
- Higher T raises kdep, kq, and kloss.
- Temperature sweeps shift the P profile and the threshold contour.
- Reaction rates are sensitive even to small temperature changes.
- The 125 °C MOR preset behaves consistently.
```

---

# Phase 7. Quencher reaction

## Goal

Add the acid–quencher neutralization. From here onwards stiff PDE
behavior is possible.

## PDE

```math
\frac{\partial H}{\partial t} = D_H \nabla^2 H - k_q H Q - k_{loss} H
```

```math
\frac{\partial Q}{\partial t} = D_Q \nabla^2 Q - k_q H Q
```

## PINN residuals

```math
r_H = \frac{\partial H}{\partial t} - D_H \nabla^2 H + k_q H Q + k_{loss} H
```

```math
r_Q = \frac{\partial Q}{\partial t} - D_Q \nabla^2 Q + k_q H Q
```

## Files

```text
reaction_diffusion_peb/src/reaction_diffusion.py
reaction_diffusion_peb/src/pinn_reaction_diffusion.py

reaction_diffusion_peb/experiments/07_quencher_reaction/run_quencher_reaction_safe.py
reaction_diffusion_peb/experiments/07_quencher_reaction/run_quencher_reaction_stiff.py
```

## Parameters

Start safe, then escalate.

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
  kq_s_inv_safe:   [1, 5, 10]
  kq_s_inv_target: [100, 300, 1000]
```

Caveat:

```text
kq in the 100 – 1000 s^-1 band is stiff.
Do not start there. First confirm correctness at kq = 1, 5, 10 s^-1,
then introduce one of:

  - much smaller dt
  - operator splitting
  - semi-implicit / implicit reaction update
  - adaptive time-stepping

before pushing kq up. PINNs also struggle in the stiff regime — do
not insist on PINN success at large kq before the FD / semi-implicit
baseline is stable.
```

## Things to verify

```text
- Q is consumed quickly where H is high.
- Sufficient Q strongly suppresses H.
- Smaller DQ keeps Q tighter than the acid.
- Larger kq degrades numerical stability.
- The PINN residual fit becomes harder as kq grows.
```

---

# Phase 8. Full reaction-diffusion

## Goal

Combine every term added so far into a single integrated model.

## Final PDE system

```math
\frac{\partial H}{\partial t} = \nabla \cdot (D_H \nabla H) - k_q H Q - k_{loss} H
```

```math
\frac{\partial Q}{\partial t} = \nabla \cdot (D_Q \nabla Q) - k_q H Q
```

```math
\frac{\partial P}{\partial t} = k_{dep} H (1 - P)
```

## Files

```text
reaction_diffusion_peb/src/reaction_diffusion.py

reaction_diffusion_peb/experiments/08_full_reaction_diffusion/run_full_model.py
```

## Parameters

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

## Things to verify

Disabling each term should reproduce the corresponding earlier phase:

```text
- Diffusion-only          → Phase 2 result
- + kloss                 → Phase 4
- + kdep                  → Phase 5
- + Arrhenius temperature → Phase 6
- + quencher              → Phase 7
```

---

# Phase 9. Dataset generation

## Goal

Accumulate FD / FFT / PINN runs as a dataset that DeepONet / FNO can
learn from. Without this phase, Phase 10 cannot start.

## Dataset inputs

```text
I(x, y)
H0(x, y)
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

## Dataset outputs

```text
H(x, y, t_final)
Q(x, y, t_final)
P(x, y, t_final)
R(x, y)
```

## Files

```text
reaction_diffusion_peb/src/dataset_builder.py

reaction_diffusion_peb/experiments/09_dataset_generation/generate_fd_dataset.py
reaction_diffusion_peb/experiments/09_dataset_generation/generate_pinn_dataset.py
reaction_diffusion_peb/experiments/09_dataset_generation/validate_dataset.py
```

## Storage format

```text
outputs/datasets/
  diffusion_dataset.npz
  acid_loss_dataset.npz
  reaction_diffusion_dataset.npz
  metadata.json
```

Example metadata:

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

## Things to verify

```text
- Sample input / output shapes are consistent.
- Parameter ranges are recorded in metadata.
- Train / validation / test splits are easy.
- FD-generated and PINN-generated subsets are distinguishable.
- Stiff-kq samples are kept in a separate subset.
```

---

# Phase 10. DeepONet / FNO operator surrogate (optional)

## Goal

Once Phase-9 datasets exist, learn the PEB operator with DeepONet
and / or FNO. This is **not** a required step — it is the optional
follow-up to data accumulation.

## Operator to learn

```text
(I(x, y), parameters)  →  H(x, y, t_final), P(x, y, t_final), R(x, y)
```

or, more compactly:

```text
H0(x, y), DH0, kdep, kloss, T, t_PEB  →  P(x, y, t_final)
```

## DeepONet layout

```text
Branch input:
  H0 field encoding
  process parameters

Trunk input:
  coordinate (x, y), optionally t

Output:
  H(x, y, t), P(x, y, t)
```

## FNO layout

```text
Input channels:
  H0(x, y)
  parameter maps: DH0, kdep, kloss, T, t_PEB

Output channels:
  H_final(x, y)
  P_final(x, y)
  R(x, y)
```

## Files

```text
reaction_diffusion_peb/experiments/10_operator_learning_optional/train_deeponet.py
reaction_diffusion_peb/experiments/10_operator_learning_optional/train_fno.py
reaction_diffusion_peb/experiments/10_operator_learning_optional/evaluate_operator_surrogate.py
```

## Things to verify

```text
- Start only after Phase 9 produced data.
- Measure surrogate error against FD / PINN truth.
- Expect parameter-extrapolation degradation.
- Evaluate stiff-kq cases on a separate split.
```

---

# Phase 11. Advanced stochastic / Petersen / z-axis

## Goal

After the core model is stable, layer in the higher-order parameters.

## Petersen nonlinear diffusion

```math
D_H = D_{H0} \exp(\alpha P)
```

or:

```math
D_H = D_{H0} \exp(\alpha \cdot \text{dose\_field})
```

Naming caveat:

```text
The original "DH = DH0 exp(αD)" formulation is ambiguous: D could be
dose, diffusion coefficient, or deprotected fraction. Pick one and
spell it out:

  P            : deprotected fraction
  dose_field   : normalized local exposure dose
  DH           : acid diffusion coefficient
```

## Temperature uniformity

```yaml
stochastic:
  temperature_uniformity_c: 0.02
```

Per-run perturbation:

```text
temperature_c_run = temperature_c + Normal(0, temperature_uniformity_c)
```

## Molecular blur / particle size

```yaml
resist:
  particle_size_nm: 1.0
```

Usage options:

```text
1. Apply a Gaussian blur to the final P field.
2. Use it as a stochastic-noise scale.
3. Treat it as the minimum meaningful grid scale.
```

## z-axis / film thickness

Initial 2D model:

```text
H(x, y, t)
Q(x, y, t)
P(x, y, t)
```

Later 3D model:

```text
H(x, y, z, t)
Q(x, y, z, t)
P(x, y, z, t)
```

PINN with z-axis:

```text
input  : x, y, z, t
output : H, Q, P
domain : z in [0, film_thickness_nm]
```

## Advanced sweep

```yaml
advanced:
  petersen_alpha: [0.5, 1.0, 2.0, 3.0]
  temperature_uniformity_c: 0.02
  molecular_blur_nm: [0.5, 1.0]
  film_thickness_nm: [20, 30]
  ensemble_runs: [10, 20]
```

---

## 12. Reference configs

### 12.1 minimal_diffusion.yaml

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

### 12.2 pinn_diffusion.yaml

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

### 12.3 acid_loss.yaml

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

### 12.4 quencher_reaction.yaml

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

### 12.5 deprotection.yaml

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

### 12.6 temperature_peb.yaml

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

### 12.7 parameter_sweep.yaml

```yaml
sweep:
  peb_temperature_c: [80, 90, 100, 110, 120, 125]
  peb_time_s: [60, 75, 90]

  Hmax_mol_dm3: [0.1, 0.2, 0.3]

  DH0_nm2_s: [0.3, 0.8, 1.5]
  DQ_ratio: [0.05, 0.1]

  kloss_s_inv: [0.001, 0.005, 0.01, 0.05]
  kdep_s_inv: [0.01, 0.05, 0.1, 0.5]

  kq_s_inv_safe:   [1, 5, 10]
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

## 13. GitHub workflow

### 13.1 Do not touch the existing project

Disallowed:

```text
- Rewriting the main README.
- Mixing this code into the main src/ tree.
- Coupling to FNO / optics modules directly.
- Starting from DeepONet / FNO.
```

Allowed:

```text
- Adding the reaction_diffusion_peb/ folder.
- Adding reaction_diffusion_peb/{README,PLAN,PARAMETER_SCOPE}.md.
- Adding reaction_diffusion_peb/configs/ .
- Adding reaction_diffusion_peb/src/ .
- Adding reaction_diffusion_peb/experiments/ .
```

### 13.2 Branch name

```text
feature/reaction-diffusion-peb
```

or:

```text
peb-pinn-reaction-module
```

### 13.3 Recommended commit / PR cadence

Each phase becomes its own commit (and ideally its own PR with metrics
+ figures, mirroring the main project's workflow):

```text
Commit 1:  Add reaction_diffusion_peb workspace skeleton
Commit 2:  Add parameter scope and sweep configs
Commit 3:  Add synthetic aerial generation and exposure model
Commit 4:  Add diffusion-only FD / FFT baseline solvers
Commit 5:  Add PINN diffusion baseline and FD / FFT / PINN comparison
Commit 6:  Add acid loss model with FD / PINN comparison
Commit 7:  Add deprotection model
Commit 8:  Add Arrhenius temperature-dependent PEB
Commit 9:  Add quencher reaction with safe kq range
Commit 10: Add full reaction-diffusion demo
Commit 11: Add dataset generation from FD / PINN simulations
Commit 12: Add optional DeepONet / FNO surrogate plan or prototype
Commit 13: Add advanced stochastic / Petersen / z-axis plan
```

### 13.4 PR title

```text
Add local Reaction-Diffusion / PEB PINN-first study module
```

### 13.5 PR description template

```markdown
## Summary

This PR adds a local `reaction_diffusion_peb/` workspace for studying
post-exposure-bake resist physics independently from the main
lithography / neural-operator pipeline.

Because no simulation dataset exists yet, the module follows a
physics-first and PINN-first workflow:

1. Build FD / FFT baselines for diffusion.
2. Train PINNs against the same PDEs and compare with FD / FFT.
3. Add acid loss, deprotection, temperature dependence, and
   quencher reaction incrementally.
4. Generate reusable simulation datasets.
5. Keep DeepONet / FNO as optional downstream operator surrogates
   after data accumulation.

## Scope

- Keep the existing project structure unchanged.
- Add the local Reaction-Diffusion / PEB workspace.
- Use synthetic exposure maps first.
- Build FD / FFT physics baselines.
- Add PINN diffusion and reaction-diffusion experiments before
  DeepONet / FNO.
- Track process, diffusion, reaction, resist, and numerical parameter
  ranges.
- Separate safe `kq` values from stiff target `kq` values.

## Out of scope

- Mask simulation.
- Fourier optics.
- Full lithography pipeline integration.
- Production-grade stiff PDE solver.
- Real calibrated CAR / MOR fitting.
- DeepONet / FNO as the first modeling step.
```

---

## 14. Execution recipes

### 14.1 Environment

```bash
cd litho-neural-operator-lab/reaction_diffusion_peb
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 14.2 Synthetic aerial

```bash
python experiments/01_synthetic_aerial/run_gaussian_spot.py
```

### 14.3 Diffusion baseline

```bash
python experiments/02_diffusion_baseline/run_diffusion_fd.py
python experiments/02_diffusion_baseline/run_diffusion_fft.py
python experiments/02_diffusion_baseline/compare_fd_fft.py
```

### 14.4 PINN diffusion

```bash
python experiments/03_pinn_diffusion/run_pinn_diffusion.py
python experiments/03_pinn_diffusion/compare_fd_fft_pinn.py
```

### 14.5 Acid loss

```bash
python experiments/04_acid_loss/run_acid_loss_fd.py
python experiments/04_acid_loss/run_acid_loss_pinn.py
python experiments/04_acid_loss/compare_fd_pinn.py
```

### 14.6 Deprotection

```bash
python experiments/05_deprotection/run_deprotection_fd.py
```

### 14.7 Temperature sweep

```bash
python experiments/06_temperature_peb/run_temperature_sweep.py
```

### 14.8 Quencher reaction

```bash
python experiments/07_quencher_reaction/run_quencher_reaction_safe.py
```

### 14.9 Dataset generation

```bash
python experiments/09_dataset_generation/generate_fd_dataset.py
python experiments/09_dataset_generation/validate_dataset.py
```

---

## 15. Minimum success criteria

The submodule is considered "working" when:

```text
1.  Synthetic exposure maps can be generated.
2.  Initial acid H0 can be derived from an exposure map.
3.  FD diffusion-only PEB shows acid blur growing with time and DH.
4.  FFT diffusion and FD diffusion agree.
5.  PINN diffusion reproduces the FD / FFT baseline at study-grade level.
6.  PINN PDE residual, IC loss, and BC loss are tracked over training.
7.  Enabling acid loss reduces total acid mass.
8.  Acid-loss PINN is comparable with FD.
9.  Deprotection P grows where H is large.
10. Higher temperature speeds up reactions through Arrhenius correction.
11. Quencher reaction is stable inside the safe kq range.
12. The target kq range is kept in a separate subset until stiff handling
    is in place.
13. Soft thresholding produces a latent resist image.
14. FD / PINN results can be written out as a dataset.
15. DeepONet / FNO start only after the dataset is in place.
```

---

## 16. Out of scope (initially)

```text
1. 3D Maxwell / RCWA / FDTD.
2. Mask 3D effects.
3. Inverse mask optimization.
4. Treating DeepONet / FNO as the first model.
5. Calibrated CAR / MOR fitting.
6. Full 3D z-axis resist simulation.
7. Production-grade stochastic LER / LWR model.
8. OPC / ILT integration.
```

Open follow-ups (after the core is stable):

```text
1. DeepONet / FNO trained on the FD / PINN dataset.
2. Petersen nonlinear diffusion.
3. Temperature-uniformity ensembles.
4. Molecular blur.
5. z-axis PINN.
6. File-based hand-off with the main optics pipeline.
```

---

## 17. Final summary

The plan in one paragraph:

```text
Leave the existing GitHub project untouched. Add a new
reaction_diffusion_peb/ folder. Run Reaction-Diffusion / PEB
experiments inside that folder only. Because there is no data yet,
use PINN before DeepONet / FNO — but never rely on the PINN alone:
always benchmark it against an FD or FFT baseline. Ramp through:
diffusion-only -> PINN diffusion -> acid loss -> deprotection ->
temperature -> quencher -> full model. Once the FD / PINN results
have been saved as a dataset, only then move to DeepONet / FNO as a
downstream operator surrogate.
```

The first files to land are:

```text
reaction_diffusion_peb/
  README.md
  PLAN.md                          (this file)
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

The first concrete milestone:

```text
Gaussian synthetic aerial image
  -> initial acid H0
  -> 60 s PEB diffusion via FD and FFT
  -> PINN diffusion training
  -> FD / FFT / PINN before / after comparison
  -> figures, metrics, and loss curves saved to outputs/
```

Once that milestone is stable, layer in acid loss, deprotection,
temperature dependence, quencher reaction, dataset generation, and
finally the optional DeepONet / FNO surrogate.
