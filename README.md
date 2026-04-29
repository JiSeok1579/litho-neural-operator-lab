# litho-neural-operator-lab

> A study-purpose lithography simulator built around Fourier optics and
> reaction-diffusion, on top of which Neural Operators (FNO / DeepONet) and
> PINNs are attached as correction or acceleration blocks. The goal is **not**
> SOTA performance — it is to understand the physics first and then learn
> where neural surrogates help and where they break.

## Status

Phases **1 – 9 are complete**. Phase 10 (active learning) is deliberately
left as future work. A separate **PEB submodule** under
[`reaction_diffusion_peb/`](./reaction_diffusion_peb) has its own plan
and is in `plan-only` state.

- [PROGRESS.md](./PROGRESS.md) — running implementation log
  (What / How / Why / Next per change, plus §H wrap-up + reopen plan).
- 132 / 132 tests green at the wrap-up commit.

**Headline result (Phase 9):** an FNO surrogate that scores 15.5 % test
complex-relative error in Phase 8 produces an optimized mask whose
predicted aerial misleads the inverse-design optimizer — re-imaging
the same mask through the true physics gives an **11 % over-shoot** in
target intensity and a 5× worse target loss than the surrogate
believed.

---

## What this lab demonstrates, end to end

```text
mask
   │  Phase 1   FFT (centered)                              ✅
   ▼
diffraction spectrum
   │  Phase 2   pupil filtering, NA cutoff                   ✅
   ▼
coherent aerial image
   │  Phase 3   differentiable inverse mask design           ✅
   │  Phase 4   partial coherence (Hopkins integral)         ✅
   ▼
aerial image (under arbitrary illumination)
   │  Phase 5   exposure + diffusion + threshold             ✅
   ▼
resist contour
   │  Phase 6   PINN benchmark vs FD / FFT                   ✅
   ▼
synthetic 3D mask correction
   │  Phase 7   closed-form correction operator + dataset    ✅
   │  Phase 8   FNO surrogate (35x over baseline)            ✅
   │  Phase 9   closed-loop inverse + surrogate validation   ✅
```

The full step-by-step study plan lives in
[`litho_neural_operator_study_plan.md`](./litho_neural_operator_study_plan.md)
(written in Korean — the source plan, kept verbatim).

---

## Phase guide

Click a phase to read its dedicated doc (goal · key files · verified
numbers · how to run · takeaway).

| Phase | Title | Status | Doc |
|---|---|---|---|
| 0 | Environment and project scaffold | ✅ | [docs/phase0](./docs/phase0_environment_and_scaffold.md) |
| 1 | Scalar diffraction | ✅ | [docs/phase1](./docs/phase1_scalar_diffraction.md) |
| 2 | Coherent aerial imaging | ✅ | [docs/phase2](./docs/phase2_coherent_aerial.md) |
| 3 | Inverse aerial optimization | ✅ | [docs/phase3](./docs/phase3_inverse_aerial.md) |
| 4 | Partial coherence (Hopkins source integration) | ✅ | [docs/phase4](./docs/phase4_partial_coherence.md) |
| 5 | Resist exposure + diffusion + threshold | ✅ | [docs/phase5](./docs/phase5_resist_diffusion.md) |
| 6 | PINN for the 2D heat equation | ✅ | [docs/phase6](./docs/phase6_pinn_diffusion.md) |
| 7 | Synthetic 3D mask correction dataset | ✅ | [docs/phase7](./docs/phase7_synthetic_3d_correction.md) |
| 8 | FNO 2D correction surrogate | ✅ | [docs/phase8](./docs/phase8_fno_surrogate.md) |
| 9 | Closed-loop surrogate-assisted inverse design | ✅ | [docs/phase9](./docs/phase9_closed_loop_surrogate.md) |
| 10 | Active learning | 💤 deferred | [PROGRESS.md §H](./PROGRESS.md) |
| — | PEB submodule (`reaction_diffusion_peb/`) | plan only | [docs/peb_submodule](./docs/peb_submodule.md) |

---

## Repository layout

```text
litho-neural-operator-lab/
├── litho_neural_operator_study_plan.md   # source plan (Korean, read-only)
├── README.md                             # this file
├── PROGRESS.md                           # running implementation log
├── docs/                                 # per-phase doc index
│   ├── phase0_environment_and_scaffold.md
│   ├── phase1_scalar_diffraction.md
│   ├── phase2_coherent_aerial.md
│   ├── phase3_inverse_aerial.md
│   ├── phase4_partial_coherence.md
│   ├── phase5_resist_diffusion.md
│   ├── phase6_pinn_diffusion.md
│   ├── phase7_synthetic_3d_correction.md
│   ├── phase8_fno_surrogate.md
│   ├── phase9_closed_loop_surrogate.md
│   └── peb_submodule.md
├── requirements.txt
├── .gitignore
│
├── configs/                              # Hydra/OmegaConf YAML configs
│   ├── scalar_imaging.yaml
│   ├── inverse_aerial.yaml
│   ├── partial_coherence.yaml
│   ├── resist.yaml
│   ├── pinn.yaml
│   └── fno.yaml
│
├── src/
│   ├── common/          # grid, fft_utils, visualization, metrics, io
│   ├── mask/            # patterns, transmission, constraints
│   ├── optics/          # scalar_diffraction, pupil, coherent / partial imaging
│   ├── inverse/         # losses, optimize_mask, regularizers
│   ├── resist/          # exposure, diffusion (FD / FFT), reaction-diffusion, threshold
│   ├── pinn/            # pinn_base, pinn_diffusion, pinn_reaction_diffusion
│   ├── neural_operator/ # fno2d, deeponet, datasets, train_*
│   └── closed_loop/     # surrogate_optimizer, active_learning
│
├── experiments/         # phase-by-phase runnable scripts
│   ├── 01_scalar_diffraction/
│   ├── 02_inverse_aerial/
│   ├── 03_partial_coherence/
│   ├── 04_resist_diffusion/
│   ├── 05_pinn_diffusion/
│   ├── 06_fno_correction/
│   └── 07_closed_loop_inverse/
│
├── outputs/             # experiment artifacts (git-ignored)
│   ├── figures/
│   ├── checkpoints/
│   ├── logs/
│   └── datasets/
│
├── notebooks/           # exploration + final demo notebook
├── tests/               # pytest unit tests (132 / 132 green)
└── reaction_diffusion_peb/   # separate PEB submodule (plan only)
```

---

## Install

```bash
# 1. virtual environment
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip wheel setuptools

# 2. PyTorch — pick the CUDA index that matches your driver / hardware.
#    See https://pytorch.org/get-started/locally/ for the right index URL.
pip install torch torchvision

# 3. the rest of the stack
pip install -r requirements.txt

# 4. sanity check
python -c "import torch; print(torch.__version__, torch.cuda.is_available()); \
x = torch.randn(1024, 1024); print((x @ x).mean().item())"

# 5. run the test suite
pytest tests/ -q
```

---

## Working rules

- **Every code change updates `PROGRESS.md`.** For each unit of work
  record *What* (which module / file), *How* (which formula or
  algorithm), *Why* (the physical / mathematical / practical reason),
  and *Next* (the very next step).
- Each phase ships as a feature branch `phase-N-<short-name>` and
  lands on `main` via a PR with auto-merge. No work goes directly
  to `main`.
- Per-phase artifacts (figures / checkpoints / datasets) live under
  `outputs/` with the phase number as a filename prefix
  (e.g. `phase2_aerial_NA_sweep.png`).
- Experiment configuration goes into `configs/*.yaml`, never
  hard-coded.
- For a typical 16 GB-class GPU, grid 256² × FNO width 64 ×
  Fourier modes 16 is a safe upper bound. Beyond that, switch to
  mixed precision (`bf16`).

---

## License / sourcing

This is a personal study repository. It does not borrow datasets or
SOTA code from external sources. The Phase 7 3D correction is
**synthetic**, so there are no licensing concerns.
