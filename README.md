# litho-neural-operator-lab

> A study-purpose lithography simulator built around Fourier optics and
> reaction-diffusion, on top of which Neural Operators (FNO / DeepONet) and
> PINNs are attached as correction or acceleration blocks. The goal is **not**
> SOTA performance — it is to understand the physics first and then learn
> where neural surrogates help and where they break.

The full step-by-step study plan lives in
[`litho_neural_operator_study_plan.md`](./litho_neural_operator_study_plan.md)
(written in Korean — this is the source plan and is intentionally kept as-is).
Implementation progress is tracked cumulatively in
[`PROGRESS.md`](./PROGRESS.md).

---

## Project rule — language

**All project documents and code comments are written in English.**
This includes `README.md`, `PROGRESS.md`, docstrings, inline comments, commit
messages, PR titles/bodies, and YAML config comments. The only Korean file is
the original study plan (`litho_neural_operator_study_plan.md`), kept verbatim
as the source of truth for the experiment design.

Why English: tooling output, external libraries, and any future collaborator /
reviewer all default to English. Mixing languages inside the codebase makes
greps and diffs harder to read.

---

## 1. Environment

| Item | Value |
|---|---|
| OS | Ubuntu 24.04.3 LTS |
| Python | 3.12.3 |
| GPU | NVIDIA RTX 5080 (Blackwell, sm_120, 16 GB VRAM) |
| CUDA driver | 580.x (CUDA 13 runtime compatible) |
| PyTorch | 2.11.0+cu128 (sm_120-compatible build) |

> ⚠️ **Blackwell caveat.** The default `pip install torch` pulls the cu126
> stable wheel, which only ships kernels up to sm_90. On RTX 5080 this fails
> at runtime with `no kernel image is available for execution`. You **must**
> install via the **cu128 index** (or cu130).

---

## 2. Install

```bash
# 1. virtual environment
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip wheel setuptools

# 2. PyTorch (cu128, Blackwell sm_120)
pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision

# 3. the rest of the stack
pip install -r requirements.txt

# 4. GPU sanity check
python -c "import torch; \
print(torch.__version__, torch.cuda.is_available(), \
torch.cuda.get_device_name(0) if torch.cuda.is_available() else None); \
x=torch.randn(1024,1024,device='cuda'); print((x@x).mean().item())"
```

---

## 3. Repository layout

```text
litho-neural-operator-lab/
├── litho_neural_operator_study_plan.md   # source plan (Korean, read-only)
├── README.md                             # this file
├── PROGRESS.md                           # running implementation log
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
└── tests/               # pytest unit tests
```

---

## 4. Phase roadmap

| Phase | Topic | Core physics | Key artifact |
|---|---|---|---|
| 1 | Scalar diffraction | Fourier optics | diffraction spectrum, aerial image |
| 2 | Coherent aerial imaging | Pupil filtering | mask vs image comparison |
| 3 | Inverse aerial optimization | Differentiable optics | optimized mask |
| 4 | Partial coherence | Source integration | per-illumination aerial |
| 5 | Resist exposure + diffusion | Reaction-diffusion | latent image, contour |
| 6 | PINN diffusion | PDE residual learning | PINN vs FD comparison |
| 7 | Synthetic 3D mask correction | 3D effect approximation | correction dataset |
| 8 | FNO / DeepONet surrogate | Operator learning | surrogate model |
| 9 | Closed-loop inverse design | Surrogate-assisted opt | target enhancement |
| 10 | Active learning (optional) | Uncertainty-driven sampling | refined surrogate |

---

## 5. Working rules

- **Every code change updates `PROGRESS.md`.** For each unit of work record
  *What* (which module / file), *How* (which formula or algorithm),
  *Why* (the physical / mathematical / practical reason), and *Next*
  (the very next step).
- Each phase ships as a feature branch `phase-N-<short-name>` and lands on
  `main` via a PR with auto-merge. No work goes directly to `main`.
- Per-phase artifacts (figures / checkpoints / datasets) live under `outputs/`
  with the phase number as a filename prefix
  (e.g. `phase2_aerial_NA_sweep.png`).
- Experiment configuration goes into `configs/*.yaml`, never hard-coded.
- VRAM budget: 16 GB. Safe upper bound is grid 256² × FNO width 64 × Fourier
  modes 16. Beyond that, use mixed precision (`bf16`).

---

## 6. License / sourcing

This is a personal study repository. It does not borrow datasets or SOTA code
from external sources. The Phase 7 3D correction is **synthetic**, so there
are no licensing concerns.
