# litho-neural-operator-lab

> A study-purpose lithography simulator built around Fourier optics and
> reaction-diffusion, on top of which Neural Operators (FNO / DeepONet) and
> PINNs are attached as correction or acceleration blocks. The goal is **not**
> SOTA performance — it is to understand the physics first and then learn
> where neural surrogates help and where they break.

## Status

Phases **1 – 9 are complete**. Phase 10 (active learning) is deliberately
left as future work. A separate **PEB submodule** under
[`reaction_diffusion_peb/`](./reaction_diffusion_peb) covers
post-exposure-bake resist physics in parallel; its Phases 1-3 are done.

- [PROGRESS.md](./PROGRESS.md) — running implementation log
  (What / How / Why / Next per change, plus §H wrap-up + reopen plan).
- 347 / 347 tests green (132 main + 215 PEB Phases 1-11).

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
| — | PEB submodule v1 (`reaction_diffusion_peb/`) — Phases 1-11 done | ✅ | [docs/peb_submodule](./docs/peb_submodule.md) |
| — | PEB v2 High-NA (`reaction_diffusion_peb_v2_high_na/`) — Stages 1-6 + Phase 2B | ✅ first-pass | [STUDY_SUMMARY](./reaction_diffusion_peb_v2_high_na/STUDY_SUMMARY.md) |

---

## PEB v2 (High-NA EUV PEB) — first-pass closeout

A second PEB submodule under [`reaction_diffusion_peb_v2_high_na/`](./reaction_diffusion_peb_v2_high_na/) reframes the v1 sandbox as a **High-NA EUV line/space process simulator**. It adds:

- pitch / line-CD / edge-roughness geometry,
- Dill-style acid generation with electron blur,
- operator-split spectral PEB solver (x-y) and a Neumann-z mirror-FFT solver (x-z),
- weak-quencher chemistry,
- CD-locked LER and PSD band metrics.

**Status**: first-pass complete. The recommended operating point is **frozen** as the internally-consistent v2 nominal OP. **The model is NOT externally calibrated** — `calibration_status.published_data_loaded` is false, and all sweeps are labelled sensitivity / controllability / hypothesis studies.

| Document | Purpose |
|---|---|
| [`STUDY_SUMMARY.md`](./reaction_diffusion_peb_v2_high_na/STUDY_SUMMARY.md) | first-pass closeout, status table, per-stage findings, frozen OP, claims boundary |
| [`README.md`](./reaction_diffusion_peb_v2_high_na/README.md) | one-page entry, frozen OP table, completed / deferred stages |
| [`RESULTS_INDEX.md`](./reaction_diffusion_peb_v2_high_na/RESULTS_INDEX.md) | per-stage table → folder, CSV, figure dir, one-line conclusion |
| [`FUTURE_WORK.md`](./reaction_diffusion_peb_v2_high_na/FUTURE_WORK.md) | gated future work (external calibration / physics extension / small pitch / reference search) |
| [`EXPERIMENT_PLAN.md`](./reaction_diffusion_peb_v2_high_na/EXPERIMENT_PLAN.md) | full per-stage spec + verified results |
| [`calibration/calibration_targets.yaml`](./reaction_diffusion_peb_v2_high_na/calibration/calibration_targets.yaml) | frozen OP + internal targets + `published_data_loaded` flag |
| [`calibration/calibration_plan.md`](./reaction_diffusion_peb_v2_high_na/calibration/calibration_plan.md) | calibration phase log (Phase 1, 2A, 2B all internal-only) |
| [`study_notes/`](./reaction_diffusion_peb_v2_high_na/study_notes) | per-stage journals (problems / decisions / results) |

Headline (frozen OP, internal-consistent):

```text
pitch=24, line_cd=12.5, dose=40 mJ/cm², σ=2 nm,
DH=0.5, time=30 s, kdep=0.5, Hmax=0.2, kloss=0.005,
quencher Q0=0.02, kq=1.0, DQ=0.0
→ CD_fixed ≈ 14.5–15 nm, LER_locked ≈ 2.5 nm,
  process window pitch ≥ 24 wide; pitch=16 closed at this chemistry.
```

This number is consistent with v2's own first-pass observations only. Quantitative agreement with experimental High-NA EUV measurements is not claimed.

Outputs (figures, logs, contour maps for ~600 sweep cells across 7 stages + 3 calibration phases) are committed under [`reaction_diffusion_peb_v2_high_na/outputs/`](./reaction_diffusion_peb_v2_high_na/outputs/).

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
