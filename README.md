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
| — | PEB v3 Screening (`reaction_diffusion_peb_v3_screening/`) — Stages 01-04 | ✅ first-pass | [v3 README](./reaction_diffusion_peb_v3_screening/README.md) |

---

## Two PEB submodules — why we split v1 and v2

> **This is a study repository.** Nothing here is calibrated to a real
> resist / scanner / dose. All numbers should be read as
> **internally-consistent** model output, not measurement predictions.

### v1 — `reaction_diffusion_peb/`  (PEB physics sandbox)

The original PEB submodule was a **physics sandbox**. Its question was
"can each PEB term be solved correctly in isolation?"

```text
exposure → diffusion (FD/FFT/PINN) → loss → deprotection → quencher
        → mass budget → Petersen → stochastic layers
                                                   → FNO surrogate
```

What v1 actually models:

- 2D x-y reaction-diffusion with Gaussian "toy" exposure maps,
- normalised dose only (no real `mJ/cm²`),
- a battery of solvers per term: FD, FFT, PINN, plus an FNO surrogate,
- 11 phases (1-11), 215 dedicated tests.

Where v1 stopped being enough:

- the geometry is a Gaussian / simple line-space — **no pitch, no CD, no
  edge roughness**;
- there is no z-direction, so **film thickness, standing wave,
  top/bottom asymmetry** cannot be studied;
- exposure is `dose=1.0` normalised, so a real `mJ/cm²` answer cannot be
  produced;
- the strong-quencher regime explored late in v1
  (`Q0=0.1, kq≥5, kdep=0.05`) collapses the `P > 0.5` contour, i.e. the
  model leaves the regime where any line-space pattern survives;
- PINN and FNO accuracy were measured against the same toy problem and
  are not reliable on real line / space.

### v2 — `reaction_diffusion_peb_v2_high_na/`  (High-NA EUV process model)

v2 keeps v1 untouched and re-poses the question as "**given a High-NA
EUV line/space target, what does the resist contour look like, and which
chemistry knob moves which metric?**"

What v2 adds on top of the v1 ideas:

| Concern in v1 | v2 fix |
|---|---|
| Gaussian exposure | line/space pattern with `pitch`, `line_cd`, edge roughness |
| Normalised dose | real `dose_mJ_cm²` + `dose_norm` separation, Dill-style acid generation |
| 2D x-y only | x-y solver + new x-z solver with Neumann-z mirror FFT |
| No electron blur | 1D / 2D Gaussian electron blur with σ as a calibration knob |
| Strong quencher collapse | weak-quencher regime (`Q0 ≤ 0.03`, `kq ≤ 5`) explored first |
| PINN / FNO as primary solver | **only FD / FFT** in v2 (PINN/FNO blocked until external calibration lands) |
| LER measured at fixed `P=0.5` | **CD-locked LER** — bisect the threshold to design CD before measuring roughness |
| No process-window concept | `unstable / merged / under_exposed / valid / robust_valid` 5-class status used as a gate |

### Why split rather than retrofit v1

```text
1. v1 is a working sandbox; we did not want to risk breaking 215 PEB tests.
2. v2 needs different physics regimes (weak quencher, real dose, x-z
   geometry) that would have required invasive changes.
3. v2 explicitly pursues a process-oriented operating point. v1 chases
   per-term solver fidelity. They have different success criteria.
4. The split lets us state a clear claims boundary: v1 results are about
   the equations; v2 results are about a frozen "v2 nominal OP" — and
   neither is calibrated to a real resist.
```

### v2 stages, what we ran

```text
Stage 1   clean geometry baseline               sigma=0, t=30
Stage 1A  sigma-compatible budget calibration   sigma in [0,3] usable at kdep=0.5 / Hmax<=0.2
Stage 1B  over-budget reference                 sigma=5 / t=60 lines collapse
Stage 2   DH × time process window              25 cells, robust DH=0.5 / t=30 chosen
Stage 3   electron blur separation              3-stage LER (design / e-blur / PEB)
Stage 4   weak quencher                         52 cells, balanced OP Q0=0.02 / kq=1
Stage 4B  CD-locked LER + small-pitch follow-up displacement-artifact vs real degradation
Stage 5   pitch × dose process window           108 cells, pitch=16 closed, pitch>=24 wide
Stage 6   x-z standing wave                     12 cells, PEB absorbs 79..60 % thin → thick
```

Plus a calibration phase that is **internal-only**:

```text
Cal Phase 1   Hmax × kdep × DH internal-consistency check
Cal Phase 2A  one-at-a-time controllability sweep
Cal Phase 2B  full atlas: x-y (141) + x-z (144) + small-pitch hypothesis (144)
```

### Headline findings

```text
1. Stage 4 / v2 OP balanced (sigma=2, Q0=0.02, kq=1, DH=0.5, t=30) is the
   internally-consistent process-window centre at pitch=24.
   CD_fixed ≈ 14.5–15 nm, CD-locked LER ≈ 2.5 nm, robust margin ~ 0.10.

2. The plan's original nominal (sigma=5, t=60) is incompatible with
   24 nm pitch — it always merges lines. We kept it as a stress
   reference, not a baseline.

3. The pitch=16 process window is closed at the v2 chemistry.
   pitch>=24 has a wide robust window over dose 28-60 mJ/cm²; the
   recommended dose ends up at 40 mJ/cm² for every workable pitch.

4. PEB diffusion absorbs the standing wave: ~79 % at thickness=15 nm,
   ~60 % at thickness=30 nm. Top/bottom asymmetry is set by the
   absorption envelope, not by the PEB equations.

5. Dropping sigma from 2 to 0 is the dominant knob that re-opens the
   small-pitch (18–20 nm) process window. Quencher tuning has a small
   or even negative effect at small pitch.

6. Among all knobs, absorption_length is the dominant lever for any
   z-direction metric (modulation, top/bottom asymmetry, side-wall
   side-wall LER, PSD mid-band) — change it and almost every z-metric
   moves > 100 % across the swept range.
```

### Calibration policy (read before citing absolute numbers)

```text
calibration_status:
  level                  : "internal-consistency only"
  published_data_loaded  : false
  v2_OP_frozen           : true
  v2_OP_freeze_date      : "2026-04-30"
```

- v2 OP is **frozen** until external (published or measured) CD / LER /
  process-window targets are loaded into
  [`calibration/calibration_targets.yaml`](./reaction_diffusion_peb_v2_high_na/calibration/calibration_targets.yaml).
- All future runs before that flip must be labelled
  **sensitivity**, **controllability**, or **hypothesis** study, never
  "calibration" or "calibrated to real".
- The procedure for opening that flip is documented in
  [`FUTURE_WORK.md`](./reaction_diffusion_peb_v2_high_na/FUTURE_WORK.md)
  Gate A.

### Where to read more (v2)

| Document | Purpose |
|---|---|
| [`README.md`](./reaction_diffusion_peb_v2_high_na/README.md) | one-page entry, frozen OP table, completed / deferred stages |
| [`STUDY_SUMMARY.md`](./reaction_diffusion_peb_v2_high_na/STUDY_SUMMARY.md) | first-pass closeout narrative, per-stage findings, claims boundary |
| [`RESULTS_INDEX.md`](./reaction_diffusion_peb_v2_high_na/RESULTS_INDEX.md) | per-stage table → folder, CSV, figure dir, one-line conclusion |
| [`FUTURE_WORK.md`](./reaction_diffusion_peb_v2_high_na/FUTURE_WORK.md) | gated future work (A external calibration / B physics / C small-pitch / D ref-search) |
| [`EXPERIMENT_PLAN.md`](./reaction_diffusion_peb_v2_high_na/EXPERIMENT_PLAN.md) | full per-stage spec + verified results |
| [`calibration/calibration_targets.yaml`](./reaction_diffusion_peb_v2_high_na/calibration/calibration_targets.yaml) | frozen OP, internal targets, `published_data_loaded` flag |
| [`calibration/calibration_plan.md`](./reaction_diffusion_peb_v2_high_na/calibration/calibration_plan.md) | per-phase log (Phase 1, 2A, 2B all internal-only) |
| [`study_notes/`](./reaction_diffusion_peb_v2_high_na/study_notes) | per-stage journals (problem → decision → result → next) |
| [`outputs/figures/xz_companions/`](./reaction_diffusion_peb_v2_high_na/outputs/figures/xz_companions/) | single-line x-z (depth) cross-sections for 11 representative configurations |

Outputs (figures, logs, contour maps for ~600 sweep cells across 8 stages + 3 calibration phases + xz companions) are committed under [`reaction_diffusion_peb_v2_high_na/outputs/`](./reaction_diffusion_peb_v2_high_na/outputs/).

---

## PEB v3 (screening) — first-pass

A third submodule under [`reaction_diffusion_peb_v3_screening/`](./reaction_diffusion_peb_v3_screening/) sits **on top of** the frozen v2 nominal physics generator. It does **not** modify the v2 OP, the calibration policy, or any `calibration_status` flag.

```text
candidate_space.yaml  →  Sobol / LHS sampler  →  10 000 candidates
                              ↓
                       budget_prefilter (analytical filters)
                              ↓
                       retained ~3 000
                              ↓
                       fd_batch_runner (re-uses the v2 helper)
                              ↓
                       labeler  →  6 status classes
                              ↓
                       sklearn RF classifier  +  multi-output RF regressor
                              ↓
                       active_learning  →  re-target FD on uncertain candidates
```

The first end-to-end run produced:
- 1 000 FD-labelled candidates (135 s, ~7.4 runs / s, 91.5 % `robust_valid`).
- A regressor with MAE 0.14 nm on `CD_locked` (R² 0.99) and 0.02 nm on `LER_CD_locked` (R² 0.93).
- An active-learning iteration that grew the defect-class minority by ~11× (16 → 186) using a single 316-FD acquisition step; `under_exposed` precision goes from 0 → 0.67 after retraining.

PINN stays out of the v3 primary screening loop. v3 may use PINN later only inside Stage 05 (autoencoder / inverse-fitting) and only after Stage 02–04 are stable.

This is **candidate screening / defect classification on the v2 nominal model**. It is **not** external calibration: the `calibration_status.published_data_loaded` flag stays `false` and v3 never overwrites v2's frozen OP.

| Document | Purpose |
|---|---|
| [`reaction_diffusion_peb_v3_screening/README.md`](./reaction_diffusion_peb_v3_screening/README.md) | one-page entry, pipeline, label schema, first-pass results |
| [`reaction_diffusion_peb_v3_screening/configs/`](./reaction_diffusion_peb_v3_screening/configs/) | label_schema, candidate_space, screening_baseline |
| [`reaction_diffusion_peb_v3_screening/src/`](./reaction_diffusion_peb_v3_screening/src/) | sampler, prefilter, FD runner, labeler, surrogates, AL |
| [`reaction_diffusion_peb_v3_screening/experiments/`](./reaction_diffusion_peb_v3_screening/experiments/) | runners for Stages 01-04 (and a placeholder 05) |
| [`reaction_diffusion_peb_v3_screening/outputs/`](./reaction_diffusion_peb_v3_screening/outputs/) | candidate JSONL, label CSV, surrogate joblib, figures, logs |

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

# 2. PyTorch — install the wheel that matches your environment.
#    See https://pytorch.org/get-started/locally/ for the right index URL.
pip install torch torchvision

# 3. the rest of the stack
pip install -r requirements.txt

# 4. sanity check
python -c "import torch; print(torch.__version__); \
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
- If memory becomes a constraint at large grids, wide FNOs, or many
  Fourier modes, switch to mixed precision (`bf16`) before adding
  more configuration knobs.

---

## License / sourcing

This is a personal study repository. It does not borrow datasets or
SOTA code from external sources. The Phase 7 3D correction is
**synthetic**, so there are no licensing concerns.
