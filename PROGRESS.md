# PROGRESS — implementation log

> A cumulative log updated on every code change.
> Purpose: at any moment, anyone (including future-me) should be able to
> answer "where are we, what was implemented, why this direction, what is
> next?" by reading this file alone.
>
> **Project rule.** All project documents and code comments are written in
> English. The only Korean file is the original study plan
> (`litho_neural_operator_study_plan.md`), kept verbatim as the source plan.
>
> For each unit of work, append four short lines under the matching phase
> section:
>
> 1. **What** — which module(s) / file(s), in what shape
> 2. **How** — formula / algorithm / assumption used
> 3. **Why** — physical / mathematical / practical reason for that direction
> 4. **Next** — the immediate next step
>
> Phase status is tracked in §B with ☐ pending / ◐ in-progress / ☑ done.
> Module-level checklists in §C are flipped ☐ → ◐ → ☑ as files land and tests
> pass.

---

## A. Infrastructure

### A.1 Environment setup — ☑ done (2026-04-29)

**What**
- Python 3.12 venv at `./.venv/`
- PyTorch **2.11.0+cu128** (cu128 wheel index — required for Blackwell sm_120)
- numpy 2.4 / scipy 1.17 / matplotlib 3.10 / h5py 3.16 / pandas 3.0
- hydra-core 1.3 / omegaconf 2.3 / tqdm 4.67 / pytest 9.0 / jupyterlab 4.5

**How**
- `python3 -m venv .venv` → `pip install --index-url …/cu128 torch torchvision`
  → `pip install -r requirements.txt`.
- GPU sanity: `torch.cuda.is_available() == True`, device `sm_120`,
  2048×2048 fp32 matmul + complex64 FFT2 + autograd all pass.

**Why**
- The cu126 stable wheel only carries kernels up to sm_90 → on RTX 5080 it
  installs fine but raises `no kernel image is available for execution` at
  runtime. cu128 (or cu130) wheels carry sm_120 PTX/SASS.

**Next**
- Begin Phase 1 with `src/common/grid.py` and `src/common/fft_utils.py`.

---

### A.2 Folder structure — ☑ done (2026-04-29)

**What**
- The tree from study plan §2:
  `src/{common,mask,optics,inverse,resist,pinn,neural_operator,closed_loop}`,
  `experiments/01..07/`,
  `outputs/{figures,checkpoints,logs,datasets}`,
  `tests/`, `notebooks/`, `configs/`.
- Empty `__init__.py` in every package.
- `.gitignore` excludes `.venv/` and `outputs/*` (with `.gitkeep` files
  preserving the empty directories).

**Why**
- The src layout keeps imports clean and lets us switch to
  `pip install -e .` later without reshuffling.

---

### A.3 Documentation baseline — ☑ done (2026-04-29)

**What**
- `README.md` — environment, install, layout, phase roadmap, working rules.
- `PROGRESS.md` (this file) — running log template.
- Project rule pinned in both files: **all project docs and code comments in
  English**; the Korean study plan is the only exception.

**Why**
- The study plan is detailed but static. PROGRESS gives a single place to
  answer "where are we now?" without re-reading 1.6k lines.

---

## B. Phase roadmap

> ☐ pending / ◐ in-progress / ☑ done

| Phase | Name | Status | Key artifact |
|---|---|---|---|
| 0 | Env / scaffold / docs | ☑ | this scaffold + README + PROGRESS |
| 1 | Scalar diffraction | ☐ | mask FFT, diffraction spectrum figures |
| 2 | Coherent aerial imaging | ☐ | pupil filtering, NA sweep |
| 3 | Inverse aerial optimization | ☐ | gradient-descent mask optimization |
| 4 | Partial coherence / source integration | ☐ | annular / dipole / quadrupole |
| 5 | Resist exposure + diffusion | ☐ | FD / FFT diffusion, threshold contour |
| 6 | PINN for diffusion | ☐ | PINN vs FD comparison |
| 7 | Synthetic 3D mask correction | ☐ | correction dataset NPZ |
| 8 | FNO / DeepONet surrogate | ☐ | surrogate `.pt` checkpoint |
| 9 | Closed-loop surrogate-assisted inverse | ☐ | surrogate vs true comparison |
| 10 | Active learning (optional) | ☐ | ensemble uncertainty |

---

## C. Per-phase module checklist

### Phase 1 — Scalar diffraction
- [ ] `src/common/grid.py` — normalized real-space and frequency grids
- [ ] `src/common/fft_utils.py` — `fft2c` / `ifft2c` (centered FFT)
- [ ] `src/common/visualization.py` — mask / spectrum / aerial plotting helpers
- [ ] `src/mask/patterns.py` — line-space, contact hole, isolated line, two-bar, elbow, random
- [ ] `src/mask/transmission.py` — binary, attenuated phase shift
- [ ] `src/optics/scalar_diffraction.py` — `diffraction_spectrum(t)`
- [ ] `experiments/01_scalar_diffraction/demo_line_space.py`
- [ ] `experiments/01_scalar_diffraction/demo_contact_hole.py`
- [ ] `tests/test_fft_utils.py`, `tests/test_patterns.py`

### Phase 2 — Coherent aerial image
- [ ] `src/optics/pupil.py` — circular pupil with NA cutoff
- [ ] `src/optics/coherent_imaging.py` — `coherent_aerial_image(mask_t, pupil)`
- [ ] `experiments/01_scalar_diffraction/demo_coherent_aerial.py`
- [ ] NA sweep figure: `outputs/figures/phase2_aerial_NA_sweep.png`

### Phase 3 — Inverse aerial optimization
- [ ] `src/inverse/losses.py` — target / forbidden / TV / binarization
- [ ] `src/inverse/regularizers.py`
- [ ] `src/inverse/optimize_mask.py` — Adam loop, binarization annealing
- [ ] `experiments/02_inverse_aerial/demo_target_spot.py`
- [ ] `experiments/02_inverse_aerial/demo_forbidden_region.py`
- [ ] `configs/inverse_aerial.yaml`

### Phase 4 — Partial coherence
- [ ] `src/optics/source.py` — coherent / annular / dipole / quadrupole / random
- [ ] `src/optics/partial_coherence.py` — incoherent sum over source points
- [ ] `experiments/03_partial_coherence/demo_source_shapes.py`
- [ ] `configs/partial_coherence.yaml`

### Phase 5 — Resist exposure + diffusion
- [ ] `src/resist/exposure.py` — Dill-style acid generation
- [ ] `src/resist/diffusion_fd.py` — explicit FD with CFL check
- [ ] `src/resist/diffusion_fft.py` — FFT heat kernel
- [ ] `src/resist/reaction_diffusion.py` — A·Q reaction term
- [ ] `src/resist/threshold.py` — soft sigmoid threshold
- [ ] `experiments/04_resist_diffusion/demo_dose_sweep.py`
- [ ] `experiments/04_resist_diffusion/demo_diffusion_length.py`
- [ ] `configs/resist.yaml`

### Phase 6 — PINN diffusion
- [ ] `src/pinn/pinn_base.py` — MLP / SIREN / Fourier feature
- [ ] `src/pinn/pinn_diffusion.py` — pure diffusion residual loss
- [ ] `src/pinn/pinn_reaction_diffusion.py`
- [ ] `experiments/05_pinn_diffusion/compare_fd_pinn.py`
- [ ] `configs/pinn.yaml`

### Phase 7 — Synthetic 3D correction dataset
- [ ] `src/neural_operator/synthetic_3d_correction_data.py`
- [ ] `experiments/06_fno_correction/generate_synthetic_dataset.py`
- [ ] `outputs/datasets/synthetic_3d_correction_train.npz`
- [ ] `outputs/datasets/synthetic_3d_correction_test.npz`

### Phase 8 — FNO / DeepONet surrogate
- [ ] `src/neural_operator/fno2d.py`
- [ ] `src/neural_operator/deeponet.py`
- [ ] `src/neural_operator/datasets.py`
- [ ] `src/neural_operator/train_fno.py` / `train_deeponet.py`
- [ ] `experiments/06_fno_correction/train_fno_correction.py`
- [ ] `experiments/06_fno_correction/train_deeponet_correction.py`
- [ ] `configs/fno.yaml`
- [ ] `outputs/checkpoints/fno_correction.pt`

### Phase 9 — Closed-loop surrogate inverse
- [ ] `src/closed_loop/surrogate_optimizer.py`
- [ ] `experiments/07_closed_loop_inverse/optimize_with_fno.py`
- [ ] surrogate vs true correction comparison figure

### Phase 10 — Active learning (optional)
- [ ] `src/closed_loop/active_learning.py`
- [ ] one of: ensemble uncertainty / distance-to-distribution / oracle validation

---

## D. Design rules (carry across phases)

1. **All optics ops use PyTorch tensors with complex dtype.**
   Don't start in NumPy and re-port for autograd in Phase 3 — go straight to
   `torch.complex64` from Phase 1.
2. **One FFT convention everywhere: centered FFT.**
   `fft2c(x) = fftshift(fft2(ifftshift(x)))`. Every module uses this exact
   form. No raw `fft2` calls outside `fft_utils.py`.
3. **Normalized units (λ = 1, pixel size = λ / k_grid).**
   No DUV / EUV physical units. Normalized math is clearer for a study repo.
4. **One module, one responsibility, one phase.**
   Cross-phase imports go through `src/common/` only.
5. **Experiment scripts are thin wrappers.**
   The body of every `experiments/*/demo_*.py` is: load YAML → call `src/`
   functions → save figure / log. No physics in the demo file.
6. **Reproducibility.**
   Every experiment seeds `torch.manual_seed` and `np.random.seed`. Output
   filenames may include the seed.
7. **PROGRESS.md update rule.**
   On every commit add a What/How/Why/Next entry to the matching phase
   section, and flip checklist items ☐ → ◐ → ☑.

---

## E. Open questions / parked decisions

> Add questions here as they come up; move them to §F once resolved.

- _(none yet)_

---

## F. Decisions / trade-offs

> One line each, dated. The point is to remember **why** later.

- **2026-04-29** Picked the cu128 PyTorch wheel (cu126 lacks sm_120). Stable
  build, so no nightly-pinning risk.
- **2026-04-29** src layout with empty `__init__.py` files — lets us switch
  to editable install (`pip install -e .`) at any time.
- **2026-04-29** All docs and code comments in English; only the source study
  plan stays in Korean. Mixing languages bloats greps and review noise.

---

## G. "Where to resume" note

> Update at the end of every session. One or two lines.

- **2026-04-29** Phase 0 done (env + scaffold + docs). Resume with Phase 1:
  start at `src/common/grid.py` and `src/common/fft_utils.py`, then
  `tests/test_fft_utils.py` to verify the centered-FFT round-trip error
  is below 1e-6.
