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

### A.5 Phase 2 — Coherent aerial imaging — ☑ done (2026-04-29)

**What**
- `src/optics/pupil.py` — `circular_pupil(grid, NA, wavelength=1.0)` returns
  a hard binary low-pass filter with cutoff `NA / wavelength` in cycles per
  length unit. `apodized_circular_pupil(..., roll_off=0.05)` adds a
  cosine-tapered edge for differentiable use later.
- `src/optics/coherent_imaging.py` — `coherent_field` returns the complex
  pupil-filtered wafer field; `coherent_aerial_image` returns its squared
  magnitude. Both are pure PyTorch and differentiable end-to-end.
- `src/common/visualization.py` gained four helpers: `show_pupil`,
  `show_aerial`, `show_aerial_sweep` (mask + N aerials in one row), and
  `show_pipeline` (mask -> |T| -> P -> |T*P| -> aerial in five panels).
- `experiments/01_scalar_diffraction/demo_coherent_aerial.py` runs at
  `n=256, extent=20 lambda, wavelength=1` and produces five figures:
  pupil sweep, the 5-panel pipeline at NA=0.6 line-space pitch=4 lambda,
  and aerial NA sweeps over line-space, contact hole, and isolated line.
- `tests/test_pupil.py` (8 tests) and `tests/test_coherent_imaging.py`
  (9 tests) — 17 new tests, 33 / 33 green overall.

**How**
- Units convention: extent counted in wavelengths, `wavelength=1.0` by
  default. `circular_pupil` uses the grid's `radial_freq()` directly so
  the cutoff is expressed in the same physical units as the grid.
- Imaging math: `E = ifft2c(fft2c(t) * P)`, `I = |E|^2 = E.real**2 + E.imag**2`.
  Building the intensity as `real**2 + imag**2` (rather than `abs()**2`)
  keeps the autograd graph entirely real, which is friendlier for the
  Phase-3 mask-optimization gradient flow.
- Optional per-panel normalization (`normalize=True`) divides by the local
  max so brightness is comparable across NAs at the cost of losing
  absolute-intensity information. Inverse design will use `False`.
- The unaligned-pitch test uses a relative criterion (low-NA std must be
  much smaller than high-NA std) instead of an absolute uniformity bound,
  because rasterization noise from a non-pixel-aligned pitch contributes
  spectral leakage at all frequencies. The aligned-pitch test does check
  perfect uniformity — it picks `extent=16, n=256, pitch=1.0` so each
  period spans exactly 16 pixels and the DFT is exact.

**Why**
- Putting `pupil` and `coherent_imaging` in separate modules keeps the
  pupil reusable for Phase 4 (partial coherence sweeps over many shifted
  pupils) without dragging the imaging pipeline along.
- The `apodized_circular_pupil` is unused in the Phase-2 demos, but having
  it ready saves a refactor when Phase-3 inverse optimization or Phase-8
  surrogate training hits the gradient pathologies of a step-function
  filter.
- The 5-panel `show_pipeline` figure is the canonical "one picture of the
  whole optics chain" that future phases can reuse to overlay corrections
  (Phase 7 / 9 will diff `show_pipeline` outputs against true physics).

**Why (this direction)**
- Phase 2 is the bridge between "FFT a mask" (Phase 1) and "optimize a
  mask" (Phase 3). Once the imaging chain is differentiable and tested,
  Phase 3 collapses to choosing a loss and an optimizer.

**Next**
- Phase 3: `src/inverse/losses.py` (target / forbidden / TV / binarization),
  `src/inverse/regularizers.py`, `src/inverse/optimize_mask.py` with an
  Adam loop and binarization annealing. Demo a single contrast spot
  (target: a small disk of intensity in the center, forbidden: a
  surrounding annulus) and a forbidden-region case. Save before / after
  mask + aerial figures and the loss-curve figure.

---

### A.4 Phase 1 — Scalar diffraction — ☑ done (2026-04-29)

**What**
- `src/common/grid.py` — `Grid2D` dataclass with normalized real / frequency
  axes (`extent`, `dx`, `df`, Nyquist, mesh + radial helpers).
- `src/common/fft_utils.py` — `fft2c`, `ifft2c`, plus `amplitude` / `phase` /
  `log_amplitude` helpers. The single FFT convention used everywhere is
  `fftshift(fft2(ifftshift(x)))` with `norm='ortho'`.
- `src/common/visualization.py` — `show_mask`, `show_spectrum`,
  `show_field_pair`, `save_figure`. The pair view supports `freq_zoom`
  (crop to central 2 * freq_zoom bins) and `amp_percentile` (clip the
  linear `|T|` colormap so the DC spike stops washing out the orders).
- `src/mask/patterns.py` — `line_space`, `contact_hole`, `isolated_line`,
  `two_bar`, `elbow`, `random_binary`. All return float32 tensors with
  values in {0, 1} on the supplied `Grid2D`.
- `src/mask/transmission.py` — `binary_transmission`,
  `attenuated_phase_shift` (Att-PSM), `alternating_phase_shift` (Alt-PSM).
- `src/optics/scalar_diffraction.py` — `diffraction_spectrum(t)` and the
  matching `reconstruct_field(T)`.
- `experiments/01_scalar_diffraction/demo_line_space.py` — pitch sweep
  {0.40, 0.20, 0.10, 0.05} at duty 0.5.
- `experiments/01_scalar_diffraction/demo_contact_hole.py` — radius sweep
  {0.20, 0.10, 0.05} plus a binary-vs-Att-PSM comparison at r = 0.10.
- `tests/test_fft_utils.py`, `tests/test_patterns.py`,
  `tests/test_scalar_diffraction.py` — 16 tests, all green.

**How**
- Centered FFT keeps DC at index `(n//2, n//2)` so frequency axes line up
  with `imshow` extent without manual unshifting.
- The Hermitian-symmetry test drops the index-0 row and column (which has
  no centered pair) before flipping; the rest must satisfy
  `T(-f) = conj(T(f))` to better than 1e-4 for fp32.
- Pitch-sweep test verifies that halving the line-space pitch roughly
  doubles the first-order frequency offset, with a 25% tolerance for DFT
  discretization.
- The contact hole spectrum reproduces a 2D sinc / Bessel-J1 pattern; the
  phase panel shows the canonical 0 / pi flip at every Bessel zero.

**Why**
- Putting the FFT and grid in `src/common/` from day one prevents every
  later phase from reinventing conventions. Phase 2 (pupil filtering) and
  Phase 3 (autograd-based mask optimization) reuse these helpers verbatim.
- Building Att-PSM in Phase 1 (even though it isn't used until Phase 4 / 8)
  gives us a non-trivial complex transmission to test conjugate symmetry
  against — a binary mask alone can't.
- Frequency zoom + percentile clip in the visualization were necessary to
  see diffraction orders at all; for pitch=0.1 on a 256² grid the orders
  sit only 10 pixels off DC and were invisible at full extent.

**Why (this direction)**
- Goal of Phase 1 is to *see* the physics: sharp edges -> high frequencies,
  pitch -> order spacing, holes -> 2D sinc. Every subsequent phase will
  attach more machinery to this same FFT pipeline, so it has to be both
  correct (tests) and visually inspectable (demos).

**Next**
- Phase 2: `src/optics/pupil.py` (circular pupil, NA cutoff) and
  `src/optics/coherent_imaging.py` (`coherent_aerial_image`). Sweep NA in
  {0.2, 0.4, 0.6, 0.8} on a vertical line-space and a contact hole; save
  `outputs/figures/phase2_aerial_NA_sweep.png`.

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
| 1 | Scalar diffraction | ☑ | mask FFT, diffraction spectrum figures |
| 2 | Coherent aerial imaging | ☑ | pupil filtering, NA sweep |
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
- [x] `src/common/grid.py` — normalized real-space and frequency grids
- [x] `src/common/fft_utils.py` — `fft2c` / `ifft2c` (centered FFT)
- [x] `src/common/visualization.py` — mask / spectrum / aerial plotting helpers (with freq zoom + percentile clip)
- [x] `src/mask/patterns.py` — line-space, contact hole, isolated line, two-bar, elbow, random
- [x] `src/mask/transmission.py` — binary, attenuated phase shift, alternating phase shift
- [x] `src/optics/scalar_diffraction.py` — `diffraction_spectrum(t)`, `reconstruct_field`
- [x] `experiments/01_scalar_diffraction/demo_line_space.py`
- [x] `experiments/01_scalar_diffraction/demo_contact_hole.py`
- [x] `tests/test_fft_utils.py`, `tests/test_patterns.py`, `tests/test_scalar_diffraction.py` — 16 pass

### Phase 2 — Coherent aerial image
- [x] `src/optics/pupil.py` — `circular_pupil` (hard cutoff) + `apodized_circular_pupil` (cosine taper for differentiable settings)
- [x] `src/optics/coherent_imaging.py` — `coherent_field`, `coherent_aerial_image(transmission, pupil, normalize=...)`
- [x] `src/common/visualization.py` extended — `show_pupil`, `show_aerial`, `show_aerial_sweep`, `show_pipeline`
- [x] `experiments/01_scalar_diffraction/demo_coherent_aerial.py` — pupil sweep + 5-panel pipeline + aerial sweeps for line-space, contact hole, isolated line
- [x] NA sweep figures: `phase2_pupil_NA_sweep.png`, `phase2_pipeline_line_space.png`, `phase2_aerial_{line_space,contact_hole,isolated_line}.png`
- [x] `tests/test_pupil.py`, `tests/test_coherent_imaging.py` — 17 added (33 total green)

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
- **2026-04-29** `Grid2D` is a frozen dataclass rather than a free function
  returning a tuple — keeps `extent` / `dx` / `df` / device / dtype tied
  together so later modules can't accidentally mix conventions.
- **2026-04-29** Demo scripts inject the repo root into `sys.path` directly.
  Cleaner alternatives (`pip install -e .` or running with `PYTHONPATH=.`)
  exist but the inline bootstrap keeps `experiments/*/demo_*.py` runnable
  with a plain `python path/to/demo.py` and no setup step.
- **2026-04-29** Aerial intensity is built as `real**2 + imag**2` rather
  than `field.abs()**2`. The two are mathematically identical, but the
  former produces a real-only autograd graph which avoids unnecessary
  complex-conjugate gradient overhead in Phase 3.
- **2026-04-29** `extent` is counted in wavelengths, with `wavelength=1.0`
  the default everywhere. Going forward, all `Grid2D(extent=...)` values
  in optics code and demos are read as multiples of lambda.

---

## G. "Where to resume" note

> Update at the end of every session. One or two lines.

- **2026-04-29** Phase 0 done (env + scaffold + docs). Resume with Phase 1:
  start at `src/common/grid.py` and `src/common/fft_utils.py`, then
  `tests/test_fft_utils.py` to verify the centered-FFT round-trip error
  is below 1e-6.
- **2026-04-29** Phase 1 done (scalar diffraction). 16 tests green, demos
  produce 9 figures under `outputs/figures/phase1_*`. Resume with Phase 2:
  add `src/optics/pupil.py` (circular pupil with NA cutoff) and
  `src/optics/coherent_imaging.py` (`coherent_aerial_image(mask_t, pupil)`),
  then `experiments/01_scalar_diffraction/demo_coherent_aerial.py` for the
  NA sweep figure.
- **2026-04-29** Phase 2 done (coherent aerial imaging). 33 tests green,
  five new figures under `outputs/figures/phase2_*` clearly show the
  cutoff cliff (line-space pitch=4 lambda goes flat at NA=0.2 and
  sinusoidal at NA>=0.4) and the resolution gain on contact hole r=0.5
  lambda. Resume with Phase 3: write `src/inverse/{losses,
  regularizers, optimize_mask}.py`, add `configs/inverse_aerial.yaml`,
  and run `experiments/02_inverse_aerial/demo_target_spot.py` to land a
  before/after pair plus a loss curve under `outputs/figures/phase3_*`.
