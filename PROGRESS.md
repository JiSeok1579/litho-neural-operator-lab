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

### A.8 Phase 5 — Resist exposure + diffusion — ☑ done (2026-04-29)

**What**
- `src/resist/exposure.py` — `acid_from_aerial(I, dose, eta=1)`
  implementing Dill saturation `A0 = 1 - exp(-eta * dose * I)`. Pure
  PyTorch, differentiable, vectorized.
- `src/resist/diffusion_fd.py` — `laplacian_5pt` (5-point stencil with
  periodic BC via ``torch.roll``), `step_diffusion_fd`, `diffuse_fd`.
  Auto-selects ``n_steps`` to satisfy CFL ``dt <= cfl_safety * dx**2 /
  (4 D)`` and rejects user-provided steps that violate it.
- `src/resist/diffusion_fft.py` — exact heat-kernel diffusion in two
  flavors: `diffuse_fft(A, grid, D, t)` and the more sweep-friendly
  `diffuse_fft_by_length(A, grid, L)` parameterized directly by the
  diffusion length ``L = sqrt(2 D t)``. Returns the real part of the
  inverse FFT (imaginary part is zero up to round-off for real input).
- `src/resist/reaction_diffusion.py` — coupled `dA/dt = D laplacian(A)
  - k A Q`, `dQ/dt = -k A Q` with explicit Euler and a coarse
  reaction-stability bound. Includes a non-negative clamp because Euler
  can dip slightly below zero near gradient peaks at large `dt`.
- `src/resist/threshold.py` — `soft_threshold` (sigmoid), the binary
  `hard_threshold`, `measure_cd_horizontal` (longest contiguous
  above-threshold run on the central row, returned in length units),
  and `thresholded_area`.
- `src/common/visualization.py` gained `show_resist_chain` (the
  aerial → acid → diffused acid → resist 4-panel chain with an
  optional threshold contour overlay) and `show_resist_sweep` (rows of
  (acid_diffused, resist) across a swept knob).
- `experiments/04_resist_diffusion/demo_dose_sweep.py` runs the chain
  for a vertical line-space (pitch 2 lambda, NA 0.6) at dose ∈ {0.5,
  1.0, 1.5, 2.0} with `L_diff = 0.10 lambda`, `A_th = 0.40`, beta = 25.
  Saves `phase5_dose_sweep.png`, `phase5_dose_chain_dose1.5.png`, and
  `phase5_dose_sweep_metrics.csv`.
- `experiments/04_resist_diffusion/demo_diffusion_length.py` fixes
  dose at 1.5 and sweeps `L_diff ∈ {0, 0.05, 0.10, 0.20, 0.40} lambda`.
  Saves `phase5_diffusion_length_sweep.png` and a metrics CSV.
- `configs/resist.yaml` — reference parameter sheet.
- `tests/test_resist.py` — 21 new tests (91 / 91 green): exposure
  saturation + zero-dose identity + monotonicity, FD Laplacian on
  constant / quadratic fields + zero-time identity + mass conservation
  + CFL rejection, FFT identity at zero D or t + constant-field
  invariance + mass conservation + agreement with FD on a smooth
  Gaussian, by-length parameterization equivalence, reaction-diffusion
  conservation of `(A - Q)` when `D = 0`, soft / hard thresholds, CD
  measurement, and area counts.

**How**
- Periodic BCs (FD) and the FFT both assume torus topology — fine for
  the periodic line-space and the central contact hole pattern this
  phase uses. Phase 7+ may need Dirichlet BCs at hard wafer edges; we
  defer until that need is real.
- The FFT decay factor uses the standard form
  `exp(-4 pi^2 D t |f|^2)` and the by-length variant absorbs
  `4 pi^2 D t = 2 pi^2 L^2`. Both reuse the existing centered FFT
  helpers, so the convention from Phase 1 is preserved.
- The FD-vs-FFT agreement test uses a smooth Gaussian initial field
  rather than a random field. With random input at near-Nyquist
  frequencies, the second-order FD truncation gives ~10% error per
  step; a Gaussian at sigma 0.25 (≈4 dx) sits comfortably in the
  resolved regime where the agreement is < 5%.
- `acid_from_aerial` is built from `1 - exp(-…)`, which keeps autograd
  through the exposure step real-valued and avoids issues that arise
  if you implement it as a sigmoid of a log-domain accumulator.

**Why**
- Splitting FD and FFT into two modules makes the phase teach both
  perspectives explicitly: FD is the textbook stencil that any PDE
  course shows; FFT is the analytic Fourier solution that scales
  better but requires a periodic grid.
- `diffuse_fft_by_length` exists because every Phase-5+ sweep that
  matters operates on the geometric scale `L`, not on `(D, t)` pair
  separately. Hiding the redundant degree of freedom prevents
  dimensional bugs in later experiments.
- `measure_cd_horizontal` is intentionally simple: it counts the
  longest above-threshold run on the y=0 row. For periodic patterns
  this gives a stable scalar that tracks CD changes; production
  lithography uses contour-following with sub-pixel interpolation,
  but a study repo doesn't need that.

**Verified results from `phase5_*_metrics.csv`**

Dose sweep (line-space pitch=2 lambda, NA=0.6, L_diff=0.10 lambda,
A_th=0.40, beta=25):

```
  dose  acid_max  acid_diffused_max  resist_max  cd_lambda  thresh_pix
  0.5   0.394     0.377              0.362       0.000      0          <- nothing prints
  1.0   0.632     0.612              0.995       0.664      21504
  1.5   0.777     0.758              1.000       0.859      27648
  2.0   0.865     0.848              1.000       0.938      30208
```

Diffusion-length sweep (dose=1.5 fixed):

```
  L_diff  acid_max  acid_min  acid_contrast  cd_lambda  thresh_pix
  0.00    0.777     0.000     1.000          0.859      27648
  0.05    0.772     0.004     0.991          0.859      27648
  0.10    0.758     0.010     0.974          0.859      27648
  0.20    0.695     0.023     0.937          0.820      26624
  0.40    0.520     0.136     0.586          0.742      23040
```

Physics confirmed:
- At dose 0.5, peak acid (0.394) sits **just below** the threshold
  (0.40), so nothing prints — the canonical "under-exposure cliff".
  Doubling the dose puts the peak well above threshold and CD jumps
  from 0 to 0.66 lambda; further dose increases widen lines smoothly.
- For small diffusion lengths (L < 0.10 lambda) CD is invariant — the
  blur is smaller than the threshold band so the contour position is
  preserved. At L = 0.20 lambda the acid bulk peak drops to 0.695 and
  the contour starts retreating; at L = 0.40 lambda the contrast
  collapses to 0.59 and CD shrinks 14% to 0.74 lambda.
- This is the "aerial vs resist" duality from study plan §5.10:
  diffusion shrinks signal **and** leakage in absolute terms, but the
  threshold cut is what determines whether sub-threshold features
  print at all — exactly the point of resist contrast modeling.

**Next**
- Phase 6: PINN diffusion. `src/pinn/{pinn_base,pinn_diffusion,
  pinn_reaction_diffusion}.py` and
  `experiments/05_pinn_diffusion/compare_fd_pinn.py`. Compare the
  trained PINN against the Phase-5 FD and FFT solutions on the same
  initial condition; report MSE, edge error, training time, and
  inference time. Save figures as `outputs/figures/phase6_*` and a
  metrics CSV.

---

### A.7 Phase 4 — Partial coherence / source integration — ☑ done (2026-04-29)

**What**
- `src/optics/source.py` — five source-shape factories
  (`coherent_source`, `annular_source`, `dipole_source`,
  `quadrupole_source`, `random_source`) plus `sigma_axis` /
  `sigma_meshgrid` / `source_points` (decode a 2D source tensor into
  a `(K, 2)` sigma list and a `(K,)` weight list).
- `src/optics/pupil.py` extended with `circular_pupil_at(grid, NA,
  center_freq, wavelength)` — same API as `circular_pupil` but
  centered at an arbitrary spatial frequency.
- `src/optics/partial_coherence.py` —
  `partial_coherent_aerial_image(transmission, grid, source, NA,
  wavelength, normalize)`. Builds a `(K, N, N)` stack of shifted
  pupils, runs the FFT chain in one batched pass, and sums weighted
  intensities. End-to-end differentiable, so Phase 9's closed loop
  can swap `coherent_aerial_image` for this with no plumbing change.
- `src/common/metrics.py` (new module) — `image_contrast`,
  `peak_intensity_in_region`, `integrated_leakage`, and a study-grade
  `normalized_image_log_slope` for NILS.
- `src/common/visualization.py` gained `show_source` (single panel in
  sigma-space) and `show_partial_coherence_sweep` (3 rows: mask
  reference, source, aerial — fixed mask, varying source).
- `experiments/03_partial_coherence/demo_source_shapes.py` runs at
  `n=256, extent=20 lambda, NA=0.4` and writes
  `phase4_sources.png`, `phase4_aerial_line_space.png`,
  `phase4_aerial_contact_hole.png`, plus a metrics CSV.
- `configs/partial_coherence.yaml` — reference parameters.
- `tests/test_source.py` (10), `tests/test_partial_coherence.py` (6),
  `tests/test_metrics.py` (5) — 21 new tests, 70 / 70 green overall.

**How**
- Source coordinates are normalized partial-coherence factors
  ``sigma in [-1, 1]^2``: ``sin(theta_source) = sigma * NA``.
  Off-axis source point at ``sigma`` shifts the equivalent pupil to
  ``sigma * NA / wavelength`` in cycles per length.
- ``sigma_axis`` is built as ``arange(-half, half+1) / half`` rather
  than ``linspace(-1, 1)`` so the center pixel is bit-exactly
  ``sigma=0``. Without this, ``coherent_source`` fed through the
  partial-coherence engine drifted ~2 % from a direct
  `coherent_aerial_image` reference because a few-bin shift swapped
  one or two boundary pixels of the pupil.
- All shifted pupils are stacked along a leading batch dim and
  multiplied with the (single) mask spectrum via broadcasting; the
  whole sum is one ``ifft2c`` call on a ``(K, N, N)`` tensor.
- Source weight normalization is **optional** (``normalize=True`` by
  default for demos so different source densities are visually
  comparable; ``False`` for inverse design where absolute intensity
  matters).

**Why**
- Putting source factories into a dedicated module makes Phase 8 (FNO
  surrogate) and Phase 9 (closed-loop) trivial: every illumination
  is just a 2D float tensor, so we can mix-and-match without changing
  the imaging engine.
- Building the Hopkins integral as a batched FFT (instead of a Python
  loop) keeps the cost flat: even with 332 source points (annular)
  the demo runs sub-second on RTX 5080.
- `circular_pupil_at` lives in `pupil.py` rather than
  `partial_coherence.py` so other phases (e.g. tilted single-plane-wave
  studies) can reuse it without importing the partial-coherence
  module.

**Why (this direction)**
- Phase 4 is the first phase where "the mask is not the only thing
  that matters" — illumination shape can rescue or destroy
  resolution. This sets up Phase 9's surrogate-assisted inverse design,
  which can co-optimize mask + source.

**Verified results from `phase4_metrics.csv` (NA=0.4, lambda=1)**

```
vertical line-space, pitch=1.5 lambda
  source       peak    contrast   leakage(3-6 lam)
  coherent     0.321   0.247       3399
  annular      0.398   0.348       3492
  dipole-x     0.668   0.809       4162   <-- best on vertical lines
  dipole-y     0.287   0.144       3390   <-- worst (perpendicular axis)
  quadrupole   0.435   0.416       3559

contact hole, r=0.5 lambda
  source       peak    contrast   leakage(3-6 lam)
  coherent     0.124   1.000       1.574  <-- best peak
  annular      0.111   0.9999      1.558
  dipole-x     0.105   0.9999      1.500
  dipole-y     0.105   0.9999      1.500
  quadrupole   0.110   0.9999      1.558
```

Physics from the table:
- Vertical line-space at pitch 1.5 lambda has its fundamental at
  fx = 0.667 cycles/lambda. On-axis NA=0.4 cannot capture it (cutoff
  is 0.4 < 0.667), so coherent contrast collapses to 0.25.
- Dipole-x at sigma=0.7 places the right pole's pupil center at fx ~
  +0.28 cycles/lambda, whose +0.4 reach hits 0.68 — just above the
  fundamental. Contrast jumps 3.3x and peak doubles.
- Dipole-y is the worst case for vertical lines: it shifts only along
  fy, leaving fx out of reach.
- For the contact hole, coherent gives the highest peak (0.124);
  off-axis sources spread energy and suppress the peak by ~10-15 %.

**Next**
- Phase 5: `src/resist/{exposure,diffusion_fd,diffusion_fft,
  reaction_diffusion,threshold}.py` plus
  `experiments/04_resist_diffusion/demo_dose_sweep.py` and
  `demo_diffusion_length.py`. Land `outputs/figures/phase5_*` and a
  metrics CSV showing CD-like width vs dose / diffusion length.

---

### A.6 Phase 3 — Inverse aerial optimization — ☑ done (2026-04-29)

**What**
- `src/inverse/losses.py` — `masked_mse(field, region, target)` and the
  two convenience aliases `target_loss` and `background_loss`. The
  helper `mean_intensity_in_region` is a non-loss diagnostic used in the
  optimization history.
- `src/inverse/regularizers.py` — `total_variation` (mean L1 first
  difference) and `binarization_penalty` (mean of `m * (1 - m)`).
- `src/inverse/optimize_mask.py` — `optimize_mask(grid, target_region,
  forbidden_region, config)` runs Adam on a real-valued raw parameter
  `theta`. Mask `m = sigmoid(alpha * theta)`; `alpha` follows a step
  schedule (default `(0,1) -> (300,4) -> (700,12)`). Returns an
  `OptimizationResult` dataclass with theta, mask, both aerials, the
  pupil, and a per-iteration history of loss components and mean
  intensities.
- `src/common/visualization.py` extended with `show_inverse_result`
  (2 rows x 4 cols: mask | aerial | y=0 row-cut with regions shaded |
  RGB overlay of regions on aerial) and `show_loss_history` (loss
  components on log-y, region-mean intensities on linear).
- `experiments/02_inverse_aerial/demo_target_spot.py` — study plan
  §3-A reproduction. Target = central disk r=1 lambda; forbidden =
  complement; NA=0.4; 800 iters; alpha=(0,1)/(300,4)/(600,12).
- `experiments/02_inverse_aerial/demo_forbidden_region.py` — same
  target but forbidden = four explicit side-lobe disks at +/-3 lambda
  along x and y; background weight bumped to 4 to compensate for the
  smaller forbidden area.
- `configs/inverse_aerial.yaml` — reference parameter sheet. Demos
  currently hard-code matching values; YAML loading deferred until a
  later phase needs sweep automation.
- `tests/test_inverse_losses.py` (10) and `tests/test_optimize_mask.py`
  (5) — 16 added, 49 / 49 green overall.

**How**
- Parameterization: `m = sigmoid(alpha * theta)`. `theta` is
  unconstrained on the real line; `alpha` is the binarization sharpness.
  `theta = 0` gives `m = 0.5` everywhere, which is a neutral starting
  point — initial aerial is constant at `|0.5|^2 = 0.25`.
- Annealing: at the alpha step boundaries the mask suddenly becomes
  more contrasted, which spikes the loss (visible in
  `phase3_*_loss.png`); the optimizer then recovers within tens of
  iterations.
- Aerial intensity is **not** normalized inside the optimizer — the
  loss is on absolute intensity so the target value of 1.0 is meaningful.
- The `target_loss` test "decreases" guard uses NA=0.6 and disables
  annealing so the budget of 80 iters is enough to reliably drop the
  loss from its initial value; the production demos use NA=0.4 with the
  full annealing schedule.

**Why**
- `sigmoid(alpha * theta)` keeps the mask differentiably bounded in
  (0, 1) without explicit projection. Compared to a clamped
  parameterization, this produces smoother gradients and avoids the
  zero-gradient plateaus of saturation.
- Target / forbidden separation is the inverse-problem framing the
  study plan (§3.3) calls for: an explicit knob for "what should be
  bright" and "what should be dark" with weights tuning the trade-off.
- TV regularization with a tiny weight (1e-3) is enough to keep the
  mask geometrically coherent without dominating the loss; raising it
  noticeably blurs the final binary pattern.
- Returning a dataclass (`OptimizationResult`) instead of a tuple
  matters once Phase 9 wraps `optimize_mask` inside a closed loop —
  named fields make the plumbing readable.

**Why (this direction)**
- Phase 3 closes the "physics direction": Phase 1 makes a spectrum,
  Phase 2 turns it into intensity, Phase 3 inverts the chain via
  gradients. Phase 4-7 add complexity (partial coherence, resist,
  3D correction) on top of this same `optimize_mask` skeleton.

**Verified results from the demos**
- `demo_target_spot`: target loss 0.5625 -> 0.0005 (~1100x), mean(I)
  in target 0.250 -> 0.992, mean(I) in forbidden (entire complement)
  0.250 -> 0.078. The optimizer parks substantial energy outside the
  target because the target is sub-Rayleigh at NA=0.4 (Rayleigh ~
  3.05 lambda, target r = 1 lambda).
- `demo_forbidden_region`: target loss 0.5625 -> 0.0066, background
  loss 0.0625 -> 0.0004. mean(I) in the four explicit forbidden disks
  drops to 0.015 — about 5x stronger suppression than the
  complement case, at the cost of slightly worse target fidelity and
  visible energy in the neutral diagonal regions. This is the
  fidelity-vs-leakage trade-off study plan §3.7 explicitly flags.

**Next**
- Phase 4: `src/optics/source.py` (coherent / annular / dipole /
  quadrupole / random-points source shapes), `src/optics/partial_coherence.py`
  (incoherent sum over shifted pupils), then a sweep demo on a
  vertical line-space and a contact hole comparing illumination
  shapes. Save under `outputs/figures/phase4_*`.

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
| 3 | Inverse aerial optimization | ☑ | gradient-descent mask optimization |
| 4 | Partial coherence / source integration | ☑ | annular / dipole / quadrupole |
| 5 | Resist exposure + diffusion | ☑ | FD / FFT diffusion, threshold contour |
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
- [x] `src/inverse/losses.py` — `masked_mse`, `target_loss`, `background_loss`, `mean_intensity_in_region`
- [x] `src/inverse/regularizers.py` — `total_variation`, `binarization_penalty`
- [x] `src/inverse/optimize_mask.py` — `optimize_mask` + `OptimizationConfig` + `LossWeights` + `OptimizationResult` (Adam, sigmoid(alpha * theta), step alpha schedule)
- [x] `src/common/visualization.py` — added `show_inverse_result` (2x4 before/after) and `show_loss_history` (2-panel)
- [x] `experiments/02_inverse_aerial/demo_target_spot.py` — central r=1 lambda disk, complement forbidden, NA=0.4
- [x] `experiments/02_inverse_aerial/demo_forbidden_region.py` — central target + 4 explicit side-lobe forbidden disks
- [x] `configs/inverse_aerial.yaml` — reference config (documentation; demos still hard-code matching values)
- [x] `tests/test_inverse_losses.py`, `tests/test_optimize_mask.py` — 16 added (49 total green)

### Phase 4 — Partial coherence
- [x] `src/optics/source.py` — `sigma_axis`, `sigma_meshgrid`, `coherent_source`, `annular_source`, `dipole_source` (x / y), `quadrupole_source` (cross / X), `random_source`, `source_points` decoder
- [x] `src/optics/pupil.py` extended with `circular_pupil_at(grid, NA, center_freq, wavelength)` — off-axis pupil for the Hopkins integral
- [x] `src/optics/partial_coherence.py` — `partial_coherent_aerial_image` (batched over source points, autograd-friendly)
- [x] `src/common/metrics.py` (new) — `image_contrast`, `peak_intensity_in_region`, `integrated_leakage`, `normalized_image_log_slope`
- [x] `src/common/visualization.py` extended with `show_source` and `show_partial_coherence_sweep` (3 rows: mask | source | aerial)
- [x] `experiments/03_partial_coherence/demo_source_shapes.py` — line-space (pitch 1.5 lambda) and contact hole (r 0.5 lambda) imaged through 5 sources
- [x] `configs/partial_coherence.yaml`
- [x] `tests/test_source.py`, `tests/test_partial_coherence.py`, `tests/test_metrics.py` — 21 added (70 total green)
- [x] Results table saved to `outputs/logs/phase4_metrics.csv`

### Phase 5 — Resist exposure + diffusion
- [x] `src/resist/exposure.py` — `acid_from_aerial(aerial, dose, eta)` (Dill saturation)
- [x] `src/resist/diffusion_fd.py` — `laplacian_5pt`, `step_diffusion_fd`, `diffuse_fd` (explicit Euler with CFL guard, periodic BC via `torch.roll`)
- [x] `src/resist/diffusion_fft.py` — `diffuse_fft` (D, t) and `diffuse_fft_by_length` (single-knob L)
- [x] `src/resist/reaction_diffusion.py` — `step_reaction_diffusion`, `evolve_reaction_diffusion` (acid + quencher with non-negative clamp)
- [x] `src/resist/threshold.py` — `soft_threshold` (sigmoid), `hard_threshold`, `measure_cd_horizontal`, `thresholded_area`
- [x] `src/common/visualization.py` extended — `show_resist_chain` (aerial → acid → diffused → resist) and `show_resist_sweep` (rows of (acid, resist))
- [x] `experiments/04_resist_diffusion/demo_dose_sweep.py` (5-A) — doses {0.5, 1.0, 1.5, 2.0}
- [x] `experiments/04_resist_diffusion/demo_diffusion_length.py` (5-B) — L_diff {0, 0.05, 0.10, 0.20, 0.40} lambda
- [x] `configs/resist.yaml`
- [x] `tests/test_resist.py` — 21 added (91 total green)
- [x] Metrics CSVs: `phase5_dose_sweep_metrics.csv`, `phase5_diffusion_length_metrics.csv`

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
- **2026-04-29** Mask parameterization is `m = sigmoid(alpha * theta)` with
  unconstrained `theta`. Picked over a clamped real-valued mask because
  the sigmoid form is differentiable everywhere and pairs naturally with
  a binarization annealing schedule on `alpha`. `theta = 0` gives a
  neutral half-tone start (`m = 0.5`), which is also why the initial
  aerial in every Phase-3 figure is a flat 0.25.
- **2026-04-29** `OptimizationResult` is a dataclass instead of a tuple.
  Phase 9 will wrap this function inside a surrogate-assisted loop where
  named fields make the plumbing far more readable than positional
  unpacks would.
- **2026-04-29** `sigma_axis` uses `arange(-half, half+1) / half` rather
  than `linspace(-1, 1)`. The float32 rounding of linspace put the center
  pixel at ~5e-8 instead of 0, which shifted the equivalent pupil enough
  to swap a few boundary pixels and broke the "delta source = coherent
  imaging" sanity check by ~2 %.
- **2026-04-29** Hopkins integral evaluated as a batched FFT over a
  `(K, N, N)` pupil stack rather than a Python loop. With ~300 source
  points (annular) on n=256 the loop cost is dominated by FFT and the
  whole demo finishes in under a second on RTX 5080.
- **2026-04-29** Diffusion exposed via two parameterizations: the raw
  ``(D, t)`` pair and the more useful ``L = sqrt(2 D t)``. Sweeps in
  Phase 5 / 6 / 9 use ``L`` because the geometric scale is the only
  thing that physically matters; ``D`` and ``t`` separately are
  redundant degrees of freedom that introduce dimensional bookkeeping
  bugs.
- **2026-04-29** FD-vs-FFT agreement test uses a smooth Gaussian rather
  than a random field. The 5-point stencil is second-order accurate in
  dx; near-Nyquist random content makes a single explicit Euler step
  drift by ~10 % even when both solvers are correct.

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
- **2026-04-29** Phase 3 done (inverse aerial optimization). 49 tests
  green. Two demos run successfully: target_spot pulls mean target
  intensity from 0.25 to 0.99 with mean forbidden 0.08; forbidden_region
  drives four explicit side-lobe disks down to mean intensity 0.015.
  Resume with Phase 4: add `src/optics/source.py` (coherent / annular /
  dipole / quadrupole / random-points source shapes) and
  `src/optics/partial_coherence.py` (incoherent sum over shifted
  pupils), then run a sweep demo on a vertical line-space and a contact
  hole — save figures as `outputs/figures/phase4_*`.
- **2026-04-29** Phase 4 done (partial coherence). 70 tests green. Five
  source shapes imaged on two masks; metrics CSV saved at
  `outputs/logs/phase4_metrics.csv`. Verified that dipole-x triples the
  contrast of vertical line-space at pitch 1.5 lambda (0.81 vs
  coherent 0.25), dipole-y is the worst, and the contact hole prefers
  coherent illumination for peak intensity. Resume with Phase 5: add
  `src/resist/{exposure,diffusion_fd,diffusion_fft,reaction_diffusion,
  threshold}.py` + the two `experiments/04_resist_diffusion/demo_*.py`
  scripts, save figures as `outputs/figures/phase5_*`, and add a
  `phase5_metrics.csv` with CD-like width vs dose / diffusion length.
- **2026-04-29** Phase 5 done (resist exposure + diffusion). 91 tests
  green. Two demos run: dose sweep finds the under-exposure cliff at
  dose=0.5 (acid_max 0.39 just below threshold 0.40 -> nothing prints,
  CD jumps from 0 to 0.66 lambda when dose reaches 1.0); diffusion
  sweep shows CD invariant up to L=0.10 lambda then shrinking 14% by
  L=0.40 lambda as acid contrast collapses from 1.00 to 0.59. Resume
  with Phase 6: write `src/pinn/{pinn_base,pinn_diffusion,
  pinn_reaction_diffusion}.py`, run
  `experiments/05_pinn_diffusion/compare_fd_pinn.py` on the same
  initial condition as Phase 5, save figures + a metrics CSV with
  PINN vs FD MSE, edge error, training time, and inference time.
