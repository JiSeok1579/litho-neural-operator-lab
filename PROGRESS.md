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
- PyTorch (CUDA build) plus the standard scientific stack: numpy / scipy /
  matplotlib / h5py / pandas / hydra-core / omegaconf / tqdm / pytest /
  jupyterlab.

**How**
- `python3 -m venv .venv` → install PyTorch from the wheel index that
  matches the local GPU's compute capability → `pip install -r
  requirements.txt`.
- GPU sanity: `torch.cuda.is_available() == True`, then a 2048×2048 fp32
  matmul + complex64 FFT2 + autograd round-trip all pass on device.

**Why**
- An older PyTorch wheel can ship without kernels for the latest GPU
  compute capabilities, which surfaces at runtime as
  `no kernel image is available for execution`. Picking the matching cuXX
  wheel resolves this without code changes.

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

### A.12 Phase 9 — Closed-loop surrogate-assisted inverse design — ☑ done (2026-04-29)

**What**
- `src/closed_loop/surrogate_optimizer.py`
  - `CorrectionFn` type — callable ``(mask, T_thin) -> T_3d``.
  - `identity_correction_fn` (case A) returns ``T_3d = T_thin``.
  - `true_correction_fn(grid, params)` (case B) precomputes the
    closed-form ``C`` once and returns ``T_thin * C``.
  - `fno_correction_fn(fno, params, device)` (case C) builds the
    9-channel input from ``mask, T_thin, theta-broadcast``, calls the
    trained FNO, and returns ``T_thin + delta_T_pred`` as a complex
    tensor. Gradients flow from the mask through the FNO (eval mode,
    weights frozen by being absent from the optimizer's parameter
    list) back to the raw mask parameter.
  - `optimize_mask_with_correction` is the Phase-3 inverse loop with
    the correction injected between `fft2c(t)` and the pupil. Same
    target / forbidden / TV / binarization losses, same Adam +
    sigmoid-alpha annealing, same `OptimizationResult`-style return
    object.
  - `evaluate_mask_under_correction` computes the aerial intensity
    of a *fixed* mask through any correction — the validation step
    that turns "what FNO believed" into "what actually happens".
- `src/common/visualization.py` gained `show_closed_loop_comparison`
  with an optional `aerial_alt` curve overlaid as a dashed line on
  the y=0 cut, used in the Phase-9 figure to show "FNO predicted"
  next to "true correction" simultaneously.
- `experiments/07_closed_loop_inverse/optimize_with_fno.py` loads the
  Phase-8 checkpoint, runs the three optimizations and the
  validation, and writes two figures + a metrics CSV.
- `tests/test_surrogate_optimizer.py` — 6 tests added (132 / 132
  green): identity, true-correction equality with direct call,
  FNO-correction shape + gradient flow through mask, smoke
  optimization with identity, evaluator helper, and a
  loss-decreases-on-disk sanity check.

**How**
- The FNO is loaded from `outputs/checkpoints/fno_correction.pt` and
  switched to ``eval()`` so any future-added BN / dropout would behave
  deterministically. Weights are not in the Adam parameter list, so
  they never update; gradients still propagate through the network
  to the mask parameter.
- The "true theta" used for cases B and C is fixed at
  ``CorrectionParams(gamma=0.2, alpha=0.3, beta=-0.2, delta=0.1,
  s=0.3, c=2.0)`` — strong enough that the correction visibly
  changes the imaging chain. The FNO has seen similar magnitudes
  during Phase-7 sampling but this exact theta is unlikely to be in
  the training set.
- The validation step takes the C-optimized mask and runs it through
  ``evaluate_mask_under_correction(grid, mask_C, true_correction_fn(...),
  NA, wavelength)``. This is the single measurement that exposes
  surrogate dishonesty.

**Why**
- A single optimizer driving all three cases is the right
  experimental control. If the optimizer changed between cases (e.g.
  different lr schedule or mask parameterization for the FNO case),
  any difference in outcomes would be confounded with optimizer
  choice rather than correction model.
- Predicting `delta_T` (Phase 8) instead of `T_3d` makes the FNO
  case naturally compatible with the imaging chain: ``T_3d_pred =
  T_thin + delta_T_pred`` is a single complex addition, no extra
  scaling.
- `evaluate_mask_under_correction` is decoupled from
  `optimize_mask_with_correction` so any future surrogate (DeepONet,
  diffusion model) can be plugged in for case C without touching the
  optimizer.

**Why (this direction)**
- Phase 9 is the lab's payoff. Phases 1-5 build the physics, Phase 6
  benchmarks PINN, Phase 7-8 train the surrogate. Phase 9 puts that
  surrogate to work inside an actual decision-making loop and asks
  the question that motivates the whole exercise: "does the
  surrogate's optimization output survive a fall back to true
  physics?" The lab's headline finding is that on a visibly
  challenging correction (~30% asymmetric shadow + 20% Gaussian
  taper), an FNO with 15% test complex-relative error produces an
  optimized mask that ends up ~11% off in target intensity when
  re-imaged through the truth.

**Verified results from `phase9_metrics.csv`**

```
case                                target  forbidden  loss_target  loss_bg
A physics-only                      0.989   0.070      0.0005       0.0341
B true correction                   0.989   0.072      0.0005       0.0351
C FNO (predicted)                   0.982   0.068      0.0021       0.0281
C validated under true correction   1.091   0.075      0.0103       0.0352
```

Findings:
- A and B converge to nearly identical mean intensities (target ~
  0.99, forbidden ~ 0.07), but their *masks* differ — case B's
  optimizer pre-compensates for the multiplicative correction so the
  post-correction aerial matches the target. The lesson: when the
  loss is on the post-correction output, the upstream mask shape
  carries the correction's signature.
- Case C's FNO believed it was producing a target intensity of 0.982
  with loss_target 0.0021. After re-evaluating the same mask under
  the true correction, the actual target intensity is 1.091 (a
  ~11 % over-shoot beyond the requested 1.0) and loss_target jumps
  ~5x to 0.0103.
- Forbidden intensity stays roughly comparable (0.068 -> 0.075), so
  the leakage-suppression part of the optimization survives the
  surrogate; the failure mode here is in the target band.
- The y=0 cut in `phase9_closed_loop_comparison.png` shows the
  "predicted" and "true" aerials of case C as two visibly separated
  curves crossing at the target threshold — the canonical "the
  optimizer was lied to" picture from study plan §9.6.

This validates exactly the lesson the lab was set up to teach: a
neural surrogate that scores well on its training-distribution test
metric (15.5 % complex relative error) can still mislead an
optimizer that runs against it, and a re-imaging pass through the
true physics is mandatory after any surrogate-assisted optimization.

**Next**
- Phase 10 (optional): active learning. Use the surrogate dishonesty
  signal to pick which (mask, theta) pairs to add to the training
  set, retrain, and check whether the gap between (C predicted) and
  (C validated) shrinks. Three uncertainty signals on the menu —
  ensemble variance, distance-to-training-distribution, or
  oracle-disagreement — but for a study repo a single pass with
  oracle disagreement (we already *have* the true correction) is the
  highest-value version.

---

### A.11 Phase 8 — FNO 2D correction surrogate — ☑ done (2026-04-29)

**What**
- `src/neural_operator/fno2d.py`
  - `SpectralConv2d(in_channels, out_channels, modes_x, modes_y)`:
    rfft2 -> low-mode (modes_x x modes_y) complex weight multiplication
    -> irfft2. Two weight tensors handle the positive- and negative-x
    Fourier bands.
  - `FNOBlock2d`: `spectral_conv(x) + 1x1 conv(x)` followed by GELU.
  - `FNO2d(in_channels, out_channels, hidden, modes_x, modes_y, n_layers)`:
    1x1 lift -> N FNO blocks -> 2-layer 1x1 head.
- `src/neural_operator/datasets.py` — `CorrectionDataset` reads the
  Phase-7 NPZ archive and assembles a 9-channel input
  ``(mask, T_thin_real, T_thin_imag, theta_0..theta_5)`` and a 2-channel
  target ``(delta_T_real, delta_T_imag)`` per sample. ``target="T_3d"``
  is also exposed for ablations that prefer absolute prediction.
- `src/neural_operator/train_fno.py` — training loop with optional
  weight decay (Adam vs AdamW), step LR schedule, and an optional
  physics-aware aerial-intensity term controlled by ``weight_aerial``.
  Also exposes the loss helpers (`spectrum_mse`,
  `complex_relative_error`, `aerial_intensity_mse`) and `evaluate_fno`.
- `src/common/visualization.py` gained `show_fno_predictions` (per-row
  panels: |Delta T| true / |Delta T| pred / |error| / |T_3d pred|) and
  `show_fno_training` (loss + complex relative error subplots).
- `experiments/06_fno_correction/train_fno_correction.py` runs the full
  recipe at ``hidden=32, modes=16, n_layers=4`` for 100 epochs of
  AdamW (lr=1e-3, weight_decay=1e-4, step LR every 35 epochs).
  Saves `outputs/checkpoints/fno_correction.pt` plus two figures and a
  metrics CSV.
- `configs/fno.yaml` — reference config.
- `tests/test_fno.py` — 10 tests added (126 / 126 green) covering
  spectral-conv shape, resolution-independent forward, FNO block /
  full-stack shapes, autograd flow, complex-relative-error contract,
  aerial-MSE zero-on-match, dataset channel layout (9-in / 2-out),
  delta_T vs T_3d target modes, and a tiny end-to-end smoke run.

**How**
- Channel layout was chosen so the FNO sees both spatial-domain
  (mask) and frequency-domain (T_thin) information, with theta as
  spatial-broadcast channels. The FNO's internal rfft2 then runs over
  this mixed-domain stack and learns the multiplicative correction
  structure. Predicting `delta_T = T_3d - T_thin` instead of `T_3d`
  directly turns the task into residual learning, which converges
  faster on the same architecture.
- Modes 16x16 truncation captures the smooth correction operator
  well: ``C(fx, fy)`` is a low-frequency function (Gaussian taper +
  polynomial phase + tanh asymmetry), so the high-frequency content
  of the spectrum is mostly preserved by the mask spectrum itself.
- Adam was upgraded to AdamW with `weight_decay=1e-4` after the first
  pass at 800 train samples showed clear overfitting (train MSE
  dropped to 5e-4 while test MSE stalled at ~4e-3). Weight decay
  helped marginally (8.1x -> 9.0x baseline ratio at 800 samples).
- Doubling the dataset to 2000 train + 400 test pushed the test
  spectrum MSE to 9e-4 and the complex relative error to 15.5%, a
  35x improvement over the identity baseline. The remaining train /
  test gap (train MSE ~1e-4 vs test ~9e-4) is the residual
  generalization gap; reducing it further is mainly a "more data"
  problem on this synthetic operator.

**Why**
- A FNO is the right surrogate family for this dataset because the
  target operator (multiplicative correction in spatial frequency
  space, smooth in (fx, fy)) is naturally captured by low-mode
  spectral convolutions. A plain CNN would have to learn the FFT
  implicitly through repeated convolutions; FNO bakes it in.
- Predicting Delta T rather than T_3d makes the regression target
  smaller in magnitude and centers it on zero, which is friendlier
  for fp32 training. Phase 9 still gets the full T_3d via
  `T_thin + delta_T_pred` at inference time.
- Adding `weight_aerial` as a configurable knob (currently 0.0) lets
  Phase 9 turn on the physics-aware aerial-image term once the
  surrogate is wired into the closed-loop optimization, without
  rewriting the trainer.

**Why (this direction)**
- Phase 8 is the operator-learning answer to Phase 6's PINN: instead
  of fitting a single PDE solution one (x, y, t) at a time, the FNO
  learns an *operator family* — every (mask, theta) pair is one
  query into the same trained network. Phase 9 puts this surrogate
  in the gradient loop of the inverse mask design from Phase 3.

**Verified results from `phase8_metrics.csv`**

```
metric                           value
n_train                          2000
n_test                           400
grid_n                           128
fno_params                       2,106,178
epochs                           100
batch_size                       8
train_time_sec                   131.9
final_test_spectrum_mse          9.038e-04
final_test_complex_rel_err       1.550e-01     (15.5 %)
final_test_aerial_mse            1.590e-03
baseline_spectrum_mse_pred_0     3.044e-02
improvement_over_baseline        35.0x
```

Findings:
- 35x improvement over the identity baseline (predicting `delta_T = 0`)
  at 100 epochs of AdamW on 2000 train samples.
- Train MSE finishes at ~1.1e-4; test MSE at ~9e-4. The remaining
  ~10x train / test gap is the dataset-size-limited generalization
  ceiling on this synthetic correction.
- Per-sample qualitative inspection (`phase8_fno_predictions.png`):
  the FNO captures the spectrum structure on contact-hole, random,
  and line-space masks; the largest absolute errors sit at the very
  tall line-space peaks (where a 5-10% relative error still produces
  ~0.7 absolute error on a tall narrow Bessel-like spike).
- The DeepONet variant from study plan §8.3 was *not* implemented in
  this phase — the FNO already serves the Phase-9 closed loop and
  the DeepONet would be a parallel ablation rather than a new lesson.
  Listed as deferred in the module checklist.

**Next**
- Phase 9: closed-loop surrogate-assisted inverse design. Take the
  trained FNO and put it inside the Phase-3 mask-optimization loop:
  mask -> T_thin -> FNO predicts delta_T -> T_3d_pred = T_thin +
  delta_T_pred -> aerial -> region-based loss -> backprop into the
  raw mask parameter. Compare three optimizations:
  (A) physics-only (no correction),
  (B) true synthetic correction (oracle),
  (C) FNO-surrogate correction.
  Re-validate (C)'s optimized mask under (B) to see whether the
  surrogate is being "tricked" by the optimizer. Save results under
  `outputs/figures/phase9_*` and a metrics CSV.

---

### A.10 Phase 7 — Synthetic 3D mask correction dataset — ☑ done (2026-04-29)

**What**
- `src/neural_operator/synthetic_3d_correction_data.py`
  - `CorrectionParams` dataclass (gamma / alpha / beta / delta / s / c)
    with `to_array` / `from_array` / `identity` helpers.
  - `correction_operator(grid, params)` builds the closed-form complex
    correction `C(fx, fy)` on a Grid2D as
    ``C = exp(-gamma * |f|^2) * exp(i * (alpha fx + beta fy +
    delta * (fx^2 - fy^2))) * (1 + s * tanh(c * fx))``.
  - `apply_3d_correction(T_thin, C)` — multiplicative pairing.
  - `sample_correction_params(rng, ranges)` draws theta uniformly within
    `DEFAULT_RANGES` (gamma 0..0.30, alpha / beta -0.40..0.40, delta
    -0.30..0.30, s -0.40..0.40, c 0..4.0).
  - `random_mask_sampler(grid, seed)` mixes 50 % block-random binary +
    25 % line-space + 25 % contact-hole arrangements.
  - `generate_dataset(grid, n_samples, output_path, seed)` produces a
    compressed NPZ with `masks`, `T_thin_real / T_thin_imag`,
    `T_3d_real / T_3d_imag`, `theta`, `theta_names`, `grid_n`,
    `grid_extent`. Returns a summary dict for logging.
  - `load_dataset(path)` mirror of the saver.
- `src/common/visualization.py` gained `show_correction_samples` for
  per-sample 5-panel rows (mask | |T_thin| | |C| | |T_3d| | |Delta T|)
  with frequency zoom + percentile clipping.
- `experiments/06_fno_correction/generate_synthetic_dataset.py` runs
  on `n=128, extent=8 lambda` and writes the train (800 samples) and
  test (200 samples) NPZs plus a 4-row preview figure and a summary
  CSV.
- `tests/test_synthetic_correction.py` — 11 tests covering identity
  parameters, multiplicative pairing, amplitude / phase symmetry
  contracts, asymmetric-shadow non-symmetry, gamma differentiability,
  CorrectionParams roundtrip, sampling-range bounds, mask-sampler
  binary contract, NPZ key set + shapes, and a recompute-from-saved-
  theta consistency check (T_3d == T_thin * correction_operator(theta)
  to better than 1e-5 relative).

**How**
- The correction operator is intentionally synthetic. Real 3D mask
  effects (absorber height, sidewall angle, oblique incidence,
  polarization, multi-layer interference) require RCWA / FDTD; we are
  building a closed-form analogue so the Phase-8 FNO surrogate can be
  benchmarked against an exact reference.
- Data layout uses real / imaginary channels separately rather than
  PyTorch's complex dtype because NPZ stores numpy arrays and numpy's
  complex64 is awkward to consume from the FNO side. Phase 8 will
  reassemble the complex form on load.
- `theta` is stored as a length-6 vector per sample. Phase 8 / 9
  broadcast each component to a spatial channel inside the dataloader
  rather than burning storage on N x 6 x H x W maps.
- Mask sampler uses three families with mixed weights so the dataset
  covers periodic / random / structured patterns. `block_size = 2..8`
  in random_binary deliberately keeps features above 1 lambda where the
  spectrum has visible diffraction orders rather than degenerating to
  white noise.

**Why**
- A single closed-form correction operator gives Phase 8 / 9 something
  exact to compare against — both the surrogate and the closed-loop
  experiment can run "true" vs "predicted" diff plots without any
  RCWA dependency.
- Storing `T_thin` and `T_3d` separately (rather than only one and the
  recipe to compute the other) wastes a bit of disk but makes the
  dataset self-contained for follow-up phases that may want to predict
  `T_3d` directly, predict `Delta T`, or experiment with both
  parameterizations.
- The mask families are kept simple (no aerial-image processing here)
  because the FNO is learning the correction operator, not the
  imaging chain.

**Verified results from `phase7_dataset_summary.csv`**

```
split  n     |T_thin|  |T_3d|   |Delta T|  gamma   alpha   beta    delta   s       c
train  800   0.1622    0.0643   0.1260     +0.150  -0.010  -0.006  +0.003  +0.014  2.018
test   200   0.1397    0.0547   0.1064     +0.148  +0.007  -0.026  +0.011  +0.009  1.955
```

Signal sanity:
- The correction is substantial: ``|Delta T| / |T_thin|`` is roughly
  78 % on train and 76 % on test, so there is real geometry for an
  FNO to fit (a low ``|Delta T|`` would mean the dataset is mostly
  identity and an ``output = input`` baseline already wins).
- ``|T_3d| / |T_thin|`` is roughly 40 %, which matches the expected
  bulk attenuation from a non-zero gamma sampled in [0, 0.3] (mean
  0.15, attenuating high-frequency content by ``exp(-0.15 * |f|^2)``).
- theta means are centered on the midpoints of their sampling ranges
  with reasonable variance — no sampler is biased.
- Data generation was IO-bound and finished in 3.6 s (train) + 0.8 s
  (test) on a CUDA GPU.

**Next**
- Phase 8: train an FNO 2D and (optionally) a DeepONet on the saved
  NPZs. Inputs ``(mask, Re(T_thin), Im(T_thin), theta-broadcast)``,
  outputs ``(Re(Delta T), Im(Delta T))`` (or ``Re(T_3d), Im(T_3d))``).
  Loss: spectrum MSE + a physics-aware aerial-image term so spectrum
  errors that don't matter for imaging do not dominate. Save trained
  checkpoints under `outputs/checkpoints/` and a metrics CSV with
  spectrum MSE, complex relative error, and downstream aerial-image
  MSE on the test split.

---

### A.9 Phase 6 — PINN for the 2D heat equation — ☑ done (2026-04-29)

**What**
- `src/pinn/pinn_base.py` — `FourierFeatures` (frozen random projection
  followed by sin/cos), `MLP`, `PINNBase` (input normalization to
  [-1,1]^3 against `x_range` / `t_range`, Fourier features, MLP head).
- `src/pinn/pinn_diffusion.py` — `PINNDiffusion(D, hard_ic=True,
  A0_callable=...)`. With ``hard_ic=True`` the network output is
  computed as ``A_pinn = A0(x, y) + t * MLP_output(x, y, t)`` so the
  initial condition is satisfied exactly at ``t=0`` by construction.
  `train_pinn_diffusion` runs Adam with optional step-LR scheduling
  and samples IC points from a regular grid + a random splash.
  Helpers: `pinn_to_grid`, `gaussian_initial_condition`,
  `gaussian_analytic_solution` (closed-form Gaussian heat-equation
  solution).
- `src/pinn/pinn_reaction_diffusion.py` — `PINNReactionDiffusion`
  with coupled `(A, Q)` outputs and `pde_residuals` returning the
  acid + quencher equation residuals (training loop is left to a
  follow-up phase since reaction-diffusion has no analytic baseline).
- `src/common/visualization.py` gained `show_pinn_vs_solvers`
  (2x4: top row solutions, bottom row absolute errors vs analytic)
  and `show_pinn_training` (loss components on log-y).
- `experiments/05_pinn_diffusion/compare_fd_pinn.py` runs the four
  solvers (analytic / FD / FFT / PINN) on the same Gaussian IC at
  `sigma=0.5, D=0.1, t_end=1` over a `n=128, extent=8` grid and writes
  `phase6_pinn_vs_solvers.png`, `phase6_pinn_training.png`, and a
  metrics CSV.
- `configs/pinn.yaml` — reference parameter sheet.
- `tests/test_pinn.py` — 14 new tests (105 / 105 green) covering
  Fourier-feature shape and reproducibility, MLP layer count, base
  forward shape and shape mismatch rejection, PDE residual shape and
  grad-input contract, Gaussian IC at the origin, analytic decay over
  time, mass conservation of the analytic solution, `pinn_to_grid`
  shape, smoke-run sanity, and reaction-diffusion residual shape.

**How**
- Input normalization to `[-1, 1]^3` lifts spectral bias: with
  `x_range=(-4, 4)`, naive Fourier features at `scale=2` produce
  encodings oscillating ~8 times across the domain, and a small MLP
  cannot integrate that smoothly to fit a localized Gaussian peak.
  Normalizing inside `forward` then using `fourier_scale=2.5` on
  unit-cube inputs gives a much cleaner encoding.
- Hard-IC parameterization (`A0(x,y) + t * MLP`) is the key
  architectural trick. The vanilla soft-IC PINN on this problem fell
  into a trivial local minimum where the network predicted ~0 for
  ``t > 0`` and only fit `A0` at `t=0`: most of the IC sample space
  has `A0 ≈ 0` so soft-IC was satisfied while the PDE residual stayed
  moderate (~5e-2). Hard-IC removes that minimum entirely — at `t=0`
  the network output is exactly `A0` regardless of weights, so all
  training pressure goes onto the PDE residual, which is the actual
  dynamics signal we want to learn.
- Initial-condition sampling combines a regular grid (small, e.g.
  8x8 = 64 points) with a random splash. With `hard_ic=True` the IC
  loss is set to zero so this sampling is only a fallback / sanity
  guard.
- LR schedule: Adam at `1e-3`, step-decay by `0.5` every 4000 iters.
  10000 total iters takes ~80 seconds on a CUDA GPU.

**Why**
- The Phase-6 pipeline reuses Phase-5 FD and FFT solvers verbatim, so
  the comparison is exactly "same equation, same IC, same grid,
  three different numerical strategies".
- Closed-form Gaussian ground truth was chosen because it lets every
  solver be benchmarked against the same exact reference, avoiding the
  circularity of "compare PINN to FD" when FD is itself the only
  reference.
- `hard_ic` is exposed as a flag rather than a default-only behavior
  because the `PINNReactionDiffusion` follow-up will need a different
  hard-IC form (one for `A`, another for `Q`); the base PINN should
  not over-commit to the diffusion-only construction.

**Why (this direction)**
- Phase 6 is the lab's first ML phase. The headline finding — that on
  a closed-form-tractable diffusion problem PINN is ~6 orders of
  magnitude worse than FFT and ~3 orders of magnitude worse than FD
  while costing 80 seconds of training — is exactly the lesson study
  plan §6.8 calls for. The PINN's value isn't winning this benchmark;
  it is in (a) handling irregular geometries / BCs that are awkward
  for grid solvers, (b) inverse parameter estimation, and (c)
  providing a continuous mesh-free representation. Phase 6 establishes
  *where the PINN's value actually sits* before Phase 8 moves on to
  operator-learning surrogates that compete with FFT on speed for a
  family of operator instances.

**Verified results from `phase6_metrics.csv`**

```
solver    MSE vs analytic   max abs err   wall-clock
analytic  0.000e+00         0.000e+00      0.000 ms / call
FD        1.330e-10         4.86e-05       7.27 ms / call
FFT       3.140e-16         1.79e-07       0.10 ms / call
PINN      4.235e-04         1.71e-01       0.37 ms / call (after 78 s training)
```

Numerical gap:
- PINN max abs err is ~3 orders worse than FD and ~6 orders worse than
  FFT. The error is concentrated at the Gaussian peak (PINN peaks at
  0.73 vs analytic 0.56 at t=1) — the network does not fully learn the
  diffusion dynamics, and a longer training schedule + SIREN-style
  activations (on the to-do list) could narrow the gap further.
- Once trained, PINN inference cost (0.37 ms / call) is in the same
  ballpark as FFT (0.10 ms) but trails FD by ~20x on this 128² grid.
  The mesh-free interpolation property remains the PINN's structural
  advantage.

**Next**
- Phase 7: synthetic 3D mask correction. Build a callable correction
  operator `C(fx, fy; theta)` that applies amplitude + phase
  modifications to the thin-mask diffraction spectrum, generate a
  paired dataset of `(mask, theta) -> (T_thin, T_3d)`, save as NPZ.
  No training yet — just the data pipeline. Land
  `outputs/datasets/synthetic_3d_correction_{train,test}.npz` and
  a few preview figures.

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
  the demo runs sub-second on a modern CUDA GPU.
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
| 6 | PINN for diffusion | ☑ | PINN vs FD comparison |
| 7 | Synthetic 3D mask correction | ☑ | correction dataset NPZ |
| 8 | FNO / DeepONet surrogate | ☑ | surrogate `.pt` checkpoint |
| 9 | Closed-loop surrogate-assisted inverse | ☑ | surrogate vs true comparison |
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
- [x] `src/pinn/pinn_base.py` — `FourierFeatures`, `MLP`, `PINNBase` (with input normalization to [-1,1]^3)
- [x] `src/pinn/pinn_diffusion.py` — `PINNDiffusion` (with optional `hard_ic=True` architectural IC enforcement), `train_pinn_diffusion` (Adam + step LR scheduler, regular IC grid + random IC mix), `pinn_to_grid`, `gaussian_initial_condition`, `gaussian_analytic_solution`
- [x] `src/pinn/pinn_reaction_diffusion.py` — coupled (A, Q) PINN with `pde_residuals`
- [x] `src/common/visualization.py` extended — `show_pinn_vs_solvers` (2x4 solutions / errors), `show_pinn_training`
- [x] `experiments/05_pinn_diffusion/compare_fd_pinn.py` — analytic Gaussian + FD + FFT + PINN, with metrics CSV
- [x] `configs/pinn.yaml`
- [x] `tests/test_pinn.py` — 14 added (105 total green)
- [x] Metrics CSV: `outputs/logs/phase6_metrics.csv`

### Phase 7 — Synthetic 3D correction dataset
- [x] `src/neural_operator/synthetic_3d_correction_data.py` — `CorrectionParams`, `correction_operator`, `apply_3d_correction`, `sample_correction_params`, `random_mask_sampler`, `generate_dataset`, `load_dataset`
- [x] `src/common/visualization.py` extended — `show_correction_samples` (rows of mask | |T_thin| | |C| | |T_3d| | |Delta T|)
- [x] `experiments/06_fno_correction/generate_synthetic_dataset.py` — preview + train/test NPZ generation
- [x] `outputs/datasets/synthetic_3d_correction_train.npz` (800 samples, n=128)
- [x] `outputs/datasets/synthetic_3d_correction_test.npz` (200 samples, n=128)
- [x] `outputs/figures/phase7_correction_samples.png`
- [x] `outputs/logs/phase7_dataset_summary.csv`
- [x] `tests/test_synthetic_correction.py` — 11 added (116 total green)

### Phase 8 — FNO / DeepONet surrogate
- [x] `src/neural_operator/fno2d.py` — `SpectralConv2d`, `FNOBlock2d`, `FNO2d`
- [x] `src/neural_operator/datasets.py` — `CorrectionDataset` (loads Phase-7 NPZ, builds 9-channel input + 2-channel target)
- [x] `src/neural_operator/train_fno.py` — `train_fno_correction` (Adam / AdamW with optional weight decay, optional aerial-image physics-aware term), `evaluate_fno`, `spectrum_mse`, `complex_relative_error`, `aerial_intensity_mse`
- [x] `src/common/visualization.py` extended — `show_fno_predictions` (per-sample 4-panel rows: |delta_T| true / pred / err / |T_3d| pred), `show_fno_training` (loss + complex rel err)
- [x] `experiments/06_fno_correction/train_fno_correction.py`
- [x] `configs/fno.yaml`
- [x] `outputs/checkpoints/fno_correction.pt`
- [x] `outputs/figures/phase8_fno_{training,predictions}.png`
- [x] `outputs/logs/phase8_metrics.csv`
- [x] `tests/test_fno.py` — 10 added (126 total green)
- [ ] DeepONet variant (deferred — FNO already serves Phase 9; can revisit later)

### Phase 9 — Closed-loop surrogate inverse
- [x] `src/closed_loop/surrogate_optimizer.py` — `optimize_mask_with_correction`, `evaluate_mask_under_correction`, three correction factories (identity / true / FNO)
- [x] `src/common/visualization.py` extended — `show_closed_loop_comparison` (rows of mask | aerial | y=0 cut | regions overlay; supports a dashed alt-aerial curve for the validation row)
- [x] `experiments/07_closed_loop_inverse/optimize_with_fno.py` — runs A / B / C plus the C-under-true validation
- [x] `outputs/figures/phase9_closed_loop_comparison.png` (4 rows: A / B / C / C-validated)
- [x] `outputs/figures/phase9_loss_history.png` (loss + region intensity per case)
- [x] `outputs/logs/phase9_metrics.csv`
- [x] `tests/test_surrogate_optimizer.py` — 6 added (132 total green)

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

- **2026-04-29** Picked a PyTorch wheel that ships kernels for the local
  GPU's compute capability. Stable build, so no nightly-pinning risk.
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
  whole demo finishes in under a second on a modern CUDA GPU.
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
- **2026-04-29** PINN inputs are normalized to `[-1, 1]^3` inside
  `PINNBase.forward`. With raw physical coordinates (extent up to 8),
  Fourier features at any reasonable `scale` produce encodings whose
  oscillations span dozens of cycles across the domain, and a small
  MLP cannot integrate that smoothly to fit a localized Gaussian. The
  normalization restores spectral bias to the wavelength scale of
  the actual target.
- **2026-04-29** PINN initial condition is enforced architecturally
  (`A_pinn = A0 + t * MLP`) rather than as a soft loss term. Soft IC
  on a localized Gaussian falls into a trivial minimum where the
  network outputs ~0 everywhere for `t > 0`: most of the random IC
  sample space already has `A0 ≈ 0`, so the soft-IC penalty is
  satisfied while the PDE residual stays at ~5e-2 with no gradient
  signal to escape. Hard-IC removes that minimum entirely.
- **2026-04-29** All hardware-specific identifiers (GPU model,
  compute capability, OS version, CUDA wheel suffix) were stripped
  from `README.md`, `PROGRESS.md`, `requirements.txt`, and the merged
  PR bodies. Past commit messages still contain those strings; the
  user has not asked for a force-push history rewrite, so the
  immutable history remains untouched.

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
- **2026-04-29** Phase 6 done (PINN diffusion). 105 tests green. PINN
  vs FD vs FFT compared against a closed-form Gaussian ground truth;
  PINN reaches MSE 4.2e-4 / max abs err 0.17 after ~80 s of training,
  which is 3 orders of magnitude worse than FD and 6 orders worse than
  FFT — confirming the study-plan §6.8 lesson that on simple diffusion
  PINNs lose to grid solvers on both accuracy and speed. The
  hard-IC architectural trick (`A_pinn = A0 + t * MLP`) was needed to
  escape a trivial soft-IC local minimum where the network predicts ~0
  for `t > 0`. Resume with Phase 7: synthetic 3D mask correction
  dataset (no training yet, just the data pipeline).
- **2026-04-29** Phase 7 done (synthetic 3D-mask correction dataset).
  116 tests green. 800 train + 200 test samples generated at n=128 in
  4.4 s on GPU. The correction is substantial (|Delta T| / |T_thin| ~
  77 %), so the dataset has real geometry for an FNO to fit. Resume
  with Phase 8: train FNO 2D (and optional DeepONet) on the saved
  NPZs; report spectrum MSE, complex relative error, and downstream
  aerial-image MSE.
- **2026-04-29** Phase 8 done (FNO 2D correction surrogate). 126 tests
  green. After re-generating the dataset at 2000 train + 400 test, a
  hidden=32 / modes=16 / 4-layer FNO trained 100 epochs of AdamW
  (weight_decay=1e-4) reaches test spectrum MSE 9.0e-4 and complex
  relative error 15.5 %, a 35x improvement over the identity baseline
  (predict `delta_T = 0`). 132 s total training. Resume with Phase 9:
  put the trained FNO inside the Phase-3 mask-optimization loop and
  compare (A) physics-only, (B) true correction, and (C)
  FNO-surrogate inverse runs. Re-validate (C) under (B) to detect
  surrogate dishonesty.
- **2026-04-29** Phase 9 done (closed-loop surrogate-assisted inverse
  design). 132 tests green. The lab's payoff finding: an FNO with
  15.5 % test complex-relative error produces an optimized mask that
  the surrogate predicts has target intensity 0.982 (loss_target
  0.0021), but when re-imaged through the true correction the same
  mask gives target intensity 1.091 (loss_target 0.0103) — about
  11 % over-shoot and 5x worse target loss. Forbidden intensity is
  preserved (0.068 -> 0.075), so the leakage-suppression part of
  the optimization survives, but the on-target dose is overstated.
  This is the canonical "surrogate dishonesty" demonstration the
  study plan §9.6 set up. Resume with Phase 10 (optional): active
  learning using oracle disagreement to refine the training set and
  shrink the predicted-vs-true gap.
