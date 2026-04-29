# PEB submodule (`reaction_diffusion_peb/`)

**Status:** Phases 1 – 5 done (synthetic aerial + exposure, FD / FFT
diffusion baselines, PINN diffusion, acid loss, deprotection P-field).
Phases 6 – 11 planned. 83 PEB tests green; total repo tests at 215 / 215.

## Goal

A separate workspace under `reaction_diffusion_peb/` that studies
**post-exposure-bake (PEB) resist physics** — acid generation, PEB
diffusion, deprotection, temperature dependence, quencher reaction —
independently from the main lithography pipeline.

The methodological choice is the opposite of the main project's
Phase 7 → 8 → 9 path. Because **no simulation dataset exists yet**,
the submodule:

1. Builds FD / FFT physics baselines first.
2. Trains a **PINN before** any DeepONet / FNO surrogate.
3. Always benchmarks the PINN against FD / FFT.
4. Only after FD / FFT / PINN runs are saved as a paired dataset
   does it move on to operator learning.

## Submodule phase plan (within `reaction_diffusion_peb/`)

| # | Topic | Status |
|---|---|---|
| 1 | Synthetic aerial + exposure → initial acid `H0` | ✅ done |
| 2 | Diffusion-only FD / FFT baselines | ✅ done |
| 3 | PINN diffusion vs FD / FFT | ✅ done |
| 4 | Acid loss `kloss` | ✅ done |
| 5 | Deprotection `kdep` (`P` field) | ✅ done |
| 6 | Arrhenius temperature dependence | planned |
| 7 | Quencher reaction (`kq` safe vs stiff target) | planned |
| 8 | Full reaction-diffusion (everything combined) | planned |
| 9 | Save FD / PINN runs as a dataset | planned |
| 10 | DeepONet / FNO operator surrogate (optional) | planned |
| 11 | Petersen / temperature uniformity / z-axis | planned |

## Key constraints (from the plan)

- **Self-contained.** Early phases do **not** import from the main
  repo's `src/{mask, optics, inverse, neural_operator, closed_loop}`.
  Inputs are synthetic exposure maps generated inside the submodule.
- **Stiff caveat.** `kq ∈ [100, 1000] s⁻¹` is the realistic range but
  is stiff. Start at `kq = 1, 5, 10` then add semi-implicit /
  operator-splitting machinery before pushing higher.
- **Naming caveat.** Never use the bare letter `D` — it is ambiguous
  between dose, diffusion coefficient, and deprotected fraction.
  Use `dose`, `DH`, `DQ`, `P` explicitly.

## Phase 1 — what's already there

```text
reaction_diffusion_peb/
  src/synthetic_aerial.py       gaussian_spot, line_space, contact_array,
                                 two_spot, normalize_intensity
  src/exposure.py                acid_generation (Dill-style)
  src/visualization.py           show_aerial_and_acid, show_dose_sweep
  configs/exposure.yaml          dose / eta / Hmax sweep parameters
  configs/minimal_diffusion.yaml reference for Phase 2

  experiments/01_synthetic_aerial/
    run_gaussian_spot.py         dose ∈ {0.5, 1, 1.5, 2.0}
    run_line_space.py            same dose sweep on smooth-edged lines

  tests/                         20 tests:
                                   - shape / orientation / periodicity
                                   - duty / contrast / smooth-edge contracts
                                   - exposure monotonicity in dose / eta / Hmax
                                   - H0 ≤ Hmax saturation
                                   - I = 0 → H0 = 0
                                   - exposure-step differentiability
```

Verified results from
`reaction_diffusion_peb/outputs/logs/peb_phase1_gaussian_metrics.csv`:

| dose | I_peak | H0_peak | H0 ≤ Hmax |
|---|---|---|---|
| 0.5 | 0.998 | 0.0786 | yes |
| 1.0 | 0.998 | 0.1263 | yes |
| 1.5 | 0.998 | 0.1553 | yes |
| 2.0 | 0.998 | 0.1728 | yes (saturating toward Hmax = 0.2) |

## Phase 2 — what's already there

```text
reaction_diffusion_peb/
  src/fft_utils.py          fft2c / ifft2c centered FFT, freq_grid_nm
                             helper (cycles/nm).
  src/diffusion_fd.py        laplacian_5pt, step_diffusion_fd,
                             diffuse_fd (CFL-guarded explicit Euler).
  src/diffusion_fft.py       diffuse_fft (D, t) and
                             diffuse_fft_by_length (single L knob).

  experiments/02_diffusion_baseline/
    run_diffusion_fd.py      DH sweep {0.3, 0.8, 1.5} nm^2/s at t=60s
    run_diffusion_fft.py     same sweep using exact heat kernel
    compare_fd_fft.py        per-DH side-by-side, abs and L2 rel error,
                             y=0 row cut, wall-clock timing
```

Verified results from
`reaction_diffusion_peb/outputs/logs/peb_phase2_compare_fd_fft_metrics.csv`:

| DH (nm²/s) | t (s) | L (nm) | FD peak | FFT peak | max\|FD-FFT\| | L2 rel err |
|---|---|---|---|---|---|---|
| 0.30 | 60 | 6.00 | 0.1087 | 0.1087 | 2.1e-06 | 2.8e-05 |
| 0.80 | 60 | 9.80 | 0.0862 | 0.0862 | 2.6e-06 | 4.7e-05 |
| 1.50 | 60 | 13.42 | 0.0661 | 0.0661 | 2.2e-06 | 5.3e-05 |

- FD and FFT agree to 5–6 decimal places (max abs err ≈ 2e-6).
- Both solvers preserve total mass exactly (no loss term).
- FFT is ~10–50× faster (sub-millisecond) than the CFL-bounded FD
  loop.
- Larger `DH` produces more blur and lower peak — exactly the heat-
  equation expectation.
- The `|FD − FFT|` panel shows an 8-petal pattern at the 1e-6 level,
  the signature of the 5-point stencil's 2nd-order truncation
  hitting where the 4th-derivative of the Gaussian peaks.

## Phase 3 — what's already there

```text
reaction_diffusion_peb/
  src/pinn_base.py            FourierFeatures, MLP, PINNBase with
                              [-1, 1]^3 input normalization built in.
  src/pinn_diffusion.py        PINNDiffusion (hard_ic=True default,
                              H_pinn = H0(x, y) + t_norm * MLP),
                              gaussian_spot_acid_callable,
                              train_pinn_diffusion, pinn_to_grid.

  experiments/03_pinn_diffusion/
    run_pinn_diffusion.py     Trains the PINN and saves
                              outputs/checkpoints/peb_phase3_pinn_diffusion.pt.
    compare_fd_fft_pinn.py    Loads the checkpoint, runs all three
                              solvers; FFT is the truth reference.
```

Verified results from
`reaction_diffusion_peb/outputs/logs/peb_phase3_compare_fd_fft_pinn_metrics.csv`:

| solver | max\|err vs FFT\| | mean\|err\| | L2 rel err | wall-clock per call |
|---|---|---|---|---|
| FFT | 0 | 0 | 0 | 0.09 ms |
| FD | 2.58e-06 | 6.01e-07 | 4.68e-05 | 13.82 ms |
| PINN | 1.85e-02 | 2.31e-03 | 0.198 | 0.30 ms (+ 42 s training) |

- **FD is ~7000× more accurate than PINN** on this benchmark, mirroring
  the main project's Phase-6 finding.
- PINN inference is ~50× faster than FD per call but pays a 42 s
  one-shot training cost. Worth it only when many evaluations are
  needed at the same `(DH, t_end)`.
- Hard-IC parameterization (`H_pinn = H0(x, y) + t_norm * MLP`) drove
  `loss_pde` to ~1e-8; the soft-IC trivial minimum that broke main
  Phase 6 is avoided by construction.
- The y=0 row cut shows PINN slightly under-diffusing the peak
  (PINN 0.105 vs truth 0.086) — error concentrated at the center.

## Phase 4 — what's already there

```text
reaction_diffusion_peb/
  src/reaction_diffusion.py           step_acid_loss_fd,
                                       diffuse_acid_loss_fd
                                       (CFL + loss-stability guarded),
                                       diffuse_acid_loss_fft (closed
                                       form), total_mass,
                                       expected_mass_decay_factor.
  src/pinn_reaction_diffusion.py       PINNDiffusionLoss (subclass of
                                       PINNDiffusion; residual adds
                                       + kloss * H).

  experiments/04_acid_loss/
    run_acid_loss_fd.py     kloss sweep {0, 0.001, 0.005, 0.01, 0.05},
                             mass_FD vs analytic exp(-kloss * t).
    run_acid_loss_pinn.py    Trains the loss-PINN and saves
                             outputs/checkpoints/peb_phase4_pinn_acid_loss.pt.
    compare_fd_pinn.py        Loads the checkpoint, runs FD / FFT /
                             PINN; FFT is still the truth reference
                             because the loss term is linear.
```

Verified results from
`reaction_diffusion_peb/outputs/logs/peb_phase4_fd_metrics.csv` and
`peb_phase4_compare_fd_pinn_metrics.csv`:

| kloss (1/s) | FD mass | analytic mass exp(-k t) | rel err |
|---|---|---|---|
| 0.000 | 144.149 | 144.149 | 0.000000 |
| 0.001 | 135.754 | 135.755 | 5e-6 |
| 0.005 | 106.776 | 106.788 | 1e-4 |
| 0.010 | 79.074 | 79.111 | 5e-4 |
| 0.050 | 7.093 | 7.177 | 1e-2 |

3-way comparison at kloss = 0.005, t = 60 s, DH = 0.8 nm²/s:

| solver | max\|err vs FFT\| | mean\|err\| | L2 rel err | wall-clock |
|---|---|---|---|---|
| FFT | 0 | 0 | 0 | 0.09 ms |
| FD | 2.46e-05 | 1.78e-06 | 2.98e-04 | 17.17 ms |
| PINN | 1.74e-02 | 2.18e-03 | 0.253 | 0.30 ms (+ 51 s training) |

- The acid-loss PDE is **linear and homogeneous** in `H`, so the FFT
  closed form survives: each Fourier mode picks up an extra
  ``exp(-k_loss * t)`` factor and total mass decays exactly by the
  same global factor.
- FD mass tracks the analytic decay to 4–5 decimal places at moderate
  ``kloss``; the 1 % gap at ``kloss = 0.05`` (95 % mass loss) is the
  expected explicit-Euler truncation error.
- PINN over-predicts both the peak (under-diffuses) and total mass
  (under-decays) — a consistent ~1.7 % under-shoot in the decay
  factor. Same magnitude of error as Phase 3.

## Phase 5 — what's already there

```text
reaction_diffusion_peb/
  src/deprotection.py                  step_acid_loss_deprotection_fd,
                                       evolve_acid_loss_deprotection_fd
                                       (CFL + loss + deprotection
                                       stability guarded; clamps P to
                                       [0, 1]),
                                       deprotected_fraction_from_H_integral
                                       (analytic per-pixel form),
                                       thresholded_area.
  src/pinn_reaction_diffusion.py        + PINNDeprotection — 2-output
                                       PINN (H, P) with hard IC
                                       enforcing H(0)=H_0 and P(0)=0.
                                       pde_residuals returns
                                       (r_H, r_P).
  src/train_pinn_deprotection.py        train_pinn_deprotection (sums
                                       H + P PDE residual MSEs);
                                       pinn_deprotection_to_grid.

  experiments/05_deprotection/
    run_deprotection_fd.py    kdep sweep {0, 0.01, 0.05, 0.1, 0.5, 1.0}
                              + time sweep at fixed kdep.
    run_deprotection_pinn.py  Trains PINNDeprotection at kdep=0.5,
                              saves checkpoint.
    compare_fd_pinn.py         Loads PINN, runs FD; FD is truth.
```

The Phase-5 scope is intentionally narrow — only the H equation from
Phase 4 plus the deprotection equation:

```text
dH/dt = D_H * laplacian(H) - k_loss * H            (Phase 4 unchanged)
dP/dt = k_dep * H * (1 - P)                        (Phase 5 only)

H(x, y, 0) = H_0(x, y)
P(x, y, 0) = 0
```

No quencher, Arrhenius, Petersen, stochastic, or DeepONet/FNO machinery
in this phase.

Verified results from
`reaction_diffusion_peb/outputs/logs/peb_phase5_fd_kdep_metrics.csv`
and the time-sweep CSV. All six study-plan §5 criteria are met:

| kdep (1/s) | P_max | P_mean | P_center | P_corner | P ∈ [0, 1] | area(P>0.5) px |
|---|---|---|---|---|---|---|
| 0.00 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | yes | 0 |
| 0.01 | 0.0533 | 0.0045 | 0.0533 | 0.0000 | yes | 0 |
| 0.05 | 0.2394 | 0.0213 | 0.2394 | 0.0000 | yes | 0 |
| 0.10 | 0.4217 | 0.0399 | 0.4217 | 0.0000 | yes | 0 |
| 0.50 | 0.9358 | 0.1310 | 0.9358 | 0.0000 | yes | 1876 |
| 1.00 | 0.9960 | 0.1843 | 0.9960 | 0.0000 | yes | 2756 |

| t_end (s) | area(P>0.5) px | P_max | P_mean |
|---|---|---|---|
| 15  | 332 | 0.5828 | 0.0514 |
| 30  | 1116 | 0.7996 | 0.0852 |
| 60  | 1876 | 0.9358 | 0.1310 |
| 120 | 2756 | 0.9847 | 0.1903 |

1. ✅ ``kdep = 0`` -> ``P`` is exactly zero everywhere.
2. ✅ The dark periphery (``H_0 ≈ 0``) keeps ``P_corner = 0``.
3. ✅ The peak pixel (``H_max``) ramps ``P`` up first.
4. ✅ ``P`` stays in ``[0, 1]`` for every kdep.
5. ✅ Larger ``kdep`` strictly increases ``P_max`` and ``P_mean``.
6. ✅ The ``P > 0.5`` contour widens monotonically with both ``kdep``
   and PEB time.

PINN comparison at ``kdep = 0.5, t = 60 s``:

| solver | max\|H err\| | max\|P err\| | P_min | P_max | area(P>0.5) | wall-clock |
|---|---|---|---|---|---|---|
| FD (truth) | 0 | 0 | 0.0000 | 0.9358 | 1876 | 24.94 ms |
| PINN | 2.30e-02 | 2.89e-01 | -0.191 | 0.9246 | 1468 | 0.30 ms (+ 58 s training) |

- The PINN learned the H equation (residual ~1e-7) and the qualitative
  P shape, but the hard-IC parameterization
  ``P_pinn = t_norm * MLP[..., 1]`` does **not enforce ``P >= 0``**
  during dynamics — the network drifted to ``P_min = -0.19`` in
  fringe regions where the PDE residual is small either way. The FD
  solver clamps to ``[0, 1]`` and stays physically consistent.
- This is a clean teaching artefact: hard-IC enforces the initial
  state exactly but does not impose dynamic constraints; later
  phases that need bounded outputs should add a soft penalty on
  ``relu(-P) + relu(P - 1)`` to the PINN loss.

## Where to start when reopening

Next milestone is Phase 6: Arrhenius temperature dependence. The H
and P equations from Phase 5 stay the same, but ``k_loss``,
``k_dep`` (and later ``k_q``) become temperature-dependent through
``k(T) = k_ref * exp(-Ea / R * (1/T - 1/T_ref))``. The PINN will see
``T`` as either a fixed scalar or an additional input feature.

## See also

- [reaction_diffusion_peb/PLAN.md](../reaction_diffusion_peb/PLAN.md)
  — full submodule plan (1764 lines).
