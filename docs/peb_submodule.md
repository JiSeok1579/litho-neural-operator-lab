# PEB submodule (`reaction_diffusion_peb/`)

**Status:** Phases 1 – 10 done (Phase 10 trains a small 2D FNO on the
safe Phase-9 archive and evaluates it on the safe-test split plus the
full stiff archive; the surrogate degrades sharply at this
sample count and parameter range — that mirrors the main project's
Phase-9 finding and is logged as an informative failure). Phase 11
planned. Open follow-ups in
[`reaction_diffusion_peb/FUTURE_WORK.md`](../reaction_diffusion_peb/FUTURE_WORK.md)
(items 1 and 4 closed; items 2 and 3 still open).
177 PEB tests green; total repo tests at 309 / 309.

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
| 6 | Arrhenius temperature dependence | ✅ done (FD-only) |
| 7 | Quencher reaction (`kq` safe vs stiff target) | ✅ done (FD-only) |
| 8 | Full reaction-diffusion (everything combined) | ✅ done (FD-only) |
| 9 | Save FD / PINN runs as a dataset | ✅ done (FD-only) |
| 10 | DeepONet / FNO operator surrogate (optional) | ✅ done (FNO-only) |
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

## Phase 6 — what's already there

```text
reaction_diffusion_peb/
  src/arrhenius.py                arrhenius_factor(T_c, T_ref_c, Ea_kj_mol),
                                  apply_arrhenius_to_rates,
                                  evolve_acid_loss_deprotection_fd_at_T
                                  (Phase-5 evolver wrapper that scales
                                  kdep / kloss before calling).

  experiments/06_temperature_peb/
    run_temperature_sweep.py    T sweep {80, 90, 100, 110, 120, 125} °C
                                + Ea=0 control (criterion 3).
    run_time_sweep.py            t sweep {60, 75, 90} s at T = T_ref.
```

PINN is intentionally not retrained in Phase 6 — the existing
Phase-5 PINN was trained for one fixed (kdep, kloss) and cannot
generalize across temperatures without taking T as an extra input.
That extension is tracked in
[`reaction_diffusion_peb/FUTURE_WORK.md`](../reaction_diffusion_peb/FUTURE_WORK.md)
item 2.

Verified results from
`reaction_diffusion_peb/outputs/logs/peb_phase6_fd_temperature_sweep_metrics.csv`
(T_ref = 100 °C, Ea = 100 kJ/mol, kdep_ref = 0.5 s⁻¹, kloss_ref = 0.005
s⁻¹, t = 60 s):

| T (°C) | factor | kdep_eff | kloss_eff | P_max | P_mean | area(P>0.5) px |
|---|---|---|---|---|---|---|
| 80 | 0.1611 | 0.0806 | 0.000806 | 0.3901 | 0.0368 | 0 |
| 90 | 0.4116 | 0.2058 | 0.002058 | 0.7051 | 0.0771 | 820 |
| 100 | **1.0000** | 0.5000 | 0.005000 | 0.9358 | 0.1310 | 1876 |
| 110 | 2.3193 | 1.1597 | 0.011597 | 0.9955 | 0.1801 | 2684 |
| 120 | 5.1539 | 2.5769 | 0.025769 | 0.9998 | 0.2101 | 3196 |
| 125 (MOR) | 7.5682 | 3.7841 | 0.037841 | 1.0000 | 0.2159 | 3276 |

`Ea = 0` control (criterion 3): for T ∈ {80, 100, 125} the factor is
1.0000, ``P_max`` is 0.9358 and threshold area is 1876 — identical
across temperatures, confirming the temperature dependence is gated
by the Arrhenius term alone.

Time sweep at T = T_ref (factor = 1):

| t (s) | P_max | P_mean | area(P>0.5) px |
|---|---|---|---|
| 60 | 0.9358 | 0.1310 | 1876 |
| 75 | 0.9588 | 0.1485 | 2136 |
| 90 | 0.9719 | 0.1638 | 2348 |

All five Phase-6 verification criteria pass:

1. ✅ T = T_ref → factor = 1.0000 exactly.
2. ✅ T > T_ref with Ea > 0 → factor > 1 (110 °C: 2.32, 125 °C: 7.57).
3. ✅ Ea = 0 → factor identically 1 at every T.
4. ✅ P_max and area(P > 0.5) increase monotonically with T
   (0.39 → 1.00 ; 0 → 3276 px).
5. ✅ 125 °C MOR preset gives a clearly faster reaction (factor 7.57,
   area 3276 px) than the 100 °C reference (1876 px).

## Phase 7 — what's already there

```text
reaction_diffusion_peb/
  src/quencher_reaction.py    step_quencher_fd, evolve_quencher_fd
                              (CFL + loss + deprotection + bimolecular
                              k_q stability guarded; clamps H/Q to ≥0
                              and P to [0, 1]),
                              evolve_quencher_fd_with_budget,
                              QuencherBudgetSnapshot,
                              stability_report (returns the four
                              dt bounds and which one is binding).
  configs/quencher_reaction.yaml   reference parameters + the
                                   kq_safe / kq_stiff sweep split.

  experiments/07_quencher_reaction/
    run_quencher_reaction_safe.py    kq sweep {1, 5, 10} s⁻¹.
    run_quencher_reaction_stiff.py   kq sweep {100, 300, 1000} s⁻¹
                                     (separate stiff demo).
```

The Phase-7 system adds a quencher field ``Q`` and a bimolecular
neutralization term to the (H, P) system from Phase 5/6:

```text
dH/dt = D_H ∇²H − k_q H Q − k_loss H
dQ/dt = D_Q ∇²Q − k_q H Q
dP/dt = k_dep H (1 − P)

H(x, y, 0) = H_0(x, y),   Q(x, y, 0) = Q_0,   P(x, y, 0) = 0
```

PINN training is intentionally deferred — the bimolecular ``H * Q``
term is a step beyond Phase 5 in difficulty (no closed form, two
coupled fields) and the existing checkpoint generalizes neither across
``kq`` nor to a non-zero ``Q``. Tracked alongside Phase-6 PINN extension
in `FUTURE_WORK.md`.

Stability — explicit Euler needs four time-step bounds, and the demo
prints which one binds:

```text
dt ≤ dx² / (4 max(D_H, D_Q))    (diffusion CFL)
dt ≤ 1 / k_loss                  (linear loss)
dt ≤ 1 / (k_dep H_max)           (deprotection)
dt ≤ 1 / (k_q max(H_max, Q_0))   (acid-quencher reaction)
```

For ``k_q ≤ 10 s⁻¹`` the diffusion CFL binds (``dt_max = 0.31 s``);
for ``k_q ≥ 100 s⁻¹`` the bimolecular term binds and the step-count
scales linearly with ``k_q``.

Verified results from
`reaction_diffusion_peb/outputs/logs/peb_phase7_quencher_safe_metrics.csv`
(safe regime — DH = 0.8, DQ = 0.08, Q_0 = 0.1, k_loss = 0.005,
k_dep = 0.5, t = 60 s):

| k_q (1/s) | binding term | H_peak | Q_min | P_max | area(P>0.5) px | H budget rel-err | Q budget rel-err | wall (s) |
|---|---|---|---|---|---|---|---|---|
| 1 | dt_diff | 0.0082 | 0.0139 | 0.6407 | 392 | 3.6e-08 | 1.0e-06 | 0.04 |
| 5 | dt_diff | 0.0027 | 0.0010 | 0.3899 | 0 | 3.9e-08 | 4.7e-07 | 0.04 |
| 10 | dt_diff | 0.0025 | 0.0001 | 0.3279 | 0 | 3.7e-08 | 5.5e-07 | 0.04 |

Verified results from
`reaction_diffusion_peb/outputs/logs/peb_phase7_quencher_stiff_metrics.csv`
(stiff regime, same other parameters):

| k_q (1/s) | binding term | n_steps | H_peak | Q_min | P_max | area(P>0.5) px | H budget rel-err | Q budget rel-err | wall (s) |
|---|---|---|---|---|---|---|---|---|---|
| 100 | dt_kq | 1515 | 0.0025 | 0.0000 | 0.2661 | 0 | 6.5e-09 | 1.3e-06 | 0.91 |
| 300 | dt_kq | 4546 | 0.0024 | 0.0000 | 0.2608 | 0 | 1.4e-08 | 3.9e-06 | 3.16 |
| 1000 | dt_kq | 15155 | 0.0024 | 0.0000 | 0.2587 | 0 | 2.1e-08 | 3.0e-05 | 11.31 |

What the numbers say:

1. ✅ ``H``, ``Q``, ``P`` stay non-negative and bounded across both
   regimes — explicit Euler with the CFL-respecting ``dt`` is stable
   even at ``k_q = 1000 s⁻¹``.
2. ✅ Increasing ``k_q`` consumes more acid (``H_peak`` drops
   monotonically) and more quencher (``Q_min`` → 0 in the bright zone).
3. ✅ Threshold area drops from 392 px at ``k_q = 1`` to 0 px for
   ``k_q ≥ 5``: at the realistic CAR / MOR rates the quencher fully
   suppresses deprotection at this dose / Q_0 ratio. Exactly the
   regime where dose / quencher loading have to be retuned.
4. ✅ Mass budgets close to float-precision: H budget tracks both
   ``k_loss`` and ``k_q`` sinks to ~3e-8; Q budget tracks the
   ``k_q`` sink to ~3e-6 (rising to 3e-5 at ``k_q = 1000`` from the
   accumulated ~15k-step round-off, still well within float32 noise).
5. ✅ Wall-clock scales linearly with ``k_q`` in the stiff regime
   (1.5k / 4.5k / 15k explicit steps) — predictable and tractable on
   CPU for a 60 s evolution.

## Phase 8 — what's already there

```text
reaction_diffusion_peb/
  src/full_reaction_diffusion.py     apply_arrhenius_to_full_rates,
                                     evolve_full_reaction_diffusion_fd_at_T,
                                     evolve_full_reaction_diffusion_fd_at_T_with_budget,
                                     stability_report_at_T.
  configs/full_reaction_diffusion.yaml   reference parameters + the
                                         T sweep.

  experiments/08_full_reaction_diffusion/
    run_full_model.py             T sweep {80, 90, 100, 110, 120} °C
                                  on the integrated (H, Q, P) system.
    run_term_disable_check.py     equivalence check vs every earlier
                                  phase at the production grid.
```

Phase 8 is the smallest possible integration step: a thin wrapper that
applies the Phase-6 Arrhenius factor to ``k_q``, ``k_dep`` and
``k_loss`` before calling the Phase-7 quencher evolver. It does not
introduce any new physics, and the entire point of the phase is the
disable-each-term equivalence check — proof that the integrated
solver reduces to every earlier phase as a limit case.

Verified results from
`reaction_diffusion_peb/outputs/logs/peb_phase8_full_T_sweep_metrics.csv`
(`DH = 0.8`, `DQ = 0.08`, `Q_0 = 0.1`, `kq_ref = 1`, `kdep_ref = 0.5`,
`kloss_ref = 0.005`, `Ea = 100 kJ/mol`, `T_ref = 100 °C`, `t = 60 s`):

| T (°C) | factor | k_q_eff | k_dep_eff | binding | H_peak | Q_min | P_max | area(P>0.5) px | H rel-err | Q rel-err |
|---|---|---|---|---|---|---|---|---|---|---|
| 80 | 0.16 | 0.16 | 0.08 | dt_diff | 0.0415 | 0.0499 | 0.2993 | 0 | 6.2e-08 | 8.3e-07 |
| 90 | 0.41 | 0.41 | 0.21 | dt_diff | 0.0213 | 0.0279 | 0.4811 | 0 | 2.3e-08 | 8.8e-07 |
| 100 | 1.00 | 1.00 | 0.50 | dt_diff | 0.0082 | 0.0139 | 0.6407 | 392 | 3.6e-08 | 1.0e-06 |
| 110 | 2.32 | 2.32 | 1.16 | dt_diff | 0.0021 | 0.0070 | 0.7538 | 540 | 7.6e-08 | 3.7e-07 |
| 120 | 5.15 | 5.15 | 2.58 | dt_diff | 0.0003 | 0.0039 | 0.8258 | 576 | 7.5e-08 | 4.0e-07 |

The 100 °C row matches the Phase-7 safe-`kq` run with `k_q = 1` exactly
(by construction — the test suite asserts bit-equality).

Verified results from
`reaction_diffusion_peb/outputs/logs/peb_phase8_term_disable_check.csv`
(production grid, `t = 60 s`):

| disabled term(s) | compared against | max\|H_full − H_ref\| | max\|P_full − P_ref\| | tol |
|---|---|---|---|---|
| T = T_ref | Phase 7 quencher | 0.000e+00 | 0.000e+00 | 1e-10 |
| Ea = 0 (T = 125 °C) | Phase 7 quencher | 0.000e+00 | 0.000e+00 | 1e-10 |
| k_q = 0 (T = 110 °C) | Phase 6 Arrhenius (H, P) | 0.000e+00 | 0.000e+00 | 1e-5 |
| k_q = 0, k_dep = 0 | Phase 4 acid loss | 0.000e+00 | 0.000e+00 | 1e-5 |
| k_q = 0, k_dep = 0, k_loss = 0 | Phase 2 pure diffusion | 8.9e-08 | 0.000e+00 | 1e-5 |

What the numbers say:

1. ✅ ``T = T_ref`` and ``Ea = 0`` are bit-identical to Phase 7 — the
   Arrhenius factor multiplies through cleanly without floating-point
   reordering.
2. ✅ ``k_q = 0`` matches the Phase-6 (H, P) evolver to floating-point
   precision; ``Q`` stays exactly uniform-and-constant.
3. ✅ Stripping ``k_q`` and ``k_dep`` recovers Phase 4 acid loss; the
   ``P`` field is identically zero.
4. ✅ Stripping all three reactions recovers Phase 2 pure diffusion to
   ~9e-8 (the tiny residual is float32 round-off in the explicit-Euler
   step — well within tolerance).
5. ✅ Per-T mass-budget identities (H budget tracks the ``k_loss`` and
   ``k_q`` sinks, Q budget tracks the ``k_q`` sink) close to ~1e-7
   across the sweep.
6. ✅ Threshold area increases monotonically with T (0 → 0 → 392 →
   540 → 576 px), and ``Q_min`` decreases monotonically — exactly the
   physics expected when both the deprotection and quencher rates
   speed up with temperature.

PINN training across the integrated system is intentionally deferred —
the bimolecular ``H * Q`` term plus a temperature input would need a
fresh training rig and is tracked alongside the Phase-6/7 PINN
extensions in `FUTURE_WORK.md`.

## Phase 9 — what's already there

```text
reaction_diffusion_peb/
  src/dataset_builder.py            SampleSpec, aerial_from_spec,
                                    random_safe_spec / random_stiff_spec,
                                    generate_sample (calls Phase-8
                                    evolver), make_split_indices,
                                    save_dataset / load_dataset,
                                    parameter_ranges.

  experiments/09_dataset_generation/
    generate_fd_dataset.py          Generates the safe and stiff
                                    .npz archives + .json metadata.
    validate_dataset.py             Asserts shapes / bounds / splits /
                                    regime-specific kq ranges; writes
                                    a per-archive summary CSV.
```

Per-sample fields stored in each ``.npz``:

```text
inputs (float32, shape (n, G, G)):
  I, H0
outputs (float32, shape (n, G, G)):
  H_final, Q_final, P_final, R                    (R is binary 0/1)
per-sample scalars (float32, shape (n,)):
  dose, eta, Hmax,
  DH_nm2_s, DQ_ratio,
  kq_ref_s_inv, kdep_ref_s_inv, kloss_ref_s_inv,
  Q0_mol_dm3,
  temperature_c, temperature_ref_c, activation_energy_kj_mol,
  t_end_s,
  aerial_param_a, aerial_param_b
per-sample int8:
  aerial_kind_code   (0=gaussian_spot, 1=line_space, 2=contact_array,
                      3=two_spot)
```

Sibling ``.json`` records ``grid_size``, ``grid_spacing_nm``,
``P_threshold``, ``solver`` (``"fd"``), ``regime`` (``"safe"`` or
``"stiff"``), ``seed``, parameter ranges, and the train / val / test
index lists. Stiff samples live in their own archive because they take
~100× longer per sample (up to 15k explicit Euler steps each at the
upper end of ``k_q``) and Phase 10 may want to train safe and stiff
surrogates separately.

Verified results from
`reaction_diffusion_peb/outputs/logs/peb_phase9_dataset_summary.csv`
(seeds 20260429 / 20260430):

| regime | file | n | grid | k_q range | mean P_max | mean R px | train / val / test |
|---|---|---|---|---|---|---|---|
| safe | `peb_phase9_safe_dataset.npz` | 64 | 128×128 | [0.5, 5.0] | 0.46 | 1019 | 52 / 6 / 6 |
| stiff | `peb_phase9_stiff_dataset.npz` | 8 | 128×128 | [100, 1000] | 0.05 | 0 | 6 / 1 / 1 |

What the numbers say:

1. ✅ Every per-sample input / output array has the expected shape and
   dtype; ``H, Q ≥ 0``, ``P ∈ [0, 1]``, ``R`` is binary (validated by
   `validate_dataset.py`).
2. ✅ Splits are non-overlapping and cover every sample exactly once;
   sizes are deterministic in the seed.
3. ✅ Safe / stiff archives are cleanly separated by ``k_q`` range
   ([0.5, 5.0] vs [100, 1000]); ``solver: fd`` and ``regime`` fields
   in metadata are the discriminators a future PINN-dataset .npz would
   distinguish itself with.
4. ✅ Aerial-pattern coverage in the safe archive: 27 ``gaussian_spot``,
   19 ``two_spot``, 12 ``line_space``, 6 ``contact_array``.
5. ✅ ``P_max`` covers a wide range in the safe archive (0.035 → 0.972)
   — the parameter ranges are wide enough to give the surrogate room
   to learn, not all converged-on-one-answer.
6. ✅ Wall-clock: 64 safe samples in ~2.5 s, 8 stiff samples in ~39 s.
   Predictable and tractable on CPU.

PINN-dataset generation is intentionally deferred — PINN training
across the integrated (H, Q, P, T) system is itself deferred — so the
matching `generate_pinn_dataset.py` script is not implemented in this
phase. The ``solver`` field in metadata is reserved as the
discriminator for when that future dataset arrives.

## Phase 10 — what's already there

```text
reaction_diffusion_peb/
  src/fno_surrogate.py         SpectralConv2d, FNOBlock, FNO2d,
                               INPUT/OUTPUT_*_NAMES, make_input_tensor,
                               make_output_tensor, ChannelStats,
                               fit_channel_stats, relative_l2,
                               per_channel_relative_l2,
                               thresholded_iou, build_fno_for_dataset,
                               manual_seed_everything.

  experiments/10_operator_learning_optional/
    train_fno.py                  FNO2d (width 32, 4 blocks, 16 modes,
                                  ~2.1 M params); 300 epochs of AdamW
                                  cosine; ~8 s wall on the safe archive.
    evaluate_operator_surrogate.py
                                  Reload checkpoint; rel-L2 + threshold
                                  IoU on the safe-test split and the
                                  full stiff archive; worst-case figure.
```

Operator learned (FNO; DeepONet is intentionally not implemented —
plan says ``DeepONet and / or FNO`` and FNO matches the regular-grid
``128 x 128`` inputs more naturally):

```text
inputs (10 channels):
  H0                                        (one 2D field)
  DH, DQ_ratio, kq_ref, kdep_ref, kloss_ref,
  Q0, T_c (centered on T_ref), Ea, t_end    (broadcast scalars)

outputs (2 channels):
  H_final, P_final                          (R recovered by thresholding)
```

Verified results from
`reaction_diffusion_peb/outputs/logs/peb_phase10_fno_evaluation.csv`
(checkpoint trained for 300 epochs on the 52-sample safe-train split):

| dataset | n | rel-L2 H_final | rel-L2 P_final | IoU(P>0.5) mean | wall |
|---|---|---|---|---|---|
| safe_test | 6 | 8.4e-01 | 2.4e-01 | 0.0000 | 149 ms first call / sub-ms after |
| stiff_full | 8 | 2.3e+02 | 9.7e+01 | 0.0000 | 0.4 ms / sample |

What the numbers say:

1. ✅ The FNO trains end-to-end without instability — train MSE drops
   from 9.6e-1 to 2.2e-4 over 300 epochs, the loss-history CSV is
   monotone after the first 30 epochs.
2. ⚠️  P-channel rel-L2 on the safe-test split is ~24 %. That is far
   from production quality — same magnitude of error as a Phase-3 PINN.
3. ❌ Threshold IoU is **zero** across both safe-test and stiff-full
   splits. The FNO output is too smooth to cross ``P = 0.5`` even when
   the FD truth does — the worst-case figure
   (`peb_phase10_fno_worst_case_safe.png`) makes this visible.
4. ❌ The stiff archive is catastrophically out-of-distribution: the
   training distribution drew ``k_q`` from ``[0.5, 5]``, while stiff
   samples have ``k_q`` from ``[100, 1000]``. The FNO cannot extrapolate
   across two orders of magnitude in a reaction rate from a 52-sample
   training set, and the rel-L2 explodes to 23 000 % / 9 700 %.
5. ✅ Inference wall-clock at ~0.4 ms/sample (after warmup) is ~5–25×
   faster than the Phase-8 FD evolver depending on regime. So the
   surrogate could be useful **once** the data and architecture
   problems are addressed; not before.

This mirrors the main project's Phase-9 closed-loop finding — a small
FNO trained on a small dataset gives the wrong answer fast. The point
of Phase 10 in the PEB submodule is making that result legible *before*
anyone tries to drop the surrogate into an inverse-design loop.

What would move the needle (FUTURE_WORK candidates, not done here):

- **More samples.** 52 → 1000+ is the obvious lever; even with the
  current architecture the rel-L2 on P should drop sharply.
- **Narrower kq range per surrogate.** Train one model per regime
  (safe / stiff) and stop expecting cross-regime extrapolation.
- **Explicit ``R`` head.** Pre-thresholding ``R`` as a separate output
  channel with a soft-IoU loss would make the threshold-crossing
  failure go away even at the same rel-L2.
- **PINN regularization.** The PDE residual is fully analytic; adding
  it as an auxiliary loss is a natural physics-informed extension.
- **DeepONet branch.** The plan also lists DeepONet; it is a different
  story for irregular sampling and is left as an open extension.

## Where to start when reopening

Phase 11 (advanced stochastic / Petersen / z-axis effects) is the
remaining major topic in the plan and is currently still planned.
Within the existing Phase-1–10 framework, the most useful single
lever for surrogate quality is regenerating Phase 9 with ~1000
samples and rerunning Phase 10 — the existing scripts handle that
without changes.

## See also

- [reaction_diffusion_peb/PLAN.md](../reaction_diffusion_peb/PLAN.md)
  — full submodule plan (1764 lines).
