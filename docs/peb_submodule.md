# PEB submodule (`reaction_diffusion_peb/`)

**Status:** Phases 1 – 2 done (synthetic aerial + exposure +
diffusion-only FD / FFT baselines). Phases 3 – 11 planned. 40 PEB
tests green; total repo tests at 172 / 172.

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
| 3 | PINN diffusion vs FD / FFT | planned |
| 4 | Acid loss `kloss` | planned |
| 5 | Deprotection `kdep` (`P` field) | planned |
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

## Where to start when reopening

The first concrete milestone (PLAN.md §17):

```text
Gaussian synthetic aerial image
  → initial acid H0
  → 60 s PEB diffusion via FD and FFT
  → PINN diffusion training
  → FD / FFT / PINN before / after comparison
  → figures, metrics, and loss curves saved to outputs/
```

## See also

- [reaction_diffusion_peb/PLAN.md](../reaction_diffusion_peb/PLAN.md)
  — full submodule plan (1764 lines).
