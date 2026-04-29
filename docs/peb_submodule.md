# PEB submodule (`reaction_diffusion_peb/`)

**Status:** Phase 1 done (synthetic aerial + exposure). Phases 2 – 11
planned. 20 PEB tests green; total repo tests at 152 / 152.

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
| 2 | Diffusion-only FD / FFT baselines | planned |
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
