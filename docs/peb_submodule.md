# PEB submodule (`reaction_diffusion_peb/`)

**Status:** plan only — no code yet (as of the wrap-up commit).

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
| 1 | Synthetic aerial + exposure → initial acid `H0` | planned |
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
