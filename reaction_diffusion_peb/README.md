# reaction_diffusion_peb — PEB submodule

A self-contained workspace for studying **post-exposure-bake (PEB)
resist physics** independently from the main lithography pipeline:

```text
synthetic aerial / exposure map
  ↓ Phase 1   exposure  →  initial acid H0
  ↓ Phase 2   diffusion-only FD / FFT baselines
  ↓ Phase 3   PINN diffusion vs FD / FFT
  ↓ Phase 4   acid loss
  ↓ Phase 5   deprotection (P field)
  ↓ Phase 6   Arrhenius temperature dependence
  ↓ Phase 7   acid–quencher reaction (stiff)
  ↓ Phase 8   full reaction-diffusion
  ↓ Phase 9   FD / PINN simulation dataset
  ↓ Phase 10  DeepONet / FNO operator surrogate (optional)
  ↓ Phase 11  advanced stochastic / Petersen / z-axis
```

The full plan with PDEs, parameter ranges, and per-phase deliverables
lives in [PLAN.md](./PLAN.md). This submodule does **not** import from
the main repo's `src/{mask, optics, inverse, neural_operator,
closed_loop}` — when the time comes to integrate, it will be through
file-based hand-off (`outputs/aerial_image.npy` →
`reaction_diffusion_peb/outputs/resist_latent.npy`).

## Status

| Phase | Status |
|---|---|
| 1 — synthetic aerial + exposure | ✅ done |
| 2 — diffusion-only FD / FFT baselines | ✅ done |
| 3 — PINN diffusion vs FD / FFT | ✅ done |
| 4 — acid loss | ✅ done |
| 5 — deprotection | ✅ done |
| 6 — Arrhenius temperature | ✅ done (FD-only; PINN deferred — see [FUTURE_WORK.md](./FUTURE_WORK.md)) |
| pre-Phase-7 diagnostics | ✅ done — mass-budget + PINN bound-penalty soft term (`weight_bound = 0.01` default), see [FUTURE_WORK.md](./FUTURE_WORK.md) items 1 & 4 |
| 7 — acid–quencher reaction | ✅ done (FD only — safe + stiff `kq` regimes; PINN training deferred — see [FUTURE_WORK.md](./FUTURE_WORK.md)) |
| 8 — full reaction-diffusion | ✅ done (FD only — Arrhenius-scaled (H, Q, P); term-disable check vs Phases 2/4/6/7 closes to machine precision; PINN deferred) |
| 9 — dataset generation | ✅ done (FD only — safe-`kq` and stiff-`kq` `.npz` archives + metadata + train/val/test splits; PINN dataset deferred) |
| 10 — DeepONet / FNO surrogate (optional) | ✅ done (FNO only — train + eval on safe / stiff Phase-9 archives; small-data + wide-parameter regime degrades sharply; informative failure logged) |
| 11 — advanced stochastic / Petersen / z-axis | planned |

## Quick start

```bash
# from repo root
pytest reaction_diffusion_peb/tests/ -q

# Phase 1: synthetic aerial -> initial acid
python reaction_diffusion_peb/experiments/01_synthetic_aerial/run_gaussian_spot.py
python reaction_diffusion_peb/experiments/01_synthetic_aerial/run_line_space.py

# Phase 2: FD / FFT diffusion baseline
python reaction_diffusion_peb/experiments/02_diffusion_baseline/run_diffusion_fd.py
python reaction_diffusion_peb/experiments/02_diffusion_baseline/run_diffusion_fft.py
python reaction_diffusion_peb/experiments/02_diffusion_baseline/compare_fd_fft.py

# Phase 3: PINN diffusion (train + 3-way compare)
python reaction_diffusion_peb/experiments/03_pinn_diffusion/run_pinn_diffusion.py
python reaction_diffusion_peb/experiments/03_pinn_diffusion/compare_fd_fft_pinn.py

# Phase 4: acid loss (FD sweep, PINN train, 3-way compare)
python reaction_diffusion_peb/experiments/04_acid_loss/run_acid_loss_fd.py
python reaction_diffusion_peb/experiments/04_acid_loss/run_acid_loss_pinn.py
python reaction_diffusion_peb/experiments/04_acid_loss/compare_fd_pinn.py

# Phase 6: Arrhenius temperature dependence (FD only)
python reaction_diffusion_peb/experiments/06_temperature_peb/run_temperature_sweep.py
python reaction_diffusion_peb/experiments/06_temperature_peb/run_time_sweep.py

# pre-Phase-7 diagnostics: mass-budget identity + PINN bound penalty
python reaction_diffusion_peb/experiments/pre_phase7_diagnostics/run_mass_budget_check.py
python reaction_diffusion_peb/experiments/pre_phase7_diagnostics/run_pinn_bound_penalty.py

# Phase 5: deprotection (kdep sweep FD, PINN train, FD-vs-PINN compare)
python reaction_diffusion_peb/experiments/05_deprotection/run_deprotection_fd.py
python reaction_diffusion_peb/experiments/05_deprotection/run_deprotection_pinn.py
python reaction_diffusion_peb/experiments/05_deprotection/compare_fd_pinn.py

# Phase 7: acid-quencher reaction (FD only; safe + stiff kq sweeps)
python reaction_diffusion_peb/experiments/07_quencher_reaction/run_quencher_reaction_safe.py
python reaction_diffusion_peb/experiments/07_quencher_reaction/run_quencher_reaction_stiff.py

# Phase 8: full reaction-diffusion (T sweep + term-disable verification)
python reaction_diffusion_peb/experiments/08_full_reaction_diffusion/run_full_model.py
python reaction_diffusion_peb/experiments/08_full_reaction_diffusion/run_term_disable_check.py

# Phase 9: dataset generation (FD only; safe + stiff .npz archives)
python reaction_diffusion_peb/experiments/09_dataset_generation/generate_fd_dataset.py
python reaction_diffusion_peb/experiments/09_dataset_generation/validate_dataset.py

# Phase 10: FNO operator surrogate (optional; train + evaluate)
python reaction_diffusion_peb/experiments/10_operator_learning_optional/train_fno.py
python reaction_diffusion_peb/experiments/10_operator_learning_optional/evaluate_operator_surrogate.py
```

Outputs land under `reaction_diffusion_peb/outputs/{figures, logs,
arrays}/` (git-ignored). Reference parameters are in
`configs/exposure.yaml` and `configs/minimal_diffusion.yaml`.

## Methodological note (the reason the submodule exists)

Because no simulation dataset exists yet, the PEB study deliberately
uses **PINN before DeepONet / FNO**, but it never trusts a PINN alone —
every PINN result is benchmarked against an FD or FFT solver. Once
those FD / PINN runs have been saved as paired data (Phase 9), only
then does the submodule move on to operator learning (Phase 10).

This is the **opposite** of the main project's Phase 7 → 8 → 9 path,
which started with a closed-form correction operator and trained an
FNO directly. Both paths are valid and they teach different lessons —
see [docs/peb_submodule.md](../docs/peb_submodule.md) for the cross-
project pointer.
