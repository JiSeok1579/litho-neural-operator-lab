# Phase 9 — Closed-loop surrogate-assisted inverse design

**Status:** ✅ done
**PR:** #11 · 6 tests added (132 / 132 total green) · **The lab's payoff phase.**

## Goal

Take the trained FNO from Phase 8 and put it inside the Phase-3
mask-optimization loop. Compare three optimizations on the same
target / forbidden / NA setting:

```text
A) physics-only      (no correction; T_3d = T_thin)
B) true correction   (oracle: T_3d = T_thin * correction_operator(theta))
C) FNO surrogate     (T_3d = T_thin + FNO_pred(mask, T_thin, theta))
```

Then **re-image case (C)'s optimized mask through the true
correction**. If the FNO has been "fooled" by the optimizer, this
re-evaluation will look very different from what the optimizer
believed during case (C).

## What landed

- `src/closed_loop/surrogate_optimizer.py`:
  - `identity_correction_fn` / `true_correction_fn` /
    `fno_correction_fn` factories produce
    `CorrectionFn` callables `(mask, T_thin) → T_3d`.
  - `optimize_mask_with_correction` is the Phase-3 inverse loop with
    the correction injected between `fft2c(t)` and the pupil. Same
    losses, same Adam + sigmoid-α annealing, same result dataclass.
  - `evaluate_mask_under_correction` computes the aerial intensity of
    a fixed mask through any correction — the **validation step** that
    turns "what FNO believed" into "what actually happens".
- Visualization: `show_closed_loop_comparison` with optional dashed
  `aerial_alt` curve so the y=0 cut can show "predicted" and "true"
  aerials simultaneously.

## Demo

```bash
python experiments/07_closed_loop_inverse/optimize_with_fno.py
```

Saves `phase9_closed_loop_comparison.png`,
`phase9_loss_history.png`, `phase9_metrics.csv`.

## Verified results

True correction theta: `gamma=0.20, alpha=0.30, beta=-0.20, delta=0.10,
s=0.30, c=2.0`.

| case | target | forbidden | loss_target | loss_bg |
|---|---|---|---|---|
| A physics-only | 0.989 | 0.070 | 0.0005 | 0.0341 |
| B true correction | 0.989 | 0.072 | 0.0005 | 0.0351 |
| C FNO (predicted) | 0.982 | 0.068 | 0.0021 | 0.0281 |
| **C validated under true correction** | **1.091** | **0.075** | **0.0103** | 0.0352 |

- Cases A and B converge to nearly identical mean intensities, but
  their **masks differ**. Case B's optimizer pre-compensates for the
  multiplicative correction so the post-correction aerial still
  matches the target.
- Case C's FNO believed it produced target intensity 0.982 with
  loss_target 0.0021. **Re-imaging the same mask through the true
  correction gives 1.091** — an ~11 % over-shoot beyond the requested
  1.0, and loss_target jumps ~5× to 0.0103.
- Forbidden intensity stays roughly comparable (0.068 → 0.075), so
  the leakage-suppression part of the optimization survives the
  surrogate. The failure mode is in the target band where the
  surrogate systematically under-predicted what the true imaging
  would deliver.

## Headline finding

A neural surrogate that scores **15.5 % test complex-relative error**
in Phase 8 produces an optimized mask that misleads the optimizer by
**11 % in target intensity** when re-imaged through the true physics.
The y=0 cut in the comparison figure shows the "predicted" and "true"
aerials of case C as two visibly separated curves — the canonical
"optimizer was lied to" picture.

This validates exactly the lesson the lab was set up to teach:

> A neural surrogate that scores well on its training-distribution
> test metric can still mislead an optimizer running against it. A
> re-imaging pass through the true physics is **mandatory** after
> any surrogate-assisted optimization.

## See also

- [PROGRESS.md §A.12](../PROGRESS.md)
- [docs/phase8_fno_surrogate.md](./phase8_fno_surrogate.md) — the FNO
  whose 15.5 % rel err became 11 % over-shoot here.
- Source: [src/closed_loop/surrogate_optimizer.py](../src/closed_loop/surrogate_optimizer.py)
