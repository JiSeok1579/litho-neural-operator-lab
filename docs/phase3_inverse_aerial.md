# Phase 3 — Inverse aerial optimization

**Status:** ✅ done
**PR:** #5 · 16 tests added (49 / 49 total green)

## Goal

Close the physics direction: Phase 1 makes a spectrum, Phase 2 turns
it into intensity, **Phase 3 inverts the chain via autograd and Adam**.
Given a target intensity in some region and a forbidden region with
zero intensity, find the binary mask that delivers them.

## What landed

- `src/inverse/losses.py` — `masked_mse(field, region, target)` plus
  the convenience aliases `target_loss` and `background_loss`, and
  the diagnostic `mean_intensity_in_region`.
- `src/inverse/regularizers.py` — `total_variation` (mean L1 first
  difference) and `binarization_penalty` (mean of `m * (1 - m)`).
- `src/inverse/optimize_mask.py` — `optimize_mask(grid, target_region,
  forbidden_region, config)` runs Adam on a real-valued raw parameter
  `theta`. Mask is `m = sigmoid(alpha * theta)`; `alpha` follows a
  step schedule (default `(0,1) → (300,4) → (700,12)`) for
  binarization annealing. Returns an `OptimizationResult` dataclass.
- Visualization: `show_inverse_result` (2 × 4 before / after) and
  `show_loss_history`.
- Reference YAML: `configs/inverse_aerial.yaml`.

## Demos

```bash
python experiments/02_inverse_aerial/demo_target_spot.py
python experiments/02_inverse_aerial/demo_forbidden_region.py
```

Each saves a before / after figure and a loss-curve figure under
`outputs/figures/phase3_*`.

## Verified results

```text
demo_target_spot   (forbidden = complement of the disk)
  target loss     0.5625 → 0.0005   (~1100x)
  mean(I) target  0.250  → 0.992
  mean(I) forbid  0.250  → 0.078

demo_forbidden_region   (forbidden = 4 explicit side-lobe disks at +/-3λ)
  target loss     0.5625 → 0.0066
  background loss 0.0625 → 0.0004
  mean(I) target  0.250  → 1.051
  mean(I) forbid  0.250  → 0.015
```

The forbidden-region case suppresses the four explicit disks ~5x more
than the complement case, at the cost of slightly worse target
fidelity (1.05 vs 0.99) and visible energy in the **neutral**
diagonal regions (where neither penalty applies). This is the
fidelity-vs-leakage trade-off study plan §3.7 explicitly flags.

## Why these design choices

- **Parameterization** `m = sigmoid(alpha * theta)` keeps the mask
  differentiably bounded in (0, 1) without clamping. `theta = 0`
  gives `m = 0.5` everywhere, which is why the initial aerial in
  every Phase-3 figure is a flat 0.25.
- **Returning a dataclass** instead of a tuple makes the Phase-9
  closed loop (which wraps this function) much more readable.

## Key takeaway

A region-weighted MSE on the post-pupil aerial, plus TV and
binarization regularizers, plus Adam with annealed `alpha` is enough
to drive sub-Rayleigh targets from a uniform half-tone start to a
clean printable mask in 800 iterations.

## See also

- [PROGRESS.md §A.6](../PROGRESS.md)
- Source: [src/inverse/optimize_mask.py](../src/inverse/optimize_mask.py)
