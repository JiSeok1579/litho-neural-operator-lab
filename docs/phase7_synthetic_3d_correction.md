# Phase 7 — Synthetic 3D mask correction dataset

**Status:** ✅ done
**PR:** #9 · 11 tests added (116 / 116 total green)

## Goal

Build the **closed-form correction operator** that Phase 8 / 9 will
treat as the "true physics" reference, plus the paired data pipeline
that feeds the FNO surrogate. Real 3D mask effects (RCWA / FDTD) are
out of scope for a study repo, so we use a synthetic operator with
the right qualitative behaviour.

## What landed

- `src/neural_operator/synthetic_3d_correction_data.py`:
  - `CorrectionParams` dataclass (`gamma`, `alpha`, `beta`, `delta`,
    `s`, `c`) with `to_array` / `from_array` / `identity` helpers.
  - `correction_operator(grid, params)` builds the closed-form
    complex correction
    ```text
    C(fx, fy) = exp(-gamma * |f|^2)
                * exp(i * (alpha fx + beta fy + delta * (fx² - fy²)))
                * (1 + s * tanh(c * fx))
    ```
  - `apply_3d_correction(T_thin, C)` — multiplicative pairing.
  - `sample_correction_params(rng, ranges)` draws `theta` uniformly
    within `DEFAULT_RANGES`.
  - `random_mask_sampler(grid, seed)` mixes 50 % block-random binary,
    25 % line-space, 25 % contact-hole arrangements.
  - `generate_dataset(grid, n_samples, output_path, seed)` produces a
    compressed NPZ archive.
- Visualization: `show_correction_samples` (rows of mask | |T_thin| |
  |C| | |T_3d| | |Delta T|).

## Demo

```bash
python experiments/06_fno_correction/generate_synthetic_dataset.py
```

Writes:

```text
outputs/datasets/synthetic_3d_correction_train.npz   (2000 samples, n=128)
outputs/datasets/synthetic_3d_correction_test.npz    (400 samples, n=128)
outputs/figures/phase7_correction_samples.png
outputs/logs/phase7_dataset_summary.csv
```

## Verified results

```text
split  n     |T_thin|  |T_3d|   |Delta T|
train  2000  0.156     0.062    0.120     (|Delta T|/|T_thin| ~77 %)
test    400  0.146     0.058    0.111     (|Delta T|/|T_thin| ~76 %)
```

- The correction is **substantial**: predicting `delta_T = 0` is far
  from optimal, so the dataset has real geometry for an FNO to fit.
- `|T_3d| / |T_thin| ~ 40 %` matches the expected bulk attenuation
  from a non-zero `gamma` sampled in `[0, 0.3]`.
- Per-component theta means sit at the midpoints of their ranges
  with reasonable variance — no sampler axis is biased.

## Key teaching point

A single closed-form correction operator gives Phase 8 / 9 something
**exact** to compare against. The surrogate's job is to predict it;
the closed loop's job is to find a mask that minimizes the post-
correction loss. Both are testable because we can always recompute
the truth.

## See also

- [PROGRESS.md §A.10](../PROGRESS.md)
- Source: [src/neural_operator/synthetic_3d_correction_data.py](../src/neural_operator/synthetic_3d_correction_data.py)
