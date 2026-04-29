# Phase 8 — FNO 2D correction surrogate

**Status:** ✅ done
**PR:** #10 · 10 tests added (126 / 126 total green)

## Goal

Train a Fourier Neural Operator that learns the closed-form
correction operator from Phase 7, so Phase 9 can swap the true
multiplicative correction for the surrogate inside an inverse-design
loop.

## What landed

- `src/neural_operator/fno2d.py`:
  - `SpectralConv2d` — `rfft2 → low-mode complex weight multiply →
    irfft2`. Two weight tensors handle the positive- and
    negative-x Fourier bands.
  - `FNOBlock2d` — `spectral_conv(x) + 1×1 conv(x)` with GELU.
  - `FNO2d` — 1×1 lift / N FNO blocks / 2-layer 1×1 head.
- `src/neural_operator/datasets.py` — `CorrectionDataset` reads the
  Phase-7 NPZ and returns 9-channel input
  `(mask, T_thin_re, T_thin_im, theta_0..theta_5)` and 2-channel
  target `(delta_T_re, delta_T_im)` per sample. Also exposes a
  `target="T_3d"` mode for ablations.
- `src/neural_operator/train_fno.py` — Adam / AdamW with optional
  weight decay, step LR schedule, optional physics-aware aerial-
  intensity term controlled by `weight_aerial`. Helpers:
  `spectrum_mse`, `complex_relative_error`, `aerial_intensity_mse`,
  `evaluate_fno`.
- Visualization: `show_fno_predictions` (per-row 4-panel comparison)
  and `show_fno_training` (loss + complex rel err).

## Demo

```bash
python experiments/06_fno_correction/train_fno_correction.py
```

Saves `outputs/checkpoints/fno_correction.pt`,
`phase8_fno_training.png`, `phase8_fno_predictions.png`,
`outputs/logs/phase8_metrics.csv`.

## Verified results

```text
metric                           value
n_train                          2000
n_test                           400
grid_n                           128
fno_params                       2,106,178
epochs                           100
batch_size                       8
train_time_sec                   131.9
final_test_spectrum_mse          9.038e-04
final_test_complex_rel_err       0.155      (15.5 %)
final_test_aerial_mse            1.590e-03
baseline_spectrum_mse_pred_0     3.044e-02
improvement_over_baseline        35.0x
```

- **35× over the identity baseline** (`pred = 0`) at 100 AdamW epochs
  on 2000 samples (132 s).
- A first pass at 800 train samples clearly overfit (train MSE 5e-4
  while test MSE stalled at 4e-3). Doubling the dataset to 2000 train
  + 400 test pushed test spectrum MSE to 9e-4 and shrank the
  generalization gap to ~10×.
- FNO captures spectrum structure across contact-hole, random, and
  line-space mask families. Largest absolute errors sit at very tall
  line-space peaks (5 – 10 % relative on a spike of magnitude ~7 still
  gives ~0.7 absolute).

## Why FNO is the right family here

- The target operator (multiplicative correction smooth in `(fx, fy)`)
  is naturally captured by **low-mode spectral convolutions**. A
  plain CNN would have to learn the FFT implicitly.
- Predicting `delta_T = T_3d - T_thin` instead of `T_3d` directly
  turns the task into residual learning, which converges faster.

## Key takeaway

A 35× improvement is real but the residual ~10× train / test gap and
the 15.5 % complex relative error become a problem in Phase 9: an
optimizer running against this surrogate will **systematically
exploit those errors** unless the optimization output is re-validated
under the true physics.

## See also

- [PROGRESS.md §A.11](../PROGRESS.md)
- Source: [src/neural_operator/](../src/neural_operator/)
