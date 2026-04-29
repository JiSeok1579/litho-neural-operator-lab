# Phase 2 — Coherent aerial imaging

**Status:** ✅ done
**PR:** #4 · 17 tests added (33 / 33 total green)

## Goal

Bridge "mask spectrum" to "wafer-plane intensity" through a circular
pupil. Make the whole chain differentiable in PyTorch so Phase 3 can
run gradient-based mask optimization on top of it.

## What landed

- `src/optics/pupil.py` — `circular_pupil(grid, NA, wavelength=1.0)`
  (hard binary low-pass) and `apodized_circular_pupil(...,
  roll_off=0.05)` (cosine taper for differentiable settings).
- `src/optics/coherent_imaging.py`:
  - `coherent_field(t, P) = ifft2c(fft2c(t) * P)`,
  - `coherent_aerial_image(t, P, normalize=True)` returning
    `|E|² = E.real**2 + E.imag**2` (kept real-only in the autograd
    graph for friendlier Phase-3 gradients).
- `src/common/visualization.py` extended with `show_pupil`,
  `show_aerial`, `show_aerial_sweep`, `show_pipeline` (5-panel
  mask → |T| → P → |T·P| → aerial).

## Demo

```bash
python experiments/01_scalar_diffraction/demo_coherent_aerial.py
```

Saves five figures under `outputs/figures/phase2_*`:
- pupil sweep over NA ∈ {0.2, 0.4, 0.6, 0.8}
- 5-panel pipeline at NA=0.6 / line-space pitch=4 λ
- aerial NA sweeps for line-space, contact hole, isolated line

## Verified physics

- Line-space pitch=4 λ goes flat at NA=0.2 (cutoff 0.2 < fundamental
  0.25), sinusoidal at NA=0.4 – 0.6, and squarer at NA=0.8 once the
  3rd harmonic at 0.75 cycles/λ is admitted.
- Contact hole r=0.5 λ sharpens from a broad blob at NA=0.2 toward
  the diffraction-limited PSF at NA=0.8.
- Isolated line side-lobes shrink as NA grows.
- An autograd test confirms gradients flow from an intensity-based
  loss back to a real-valued mask parameter — Phase 3 unblocked.

## Convention pinned in this phase

- Grid extent is counted in **wavelengths** (default `wavelength=1.0`
  means `Grid2D.extent` is read in λ).
- Cutoff frequency = `NA / wavelength` (cycles per length unit).

## Key takeaway

Once `coherent_aerial_image` is differentiable end-to-end, every
inverse-design or surrogate-training task downstream is just a choice
of loss and optimizer — no plumbing changes.

## See also

- [PROGRESS.md §A.5](../PROGRESS.md)
- Source: [src/optics/pupil.py](../src/optics/pupil.py),
  [src/optics/coherent_imaging.py](../src/optics/coherent_imaging.py)
