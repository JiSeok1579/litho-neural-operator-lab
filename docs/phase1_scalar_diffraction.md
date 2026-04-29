# Phase 1 — Scalar diffraction

**Status:** ✅ done
**PR:** #2 · 16 tests added (16 / 16 green)

## Goal

Build the Fourier-optics foundation. Every later phase reuses the same
`Grid2D` and centered-FFT helpers, so this phase is mostly about
nailing one convention and proving it with tests.

## What landed

- `src/common/grid.py` — `Grid2D` dataclass with normalized real- and
  frequency-space axes, `dx` / `df` / Nyquist, `meshgrid`,
  `radial_freq`.
- `src/common/fft_utils.py` — `fft2c`, `ifft2c` (centered FFT,
  `norm='ortho'`), plus `amplitude` / `phase` / `log_amplitude`. No
  raw `fft2` calls anywhere else.
- `src/common/visualization.py` — `show_mask`, `show_spectrum`,
  `show_field_pair`, `save_figure`. Pair view supports `freq_zoom`
  and `amp_percentile` so diffraction orders sitting near DC are
  actually visible (without these, the orders for pitch=0.1 on a 256²
  grid disappeared in the full canvas).
- `src/mask/patterns.py` — `line_space`, `contact_hole`,
  `isolated_line`, `two_bar`, `elbow`, `random_binary`.
- `src/mask/transmission.py` — `binary_transmission`,
  `attenuated_phase_shift` (Att-PSM), `alternating_phase_shift`
  (Alt-PSM).
- `src/optics/scalar_diffraction.py` — `diffraction_spectrum(t)` and
  `reconstruct_field(T)`.
- Demos under `experiments/01_scalar_diffraction/`.

## Demos

```bash
python experiments/01_scalar_diffraction/demo_line_space.py
python experiments/01_scalar_diffraction/demo_contact_hole.py
```

Saves nine figures under `outputs/figures/phase1_*`:
- pitch sweep on line-space at duty 0.5
- radius sweep on contact hole
- binary vs Att-PSM comparison at r = 0.10

## Verified physics

- Sharp on / off edges spread spectral energy into many diffraction
  orders.
- Halving the line-space pitch roughly **doubles** the first-order
  frequency offset (the order-spacing test passes within 25 %).
- An isolated contact hole produces a 2D Bessel-J1 / Airy ring pattern
  with phase flips at every Bessel zero.
- For a real-valued binary mask, the centered spectrum is Hermitian:
  `T(-f) = conj(T(f))` to better than 1e-4 (fp32).

## Key takeaway

Lock in the FFT convention once, here, and never reinvent it. Every
later phase relies on `fft2c(x) = fftshift(fft2(ifftshift(x)))` with
DC at the centered index `(N/2, N/2)`.

## See also

- [PROGRESS.md §A.4](../PROGRESS.md)
- Source: [src/common/fft_utils.py](../src/common/fft_utils.py),
  [src/optics/scalar_diffraction.py](../src/optics/scalar_diffraction.py)
