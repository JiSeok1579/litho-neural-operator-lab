# Phase 4 — Partial coherence (Hopkins source integration)

**Status:** ✅ done
**PR:** #6 · 21 tests added (70 / 70 total green)

## Goal

Move from "single plane wave" to **distributed illumination**. Each
source point in σ-space contributes a shifted-pupil aerial, and the
total aerial is the incoherent sum.

## What landed

- `src/optics/source.py` — five source-shape factories
  (`coherent_source`, `annular_source`, `dipole_source`,
  `quadrupole_source`, `random_source`) plus `sigma_axis`,
  `sigma_meshgrid`, and a `source_points` decoder that turns a 2D
  source tensor into a `(K, 2)` σ list and `(K,)` weights.
- `src/optics/pupil.py` extended with
  `circular_pupil_at(grid, NA, center_freq, wavelength)`.
- `src/optics/partial_coherence.py` — `partial_coherent_aerial_image`
  evaluates the Hopkins integral as a **batched FFT** over a
  `(K, N, N)` shifted-pupil stack rather than a Python loop. Even
  with ~300 source points (annular) the demo finishes sub-second.
- `src/common/metrics.py` (new module) — `image_contrast`,
  `peak_intensity_in_region`, `integrated_leakage`,
  `normalized_image_log_slope` (study-grade NILS).
- Visualization: `show_source`, `show_partial_coherence_sweep`.

## Demo

```bash
python experiments/03_partial_coherence/demo_source_shapes.py
```

Saves three figures and `outputs/logs/phase4_metrics.csv`.

## Verified results

Vertical line-space (pitch=1.5 λ, NA=0.4) — fundamental at
fx=0.667 cycles/λ is unreachable on-axis at NA=0.4 (cutoff 0.4):

| source | peak | contrast | leakage(3-6 λ) |
|---|---|---|---|
| coherent | 0.321 | 0.247 | 3399 |
| annular | 0.398 | 0.348 | 3492 |
| **dipole-x** | **0.668** | **0.809** | 4162 |
| dipole-y | 0.287 | 0.144 | 3390 |
| quadrupole | 0.435 | 0.416 | 3559 |

Dipole-x at σ=0.7 places the right pole's pupil center at
fx ~ +0.28 cycles/λ, whose 0.4-radius reach hits 0.68 — just
captures the +1 order. Dipole-y shifts only along fy and cannot help
vertical lines.

Contact hole r=0.5 λ at NA=0.4: coherent gives the highest peak
(0.124); off-axis sources spread energy and suppress the peak by
10 – 15 %.

## Bug captured here

`linspace(-1, 1, 31)` in float32 placed the center pixel at ~5e-8
instead of 0, which **shifted the equivalent pupil** enough to swap
a few boundary pixels and broke the "delta source = coherent imaging"
sanity check by ~2 %. Fix: build `sigma_axis` as
`arange(-half, half+1) / half` so the center is bit-exactly zero.

## Key takeaway

Same mask, completely different aerial depending on the source
shape. **Source design is as much a degree of freedom as mask design**
— Phase 9's closed loop will eventually want to co-optimize them.

## See also

- [PROGRESS.md §A.7](../PROGRESS.md)
- Source: [src/optics/source.py](../src/optics/source.py),
  [src/optics/partial_coherence.py](../src/optics/partial_coherence.py)
