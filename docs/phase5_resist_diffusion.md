# Phase 5 — Resist exposure, diffusion, threshold

**Status:** ✅ done
**PR:** #7 · 21 tests added (91 / 91 total green)

## Goal

Add the **optical-to-chemical chain** that turns an aerial intensity
into a printed resist contour:

```text
aerial  →  Dill exposure (acid)  →  diffusion (FD or FFT)
                                 →  sigmoid threshold (resist)
```

## What landed

- `src/resist/exposure.py` — `acid_from_aerial` (Dill saturation
  `A0 = 1 - exp(-eta * dose * I)`).
- `src/resist/diffusion_fd.py` — `laplacian_5pt` with periodic BC via
  `torch.roll`, `step_diffusion_fd`, `diffuse_fd`. Auto-picks `dt` to
  satisfy CFL `dt ≤ cfl_safety * dx² / (4 D)` and rejects
  user-provided steps that violate it.
- `src/resist/diffusion_fft.py` — exact heat-kernel diffusion in two
  flavors: `diffuse_fft(D, t)` and the more sweep-friendly
  `diffuse_fft_by_length(L)` parameterized directly by the
  diffusion length `L = sqrt(2 D t)`.
- `src/resist/reaction_diffusion.py` — coupled
  `dA/dt = D_A laplacian(A) - k A Q`, `dQ/dt = -k A Q` with explicit
  Euler and a non-negative clamp.
- `src/resist/threshold.py` — `soft_threshold` (sigmoid),
  `hard_threshold`, `measure_cd_horizontal` (longest above-threshold
  run on the central row, returned in length units), and
  `thresholded_area`.
- Visualization: `show_resist_chain` (4-panel chain with optional
  threshold contour) and `show_resist_sweep`.

## Demos

```bash
python experiments/04_resist_diffusion/demo_dose_sweep.py
python experiments/04_resist_diffusion/demo_diffusion_length.py
```

Saves `phase5_dose_sweep.png`, `phase5_dose_chain_dose1.5.png`,
`phase5_diffusion_length_sweep.png` and two metrics CSVs.

## Verified results

Dose sweep (line-space pitch=2 λ, NA=0.6, L_diff=0.10 λ, A_th=0.40):

| dose | acid_max | acid_diff_max | resist_max | CD (λ) |
|---|---|---|---|---|
| 0.5 | 0.394 | 0.377 | 0.362 | **0.000** |
| 1.0 | 0.632 | 0.612 | 0.995 | 0.664 |
| 1.5 | 0.777 | 0.758 | 1.000 | 0.859 |
| 2.0 | 0.865 | 0.848 | 1.000 | 0.938 |

Diffusion-length sweep (dose=1.5):

| L_diff | acid_max | acid_min | acid contrast | CD (λ) |
|---|---|---|---|---|
| 0.00 | 0.777 | 0.000 | 1.000 | 0.859 |
| 0.05 | 0.772 | 0.004 | 0.991 | 0.859 |
| 0.10 | 0.758 | 0.010 | 0.974 | 0.859 |
| 0.20 | 0.695 | 0.023 | 0.937 | 0.820 |
| 0.40 | 0.520 | 0.136 | 0.586 | 0.742 |

## Verified physics

- At dose 0.5 the peak acid (0.394) sits **just below** the threshold
  (0.40) so nothing prints — the canonical under-exposure cliff.
- For small diffusion lengths (L < 0.10 λ) CD is invariant; at
  L = 0.40 λ the acid contrast collapses to 0.59 and CD shrinks 14 %.

## Key takeaway

Diffusion shrinks **both** signal and leakage in absolute terms, but
the threshold cut is what determines whether sub-threshold features
print at all — exactly the aerial-vs-resist duality study plan §5.10
emphasizes.

## See also

- [PROGRESS.md §A.8](../PROGRESS.md)
- Source: [src/resist/](../src/resist/)
