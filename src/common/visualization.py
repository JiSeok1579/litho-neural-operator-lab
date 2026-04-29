"""Plotting helpers for masks, spectra, and aerial images.

All helpers expect 2D real or complex tensors and return a matplotlib
``Figure``. They never touch GPU state; tensors are detached, moved to CPU,
and converted to NumPy inside the helpers.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from .fft_utils import amplitude, log_amplitude, phase


def _to_numpy(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().numpy()


def _crop_center(img: np.ndarray, half_size: int) -> np.ndarray:
    """Crop a 2D array to a (2*half_size) square centered on the array center."""
    n = img.shape[-1]
    c = n // 2
    lo = max(0, c - half_size)
    hi = min(n, c + half_size)
    return img[lo:hi, lo:hi]


def show_mask(
    mask: torch.Tensor,
    extent: float | None = None,
    title: str = "mask",
    cmap: str = "gray",
):
    fig, ax = plt.subplots(figsize=(4, 4))
    img = _to_numpy(mask.real if torch.is_complex(mask) else mask)
    ext = None if extent is None else (-extent / 2, extent / 2, -extent / 2, extent / 2)
    ax.imshow(img, cmap=cmap, extent=ext, origin="lower")
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.tight_layout()
    return fig


def show_spectrum(
    spec: torch.Tensor,
    df: float | None = None,
    title_prefix: str = "spectrum",
):
    """Four-panel: amplitude, log amplitude, phase, real part."""
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    n = spec.shape[-1]
    if df is None:
        ext = None
    else:
        f_max = (n / 2) * df
        ext = (-f_max, f_max, -f_max, f_max)

    panels = [
        ("|T|", _to_numpy(amplitude(spec)), "viridis"),
        ("log10 |T|", _to_numpy(log_amplitude(spec)), "viridis"),
        ("phase(T)", _to_numpy(phase(spec)), "twilight"),
        ("Re(T)", _to_numpy(spec.real), "RdBu"),
    ]
    for ax, (name, img, cmap) in zip(axes, panels):
        im = ax.imshow(img, cmap=cmap, extent=ext, origin="lower")
        ax.set_title(f"{title_prefix} - {name}")
        ax.set_xlabel("fx")
        ax.set_ylabel("fy")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


def show_field_pair(
    mask: torch.Tensor,
    spec: torch.Tensor,
    extent: float | None = None,
    df: float | None = None,
    title: str = "",
    freq_zoom: int | None = 32,
    amp_percentile: float = 99.0,
):
    """One-row 1x4 figure: mask | |T| | log|T| | phase(T).

    ``freq_zoom`` crops the spectrum panels to the central ``2 * freq_zoom``
    bins so that diffraction orders sitting near DC are actually visible
    instead of being lost in a 256-pixel-wide DFT canvas. Pass ``None`` to
    show the full spectrum.

    ``amp_percentile`` clips the linear ``|T|`` colormap at that percentile,
    which prevents the DC spike from washing out everything else.
    """
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    n = spec.shape[-1]
    real_ext = None if extent is None else (-extent / 2, extent / 2, -extent / 2, extent / 2)

    amp_img = _to_numpy(amplitude(spec))
    log_img = _to_numpy(log_amplitude(spec))
    phase_img = _to_numpy(phase(spec))

    if freq_zoom is not None:
        amp_img = _crop_center(amp_img, freq_zoom)
        log_img = _crop_center(log_img, freq_zoom)
        phase_img = _crop_center(phase_img, freq_zoom)
        if df is not None:
            f_max = freq_zoom * df
            freq_ext = (-f_max, f_max, -f_max, f_max)
        else:
            freq_ext = None
    else:
        if df is None:
            freq_ext = None
        else:
            f_max = (n / 2) * df
            freq_ext = (-f_max, f_max, -f_max, f_max)

    axes[0].imshow(_to_numpy(mask.real if torch.is_complex(mask) else mask),
                   cmap="gray", extent=real_ext, origin="lower")
    axes[0].set_title("mask")
    axes[0].set_xlabel("x"); axes[0].set_ylabel("y")

    amp_vmax = float(np.percentile(amp_img, amp_percentile)) if amp_percentile < 100 else None
    panels = [
        ("|T|  (clip @ p{:g})".format(amp_percentile), amp_img, "viridis", 0.0, amp_vmax),
        ("log10 |T|", log_img, "viridis", None, None),
        ("phase(T)", phase_img, "twilight", -np.pi, np.pi),
    ]
    for ax, (name, img, cmap, vmin, vmax) in zip(axes[1:], panels):
        im = ax.imshow(img, cmap=cmap, extent=freq_ext, origin="lower",
                       vmin=vmin, vmax=vmax)
        ax.set_title(name)
        ax.set_xlabel("fx"); ax.set_ylabel("fy")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return fig


def show_pupil(
    pupil: torch.Tensor,
    df: float | None = None,
    title: str = "pupil",
):
    """Single-panel binary or apodized pupil view in frequency space."""
    fig, ax = plt.subplots(figsize=(4, 4))
    n = pupil.shape[-1]
    ext = None if df is None else (-(n / 2) * df, (n / 2) * df, -(n / 2) * df, (n / 2) * df)
    im = ax.imshow(_to_numpy(pupil), cmap="gray", extent=ext, origin="lower",
                   vmin=0.0, vmax=1.0)
    ax.set_title(title)
    ax.set_xlabel("fx"); ax.set_ylabel("fy")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


def show_aerial(
    aerial: torch.Tensor,
    extent: float | None = None,
    title: str = "aerial intensity",
    vmax: float | None = 1.0,
    cmap: str = "inferno",
):
    """Single-panel aerial-image view (real, non-negative)."""
    fig, ax = plt.subplots(figsize=(4, 4))
    real_ext = None if extent is None else (-extent / 2, extent / 2, -extent / 2, extent / 2)
    im = ax.imshow(_to_numpy(aerial), cmap=cmap, extent=real_ext, origin="lower",
                   vmin=0.0, vmax=vmax)
    ax.set_title(title)
    ax.set_xlabel("x"); ax.set_ylabel("y")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


def show_aerial_sweep(
    mask: torch.Tensor,
    aerials: list[torch.Tensor],
    NAs: list[float],
    extent: float | None = None,
    suptitle: str = "",
    cmap: str = "inferno",
):
    """One-row figure: mask | aerial(NA_0) | aerial(NA_1) | ...

    Each aerial is assumed normalized to [0, 1] (e.g. via
    ``coherent_aerial_image(..., normalize=True)``); the colormap is fixed
    to vmin=0, vmax=1 so brightness is comparable across panels.
    """
    n_cols = 1 + len(aerials)
    fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4))
    real_ext = None if extent is None else (-extent / 2, extent / 2, -extent / 2, extent / 2)

    mask_img = _to_numpy(mask.real if torch.is_complex(mask) else mask)
    axes[0].imshow(mask_img, cmap="gray", extent=real_ext, origin="lower",
                   vmin=0.0, vmax=1.0)
    axes[0].set_title("mask"); axes[0].set_xlabel("x"); axes[0].set_ylabel("y")

    for ax, aerial, NA in zip(axes[1:], aerials, NAs):
        im = ax.imshow(_to_numpy(aerial), cmap=cmap, extent=real_ext, origin="lower",
                       vmin=0.0, vmax=1.0)
        ax.set_title(f"aerial  NA={NA:g}")
        ax.set_xlabel("x"); ax.set_ylabel("y")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    if suptitle:
        fig.suptitle(suptitle)
    fig.tight_layout()
    return fig


def show_pipeline(
    mask: torch.Tensor,
    spectrum: torch.Tensor,
    pupil: torch.Tensor,
    filtered_spectrum: torch.Tensor,
    aerial: torch.Tensor,
    extent: float | None = None,
    df: float | None = None,
    freq_zoom: int | None = 32,
    suptitle: str = "",
):
    """Five-panel pipeline view: mask -> |T| -> pupil -> |T*P| -> aerial."""
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    n = spectrum.shape[-1]
    real_ext = None if extent is None else (-extent / 2, extent / 2, -extent / 2, extent / 2)
    if freq_zoom is not None and df is not None:
        f_max = freq_zoom * df
        freq_ext = (-f_max, f_max, -f_max, f_max)
        amp_T = _crop_center(_to_numpy(spectrum.abs()), freq_zoom)
        amp_TP = _crop_center(_to_numpy(filtered_spectrum.abs()), freq_zoom)
        pup = _crop_center(_to_numpy(pupil), freq_zoom)
    else:
        if df is None:
            freq_ext = None
        else:
            f_max = (n / 2) * df
            freq_ext = (-f_max, f_max, -f_max, f_max)
        amp_T = _to_numpy(spectrum.abs())
        amp_TP = _to_numpy(filtered_spectrum.abs())
        pup = _to_numpy(pupil)

    mask_img = _to_numpy(mask.real if torch.is_complex(mask) else mask)
    axes[0].imshow(mask_img, cmap="gray", extent=real_ext, origin="lower", vmin=0.0, vmax=1.0)
    axes[0].set_title("mask"); axes[0].set_xlabel("x"); axes[0].set_ylabel("y")

    vmax_T = float(np.percentile(amp_T, 99.0))
    axes[1].imshow(amp_T, cmap="viridis", extent=freq_ext, origin="lower", vmin=0, vmax=vmax_T)
    axes[1].set_title("|T|  (clip @ p99)")
    axes[1].set_xlabel("fx"); axes[1].set_ylabel("fy")

    axes[2].imshow(pup, cmap="gray", extent=freq_ext, origin="lower", vmin=0.0, vmax=1.0)
    axes[2].set_title("pupil  P")
    axes[2].set_xlabel("fx"); axes[2].set_ylabel("fy")

    vmax_TP = float(np.percentile(amp_TP, 99.0)) if amp_TP.max() > 0 else 1.0
    axes[3].imshow(amp_TP, cmap="viridis", extent=freq_ext, origin="lower", vmin=0, vmax=vmax_TP)
    axes[3].set_title("|T * P|  (clip @ p99)")
    axes[3].set_xlabel("fx"); axes[3].set_ylabel("fy")

    axes[4].imshow(_to_numpy(aerial), cmap="inferno", extent=real_ext, origin="lower",
                   vmin=0.0, vmax=1.0)
    axes[4].set_title("aerial  |E|^2")
    axes[4].set_xlabel("x"); axes[4].set_ylabel("y")

    if suptitle:
        fig.suptitle(suptitle)
    fig.tight_layout()
    return fig


def show_inverse_result(
    initial_mask: torch.Tensor,
    final_mask: torch.Tensor,
    initial_aerial: torch.Tensor,
    final_aerial: torch.Tensor,
    target_region: torch.Tensor,
    forbidden_region: torch.Tensor,
    extent: float | None = None,
    target_value: float = 1.0,
    suptitle: str = "",
):
    """2-row x 4-col before / after view of an inverse-mask optimization.

    Row 0 (initial): mask | aerial | aerial vs target row-cut | regions overlay
    Row 1 (final):   mask | aerial | aerial vs target row-cut | regions overlay
    """
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    real_ext = None if extent is None else (-extent / 2, extent / 2, -extent / 2, extent / 2)

    rows = [
        ("initial", initial_mask, initial_aerial),
        ("final", final_mask, final_aerial),
    ]
    for r, (label, m, a) in enumerate(rows):
        # mask
        axes[r, 0].imshow(_to_numpy(m.real if torch.is_complex(m) else m),
                          cmap="gray", extent=real_ext, origin="lower",
                          vmin=0.0, vmax=1.0)
        axes[r, 0].set_title(f"{label} mask")
        axes[r, 0].set_xlabel("x"); axes[r, 0].set_ylabel("y")

        # aerial
        a_max = float(a.max().item())
        im = axes[r, 1].imshow(_to_numpy(a), cmap="inferno", extent=real_ext,
                               origin="lower", vmin=0.0, vmax=max(a_max, 1e-6))
        axes[r, 1].set_title(f"{label} aerial  (peak={a_max:.3f})")
        axes[r, 1].set_xlabel("x"); axes[r, 1].set_ylabel("y")
        fig.colorbar(im, ax=axes[r, 1], fraction=0.046, pad=0.04)

        # row cut through y=0 (middle row of the array)
        n = a.shape[-1]
        cut = _to_numpy(a[n // 2, :])
        x_axis = (np.arange(n) - n / 2) * (extent / n if extent else 1.0)
        axes[r, 2].plot(x_axis, cut, label="aerial", color="C1")
        axes[r, 2].axhline(target_value, color="C2", linestyle="--", label=f"target={target_value}")
        # mark target region extent along the cut
        tr_cut = _to_numpy(target_region[n // 2, :])
        axes[r, 2].fill_between(x_axis, 0, max(target_value, a_max), where=tr_cut > 0.5,
                                alpha=0.15, color="green", label="target")
        fr_cut = _to_numpy(forbidden_region[n // 2, :])
        axes[r, 2].fill_between(x_axis, 0, max(target_value, a_max), where=fr_cut > 0.5,
                                alpha=0.10, color="red", label="forbidden")
        axes[r, 2].set_title(f"{label} aerial  (y=0 cut)")
        axes[r, 2].set_xlabel("x"); axes[r, 2].set_ylabel("intensity")
        axes[r, 2].legend(loc="upper right", fontsize=8)
        axes[r, 2].grid(alpha=0.3)

        # regions overlay (target green, forbidden red, on top of aerial)
        rgb = np.stack([
            _to_numpy(forbidden_region) * 0.6,                 # R
            _to_numpy(target_region) * 0.6,                    # G
            np.zeros_like(_to_numpy(target_region)),           # B
        ], axis=-1)
        bg = _to_numpy(a) / max(a_max, 1e-12)
        gray = np.stack([bg, bg, bg], axis=-1) * 0.4
        composite = np.clip(gray + rgb, 0, 1)
        axes[r, 3].imshow(composite, extent=real_ext, origin="lower")
        axes[r, 3].set_title(f"{label} regions on aerial")
        axes[r, 3].set_xlabel("x"); axes[r, 3].set_ylabel("y")

    if suptitle:
        fig.suptitle(suptitle)
    fig.tight_layout()
    return fig


def show_loss_history(history: list[dict], suptitle: str = ""):
    """Two-panel loss / intensity history figure.

    Panel A (log y): total / target / background / tv / binarization losses
    Panel B (linear): mean target intensity, mean forbidden intensity
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    its = [h["iter"] for h in history]

    keys = [
        ("loss_total", "total", "black", "-"),
        ("loss_target", "target", "C0", "-"),
        ("loss_background", "background", "C3", "-"),
        ("loss_tv", "tv", "C2", "--"),
        ("loss_binarization", "binarization", "C4", ":"),
    ]
    for key, label, color, ls in keys:
        ys = [h[key] for h in history]
        if max(ys) > 0:
            axes[0].plot(its, ys, label=label, color=color, linestyle=ls)
    axes[0].set_yscale("log")
    axes[0].set_title("loss components")
    axes[0].set_xlabel("iteration"); axes[0].set_ylabel("loss (log)")
    axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)

    target_int = [h["mean_intensity_target"] for h in history]
    bg_int = [h["mean_intensity_background"] for h in history]
    axes[1].plot(its, target_int, label="mean(I) in target", color="C2")
    axes[1].plot(its, bg_int, label="mean(I) in forbidden", color="C3")
    axes[1].set_title("mean aerial intensity per region")
    axes[1].set_xlabel("iteration"); axes[1].set_ylabel("intensity")
    axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)

    if suptitle:
        fig.suptitle(suptitle)
    fig.tight_layout()
    return fig


def show_source(
    source: torch.Tensor,
    title: str = "source",
    sigma_max: float = 1.0,
):
    fig, ax = plt.subplots(figsize=(4, 4))
    img = _to_numpy(source)
    ext = (-sigma_max, sigma_max, -sigma_max, sigma_max)
    ax.imshow(img, cmap="gray", extent=ext, origin="lower",
              vmin=0.0, vmax=max(float(img.max()), 1e-6))
    ax.set_title(title)
    ax.set_xlabel("sigma_x"); ax.set_ylabel("sigma_y")
    ax.set_aspect("equal")
    fig.tight_layout()
    return fig


def show_partial_coherence_sweep(
    mask: torch.Tensor,
    sources: list[torch.Tensor],
    aerials: list[torch.Tensor],
    source_names: list[str],
    extent: float | None = None,
    sigma_max: float = 1.0,
    suptitle: str = "",
):
    """3 rows x N columns: (mask repeated) | source | aerial intensity.

    Aerial colormap is fixed to vmin=0, vmax=1 because Phase 4 demos
    feed in normalized aerials so brightness is comparable across sources.
    """
    n = len(sources)
    fig, axes = plt.subplots(3, n, figsize=(4 * n, 12))
    if n == 1:
        axes = axes.reshape(3, 1)
    real_ext = None if extent is None else (-extent / 2, extent / 2, -extent / 2, extent / 2)
    sigma_ext = (-sigma_max, sigma_max, -sigma_max, sigma_max)
    mask_img = _to_numpy(mask.real if torch.is_complex(mask) else mask)

    for k, (src, aerial, name) in enumerate(zip(sources, aerials, source_names)):
        # Row 0: mask (same in every column for visual context)
        axes[0, k].imshow(mask_img, cmap="gray", extent=real_ext, origin="lower",
                          vmin=0.0, vmax=1.0)
        axes[0, k].set_title(f"mask\nsource: {name}")
        axes[0, k].set_xlabel("x"); axes[0, k].set_ylabel("y")

        # Row 1: source in sigma-space
        s_img = _to_numpy(src)
        axes[1, k].imshow(s_img, cmap="gray", extent=sigma_ext, origin="lower",
                          vmin=0.0, vmax=max(float(s_img.max()), 1e-6))
        axes[1, k].set_title(f"source: {name}")
        axes[1, k].set_xlabel("sigma_x"); axes[1, k].set_ylabel("sigma_y")
        axes[1, k].set_aspect("equal")

        # Row 2: aerial intensity (peak shown in title)
        a = _to_numpy(aerial)
        a_max = float(a.max())
        im = axes[2, k].imshow(a, cmap="inferno", extent=real_ext, origin="lower",
                               vmin=0.0, vmax=max(a_max, 1e-6))
        axes[2, k].set_title(f"aerial  peak={a_max:.3f}")
        axes[2, k].set_xlabel("x"); axes[2, k].set_ylabel("y")
        fig.colorbar(im, ax=axes[2, k], fraction=0.046, pad=0.04)

    if suptitle:
        fig.suptitle(suptitle)
    fig.tight_layout()
    return fig


def show_resist_chain(
    aerial: torch.Tensor,
    acid_initial: torch.Tensor,
    acid_diffused: torch.Tensor,
    resist: torch.Tensor,
    extent: float | None = None,
    threshold: float | None = None,
    suptitle: str = "",
):
    """4-panel chain: aerial -> acid -> diffused acid -> resist.

    For ``threshold`` not None, an isocontour is overlaid on the diffused
    acid panel to mark the developer cut.
    """
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    real_ext = None if extent is None else (-extent / 2, extent / 2, -extent / 2, extent / 2)
    panels = [
        ("aerial  |E|^2", aerial, "inferno", 0.0, max(float(aerial.max().item()), 1e-6)),
        ("acid  A_0", acid_initial, "magma", 0.0, max(float(acid_initial.max().item()), 1e-6)),
        ("acid (after diffusion)", acid_diffused, "magma", 0.0, max(float(acid_diffused.max().item()), 1e-6)),
        ("resist  R = sigmoid(beta(A - A_th))", resist, "Greens", 0.0, 1.0),
    ]
    for ax, (name, img, cmap, vmin, vmax) in zip(axes, panels):
        im = ax.imshow(_to_numpy(img), cmap=cmap, extent=real_ext, origin="lower",
                       vmin=vmin, vmax=vmax)
        ax.set_title(name)
        ax.set_xlabel("x"); ax.set_ylabel("y")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    if threshold is not None:
        # Add a contour at the threshold on the diffused-acid panel.
        n = acid_diffused.shape[-1]
        if extent is not None:
            xs = np.linspace(-extent / 2, extent / 2, n)
        else:
            xs = np.arange(n)
        axes[2].contour(xs, xs, _to_numpy(acid_diffused), levels=[threshold],
                        colors="cyan", linewidths=1.0)

    if suptitle:
        fig.suptitle(suptitle)
    fig.tight_layout()
    return fig


def show_resist_sweep(
    rows_data: list[tuple[str, torch.Tensor, torch.Tensor]],
    extent: float | None = None,
    suptitle: str = "",
    cmap_acid: str = "magma",
    cmap_resist: str = "Greens",
):
    """Stack of (label, acid_diffused, resist) rows in a 2-column layout.

    Each row plots the diffused acid on the left and the resist (sigmoid
    threshold) on the right, with the label in the row title. Used for
    dose / diffusion-length sweeps.
    """
    n_rows = len(rows_data)
    fig, axes = plt.subplots(n_rows, 2, figsize=(8, 4 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, 2)
    real_ext = None if extent is None else (-extent / 2, extent / 2, -extent / 2, extent / 2)

    for r, (label, acid, resist) in enumerate(rows_data):
        a_max = max(float(acid.max().item()), 1e-6)
        im = axes[r, 0].imshow(_to_numpy(acid), cmap=cmap_acid, extent=real_ext,
                               origin="lower", vmin=0.0, vmax=a_max)
        axes[r, 0].set_title(f"{label}  acid (diffused)")
        axes[r, 0].set_xlabel("x"); axes[r, 0].set_ylabel("y")
        fig.colorbar(im, ax=axes[r, 0], fraction=0.046, pad=0.04)

        im2 = axes[r, 1].imshow(_to_numpy(resist), cmap=cmap_resist, extent=real_ext,
                                origin="lower", vmin=0.0, vmax=1.0)
        axes[r, 1].set_title(f"{label}  resist")
        axes[r, 1].set_xlabel("x"); axes[r, 1].set_ylabel("y")
        fig.colorbar(im2, ax=axes[r, 1], fraction=0.046, pad=0.04)

    if suptitle:
        fig.suptitle(suptitle)
    fig.tight_layout()
    return fig


def show_pinn_vs_solvers(
    A_truth: torch.Tensor,
    A_fd: torch.Tensor,
    A_fft: torch.Tensor,
    A_pinn: torch.Tensor,
    extent: float | None = None,
    suptitle: str = "",
):
    """2x4 figure: top row = solutions, bottom row = absolute errors.

    All four solutions share a colormap range; absolute errors share a
    second range. The "analytic" panel in the bottom row is reserved as
    a label slot showing 0 (reference).
    """
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    real_ext = None if extent is None else (-extent / 2, extent / 2, -extent / 2, extent / 2)

    sols = [
        ("analytic", A_truth),
        ("FD", A_fd),
        ("FFT", A_fft),
        ("PINN", A_pinn),
    ]
    sol_max = max(float(s[1].max().item()) for s in sols)
    err_max = max(float((s[1] - A_truth).abs().max().item()) for s in sols if s[0] != "analytic")
    err_max = max(err_max, 1e-12)

    for c, (name, A) in enumerate(sols):
        im = axes[0, c].imshow(_to_numpy(A), cmap="viridis", extent=real_ext,
                               origin="lower", vmin=0.0, vmax=sol_max)
        axes[0, c].set_title(f"{name}  peak={float(A.max().item()):.4f}")
        axes[0, c].set_xlabel("x"); axes[0, c].set_ylabel("y")
        fig.colorbar(im, ax=axes[0, c], fraction=0.046, pad=0.04)

        if name == "analytic":
            err = torch.zeros_like(A)
            title = "(reference)"
        else:
            err = (A - A_truth).abs()
            title = f"|{name} - analytic|  max={float(err.max().item()):.4e}"
        im = axes[1, c].imshow(_to_numpy(err), cmap="inferno", extent=real_ext,
                               origin="lower", vmin=0.0, vmax=err_max)
        axes[1, c].set_title(title)
        axes[1, c].set_xlabel("x"); axes[1, c].set_ylabel("y")
        fig.colorbar(im, ax=axes[1, c], fraction=0.046, pad=0.04)

    if suptitle:
        fig.suptitle(suptitle)
    fig.tight_layout()
    return fig


def show_pinn_training(history: list[dict], suptitle: str = ""):
    """Loss history for PINN training. ``history`` is a list of dicts with
    keys ``iter``, ``loss_total``, ``loss_pde``, ``loss_ic``."""
    its = [h["iter"] for h in history]
    fig, ax = plt.subplots(figsize=(8, 4))
    for key, label, color, ls in [
        ("loss_total", "total", "black", "-"),
        ("loss_pde", "PDE residual", "C0", "-"),
        ("loss_ic", "initial condition", "C3", "-"),
    ]:
        ys = [max(h[key], 1e-20) for h in history]
        ax.plot(its, ys, label=label, color=color, linestyle=ls)
    ax.set_yscale("log")
    ax.set_title("PINN training")
    ax.set_xlabel("iteration"); ax.set_ylabel("loss (log)")
    ax.legend()
    ax.grid(alpha=0.3)
    if suptitle:
        fig.suptitle(suptitle)
    fig.tight_layout()
    return fig


def show_correction_samples(
    samples: list[dict],
    extent: float | None = None,
    df: float | None = None,
    freq_zoom: int | None = 16,
    suptitle: str = "",
):
    """One row per sample: mask | |T_thin| | |C| | |T_3d| | |Delta T|.

    ``samples`` is a list of dicts with keys ``mask``, ``T_thin``, ``C``,
    ``T_3d`` (all 2D torch tensors). Spectra are zoomed to the central
    ``2 * freq_zoom`` bins and clipped at the 99th percentile so weak
    side-lobes are visible.
    """
    n_rows = len(samples)
    fig, axes = plt.subplots(n_rows, 5, figsize=(20, 4 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, 5)
    real_ext = None if extent is None else (-extent / 2, extent / 2, -extent / 2, extent / 2)

    for r, s in enumerate(samples):
        mask = s["mask"]; T_thin = s["T_thin"]; C = s["C"]; T_3d = s["T_3d"]
        delta = T_3d - T_thin
        n = T_thin.shape[-1]

        amp_T_thin = _to_numpy(T_thin.abs())
        amp_C = _to_numpy(C.abs())
        amp_T_3d = _to_numpy(T_3d.abs())
        amp_delta = _to_numpy(delta.abs())
        if freq_zoom is not None:
            amp_T_thin = _crop_center(amp_T_thin, freq_zoom)
            amp_C = _crop_center(amp_C, freq_zoom)
            amp_T_3d = _crop_center(amp_T_3d, freq_zoom)
            amp_delta = _crop_center(amp_delta, freq_zoom)
            if df is not None:
                fmax = freq_zoom * df
                freq_ext = (-fmax, fmax, -fmax, fmax)
            else:
                freq_ext = None
        else:
            if df is None:
                freq_ext = None
            else:
                fmax = (n / 2) * df
                freq_ext = (-fmax, fmax, -fmax, fmax)

        mask_img = _to_numpy(mask.real if torch.is_complex(mask) else mask)
        axes[r, 0].imshow(mask_img, cmap="gray", extent=real_ext, origin="lower",
                          vmin=0.0, vmax=1.0)
        axes[r, 0].set_title("mask"); axes[r, 0].set_xlabel("x"); axes[r, 0].set_ylabel("y")

        for ax, name, img in (
            (axes[r, 1], "|T_thin|", amp_T_thin),
            (axes[r, 2], "|C|",      amp_C),
            (axes[r, 3], "|T_3d|",   amp_T_3d),
            (axes[r, 4], "|Delta T|", amp_delta),
        ):
            vmax = float(np.percentile(img, 99.0))
            im = ax.imshow(img, cmap="viridis", extent=freq_ext, origin="lower",
                           vmin=0.0, vmax=max(vmax, 1e-12))
            ax.set_title(name); ax.set_xlabel("fx"); ax.set_ylabel("fy")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    if suptitle:
        fig.suptitle(suptitle)
    fig.tight_layout()
    return fig


def show_fno_predictions(
    samples: list[dict],
    df: float | None = None,
    freq_zoom: int | None = 16,
    suptitle: str = "",
):
    """Per-sample 4-panel row: |delta_T true| | |delta_T pred| | |error| | |T_3d true vs pred ratio|.

    ``samples`` is a list of dicts with keys
    ``delta_T_true``, ``delta_T_pred``, ``T_thin``, ``params_str`` (str).
    """
    n = len(samples)
    fig, axes = plt.subplots(n, 4, figsize=(16, 4 * n))
    if n == 1:
        axes = axes.reshape(1, 4)

    for r, s in enumerate(samples):
        dt_true = s["delta_T_true"]
        dt_pred = s["delta_T_pred"]
        T_thin = s["T_thin"]
        params_str = s.get("params_str", "")

        amp_true = _to_numpy(dt_true.abs())
        amp_pred = _to_numpy(dt_pred.abs())
        amp_err = _to_numpy((dt_pred - dt_true).abs())
        T_3d_true = T_thin + dt_true
        T_3d_pred = T_thin + dt_pred
        amp_T_true = _to_numpy(T_3d_true.abs())
        amp_T_pred = _to_numpy(T_3d_pred.abs())

        nside = amp_true.shape[-1]
        if freq_zoom is not None:
            amp_true = _crop_center(amp_true, freq_zoom)
            amp_pred = _crop_center(amp_pred, freq_zoom)
            amp_err = _crop_center(amp_err, freq_zoom)
            amp_T_true = _crop_center(amp_T_true, freq_zoom)
            amp_T_pred = _crop_center(amp_T_pred, freq_zoom)
            if df is not None:
                fmax = freq_zoom * df
                ext = (-fmax, fmax, -fmax, fmax)
            else:
                ext = None
        else:
            if df is None:
                ext = None
            else:
                fmax = (nside / 2) * df
                ext = (-fmax, fmax, -fmax, fmax)

        # Use the same vmax for true vs pred panels for fairness
        delta_max = max(float(amp_true.max()), float(amp_pred.max()), 1e-12)

        im0 = axes[r, 0].imshow(amp_true, cmap="viridis", extent=ext, origin="lower",
                                vmin=0, vmax=delta_max)
        axes[r, 0].set_title(f"|Delta T| true   {params_str}" if r == 0 and params_str else "|Delta T| true")
        axes[r, 0].set_xlabel("fx"); axes[r, 0].set_ylabel("fy")
        fig.colorbar(im0, ax=axes[r, 0], fraction=0.046, pad=0.04)

        im1 = axes[r, 1].imshow(amp_pred, cmap="viridis", extent=ext, origin="lower",
                                vmin=0, vmax=delta_max)
        axes[r, 1].set_title("|Delta T| pred")
        axes[r, 1].set_xlabel("fx"); axes[r, 1].set_ylabel("fy")
        fig.colorbar(im1, ax=axes[r, 1], fraction=0.046, pad=0.04)

        im2 = axes[r, 2].imshow(amp_err, cmap="inferno", extent=ext, origin="lower",
                                vmin=0, vmax=max(float(amp_err.max()), 1e-12))
        axes[r, 2].set_title(f"|err|  max={float(amp_err.max()):.4f}")
        axes[r, 2].set_xlabel("fx"); axes[r, 2].set_ylabel("fy")
        fig.colorbar(im2, ax=axes[r, 2], fraction=0.046, pad=0.04)

        T_max = max(float(amp_T_true.max()), float(amp_T_pred.max()), 1e-12)
        im3 = axes[r, 3].imshow(amp_T_pred, cmap="viridis", extent=ext, origin="lower",
                                vmin=0, vmax=T_max)
        axes[r, 3].set_title("|T_3d| pred")
        axes[r, 3].set_xlabel("fx"); axes[r, 3].set_ylabel("fy")
        fig.colorbar(im3, ax=axes[r, 3], fraction=0.046, pad=0.04)

    if suptitle:
        fig.suptitle(suptitle)
    fig.tight_layout()
    return fig


def show_fno_training(
    epochs: list[int],
    train_losses: list[float],
    test_losses: list[float],
    test_complex_rel_err: list[float] | None = None,
    suptitle: str = "",
):
    """Training history with train / test loss on log y plus optional
    complex-relative-error subplot."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(epochs, train_losses, label="train spectrum MSE", color="C0")
    axes[0].plot(epochs, test_losses, label="test spectrum MSE",
                 color="C3", linestyle="--")
    axes[0].set_yscale("log")
    axes[0].set_title("loss")
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("MSE (log)")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    if test_complex_rel_err is not None:
        axes[1].plot(epochs, test_complex_rel_err, color="C2")
        axes[1].set_title("test complex relative error")
        axes[1].set_xlabel("epoch"); axes[1].set_ylabel("rel err")
        axes[1].grid(alpha=0.3)
    else:
        axes[1].axis("off")

    if suptitle:
        fig.suptitle(suptitle)
    fig.tight_layout()
    return fig


def show_closed_loop_comparison(
    rows: list[dict],
    extent: float | None = None,
    target_value: float = 1.0,
    suptitle: str = "",
):
    """One row per scenario in ``rows``.

    Each row dict expects: ``label``, ``mask``, ``aerial``,
    ``target_region``, ``forbidden_region`` (and optionally
    ``aerial_alt`` for the validation row in Phase 9).

    Layout per row: mask | aerial | y=0 cut | regions overlay (4 cols).
    """
    n = len(rows)
    fig, axes = plt.subplots(n, 4, figsize=(16, 4 * n))
    if n == 1:
        axes = axes.reshape(1, 4)
    real_ext = None if extent is None else (-extent / 2, extent / 2,
                                            -extent / 2, extent / 2)

    for r, row in enumerate(rows):
        label = row["label"]
        mask = row["mask"]
        aerial = row["aerial"]
        target_region = row["target_region"]
        forbidden_region = row["forbidden_region"]

        # mask
        axes[r, 0].imshow(_to_numpy(mask.real if torch.is_complex(mask) else mask),
                          cmap="gray", extent=real_ext, origin="lower",
                          vmin=0.0, vmax=1.0)
        axes[r, 0].set_title(f"{label}: mask")
        axes[r, 0].set_xlabel("x"); axes[r, 0].set_ylabel("y")

        # aerial
        a_max = float(aerial.max().item())
        im = axes[r, 1].imshow(_to_numpy(aerial), cmap="inferno",
                               extent=real_ext, origin="lower",
                               vmin=0.0, vmax=max(a_max, 1e-6))
        axes[r, 1].set_title(f"{label}: aerial  peak={a_max:.3f}")
        axes[r, 1].set_xlabel("x"); axes[r, 1].set_ylabel("y")
        fig.colorbar(im, ax=axes[r, 1], fraction=0.046, pad=0.04)

        # row cut
        nside = aerial.shape[-1]
        cut = _to_numpy(aerial[nside // 2, :])
        x_axis = np.linspace(-extent / 2, extent / 2, nside) if extent else np.arange(nside)
        axes[r, 2].plot(x_axis, cut, label="aerial", color="C1")
        if "aerial_alt" in row and row["aerial_alt"] is not None:
            cut_alt = _to_numpy(row["aerial_alt"][nside // 2, :])
            axes[r, 2].plot(x_axis, cut_alt, label=row.get("aerial_alt_label", "aerial alt"),
                            color="C0", linestyle="--")
        axes[r, 2].axhline(target_value, color="C2", linestyle=":",
                           label=f"target={target_value}")
        tr_cut = _to_numpy(target_region[nside // 2, :])
        upper = max(target_value, a_max)
        axes[r, 2].fill_between(x_axis, 0, upper, where=tr_cut > 0.5,
                                alpha=0.15, color="green")
        fr_cut = _to_numpy(forbidden_region[nside // 2, :])
        axes[r, 2].fill_between(x_axis, 0, upper, where=fr_cut > 0.5,
                                alpha=0.10, color="red")
        axes[r, 2].set_title(f"{label}: y=0 cut")
        axes[r, 2].set_xlabel("x"); axes[r, 2].set_ylabel("intensity")
        axes[r, 2].legend(loc="upper right", fontsize=8)
        axes[r, 2].grid(alpha=0.3)

        # regions overlay
        rgb = np.stack([
            _to_numpy(forbidden_region) * 0.6,
            _to_numpy(target_region) * 0.6,
            np.zeros_like(_to_numpy(target_region)),
        ], axis=-1)
        bg = _to_numpy(aerial) / max(a_max, 1e-12)
        gray = np.stack([bg, bg, bg], axis=-1) * 0.4
        composite = np.clip(gray + rgb, 0, 1)
        axes[r, 3].imshow(composite, extent=real_ext, origin="lower")
        axes[r, 3].set_title(f"{label}: regions on aerial")
        axes[r, 3].set_xlabel("x"); axes[r, 3].set_ylabel("y")

    if suptitle:
        fig.suptitle(suptitle)
    fig.tight_layout()
    return fig


def save_figure(fig, path: str | Path, dpi: int = 150) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path
