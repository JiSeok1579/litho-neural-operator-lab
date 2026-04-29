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


def save_figure(fig, path: str | Path, dpi: int = 150) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path
