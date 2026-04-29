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


def save_figure(fig, path: str | Path, dpi: int = 150) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path
