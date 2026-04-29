"""Plotting helpers shared by the PEB submodule's experiments.

Self-contained on purpose — no imports from the main repo's
``src/common/visualization.py``.
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def _to_numpy(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().numpy()


def show_aerial_and_acid(
    I: torch.Tensor,
    H0: torch.Tensor,
    Hmax: float,
    suptitle: str = "",
    cmap_aerial: str = "inferno",
    cmap_acid: str = "magma",
):
    """2-panel figure: aerial intensity ``I`` | initial acid ``H0``."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    im0 = axes[0].imshow(_to_numpy(I), cmap=cmap_aerial, origin="lower",
                         vmin=0.0, vmax=1.0)
    axes[0].set_title(f"aerial I(x, y)   peak={float(I.max().item()):.3f}")
    axes[0].set_xlabel("x [px]"); axes[0].set_ylabel("y [px]")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(_to_numpy(H0), cmap=cmap_acid, origin="lower",
                         vmin=0.0, vmax=Hmax)
    axes[1].set_title(f"acid H_0(x, y)   peak={float(H0.max().item()):.4f} mol/dm^3")
    axes[1].set_xlabel("x [px]"); axes[1].set_ylabel("y [px]")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    if suptitle:
        fig.suptitle(suptitle)
    fig.tight_layout()
    return fig


def show_dose_sweep(
    I: torch.Tensor,
    H0_list: list[torch.Tensor],
    doses: list[float],
    Hmax: float,
    suptitle: str = "",
):
    """One-row figure: aerial in column 0, then ``H_0`` for each dose."""
    n_cols = 1 + len(H0_list)
    fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4))
    im0 = axes[0].imshow(_to_numpy(I), cmap="inferno", origin="lower",
                         vmin=0.0, vmax=1.0)
    axes[0].set_title(f"aerial I(x, y)   peak={float(I.max().item()):.3f}")
    axes[0].set_xlabel("x [px]"); axes[0].set_ylabel("y [px]")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    for ax, H0, dose in zip(axes[1:], H0_list, doses):
        im = ax.imshow(_to_numpy(H0), cmap="magma", origin="lower",
                       vmin=0.0, vmax=Hmax)
        ax.set_title(f"dose={dose:g}   peak={float(H0.max().item()):.4f}")
        ax.set_xlabel("x [px]"); ax.set_ylabel("y [px]")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    if suptitle:
        fig.suptitle(suptitle)
    fig.tight_layout()
    return fig


def show_diffusion_chain(
    H0: torch.Tensor,
    H_after: torch.Tensor,
    dx_nm: float,
    t_end_s: float,
    Hmax: float | None = None,
    suptitle: str = "",
):
    """3-panel: initial ``H_0`` | ``H(t)`` after diffusion | difference."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    n = H0.shape[-1]
    extent_nm = n * dx_nm
    real_ext = (-extent_nm / 2, extent_nm / 2, -extent_nm / 2, extent_nm / 2)
    vmax = float(Hmax) if Hmax is not None else max(
        float(H0.max().item()), float(H_after.max().item()), 1e-12)

    im0 = axes[0].imshow(_to_numpy(H0), cmap="magma", extent=real_ext,
                         origin="lower", vmin=0.0, vmax=vmax)
    axes[0].set_title(f"H_0(x, y)   peak={float(H0.max().item()):.4f}")
    axes[0].set_xlabel("x [nm]"); axes[0].set_ylabel("y [nm]")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(_to_numpy(H_after), cmap="magma", extent=real_ext,
                         origin="lower", vmin=0.0, vmax=vmax)
    axes[1].set_title(f"H(t={t_end_s:g} s)   peak={float(H_after.max().item()):.4f}")
    axes[1].set_xlabel("x [nm]"); axes[1].set_ylabel("y [nm]")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    diff = (H_after - H0)
    diff_max = float(diff.abs().max().item()) or 1e-12
    im2 = axes[2].imshow(_to_numpy(diff), cmap="RdBu_r", extent=real_ext,
                         origin="lower", vmin=-diff_max, vmax=+diff_max)
    axes[2].set_title(f"H(t) - H_0   max|diff|={diff_max:.4f}")
    axes[2].set_xlabel("x [nm]"); axes[2].set_ylabel("y [nm]")
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    if suptitle:
        fig.suptitle(suptitle)
    fig.tight_layout()
    return fig


def show_fd_vs_fft(
    H0: torch.Tensor,
    H_fd: torch.Tensor,
    H_fft: torch.Tensor,
    dx_nm: float,
    t_end_s: float,
    Hmax: float | None = None,
    suptitle: str = "",
):
    """2-row figure.

    Top row : H_0 | FD result | FFT result | |FD - FFT|
    Bottom : a y=0 row cut comparing H_0, FD, FFT line traces.
    """
    fig = plt.figure(figsize=(16, 7))
    gs = fig.add_gridspec(2, 4, height_ratios=[2, 1])
    n = H0.shape[-1]
    extent_nm = n * dx_nm
    real_ext = (-extent_nm / 2, extent_nm / 2, -extent_nm / 2, extent_nm / 2)
    vmax = float(Hmax) if Hmax is not None else max(
        float(H0.max().item()), float(H_fd.max().item()),
        float(H_fft.max().item()), 1e-12)
    diff = (H_fd - H_fft)
    diff_max = float(diff.abs().max().item()) or 1e-12

    panels = [
        ("H_0", H0, "magma", 0.0, vmax),
        (f"FD  t={t_end_s:g} s", H_fd, "magma", 0.0, vmax),
        (f"FFT t={t_end_s:g} s", H_fft, "magma", 0.0, vmax),
        (f"|FD - FFT|  max={diff_max:.3e}", diff.abs(), "inferno", 0.0,
         max(diff_max, 1e-12)),
    ]
    for k, (name, img, cmap, lo, hi) in enumerate(panels):
        ax = fig.add_subplot(gs[0, k])
        im = ax.imshow(_to_numpy(img), cmap=cmap, extent=real_ext,
                       origin="lower", vmin=lo, vmax=hi)
        ax.set_title(name); ax.set_xlabel("x [nm]"); ax.set_ylabel("y [nm]")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = fig.add_subplot(gs[1, :])
    x_axis = (np.arange(n) - n / 2) * dx_nm
    row = n // 2
    ax.plot(x_axis, _to_numpy(H0[row]),    label="H_0",   color="black")
    ax.plot(x_axis, _to_numpy(H_fd[row]),  label="FD",    color="C3", linestyle="--")
    ax.plot(x_axis, _to_numpy(H_fft[row]), label="FFT",   color="C0", linestyle="-.")
    ax.set_title(f"y=0 row cut")
    ax.set_xlabel("x [nm]"); ax.set_ylabel("H [mol/dm^3]")
    ax.legend(); ax.grid(alpha=0.3)

    if suptitle:
        fig.suptitle(suptitle)
    fig.tight_layout()
    return fig


def show_diffusion_dh_sweep(
    H0: torch.Tensor,
    H_results: list[torch.Tensor],
    DH_values: list[float],
    dx_nm: float,
    t_end_s: float,
    Hmax: float | None = None,
    suptitle: str = "",
):
    """One row: H_0 followed by H(t) for each DH value."""
    n_cols = 1 + len(H_results)
    fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4))
    n = H0.shape[-1]
    extent_nm = n * dx_nm
    real_ext = (-extent_nm / 2, extent_nm / 2, -extent_nm / 2, extent_nm / 2)
    vmax = float(Hmax) if Hmax is not None else max(
        float(H0.max().item()),
        max(float(h.max().item()) for h in H_results),
        1e-12,
    )

    im = axes[0].imshow(_to_numpy(H0), cmap="magma", extent=real_ext,
                       origin="lower", vmin=0.0, vmax=vmax)
    axes[0].set_title(f"H_0   peak={float(H0.max().item()):.4f}")
    axes[0].set_xlabel("x [nm]"); axes[0].set_ylabel("y [nm]")
    fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)

    for ax, H, DH in zip(axes[1:], H_results, DH_values):
        L_nm = math.sqrt(2.0 * DH * t_end_s)
        im = ax.imshow(_to_numpy(H), cmap="magma", extent=real_ext,
                       origin="lower", vmin=0.0, vmax=vmax)
        ax.set_title(
            f"DH={DH:g} nm^2/s   L={L_nm:.2f} nm   peak={float(H.max().item()):.4f}"
        )
        ax.set_xlabel("x [nm]"); ax.set_ylabel("y [nm]")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

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
