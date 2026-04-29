"""Plotting helpers shared by the PEB submodule's experiments.

Self-contained on purpose — no imports from the main repo's
``src/common/visualization.py``.
"""

from __future__ import annotations

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


def save_figure(fig, path: str | Path, dpi: int = 150) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path
