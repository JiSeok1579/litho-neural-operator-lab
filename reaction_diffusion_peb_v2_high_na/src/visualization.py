"""Plotting helpers for v2 baseline."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_field(
    field: np.ndarray,
    x_nm: np.ndarray,
    y_nm: np.ndarray,
    title: str,
    out_path: str,
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    cbar_label: str = "",
) -> None:
    fig, ax = plt.subplots(figsize=(5.0, 4.5))
    im = ax.imshow(
        field,
        origin="lower",
        extent=[x_nm[0], x_nm[-1], y_nm[0], y_nm[-1]],
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        aspect="equal",
    )
    ax.set_xlabel("x [nm]")
    ax.set_ylabel("y [nm]")
    ax.set_title(title)
    cb = fig.colorbar(im, ax=ax)
    cb.set_label(cbar_label)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_contour_overlay(
    P_field: np.ndarray,
    x_nm: np.ndarray,
    y_nm: np.ndarray,
    threshold: float,
    initial_edges: tuple[np.ndarray, np.ndarray, np.ndarray] | None,
    final_edges: tuple[np.ndarray, np.ndarray, np.ndarray] | None,
    title: str,
    out_path: str,
) -> None:
    """Overlay initial/final edge curves on the P field.

    *_edges = (line_centers, left[n_lines, ny], right[n_lines, ny])
    """
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(
        P_field,
        origin="lower",
        extent=[x_nm[0], x_nm[-1], y_nm[0], y_nm[-1]],
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
        aspect="equal",
    )
    ax.contour(
        x_nm,
        y_nm,
        P_field,
        levels=[threshold],
        colors="red",
        linewidths=1.2,
    )
    if initial_edges is not None:
        _, left0, right0 = initial_edges
        for li in range(left0.shape[0]):
            ax.plot(left0[li], y_nm, "w--", lw=0.7)
            ax.plot(right0[li], y_nm, "w--", lw=0.7)
    if final_edges is not None:
        _, leftf, rightf = final_edges
        for li in range(leftf.shape[0]):
            ax.plot(leftf[li], y_nm, "k-", lw=0.6, alpha=0.7)
            ax.plot(rightf[li], y_nm, "k-", lw=0.6, alpha=0.7)
    ax.set_xlabel("x [nm]")
    ax.set_ylabel("y [nm]")
    ax.set_title(title)
    cb = fig.colorbar(im, ax=ax)
    cb.set_label("P")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
