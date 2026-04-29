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


def show_pinn_training(history, suptitle: str = ""):
    """Loss-history figure for ``train_pinn_diffusion``.

    ``history`` may be a TrainingHistory dataclass or a list of dicts
    with keys ``iter``, ``loss_total``, ``loss_pde``, ``loss_ic``.
    """
    if hasattr(history, "iters"):
        its = history.iters
        loss_total = history.loss_total
        loss_pde = history.loss_pde
        loss_ic = history.loss_ic
    else:
        its = [h["iter"] for h in history]
        loss_total = [h["loss_total"] for h in history]
        loss_pde = [h["loss_pde"] for h in history]
        loss_ic = [h["loss_ic"] for h in history]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(its, [max(v, 1e-20) for v in loss_total], label="total", color="black")
    ax.plot(its, [max(v, 1e-20) for v in loss_pde], label="PDE residual", color="C0")
    ax.plot(its, [max(v, 1e-20) for v in loss_ic], label="IC", color="C3", linestyle="--")
    ax.set_yscale("log")
    ax.set_title("PINN training")
    ax.set_xlabel("iteration"); ax.set_ylabel("loss (log)")
    ax.legend(); ax.grid(alpha=0.3)
    if suptitle:
        fig.suptitle(suptitle)
    fig.tight_layout()
    return fig


def show_fd_fft_pinn(
    H0: torch.Tensor,
    H_fd: torch.Tensor,
    H_fft: torch.Tensor,
    H_pinn: torch.Tensor,
    dx_nm: float,
    t_end_s: float,
    Hmax: float | None = None,
    suptitle: str = "",
):
    """3-row figure: solutions, errors vs FFT, y=0 row cut.

    Row 0 panels : H_0 | FD | FFT | PINN
    Row 1 panels : (reference) | |FD - FFT| | (reference) | |PINN - FFT|
    Row 2 (full) : y=0 row cut comparing all four traces.
    """
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 4, height_ratios=[2, 2, 1])
    n = H0.shape[-1]
    extent_nm = n * dx_nm
    real_ext = (-extent_nm / 2, extent_nm / 2, -extent_nm / 2, extent_nm / 2)

    sol_max = max(
        float(H0.max().item()), float(H_fd.max().item()),
        float(H_fft.max().item()), float(H_pinn.max().item()), 1e-12,
    )
    if Hmax is not None:
        sol_max = max(sol_max, Hmax)

    err_fd = (H_fd - H_fft).abs()
    err_pinn = (H_pinn - H_fft).abs()
    err_max = max(float(err_fd.max().item()), float(err_pinn.max().item()), 1e-12)

    sols = [
        ("H_0", H0, "magma"),
        (f"FD  t={t_end_s:g} s", H_fd, "magma"),
        (f"FFT t={t_end_s:g} s", H_fft, "magma"),
        (f"PINN t={t_end_s:g} s", H_pinn, "magma"),
    ]
    for k, (name, img, cmap) in enumerate(sols):
        ax = fig.add_subplot(gs[0, k])
        im = ax.imshow(_to_numpy(img), cmap=cmap, extent=real_ext,
                       origin="lower", vmin=0.0, vmax=sol_max)
        ax.set_title(f"{name}   peak={float(img.max().item()):.4f}")
        ax.set_xlabel("x [nm]"); ax.set_ylabel("y [nm]")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    err_panels = [
        ("(reference)", torch.zeros_like(H0), "Greys"),
        (f"|FD - FFT|  max={float(err_fd.max().item()):.3e}", err_fd, "inferno"),
        ("(reference)", torch.zeros_like(H0), "Greys"),
        (f"|PINN - FFT|  max={float(err_pinn.max().item()):.3e}", err_pinn, "inferno"),
    ]
    for k, (name, img, cmap) in enumerate(err_panels):
        ax = fig.add_subplot(gs[1, k])
        im = ax.imshow(_to_numpy(img), cmap=cmap, extent=real_ext,
                       origin="lower", vmin=0.0, vmax=err_max)
        ax.set_title(name)
        ax.set_xlabel("x [nm]"); ax.set_ylabel("y [nm]")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = fig.add_subplot(gs[2, :])
    x_axis = (np.arange(n) - n / 2) * dx_nm
    row = n // 2
    ax.plot(x_axis, _to_numpy(H0[row]),    label="H_0",  color="black")
    ax.plot(x_axis, _to_numpy(H_fd[row]),  label="FD",   color="C3", linestyle="--")
    ax.plot(x_axis, _to_numpy(H_fft[row]), label="FFT",  color="C0", linestyle="-.")
    ax.plot(x_axis, _to_numpy(H_pinn[row]), label="PINN", color="C2")
    ax.set_title("y=0 row cut")
    ax.set_xlabel("x [nm]"); ax.set_ylabel("H [mol/dm^3]")
    ax.legend(); ax.grid(alpha=0.3)

    if suptitle:
        fig.suptitle(suptitle)
    fig.tight_layout()
    return fig


def show_deprotection_chain(
    H0: torch.Tensor,
    H_t: torch.Tensor,
    P_t: torch.Tensor,
    dx_nm: float,
    t_end_s: float,
    P_threshold: float = 0.5,
    Hmax: float | None = None,
    suptitle: str = "",
):
    """4-panel figure: H_0 | H(t) | P(t) | thresholded P > P_threshold."""
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    n = H0.shape[-1]
    extent_nm = n * dx_nm
    real_ext = (-extent_nm / 2, extent_nm / 2, -extent_nm / 2, extent_nm / 2)
    H_max_val = float(Hmax) if Hmax is not None else max(
        float(H0.max().item()), float(H_t.max().item()), 1e-12)

    panels = [
        ("H_0", H0, "magma", 0.0, H_max_val),
        (f"H(t={t_end_s:g} s)", H_t, "magma", 0.0, H_max_val),
        (f"P(t={t_end_s:g} s)", P_t, "Greens", 0.0, 1.0),
    ]
    for k, (name, img, cmap, lo, hi) in enumerate(panels):
        im = axes[k].imshow(_to_numpy(img), cmap=cmap, extent=real_ext,
                            origin="lower", vmin=lo, vmax=hi)
        axes[k].set_title(f"{name}   peak={float(img.max().item()):.4f}")
        axes[k].set_xlabel("x [nm]"); axes[k].set_ylabel("y [nm]")
        fig.colorbar(im, ax=axes[k], fraction=0.046, pad=0.04)

    R = (P_t > P_threshold).to(P_t.dtype)
    n_pixels = int(R.sum().item())
    im = axes[3].imshow(_to_numpy(R), cmap="Greens", extent=real_ext,
                        origin="lower", vmin=0.0, vmax=1.0)
    axes[3].set_title(f"P > {P_threshold} (area = {n_pixels} px)")
    axes[3].set_xlabel("x [nm]"); axes[3].set_ylabel("y [nm]")
    fig.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)

    if suptitle:
        fig.suptitle(suptitle)
    fig.tight_layout()
    return fig


def show_deprotection_kdep_sweep(
    H0: torch.Tensor,
    P_results: list[torch.Tensor],
    kdep_values: list[float],
    dx_nm: float,
    t_end_s: float,
    P_threshold: float = 0.5,
    suptitle: str = "",
):
    """One row: H_0 followed by ``P(t_end)`` for each kdep value.

    Threshold contour is overlaid on each P panel.
    """
    n_cols = 1 + len(P_results)
    fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4))
    n = H0.shape[-1]
    extent_nm = n * dx_nm
    real_ext = (-extent_nm / 2, extent_nm / 2, -extent_nm / 2, extent_nm / 2)
    Hmax_val = max(float(H0.max().item()), 1e-12)

    im = axes[0].imshow(_to_numpy(H0), cmap="magma", extent=real_ext,
                       origin="lower", vmin=0.0, vmax=Hmax_val)
    axes[0].set_title(f"H_0   peak={Hmax_val:.4f}")
    axes[0].set_xlabel("x [nm]"); axes[0].set_ylabel("y [nm]")
    fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)

    xs = np.linspace(-extent_nm / 2, extent_nm / 2, n)
    for ax, P, kdep in zip(axes[1:], P_results, kdep_values):
        im = ax.imshow(_to_numpy(P), cmap="Greens", extent=real_ext,
                       origin="lower", vmin=0.0, vmax=1.0)
        n_pix = int((P > P_threshold).sum().item())
        ax.set_title(f"P  kdep={kdep:g} 1/s  area={n_pix} px")
        ax.contour(xs, xs, _to_numpy(P), levels=[P_threshold],
                   colors="black", linewidths=1.0)
        ax.set_xlabel("x [nm]"); ax.set_ylabel("y [nm]")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    if suptitle:
        fig.suptitle(suptitle)
    fig.tight_layout()
    return fig


def show_deprotection_fd_vs_pinn(
    H0: torch.Tensor,
    H_fd: torch.Tensor, P_fd: torch.Tensor,
    H_pinn: torch.Tensor, P_pinn: torch.Tensor,
    dx_nm: float,
    t_end_s: float,
    P_threshold: float = 0.5,
    suptitle: str = "",
):
    """2x4 figure comparing FD (truth) and PINN (H, P) outputs.

    Row 0 : H_0 | H_FD | H_PINN | |H_PINN - H_FD|
    Row 1 : (P=0)|  P_FD | P_PINN | |P_PINN - P_FD|
    """
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    n = H0.shape[-1]
    extent_nm = n * dx_nm
    real_ext = (-extent_nm / 2, extent_nm / 2, -extent_nm / 2, extent_nm / 2)
    H_max_val = max(float(H0.max().item()),
                    float(H_fd.max().item()),
                    float(H_pinn.max().item()), 1e-12)
    err_H = (H_pinn - H_fd).abs()
    err_P = (P_pinn - P_fd).abs()
    H_err_max = max(float(err_H.max().item()), 1e-12)
    P_err_max = max(float(err_P.max().item()), 1e-12)

    # Row 0: H field
    h_panels = [
        ("H_0", H0, "magma", 0.0, H_max_val),
        (f"H FD t={t_end_s:g} s", H_fd, "magma", 0.0, H_max_val),
        (f"H PINN t={t_end_s:g} s", H_pinn, "magma", 0.0, H_max_val),
        (f"|H PINN - H FD|  max={H_err_max:.3e}", err_H, "inferno", 0.0, H_err_max),
    ]
    for k, (name, img, cmap, lo, hi) in enumerate(h_panels):
        im = axes[0, k].imshow(_to_numpy(img), cmap=cmap, extent=real_ext,
                               origin="lower", vmin=lo, vmax=hi)
        axes[0, k].set_title(name)
        axes[0, k].set_xlabel("x [nm]"); axes[0, k].set_ylabel("y [nm]")
        fig.colorbar(im, ax=axes[0, k], fraction=0.046, pad=0.04)

    # Row 1: P field
    p_panels = [
        ("P_0 = 0", torch.zeros_like(H0), "Greens", 0.0, 1.0),
        (f"P FD t={t_end_s:g} s", P_fd, "Greens", 0.0, 1.0),
        (f"P PINN t={t_end_s:g} s", P_pinn, "Greens", 0.0, 1.0),
        (f"|P PINN - P FD|  max={P_err_max:.3e}", err_P, "inferno", 0.0, P_err_max),
    ]
    for k, (name, img, cmap, lo, hi) in enumerate(p_panels):
        im = axes[1, k].imshow(_to_numpy(img), cmap=cmap, extent=real_ext,
                               origin="lower", vmin=lo, vmax=hi)
        axes[1, k].set_title(name)
        axes[1, k].set_xlabel("x [nm]"); axes[1, k].set_ylabel("y [nm]")
        fig.colorbar(im, ax=axes[1, k], fraction=0.046, pad=0.04)

    if suptitle:
        fig.suptitle(suptitle)
    fig.tight_layout()
    return fig


def show_quencher_chain(
    H0: torch.Tensor,
    H_t: torch.Tensor,
    Q_t: torch.Tensor,
    P_t: torch.Tensor,
    Q0_mol_dm3: float,
    dx_nm: float,
    t_end_s: float,
    P_threshold: float = 0.5,
    Hmax: float | None = None,
    suptitle: str = "",
):
    """5-panel figure for one quencher run: H_0 | H(t) | Q(t) | P(t) | P > thr."""
    fig, axes = plt.subplots(1, 5, figsize=(22, 4))
    n = H0.shape[-1]
    extent_nm = n * dx_nm
    real_ext = (-extent_nm / 2, extent_nm / 2, -extent_nm / 2, extent_nm / 2)
    H_max_val = float(Hmax) if Hmax is not None else max(
        float(H0.max().item()), float(H_t.max().item()), 1e-12)
    Q_max_val = max(float(Q0_mol_dm3), float(Q_t.max().item()), 1e-12)

    panels = [
        ("H_0", H0, "magma", 0.0, H_max_val),
        (f"H(t={t_end_s:g} s)", H_t, "magma", 0.0, H_max_val),
        (f"Q(t={t_end_s:g} s)", Q_t, "Blues", 0.0, Q_max_val),
        (f"P(t={t_end_s:g} s)", P_t, "Greens", 0.0, 1.0),
    ]
    for k, (name, img, cmap, lo, hi) in enumerate(panels):
        im = axes[k].imshow(_to_numpy(img), cmap=cmap, extent=real_ext,
                            origin="lower", vmin=lo, vmax=hi)
        axes[k].set_title(f"{name}   peak={float(img.max().item()):.4f}")
        axes[k].set_xlabel("x [nm]"); axes[k].set_ylabel("y [nm]")
        fig.colorbar(im, ax=axes[k], fraction=0.046, pad=0.04)

    R = (P_t > P_threshold).to(P_t.dtype)
    n_pixels = int(R.sum().item())
    im = axes[4].imshow(_to_numpy(R), cmap="Greens", extent=real_ext,
                        origin="lower", vmin=0.0, vmax=1.0)
    axes[4].set_title(f"P > {P_threshold} (area = {n_pixels} px)")
    axes[4].set_xlabel("x [nm]"); axes[4].set_ylabel("y [nm]")
    fig.colorbar(im, ax=axes[4], fraction=0.046, pad=0.04)

    if suptitle:
        fig.suptitle(suptitle)
    fig.tight_layout()
    return fig


def show_quencher_kq_sweep(
    H0: torch.Tensor,
    P_results: list[torch.Tensor],
    Q_results: list[torch.Tensor],
    kq_values: list[float],
    Q0_mol_dm3: float,
    dx_nm: float,
    t_end_s: float,
    P_threshold: float = 0.5,
    suptitle: str = "",
):
    """2-row figure: row 0 = P(t) per kq with threshold contour;
    row 1 = Q(t) per kq. Column 0 holds H_0 (top) and Q_0 reference (bottom)."""
    n_cols = 1 + len(P_results)
    fig, axes = plt.subplots(2, n_cols, figsize=(4 * n_cols, 8))
    n = H0.shape[-1]
    extent_nm = n * dx_nm
    real_ext = (-extent_nm / 2, extent_nm / 2, -extent_nm / 2, extent_nm / 2)
    Hmax_val = max(float(H0.max().item()), 1e-12)
    Q_max_val = max(float(Q0_mol_dm3), 1e-12)

    im = axes[0, 0].imshow(_to_numpy(H0), cmap="magma", extent=real_ext,
                           origin="lower", vmin=0.0, vmax=Hmax_val)
    axes[0, 0].set_title(f"H_0   peak={Hmax_val:.4f}")
    axes[0, 0].set_xlabel("x [nm]"); axes[0, 0].set_ylabel("y [nm]")
    fig.colorbar(im, ax=axes[0, 0], fraction=0.046, pad=0.04)

    Q0_ref = torch.full_like(H0, float(Q0_mol_dm3))
    im = axes[1, 0].imshow(_to_numpy(Q0_ref), cmap="Blues", extent=real_ext,
                           origin="lower", vmin=0.0, vmax=Q_max_val)
    axes[1, 0].set_title(f"Q_0 = {Q0_mol_dm3:g} (uniform)")
    axes[1, 0].set_xlabel("x [nm]"); axes[1, 0].set_ylabel("y [nm]")
    fig.colorbar(im, ax=axes[1, 0], fraction=0.046, pad=0.04)

    xs = np.linspace(-extent_nm / 2, extent_nm / 2, n)
    for col, (P, Q, kq) in enumerate(zip(P_results, Q_results, kq_values), start=1):
        im = axes[0, col].imshow(_to_numpy(P), cmap="Greens", extent=real_ext,
                                 origin="lower", vmin=0.0, vmax=1.0)
        n_pix = int((P > P_threshold).sum().item())
        axes[0, col].set_title(f"P  kq={kq:g}  area={n_pix} px")
        axes[0, col].contour(xs, xs, _to_numpy(P), levels=[P_threshold],
                             colors="black", linewidths=1.0)
        axes[0, col].set_xlabel("x [nm]"); axes[0, col].set_ylabel("y [nm]")
        fig.colorbar(im, ax=axes[0, col], fraction=0.046, pad=0.04)

        im = axes[1, col].imshow(_to_numpy(Q), cmap="Blues", extent=real_ext,
                                 origin="lower", vmin=0.0, vmax=Q_max_val)
        axes[1, col].set_title(
            f"Q  kq={kq:g}  min={float(Q.min().item()):.4f}"
        )
        axes[1, col].set_xlabel("x [nm]"); axes[1, col].set_ylabel("y [nm]")
        fig.colorbar(im, ax=axes[1, col], fraction=0.046, pad=0.04)

    if suptitle:
        fig.suptitle(suptitle)
    fig.tight_layout()
    return fig


def show_quencher_mass_budget(
    histories: list[list],
    kq_values: list[float],
    suptitle: str = "",
):
    """Plot the H/Q mass-budget relative errors over time.

    ``histories[i]`` is a list of :class:`QuencherBudgetSnapshot` for kq[i].
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for hist, kq in zip(histories, kq_values):
        ts = [s.t_s for s in hist]
        h_err = [max(s.H_budget_relative_error, 1e-20) for s in hist]
        q_err = [max(s.Q_budget_relative_error, 1e-20) for s in hist]
        axes[0].plot(ts, h_err, label=f"kq={kq:g}", marker="o", markersize=3)
        axes[1].plot(ts, q_err, label=f"kq={kq:g}", marker="o", markersize=3)
    for ax, name in zip(axes, ("H budget rel-err", "Q budget rel-err")):
        ax.set_yscale("log")
        ax.set_title(name)
        ax.set_xlabel("t [s]")
        ax.set_ylabel("|budget - initial| / initial")
        ax.grid(alpha=0.3, which="both")
        ax.legend()
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
