"""Resist threshold and CD-like measurement helpers."""

from __future__ import annotations

import torch


def soft_threshold(A: torch.Tensor, A_th: float, beta: float = 20.0) -> torch.Tensor:
    """Differentiable threshold: ``R = sigmoid(beta * (A - A_th))``.

    Large ``beta`` approaches a step function. ``beta = 20`` is a good
    default for normalized acid concentrations in [0, 1] — the transition
    band is roughly ``4 / beta = 0.2`` units wide.
    """
    if beta <= 0:
        raise ValueError("beta must be positive")
    return torch.sigmoid(beta * (A - A_th))


def hard_threshold(A: torch.Tensor, A_th: float) -> torch.Tensor:
    """Non-differentiable {0, 1} threshold (analysis only)."""
    return (A > A_th).to(A.dtype)


def measure_cd_horizontal(
    field: torch.Tensor,
    threshold: float,
    dx: float,
) -> float:
    """Approximate critical dimension along the central row.

    Counts the longest contiguous run of pixels above ``threshold`` along
    the middle row of the 2D field and returns its physical length
    ``run_length * dx``. Returns 0 if no pixel exceeds the threshold.
    """
    if field.ndim != 2:
        raise ValueError("expected 2D field")
    if dx <= 0:
        raise ValueError("dx must be positive")
    n = field.shape[-1]
    row = field[n // 2]
    above = (row > threshold).to(torch.uint8).cpu().numpy()
    max_run = 0
    cur_run = 0
    for v in above:
        if v == 1:
            cur_run += 1
            if cur_run > max_run:
                max_run = cur_run
        else:
            cur_run = 0
    return max_run * dx


def thresholded_area(field: torch.Tensor, threshold: float) -> int:
    return int((field > threshold).sum().item())
