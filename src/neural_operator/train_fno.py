"""Training loop for the Phase-8 FNO 2D correction surrogate.

Loss:

    L = MSE(delta_T_pred, delta_T_true)
        + weight_aerial * MSE(|E_pred|^2, |E_true|^2)

where ``E_pred = ifft2c(T_thin + delta_T_pred)`` and likewise for the
truth. ``weight_aerial = 0`` (default) keeps the loss a plain spectrum
regression; setting it positive adds a "physics-aware" term that puts
extra pressure where spectrum errors actually move the imaging output.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.common.fft_utils import ifft2c
from src.neural_operator.fno2d import FNO2d


def _delta_to_complex(delta_T: torch.Tensor) -> torch.Tensor:
    """``(B, 2, H, W) -> (B, H, W) complex``."""
    return torch.complex(delta_T[:, 0], delta_T[:, 1])


def _t_thin_complex(x: torch.Tensor) -> torch.Tensor:
    """Recover the complex T_thin from the input channel stack.

    Channel layout (from :class:`CorrectionDataset`): 0=mask, 1=T_thin_real,
    2=T_thin_imag, 3..=theta-broadcast.
    """
    return torch.complex(x[:, 1], x[:, 2])


def spectrum_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean squared error over (real, imag) channels combined."""
    return ((pred - target) ** 2).mean()


def complex_relative_error(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """``||pred - target|| / ||target||`` summed over the complex magnitude."""
    diff = pred - target
    num = (diff[:, 0] ** 2 + diff[:, 1] ** 2).mean().sqrt()
    den = (target[:, 0] ** 2 + target[:, 1] ** 2).mean().sqrt() + 1e-12
    return num / den


def aerial_intensity_mse(
    delta_pred: torch.Tensor,
    delta_true: torch.Tensor,
    T_thin: torch.Tensor,
) -> torch.Tensor:
    """Compare ``|ifft2c(T_thin + delta_pred)|^2`` to the truth-version."""
    pred_T_3d = T_thin + _delta_to_complex(delta_pred)
    true_T_3d = T_thin + _delta_to_complex(delta_true)
    pred_field = ifft2c(pred_T_3d)
    true_field = ifft2c(true_T_3d)
    pred_I = pred_field.real ** 2 + pred_field.imag ** 2
    true_I = true_field.real ** 2 + true_field.imag ** 2
    return ((pred_I - true_I) ** 2).mean()


@dataclass
class FNOTrainResult:
    train_losses: list[float]
    test_losses: list[float]
    test_complex_rel_err: list[float]
    test_aerial_mse: list[float]
    epochs: list[int]
    train_time_sec: float
    final_lr: float


def train_fno_correction(
    model: FNO2d,
    train_loader: DataLoader,
    test_loader: DataLoader,
    n_epochs: int = 50,
    lr: float = 1.0e-3,
    weight_decay: float = 0.0,
    weight_aerial: float = 0.0,
    lr_decay_step: int | None = None,
    lr_decay_gamma: float = 0.5,
    device: torch.device | None = None,
    log_every: int = 1,
) -> FNOTrainResult:
    if device is None:
        device = next(model.parameters()).device
    if weight_decay > 0:
        optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optim = torch.optim.Adam(model.parameters(), lr=lr)
    if lr_decay_step is not None:
        scheduler = torch.optim.lr_scheduler.StepLR(
            optim, step_size=int(lr_decay_step), gamma=float(lr_decay_gamma)
        )
    else:
        scheduler = None

    train_losses: list[float] = []
    test_losses: list[float] = []
    test_complex_rel_err: list[float] = []
    test_aerial_mse: list[float] = []
    epochs_logged: list[int] = []

    t0 = time.time()
    for epoch in range(n_epochs):
        model.train()
        running = 0.0
        n_batches = 0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            pred = model(x)
            loss = spectrum_mse(pred, y)
            if weight_aerial > 0:
                T_thin = _t_thin_complex(x)
                loss = loss + weight_aerial * aerial_intensity_mse(pred, y, T_thin)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            running += float(loss.item())
            n_batches += 1
        if scheduler is not None:
            scheduler.step()
        train_avg = running / max(1, n_batches)

        if (epoch % log_every == 0) or (epoch == n_epochs - 1):
            te_loss, te_rel, te_aerial = evaluate_fno(
                model, test_loader, device=device, weight_aerial=weight_aerial,
            )
            train_losses.append(train_avg)
            test_losses.append(te_loss)
            test_complex_rel_err.append(te_rel)
            test_aerial_mse.append(te_aerial)
            epochs_logged.append(epoch)

    if device.type == "cuda":
        torch.cuda.synchronize()
    train_time = time.time() - t0

    return FNOTrainResult(
        train_losses=train_losses,
        test_losses=test_losses,
        test_complex_rel_err=test_complex_rel_err,
        test_aerial_mse=test_aerial_mse,
        epochs=epochs_logged,
        train_time_sec=train_time,
        final_lr=float(optim.param_groups[0]["lr"]),
    )


@torch.no_grad()
def evaluate_fno(
    model: FNO2d,
    test_loader: DataLoader,
    device: torch.device | None = None,
    weight_aerial: float = 0.0,
) -> tuple[float, float, float]:
    """Return ``(spectrum_mse, complex_relative_error, aerial_mse)`` on the
    test loader."""
    if device is None:
        device = next(model.parameters()).device
    model.eval()
    sum_mse = 0.0
    sum_rel = 0.0
    sum_aerial = 0.0
    n = 0
    for x, y in test_loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        pred = model(x)
        bsize = x.shape[0]
        sum_mse += float(spectrum_mse(pred, y).item()) * bsize
        sum_rel += float(complex_relative_error(pred, y).item()) * bsize
        T_thin = _t_thin_complex(x)
        sum_aerial += float(aerial_intensity_mse(pred, y, T_thin).item()) * bsize
        n += bsize
    return sum_mse / max(1, n), sum_rel / max(1, n), sum_aerial / max(1, n)
