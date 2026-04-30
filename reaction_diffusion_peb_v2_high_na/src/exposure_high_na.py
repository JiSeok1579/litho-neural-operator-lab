"""Dill-style acid generation for High-NA EUV PEB v2."""
from __future__ import annotations

import numpy as np


def dill_acid_generation(
    intensity: np.ndarray,
    dose_norm: float,
    eta: float,
    Hmax: float,
) -> np.ndarray:
    """H0 = Hmax * (1 - exp(-eta * dose_norm * I))."""
    return Hmax * (1.0 - np.exp(-eta * dose_norm * intensity))


def normalize_dose(dose_mJ_cm2: float, reference_dose_mJ_cm2: float) -> float:
    if reference_dose_mJ_cm2 <= 0.0:
        raise ValueError("reference_dose_mJ_cm2 must be positive")
    return float(dose_mJ_cm2) / float(reference_dose_mJ_cm2)
