"""Shared helpers for calibration sweeps."""
from __future__ import annotations

import copy

import numpy as np


def classify(r: dict) -> str:
    """Stage-5 classifier reused across calibration sweeps."""
    if not (r["cond_metrics"] and np.isfinite(r["P_max"]) and np.isfinite(r["P_min"])
            and np.isfinite(r["LER_after_PEB_P_nm"])):
        return "unstable"
    if r["H_min"] < -1e-6 or r["P_min"] < -1e-6 or r["P_max"] > 1.0 + 1e-6:
        return "unstable"
    if (r["P_space_center_mean"] >= 0.50
            or r["area_frac"] >= 0.90
            or (np.isfinite(r["CD_pitch_frac"]) and r["CD_pitch_frac"] >= 0.85)):
        return "merged"
    if r["P_line_center_mean"] < 0.65:
        return "under_exposed"
    if r["contrast"] <= 0.15:
        return "low_contrast"
    if r["P_line_margin"] >= 0.05:
        return "robust_valid"
    return "valid"


def apply_xy_overrides(cfg: dict, overrides: dict) -> dict:
    """Return a deep-copied cfg with the given x-y overrides applied.

    Supported keys: dose_mJ_cm2, electron_blur_sigma_nm, DH_nm2_s, time_s,
    Q0_mol_dm3, kq_s_inv, kdep_s_inv, Hmax_mol_dm3, pitch_nm.
    Pitch overrides also resize domain_x_nm to pitch * 5 (FFT-seam-safe).
    """
    out = copy.deepcopy(cfg)
    for k, v in overrides.items():
        if k == "dose_mJ_cm2":
            out["exposure"]["dose_mJ_cm2"] = float(v)
            out["exposure"]["dose_norm"] = float(v) / float(out["exposure"]["reference_dose_mJ_cm2"])
        elif k == "electron_blur_sigma_nm":
            out["exposure"]["electron_blur_sigma_nm"] = float(v)
            out["exposure"]["electron_blur_enabled"] = float(v) > 0.0
        elif k in {"DH_nm2_s", "time_s", "kdep_s_inv"}:
            out["peb"][k] = float(v)
        elif k == "Hmax_mol_dm3":
            out["exposure"]["Hmax_mol_dm3"] = float(v)
        elif k == "Q0_mol_dm3":
            out["quencher"]["Q0_mol_dm3"] = float(v)
            out["quencher"]["enabled"] = float(v) > 0.0
        elif k == "kq_s_inv":
            out["quencher"]["kq_s_inv"] = float(v)
        elif k == "pitch_nm":
            out["geometry"]["pitch_nm"] = float(v)
            out["geometry"]["half_pitch_nm"] = 0.5 * float(v)
            out["geometry"]["domain_x_nm"] = float(v) * 5.0
        else:
            raise KeyError(f"unsupported override key: {k}")
    return out


def metric_value(r: dict, key: str):
    return r.get(key, float("nan"))
