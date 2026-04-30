"""Cheap analytical pre-filters that score each candidate without running FD.

Score is built from heuristics that match the failure modes the v2 sweeps
already documented:
    diffusion length vs half-pitch (merged when too large),
    H budget vs deprotection (under_exposed when too small),
    duty cycle (CD/pitch close to 1.0 → already at merged threshold),
    sigma vs pitch fundamental (e-blur kills the line/space contrast).

Higher score = more likely to land in a useful (robust_valid / margin_risk
/ roughness_degraded) cell, lower score = more likely to be merged or
under_exposed.

The score is heuristic only — its purpose is to rank candidates so the FD
budget skips obvious failures, not to predict the final label exactly.
"""
from __future__ import annotations

import math

import numpy as np


def diffusion_length_nm(DH_nm2_s: float, time_s: float) -> float:
    return float(math.sqrt(2.0 * DH_nm2_s * time_s))


def fundamental_blur_attenuation(pitch_nm: float, sigma_nm: float) -> float:
    """exp(-(2 pi / pitch)^2 * sigma^2 / 2) — fundamental-mode attenuation."""
    if sigma_nm <= 0.0:
        return 1.0
    k = 2.0 * math.pi / pitch_nm
    return float(math.exp(-0.5 * (k * sigma_nm) ** 2))


def estimate_h_peak(Hmax: float, eta: float, dose_norm: float) -> float:
    """Dill saturation at the line-center pixel (where I=1 before blur)."""
    return float(Hmax * (1.0 - math.exp(-eta * dose_norm)))


def score_candidate(c: dict) -> dict:
    """Return a dict with intermediate metrics + a single combined score in [0, 1].

    Heuristics:
      diff_term       in [0, 1] — favours diffusion length comparable to half-pitch
      h_term          in [0, 1] — favours H_peak * kdep * t in a reasonable range
      duty_term       in [0, 1] — penalises CD/pitch above 0.78 (the v2 cliff)
      blur_term       in [0, 1] — favours sigma so that fundamental survives > 0.3
      space_diff_term in [0, 1] — penalises diffusion length larger than space width
    """
    pitch = float(c["pitch_nm"])
    line_cd = float(c["line_cd_nm"])
    space = pitch - line_cd
    half_pitch = pitch / 2.0

    DH = float(c["DH_nm2_s"])
    t = float(c["time_s"])
    Ld = diffusion_length_nm(DH, t)

    # 1) diffusion length vs half-pitch: best around 0.3–0.6 of half-pitch.
    r = Ld / half_pitch
    if r <= 0.0:
        diff_term = 0.0
    else:
        # tent-shaped score peaking at r=0.45.
        diff_term = max(0.0, 1.0 - abs(r - 0.45) / 0.6)

    # 2) H budget: H_peak × kdep × t needs to be enough to deprotect.
    Hmax = float(c["Hmax_mol_dm3"])
    eta = float(c["eta"])
    dose_norm = float(c["dose_norm"])
    kdep = float(c["kdep_s_inv"])
    H_peak = estimate_h_peak(Hmax, eta, dose_norm)
    deprotect = H_peak * kdep * t  # dimensionless; > 0.5 typically gives P_line > 0.65
    h_term = float(np.clip((deprotect - 0.4) / 1.5, 0.0, 1.0))

    # 3) duty cycle: CD/pitch above 0.78 leaves no inter-line space (Stage 5
    # found pitch=16 closes for that reason).
    duty = line_cd / pitch
    duty_term = float(np.clip((0.78 - duty) / 0.32, 0.0, 1.0))

    # 4) e-blur attenuation of the fundamental line/space mode.
    sigma = float(c["sigma_nm"])
    att = fundamental_blur_attenuation(pitch, sigma)
    # fundamental survival above 0.30 is a soft requirement.
    blur_term = float(np.clip((att - 0.30) / 0.45, 0.0, 1.0))

    # 5) diffusion length vs space width: if Ld > 0.7 * space the lines start
    # bleeding into each other (matches the v2 small-pitch finding).
    if space <= 0.0:
        space_diff_term = 0.0
    else:
        ratio = Ld / space
        space_diff_term = float(np.clip(1.0 - max(0.0, ratio - 0.7) / 1.5, 0.0, 1.0))

    # Combined score: arithmetic mean of the soft terms; a multiplicative
    # gate ensures that a single hard failure (duty / blur / space_diff = 0)
    # collapses the score to ~0 even if other terms look fine.
    soft = 0.25 * diff_term + 0.25 * h_term + 0.20 * duty_term + 0.15 * blur_term + 0.15 * space_diff_term
    gate = (
        max(diff_term, 1e-3)
        * max(h_term, 1e-3)
        * max(duty_term, 1e-3)
        * max(blur_term, 1e-3)
        * max(space_diff_term, 1e-3)
    ) ** 0.2
    score = 0.6 * soft + 0.4 * gate

    return {
        "diffusion_length_nm": Ld,
        "fundamental_attenuation": att,
        "duty": duty,
        "H_peak_estimate": H_peak,
        "deprotect_budget": deprotect,
        "diff_term": diff_term,
        "h_term": h_term,
        "duty_term": duty_term,
        "blur_term": blur_term,
        "space_diff_term": space_diff_term,
        "prefilter_score": score,
    }


def score_all(candidates: list[dict]) -> list[dict]:
    """Annotate each candidate (in-place mirror) with the prefilter result."""
    out = []
    for c in candidates:
        s = score_candidate(c)
        out.append({**c, **s})
    return out


def select_top_n(candidates_with_score: list[dict], n: int) -> list[dict]:
    sorted_c = sorted(
        candidates_with_score,
        key=lambda c: -c["prefilter_score"],
    )
    return sorted_c[: max(0, n)]
