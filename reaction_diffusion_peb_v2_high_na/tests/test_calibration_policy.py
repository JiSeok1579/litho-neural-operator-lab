"""Lightweight tests that pin the calibration policy.

These guard the user-required closeout invariants:
    - calibration_status.published_data_loaded == false
    - calibration_status.v2_OP_frozen          == true
    - frozen_nominal_OP exists with the agreed values
    - DH=0.80 internal_best_score_candidate exists, not as the active OP

If any of these flip without an explicit policy decision, the test will
catch it and force the change to be reviewed.
"""
from __future__ import annotations

from pathlib import Path

import yaml


REPO = Path(__file__).resolve().parents[1]
TARGETS_PATH = REPO / "calibration" / "calibration_targets.yaml"


def _load() -> dict:
    return yaml.safe_load(TARGETS_PATH.read_text())


def test_calibration_status_internal_only():
    cfg = _load()
    cs = cfg["calibration_status"]
    assert cs["level"] == "internal-consistency only", cs
    assert cs["published_data_loaded"] is False
    assert cs["v2_OP_frozen"] is True


def test_reference_block_marks_not_loaded():
    cfg = _load()
    ref = cfg["reference"]
    assert ref["published_data_loaded"] is False, (
        "reference.published_data_loaded must stay false until external data is loaded"
    )


def test_frozen_nominal_OP_matches_user_spec():
    cfg = _load()
    op = cfg["frozen_nominal_OP"]
    expected = {
        "pitch_nm":       24,
        "line_cd_nm":     12.5,
        "dose_mJ_cm2":    40,
        "sigma_nm":       2,
        "time_s":         30,
        "kdep_s_inv":     0.5,
        "Hmax_mol_dm3":   0.2,
        "kloss_s_inv":    0.005,
        "DH_nm2_s":       0.5,
        "Q0_mol_dm3":     0.02,
        "kq_s_inv":       1.0,
        "DQ_nm2_s":       0.0,
        "domain_x_nm":    120,
    }
    for key, value in expected.items():
        assert op[key] == value, f"{key}: expected {value}, got {op[key]}"


def test_internal_best_score_candidate_recorded_only():
    cfg = _load()
    cand = cfg["internal_best_score_candidate"]
    assert cand["DH_nm2_s"] == 0.80
    assert cand["Hmax_mol_dm3"] == 0.20
    assert cand["kdep_s_inv"] == 0.50
    # Active OP keeps DH=0.5, never adopted from this candidate.
    op = cfg["frozen_nominal_OP"]
    assert op["DH_nm2_s"] == 0.5


def test_legacy_operating_point_block_matches_frozen_OP():
    """The legacy operating_point block is kept for backward compat only and
    must mirror frozen_nominal_OP exactly."""
    cfg = _load()
    op = cfg["operating_point"]
    fop = cfg["frozen_nominal_OP"]
    for key in [
        "pitch_nm", "line_cd_nm", "dose_mJ_cm2", "time_s",
        "kdep_s_inv", "Hmax_mol_dm3", "kloss_s_inv", "DH_nm2_s",
        "Q0_mol_dm3", "kq_s_inv", "DQ_nm2_s",
        "domain_x_nm", "domain_y_nm", "grid_spacing_nm",
        "edge_roughness_amp_nm", "edge_roughness_corr_nm",
        "edge_roughness_seed",
    ]:
        assert op[key] == fop[key], f"legacy operating_point.{key} drifted from frozen_nominal_OP"
    # legacy block uses the explicit-name electron_blur_sigma_nm, mirror sigma_nm.
    assert op["electron_blur_sigma_nm"] == fop["sigma_nm"]
