"""Batch runner that turns a list of v3 candidates into FD outputs using
the v2 helper. The v2 OP and the v2 helper code path are unchanged; only
the cfg dict that we pass in is built per candidate.
"""
from __future__ import annotations

import copy
import time
from pathlib import Path
from typing import Iterable

from reaction_diffusion_peb_v2_high_na.experiments.run_sigma_sweep_helpers import (
    run_one_with_overrides,
)

from .labeler import LabelThresholds, label_one


def _candidate_to_cfg(candidate: dict) -> dict:
    """Build the cfg dict that v2's run_one_with_overrides expects."""
    pitch_nm = float(candidate["pitch_nm"])
    line_cd_nm = float(candidate["line_cd_nm"])
    domain_x_nm = float(candidate["domain_x_nm"])

    cfg = {
        "run": {"name": f"v3_candidate_{candidate.get('_id', '?')}",
                "seed": int(candidate.get("edge_roughness_seed", 7))},
        "geometry": {
            "pattern": "line_space",
            "pitch_nm": pitch_nm,
            "half_pitch_nm": pitch_nm / 2.0,
            "line_cd_nm": line_cd_nm,
            "grid_spacing_nm": float(candidate["grid_spacing_nm"]),
            "domain_x_nm": domain_x_nm,
            "domain_y_nm": float(candidate["domain_y_nm"]),
            "edge_roughness_enabled": True,
            "edge_roughness_amp_nm": float(candidate["edge_roughness_amp_nm"]),
            "edge_roughness_corr_nm": float(candidate["edge_roughness_corr_nm"]),
        },
        "exposure": {
            "wavelength_nm": 13.5,
            "dose_mJ_cm2": float(candidate["dose_mJ_cm2"]),
            "reference_dose_mJ_cm2": float(candidate["reference_dose_mJ_cm2"]),
            "dose_norm": float(candidate["dose_norm"]),
            "eta": float(candidate["eta"]),
            "Hmax_mol_dm3": float(candidate["Hmax_mol_dm3"]),
            "electron_blur_enabled": float(candidate["sigma_nm"]) > 0.0,
            "electron_blur_sigma_nm": float(candidate["sigma_nm"]),
        },
        "peb": {
            "time_s": float(candidate["time_s"]),
            "temperature_C": 100.0,
            "DH_nm2_s": float(candidate["DH_nm2_s"]),
            "kloss_s_inv": float(candidate["kloss_s_inv"]),
            "kdep_s_inv": float(candidate["kdep_s_inv"]),
            "dt_s": float(candidate["dt_s"]),
        },
        "quencher": {
            "enabled": float(candidate["Q0_mol_dm3"]) > 0.0,
            "Q0_mol_dm3": float(candidate["Q0_mol_dm3"]),
            "DQ_nm2_s": float(candidate["DQ_nm2_s"]),
            "kq_s_inv": float(candidate["kq_s_inv"]),
        },
        "development": {"method": "threshold", "P_threshold": float(candidate["P_threshold"])},
    }
    return cfg


def run_one_candidate(candidate: dict, thresholds: LabelThresholds) -> dict:
    """Build cfg, run helper, label, and return a flat dict ready for CSV."""
    cfg = _candidate_to_cfg(candidate)
    try:
        r = run_one_with_overrides(
            cfg,
            sigma_nm=cfg["exposure"]["electron_blur_sigma_nm"],
            time_s=cfg["peb"]["time_s"],
            DH_nm2_s=cfg["peb"]["DH_nm2_s"],
            kdep_s_inv=cfg["peb"]["kdep_s_inv"],
            Hmax_mol_dm3=cfg["exposure"]["Hmax_mol_dm3"],
            quencher_enabled=cfg["quencher"]["enabled"],
            Q0_mol_dm3=cfg["quencher"]["Q0_mol_dm3"],
            DQ_nm2_s=cfg["quencher"]["DQ_nm2_s"],
            kq_s_inv=cfg["quencher"]["kq_s_inv"],
        )
    except Exception as e:  # noqa: BLE001 — solver bugs become numerical_invalid rows
        return {
            **candidate,
            "label": "numerical_invalid",
            "fd_error": repr(e),
        }

    # Drop the heavy fields (numpy arrays) before flattening for CSV.
    drop_keys = {k for k in r.keys() if k.startswith("_")}
    flat = {k: v for k, v in r.items() if k not in drop_keys}
    flat.update({k: candidate.get(k) for k in candidate.keys() if k not in flat})

    label = label_one(r, t=thresholds)
    flat["label"] = label
    return flat


def run_batch(
    candidates: Iterable[dict],
    thresholds: LabelThresholds,
    out_csv: str | Path | None = None,
    progress_every: int = 50,
) -> list[dict]:
    """Run FD over `candidates`. If `out_csv` is provided, stream rows so a
    crash mid-batch leaves a partial CSV on disk."""
    from .metrics_io import LABEL_CSV_COLUMNS
    import csv as _csv

    rows: list[dict] = []
    f = None
    writer = None
    if out_csv is not None:
        out_csv = Path(out_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        f = open(out_csv, "w", newline="")
        writer = _csv.DictWriter(f, fieldnames=LABEL_CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()

    t0 = time.time()
    try:
        for i, candidate in enumerate(candidates):
            row = run_one_candidate(candidate, thresholds)
            rows.append(row)
            if writer is not None:
                writer.writerow(row)
                f.flush()
            if (i + 1) % progress_every == 0:
                dt = time.time() - t0
                rate = (i + 1) / max(dt, 1e-9)
                print(f"  fd_batch: {i+1} runs done, {dt:.1f}s ({rate:.2f} runs/s)")
    finally:
        if f is not None:
            f.close()

    return rows
