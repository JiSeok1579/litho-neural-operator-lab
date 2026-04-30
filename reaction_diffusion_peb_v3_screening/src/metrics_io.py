"""I/O helpers for v3 candidates / labels / metrics / models."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np


# ---------- candidates ----------

def write_candidates_jsonl(candidates: Iterable[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for c in candidates:
            f.write(json.dumps(c) + "\n")


def read_candidates_jsonl(path: str | Path) -> list[dict]:
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


# ---------- labels + metrics CSV ----------

LABEL_CSV_COLUMNS = [
    "_id", "label",
    # candidate parameters (pin the exact set used downstream)
    "pitch_nm", "line_cd_nm", "line_cd_ratio", "domain_x_nm",
    "dose_mJ_cm2", "sigma_nm", "DH_nm2_s", "time_s",
    "Hmax_mol_dm3", "kdep_s_inv", "Q0_mol_dm3", "kq_s_inv",
    "abs_len_nm",
    # FD-derived metrics that go to the regressor
    "CD_locked_nm", "LER_CD_locked_nm",
    "area_frac", "P_line_margin",
    # ancillary metrics + bounds
    "CD_final_nm", "CD_pitch_frac",
    "LER_after_PEB_P_nm", "LER_design_initial_nm",
    "P_space_center_mean", "P_line_center_mean",
    "contrast",
    "psd_locked_mid", "psd_design_mid",
    "H_min", "P_min", "P_max",
    "prefilter_score",
    "roughness_trigger",
]


def write_labels_csv(rows: list[dict], path: str | Path,
                     columns: list[str] = LABEL_CSV_COLUMNS) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def read_labels_csv(path: str | Path) -> list[dict]:
    out = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            out.append(row)
    return out


# ---------- model joblib ----------

def save_model(model, path: str | Path, metadata: dict | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"model": model, "metadata": metadata or {}}
    joblib.dump(payload, path, compress=3)


def load_model(path: str | Path):
    payload = joblib.load(path)
    return payload["model"], payload.get("metadata", {})


# ---------- feature vector ----------

FEATURE_KEYS = [
    "pitch_nm", "line_cd_ratio",
    "dose_mJ_cm2", "sigma_nm",
    "DH_nm2_s", "time_s",
    "Hmax_mol_dm3", "kdep_s_inv",
    "Q0_mol_dm3", "kq_s_inv",
    "abs_len_nm",
]


def build_feature_matrix(candidates: list[dict],
                         feature_keys: list[str] = FEATURE_KEYS) -> np.ndarray:
    X = np.zeros((len(candidates), len(feature_keys)), dtype=np.float64)
    for i, c in enumerate(candidates):
        for j, k in enumerate(feature_keys):
            X[i, j] = float(c[k])
    return X


REGRESSION_TARGETS = ["CD_locked_nm", "LER_CD_locked_nm", "area_frac", "P_line_margin"]


def build_regression_target_matrix(rows: list[dict],
                                   targets: list[str] = REGRESSION_TARGETS) -> np.ndarray:
    Y = np.zeros((len(rows), len(targets)), dtype=np.float64)
    for i, r in enumerate(rows):
        for j, k in enumerate(targets):
            v = r.get(k)
            try:
                Y[i, j] = float(v)
            except (TypeError, ValueError):
                Y[i, j] = np.nan
    return Y
