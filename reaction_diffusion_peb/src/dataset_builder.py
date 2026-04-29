"""Phase 9 — dataset generation utilities.

Drives the Phase-8 integrated FD evolver across randomized parameters
and packages the (input parameters, input fields, output fields)
tuples as ``.npz`` files so Phase 10 (DeepONet / FNO surrogate) has
something to learn from.

Per-sample fields:

    Inputs (2D float32, shape (G, G)):
        I        — aerial intensity                       in [0, 1]
        H0       — initial acid                           [mol/dm^3]

    Outputs (2D float32, shape (G, G)):
        H_final  — acid at t_end_s                        [mol/dm^3]
        Q_final  — quencher at t_end_s                    [mol/dm^3]
        P_final  — deprotected fraction at t_end_s        [0, 1]
        R        — thresholded resist mask (P > P_threshold)  {0, 1}

    Per-sample scalars (float32):
        dose, eta, Hmax,
        DH, DQ_ratio,
        kq_ref, kdep_ref, kloss_ref,
        Q0,
        temperature_c, temperature_ref_c, activation_energy_kj_mol,
        t_end_s,
        aerial_kind_code (int8, see ``AERIAL_KIND_CODES``).

The dataset_builder does not own any physics — it always calls the
Phase-8 ``evolve_full_reaction_diffusion_fd_at_T`` so the dataset
inherits the same (validated, term-disable-checked) source of truth.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from reaction_diffusion_peb.src.exposure import acid_generation
from reaction_diffusion_peb.src.full_reaction_diffusion import (
    evolve_full_reaction_diffusion_fd_at_T,
)
from reaction_diffusion_peb.src.synthetic_aerial import (
    contact_array,
    gaussian_spot,
    line_space,
    normalize_intensity,
    two_spot,
)


# Aerial pattern integer codes — kept stable across releases so old
# .npz files remain readable.
AERIAL_KIND_CODES: dict[str, int] = {
    "gaussian_spot": 0,
    "line_space": 1,
    "contact_array": 2,
    "two_spot": 3,
}
AERIAL_KIND_FROM_CODE: dict[int, str] = {v: k for k, v in AERIAL_KIND_CODES.items()}


# --------------------------------------------------------------------------
# spec
# --------------------------------------------------------------------------

@dataclass
class SampleSpec:
    """Full specification of one Phase-9 sample.

    Holds every input parameter the integrated Phase-8 FD evolver and
    the aerial generator need. Ranges are sampled by the helpers below.
    """

    # aerial-pattern parameters
    aerial_kind: str = "gaussian_spot"
    aerial_param_a: float = 12.0   # sigma_px / pitch_px / pitch_px / sigma_px
    aerial_param_b: float = 0.0    # 0 / duty / sigma_px / separation_px

    # Dill exposure parameters
    dose: float = 1.0
    eta: float = 1.0
    Hmax: float = 0.2

    # diffusion
    DH_nm2_s: float = 0.8
    DQ_ratio: float = 0.1

    # reaction (reference rates; Arrhenius scales them)
    kq_ref_s_inv: float = 1.0
    kdep_ref_s_inv: float = 0.5
    kloss_ref_s_inv: float = 0.005

    # quencher initial concentration
    Q0_mol_dm3: float = 0.1

    # temperature
    temperature_c: float = 100.0
    temperature_ref_c: float = 100.0
    activation_energy_kj_mol: float = 100.0

    # PEB time
    t_end_s: float = 60.0


def aerial_from_spec(spec: SampleSpec, grid_size: int) -> torch.Tensor:
    """Build the aerial intensity I(x, y) from a spec."""
    kind = spec.aerial_kind
    a = spec.aerial_param_a
    b = spec.aerial_param_b
    if kind == "gaussian_spot":
        return gaussian_spot(grid_size, sigma_px=a)
    if kind == "line_space":
        return line_space(grid_size, pitch_px=a, duty=b if b > 0 else 0.5)
    if kind == "contact_array":
        return normalize_intensity(contact_array(grid_size, pitch_px=a, sigma_px=b))
    if kind == "two_spot":
        return normalize_intensity(
            two_spot(grid_size, sigma_px=a, separation_px=b)
        )
    raise ValueError(f"unknown aerial_kind: {kind!r}")


# --------------------------------------------------------------------------
# random spec sampling
# --------------------------------------------------------------------------

DEFAULT_AERIAL_KIND_WEIGHTS: dict[str, float] = {
    "gaussian_spot": 0.4,
    "two_spot": 0.3,
    "line_space": 0.2,
    "contact_array": 0.1,
}


def _sample_aerial(rng: np.random.Generator, grid_size: int,
                   weights: dict[str, float] | None = None
                   ) -> tuple[str, float, float]:
    if weights is None:
        weights = DEFAULT_AERIAL_KIND_WEIGHTS
    kinds = list(weights.keys())
    probs = np.array([weights[k] for k in kinds], dtype=np.float64)
    probs = probs / probs.sum()
    kind = str(rng.choice(kinds, p=probs))
    if kind == "gaussian_spot":
        return kind, float(rng.uniform(6.0, 18.0)), 0.0
    if kind == "line_space":
        pitch = float(rng.uniform(16.0, 48.0))
        duty = float(rng.uniform(0.3, 0.7))
        return kind, pitch, duty
    if kind == "contact_array":
        pitch = float(rng.uniform(20.0, 48.0))
        sigma = float(rng.uniform(3.0, 8.0))
        return kind, pitch, sigma
    if kind == "two_spot":
        sigma = float(rng.uniform(4.0, 10.0))
        sep = float(rng.uniform(20.0, 60.0))
        return kind, sigma, sep
    raise AssertionError(f"unhandled kind {kind!r}")


def random_safe_spec(rng: np.random.Generator, grid_size: int = 128) -> SampleSpec:
    """Sample a spec from the safe-kq operating regime."""
    kind, a, b = _sample_aerial(rng, grid_size)
    return SampleSpec(
        aerial_kind=kind, aerial_param_a=a, aerial_param_b=b,
        dose=float(rng.uniform(0.5, 1.5)),
        eta=float(rng.uniform(0.7, 1.3)),
        Hmax=float(rng.uniform(0.15, 0.25)),
        DH_nm2_s=float(rng.uniform(0.4, 1.2)),
        DQ_ratio=float(rng.uniform(0.05, 0.2)),
        kq_ref_s_inv=float(rng.uniform(0.5, 5.0)),
        kdep_ref_s_inv=float(rng.uniform(0.2, 0.8)),
        kloss_ref_s_inv=float(rng.uniform(0.001, 0.01)),
        Q0_mol_dm3=float(rng.uniform(0.05, 0.15)),
        temperature_c=float(rng.uniform(85.0, 115.0)),
        temperature_ref_c=100.0,
        activation_energy_kj_mol=float(rng.uniform(60.0, 130.0)),
        t_end_s=float(rng.uniform(45.0, 75.0)),
    )


def random_stiff_spec(rng: np.random.Generator, grid_size: int = 128) -> SampleSpec:
    """Sample a spec from the stiff-kq operating regime (CAR / MOR-realistic).

    Identical to ``random_safe_spec`` except ``kq_ref`` is drawn from
    ``[100, 1000] 1/s``. These samples are expensive (12k+ explicit
    Euler steps each at the upper end) so the demos generate only a
    small handful of stiff samples.
    """
    base = random_safe_spec(rng, grid_size)
    base.kq_ref_s_inv = float(rng.uniform(100.0, 1000.0))
    return base


# --------------------------------------------------------------------------
# generate one sample
# --------------------------------------------------------------------------

@dataclass
class SampleArrays:
    """Per-sample numpy arrays plus the spec used to generate them."""

    spec: SampleSpec
    I: np.ndarray
    H0: np.ndarray
    H_final: np.ndarray
    Q_final: np.ndarray
    P_final: np.ndarray
    R: np.ndarray


def generate_sample(
    spec: SampleSpec,
    grid_size: int = 128,
    dx_nm: float = 1.0,
    P_threshold: float = 0.5,
) -> SampleArrays:
    """Run the Phase-8 integrated FD evolver for one spec; return arrays.

    All output fields are returned as ``float32`` numpy arrays of shape
    ``(grid_size, grid_size)``. ``R`` is the thresholded mask (0/1) at
    ``P > P_threshold``.
    """
    I = aerial_from_spec(spec, grid_size=grid_size)
    H0 = acid_generation(I, dose=spec.dose, eta=spec.eta, Hmax=spec.Hmax)
    H, Q, P = evolve_full_reaction_diffusion_fd_at_T(
        H0,
        Q0_mol_dm3=spec.Q0_mol_dm3,
        DH_nm2_s=spec.DH_nm2_s,
        DQ_nm2_s=spec.DQ_ratio * spec.DH_nm2_s,
        kq_ref_s_inv=spec.kq_ref_s_inv,
        kloss_ref_s_inv=spec.kloss_ref_s_inv,
        kdep_ref_s_inv=spec.kdep_ref_s_inv,
        temperature_c=spec.temperature_c,
        temperature_ref_c=spec.temperature_ref_c,
        activation_energy_kj_mol=spec.activation_energy_kj_mol,
        t_end_s=spec.t_end_s,
        dx_nm=dx_nm,
    )
    R = (P > P_threshold).to(P.dtype)
    return SampleArrays(
        spec=spec,
        I=I.detach().cpu().numpy().astype(np.float32),
        H0=H0.detach().cpu().numpy().astype(np.float32),
        H_final=H.detach().cpu().numpy().astype(np.float32),
        Q_final=Q.detach().cpu().numpy().astype(np.float32),
        P_final=P.detach().cpu().numpy().astype(np.float32),
        R=R.detach().cpu().numpy().astype(np.float32),
    )


# --------------------------------------------------------------------------
# split indices
# --------------------------------------------------------------------------

def make_split_indices(
    n_samples: int,
    fractions: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 0,
) -> dict[str, list[int]]:
    """Return non-overlapping ``train / val / test`` index lists.

    Indices are shuffled deterministically by ``seed`` and partitioned
    by the given fractions. The leftover from rounding goes to ``train``
    so val/test do not float by 1 sample between runs.
    """
    if n_samples < 3:
        raise ValueError("need at least 3 samples to form three splits")
    if abs(sum(fractions) - 1.0) > 1e-6:
        raise ValueError("fractions must sum to 1")
    f_train, f_val, f_test = fractions
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_samples).tolist()
    n_val = max(1, int(round(f_val * n_samples)))
    n_test = max(1, int(round(f_test * n_samples)))
    n_train = n_samples - n_val - n_test
    if n_train < 1:
        raise ValueError("split sizes left no train samples; "
                         "raise n_samples or shrink val/test fractions")
    return {
        "train": perm[:n_train],
        "val": perm[n_train:n_train + n_val],
        "test": perm[n_train + n_val:],
    }


# --------------------------------------------------------------------------
# save / load
# --------------------------------------------------------------------------

DATASET_FIELD_NAMES = (
    "I", "H0", "H_final", "Q_final", "P_final", "R",
)
DATASET_SCALAR_NAMES = (
    "dose", "eta", "Hmax",
    "DH_nm2_s", "DQ_ratio",
    "kq_ref_s_inv", "kdep_ref_s_inv", "kloss_ref_s_inv",
    "Q0_mol_dm3",
    "temperature_c", "temperature_ref_c", "activation_energy_kj_mol",
    "t_end_s",
    "aerial_param_a", "aerial_param_b",
)


def save_dataset(
    path: str | Path,
    samples: list[SampleArrays],
    metadata: dict[str, Any],
) -> Path:
    """Save a list of samples to one ``.npz`` archive plus a sibling
    ``.json`` metadata file.

    ``.npz`` layout:
        I, H0, H_final, Q_final, P_final, R   stacked float32 (n, G, G)
        dose, eta, ...                         float32 (n,)
        aerial_kind_code                       int8 (n,)

    ``.json`` carries ranges, splits, solver, regime, seed.
    """
    if not samples:
        raise ValueError("samples must be non-empty")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    n = len(samples)
    fields = {name: np.stack([getattr(s, name) for s in samples])
              for name in DATASET_FIELD_NAMES}
    scalars = {
        name: np.array(
            [getattr(s.spec, _scalar_attr_for(name)) for s in samples],
            dtype=np.float32,
        )
        for name in DATASET_SCALAR_NAMES
    }
    aerial_kind_code = np.array(
        [AERIAL_KIND_CODES[s.spec.aerial_kind] for s in samples],
        dtype=np.int8,
    )
    np.savez(
        path,
        **fields,
        **scalars,
        aerial_kind_code=aerial_kind_code,
    )
    meta_path = path.with_suffix(".json")
    full_meta = dict(metadata)
    full_meta.setdefault("n_samples", n)
    full_meta.setdefault("aerial_kind_codes", AERIAL_KIND_CODES)
    full_meta.setdefault("field_names", list(DATASET_FIELD_NAMES))
    full_meta.setdefault("scalar_names", list(DATASET_SCALAR_NAMES))
    with open(meta_path, "w") as f:
        json.dump(full_meta, f, indent=2)
    return path


def _scalar_attr_for(name: str) -> str:
    """Map dataset-scalar name to ``SampleSpec`` attribute (handles the
    shorter ``DH``/``Hmax`` aliases)."""
    if name == "Hmax":
        return "Hmax"
    return name


def load_dataset(path: str | Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Load a Phase-9 dataset and its metadata. Returns ``(arrays, meta)``."""
    path = Path(path)
    arrays = dict(np.load(path))
    meta_path = path.with_suffix(".json")
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
    else:
        meta = {}
    return arrays, meta


# --------------------------------------------------------------------------
# parameter-range summary
# --------------------------------------------------------------------------

def parameter_ranges(samples: list[SampleArrays]) -> dict[str, dict[str, float]]:
    """Compute per-scalar (min, max, mean) for the provided samples."""
    out: dict[str, dict[str, float]] = {}
    for name in DATASET_SCALAR_NAMES:
        vals = np.array(
            [getattr(s.spec, _scalar_attr_for(name)) for s in samples],
            dtype=np.float64,
        )
        out[name] = {
            "min": float(vals.min()),
            "max": float(vals.max()),
            "mean": float(vals.mean()),
        }
    aerial_counts: dict[str, int] = {}
    for s in samples:
        aerial_counts[s.spec.aerial_kind] = aerial_counts.get(s.spec.aerial_kind, 0) + 1
    return {"scalars": out, "aerial_kind_counts": aerial_counts}


def spec_to_dict(spec: SampleSpec) -> dict[str, Any]:
    return asdict(spec)
