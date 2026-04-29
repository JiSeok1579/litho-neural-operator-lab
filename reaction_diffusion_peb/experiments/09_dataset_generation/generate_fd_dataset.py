"""PEB Phase-9 demo: generate the FD dataset.

Run:
    python reaction_diffusion_peb/experiments/09_dataset_generation/generate_fd_dataset.py

Drives the Phase-8 integrated FD evolver across randomized parameters
and writes two ``.npz`` archives:

  outputs/datasets/peb_phase9_safe_dataset.npz   — kq_ref in [0.5, 5]
  outputs/datasets/peb_phase9_stiff_dataset.npz  — kq_ref in [100, 1000]

Each archive comes with a sibling ``.json`` metadata file recording
parameter ranges, train / val / test indices, the solver, and the
seed. Stiff samples are kept in a separate file because they take
~100x longer per sample (15k explicit Euler steps each at the upper
end) and because Phase 10 may want to train safe and stiff surrogates
separately.

PINN dataset generation is intentionally deferred — the integrated
PINN training is itself deferred (see FUTURE_WORK.md), so the
``solver: fd`` field in metadata is the discriminator that future
PINN-dataset .npz files will distinguish themselves with.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np

from reaction_diffusion_peb.src.dataset_builder import (
    generate_sample,
    make_split_indices,
    parameter_ranges,
    random_safe_spec,
    random_stiff_spec,
    save_dataset,
    spec_to_dict,
)

OUT_DATA = Path("reaction_diffusion_peb/outputs/datasets")

GRID_SIZE = 128
DX_NM = 1.0
P_THRESHOLD = 0.5

# Modest counts: enough to validate splits and keep regen quick.
# Phase 10 will likely call back here with a larger n.
N_SAFE = 64
N_STIFF = 8

SEED_SAFE = 20260429       # today
SEED_STIFF = 20260430


def _run_one_dataset(
    n_samples: int, seed: int, regime: str,
    sampler, out_name: str,
) -> Path:
    rng = np.random.default_rng(seed)
    print(f"\n[{regime}] generating {n_samples} samples (seed={seed})")
    samples = []
    t0 = time.perf_counter()
    for i in range(n_samples):
        spec = sampler(rng, grid_size=GRID_SIZE)
        sample = generate_sample(
            spec, grid_size=GRID_SIZE, dx_nm=DX_NM,
            P_threshold=P_THRESHOLD,
        )
        samples.append(sample)
        if (i + 1) % max(1, n_samples // 8) == 0 or i == n_samples - 1:
            elapsed = time.perf_counter() - t0
            print(f"  [{regime}] sample {i + 1:4d}/{n_samples}   "
                  f"elapsed={elapsed:7.1f} s")
    elapsed = time.perf_counter() - t0
    splits = make_split_indices(n_samples, seed=seed)
    ranges = parameter_ranges(samples)
    meta = {
        "grid_size": GRID_SIZE,
        "grid_spacing_nm": DX_NM,
        "P_threshold": P_THRESHOLD,
        "solver": "fd",
        "regime": regime,
        "seed": seed,
        "wall_clock_s": elapsed,
        "parameter_ranges": ranges,
        "splits": splits,
    }
    out_path = OUT_DATA / out_name
    save_dataset(out_path, samples, meta)
    print(f"  [{regime}] wrote {out_path} ({n_samples} samples, "
          f"{elapsed:.1f} s wall)")
    print(f"  [{regime}] split sizes: "
          f"train={len(splits['train'])}  val={len(splits['val'])}  "
          f"test={len(splits['test'])}")
    print(f"  [{regime}] aerial_kind_counts: {ranges['aerial_kind_counts']}")
    # First-sample sanity print
    first = samples[0]
    print(f"  [{regime}] sample[0] spec excerpt: "
          f"kind={first.spec.aerial_kind} "
          f"kq_ref={first.spec.kq_ref_s_inv:.3g} "
          f"T={first.spec.temperature_c:.1f} C "
          f"t={first.spec.t_end_s:.1f} s")
    print(f"  [{regime}] sample[0] outputs: "
          f"P_max={first.P_final.max():.4f} "
          f"R_pixels={int(first.R.sum())}")
    return out_path


def main() -> None:
    OUT_DATA.mkdir(parents=True, exist_ok=True)

    safe_path = _run_one_dataset(
        n_samples=N_SAFE, seed=SEED_SAFE, regime="safe",
        sampler=random_safe_spec,
        out_name="peb_phase9_safe_dataset.npz",
    )
    stiff_path = _run_one_dataset(
        n_samples=N_STIFF, seed=SEED_STIFF, regime="stiff",
        sampler=random_stiff_spec,
        out_name="peb_phase9_stiff_dataset.npz",
    )

    print("\nWrote datasets:")
    print(f"  {safe_path}")
    print(f"  {stiff_path}")
    print(f"  metadata sidecars at the matching .json paths.")


if __name__ == "__main__":
    main()
