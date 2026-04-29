"""PEB Phase-9 (Phase-10 prerequisite): generate the **large** safe dataset.

Run:
    python reaction_diffusion_peb/experiments/09_dataset_generation/generate_safe_large_dataset.py

Same builder as ``generate_fd_dataset.py`` but with ``N_SAFE = 1024`` so
Phase 10 has enough samples to start telling apart the small-data
failure mode from the architecture / target-definition failure modes.
The original 64-sample archive at
``outputs/datasets/peb_phase9_safe_dataset.npz`` is left untouched
so the Phase-9 numbers stay reproducible.

Stiff samples are not regenerated — Phase 10's ablation keeps the
existing stiff archive as the OOD evaluation set.
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
    save_dataset,
)

OUT_DATA = Path("reaction_diffusion_peb/outputs/datasets")

GRID_SIZE = 128
DX_NM = 1.0
P_THRESHOLD = 0.5

N_SAFE = 1024
SEED_SAFE = 20260430


def main() -> None:
    OUT_DATA.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED_SAFE)
    print(f"[safe-large] generating {N_SAFE} samples (seed={SEED_SAFE})")

    samples = []
    t0 = time.perf_counter()
    log_every = max(1, N_SAFE // 16)
    for i in range(N_SAFE):
        spec = random_safe_spec(rng, grid_size=GRID_SIZE)
        samples.append(generate_sample(
            spec, grid_size=GRID_SIZE, dx_nm=DX_NM,
            P_threshold=P_THRESHOLD,
        ))
        if (i + 1) % log_every == 0 or i == N_SAFE - 1:
            elapsed = time.perf_counter() - t0
            print(f"  sample {i + 1:4d}/{N_SAFE}   "
                  f"elapsed={elapsed:7.1f} s")
    elapsed = time.perf_counter() - t0
    splits = make_split_indices(N_SAFE, seed=SEED_SAFE)
    ranges = parameter_ranges(samples)
    meta = {
        "grid_size": GRID_SIZE,
        "grid_spacing_nm": DX_NM,
        "P_threshold": P_THRESHOLD,
        "solver": "fd",
        "regime": "safe",
        "size_label": "large",
        "seed": SEED_SAFE,
        "wall_clock_s": elapsed,
        "parameter_ranges": ranges,
        "splits": splits,
    }
    out_path = OUT_DATA / "peb_phase9_safe_large_dataset.npz"
    save_dataset(out_path, samples, meta)
    print(f"  wrote {out_path} ({N_SAFE} samples, {elapsed:.1f} s wall)")
    print(f"  split sizes: train={len(splits['train'])}  "
          f"val={len(splits['val'])}  test={len(splits['test'])}")
    print(f"  aerial_kind_counts: {ranges['aerial_kind_counts']}")


if __name__ == "__main__":
    main()
