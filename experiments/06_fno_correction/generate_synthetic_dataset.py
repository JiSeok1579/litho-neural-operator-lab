"""Phase 7 demo: build the synthetic 3D-mask correction paired dataset.

Run:
    python experiments/06_fno_correction/generate_synthetic_dataset.py

Generates two NPZ archives that Phase 8 / 9 will consume:

    outputs/datasets/synthetic_3d_correction_train.npz   (default 800 samples)
    outputs/datasets/synthetic_3d_correction_test.npz    (default 200 samples)

Each archive stores ``masks`` (N, H, W), ``T_thin_real / T_thin_imag``,
``T_3d_real / T_3d_imag``, ``theta`` (N, 6), and metadata
(``theta_names``, ``grid_n``, ``grid_extent``). All on the same grid
(``n=128, extent=8 lambda``).

Also writes:

    outputs/figures/phase7_correction_samples.png     — 4-row preview
    outputs/logs/phase7_dataset_summary.csv          — per-archive stats

Physical takeaways logged in PROGRESS.md, but the gist is: the synthetic
3D correction is a multiplicative, spatially-smooth operator in
frequency space whose effect on a real-valued mask is large enough to
be worth learning (mean |Delta T| about 30-50 % of mean |T_thin|).
"""

from __future__ import annotations

import csv
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from src.common.fft_utils import fft2c
from src.common.grid import Grid2D
from src.common.visualization import (
    save_figure,
    show_correction_samples,
)
from src.mask.transmission import binary_transmission
from src.neural_operator.synthetic_3d_correction_data import (
    CorrectionParams,
    correction_operator,
    generate_dataset,
    random_mask_sampler,
    sample_correction_params,
)

OUT_DATA = Path("outputs/datasets")
OUT_FIG = Path("outputs/figures")
OUT_LOG = Path("outputs/logs")
N = 128
EXTENT = 8.0
N_TRAIN = 2000
N_TEST = 400
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _preview(grid: Grid2D, n_preview: int = 4, seed: int = 7) -> None:
    """Build n_preview samples by hand and plot them so we can eyeball
    the dataset distribution before committing to a long generation run.
    """
    rng = random.Random(seed)
    samples = []
    for _ in range(n_preview):
        mask_seed = rng.randrange(0, 2**31 - 1)
        mask = random_mask_sampler(grid, seed=mask_seed)
        params = sample_correction_params(rng)
        T_thin = fft2c(binary_transmission(mask))
        C = correction_operator(grid, params)
        T_3d = T_thin * C
        samples.append({
            "mask": mask, "T_thin": T_thin, "C": C, "T_3d": T_3d,
            "params": params,
        })

    label_lines = []
    for k, s in enumerate(samples):
        p = s["params"]
        label_lines.append(
            f"s{k}: gamma={p.gamma:+.3f} alpha={p.alpha:+.3f} "
            f"beta={p.beta:+.3f} delta={p.delta:+.3f} "
            f"s={p.s:+.3f} c={p.c:.3f}"
        )
    suptitle = "phase 7 correction samples\n" + "\n".join(label_lines)
    fig = show_correction_samples(
        samples, extent=grid.extent, df=grid.df, freq_zoom=24,
        suptitle=suptitle,
    )
    out = save_figure(fig, OUT_FIG / "phase7_correction_samples.png")
    print(f"  wrote {out}")


def main() -> None:
    grid = Grid2D(n=N, extent=EXTENT, device=DEVICE)

    _preview(grid)

    print()
    print(f"generating train dataset (n={N_TRAIN})...")
    t0 = time.time()
    s_train = generate_dataset(
        grid=grid,
        n_samples=N_TRAIN,
        output_path=OUT_DATA / "synthetic_3d_correction_train.npz",
        seed=0,
    )
    print(f"  done in {time.time() - t0:.1f} s")

    print()
    print(f"generating test dataset (n={N_TEST})...")
    t0 = time.time()
    s_test = generate_dataset(
        grid=grid,
        n_samples=N_TEST,
        output_path=OUT_DATA / "synthetic_3d_correction_test.npz",
        seed=1,
    )
    print(f"  done in {time.time() - t0:.1f} s")

    OUT_LOG.mkdir(parents=True, exist_ok=True)
    log = OUT_LOG / "phase7_dataset_summary.csv"
    rows = [
        ["split", "n_samples", "T_thin_l2_mean", "T_3d_l2_mean",
         "delta_T_l2_mean",
         "gamma_mean", "alpha_mean", "beta_mean",
         "delta_mean", "s_mean", "c_mean"],
    ]
    for label, s in (("train", s_train), ("test", s_test)):
        rows.append([
            label,
            s["n_samples"],
            f"{s['T_thin_l2_mean']:.4f}",
            f"{s['T_3d_l2_mean']:.4f}",
            f"{s['delta_T_l2_mean']:.4f}",
            f"{s['theta_means']['gamma']:+.4f}",
            f"{s['theta_means']['alpha']:+.4f}",
            f"{s['theta_means']['beta']:+.4f}",
            f"{s['theta_means']['delta']:+.4f}",
            f"{s['theta_means']['s']:+.4f}",
            f"{s['theta_means']['c']:.4f}",
        ])
    with open(log, "w", newline="") as f:
        csv.writer(f).writerows(rows)
    print(f"  wrote {log}")
    print()
    print("summary:")
    widths = [max(len(str(r[c])) for r in rows) for c in range(len(rows[0]))]
    for r in rows:
        print("  " + "  ".join(str(v).ljust(w) for v, w in zip(r, widths)))


if __name__ == "__main__":
    main()
