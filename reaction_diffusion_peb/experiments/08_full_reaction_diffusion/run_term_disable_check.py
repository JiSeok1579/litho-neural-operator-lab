"""PEB Phase-8 verification: disabling each term reproduces the
corresponding earlier phase.

Run:
    python reaction_diffusion_peb/experiments/08_full_reaction_diffusion/run_term_disable_check.py

For each disabled-term case the integrated Phase-8 evolver is compared
against the earlier-phase evolver that physically owns that limit.
The test suite already verifies these contracts at small grid; this
script reruns them at the production 128 x 128 grid and dumps a CSV
so the equivalence is visible in the experiment outputs too.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import torch

from reaction_diffusion_peb.src.arrhenius import (
    evolve_acid_loss_deprotection_fd_at_T,
)
from reaction_diffusion_peb.src.diffusion_fd import diffuse_fd
from reaction_diffusion_peb.src.exposure import acid_generation
from reaction_diffusion_peb.src.full_reaction_diffusion import (
    evolve_full_reaction_diffusion_fd_at_T,
)
from reaction_diffusion_peb.src.quencher_reaction import evolve_quencher_fd
from reaction_diffusion_peb.src.reaction_diffusion import diffuse_acid_loss_fd
from reaction_diffusion_peb.src.synthetic_aerial import gaussian_spot

OUT_LOG = Path("reaction_diffusion_peb/outputs/logs")

GRID_SIZE = 128
DX_NM = 1.0
SIGMA_PX = 12.0
HMAX = 0.2
ETA = 1.0
DOSE = 1.0

DH = 0.8
DQ = 0.1 * DH
KDEP = 0.5
KLOSS = 0.005
KQ = 1.0
Q0 = 0.1
T_END_S = 60.0
T_REF_C = 100.0
EA = 100.0


def linf(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().max().item())


def main() -> None:
    I = gaussian_spot(GRID_SIZE, sigma_px=SIGMA_PX)
    H0 = acid_generation(I, dose=DOSE, eta=ETA, Hmax=HMAX)

    rows = [["case", "compared_against",
             "max|H_full - H_ref|", "max|P_full - P_ref|",
             "tolerance", "pass"]]

    # 1. T = T_ref -> Phase 7 quencher
    H_full, Q_full, P_full = evolve_full_reaction_diffusion_fd_at_T(
        H0, Q0_mol_dm3=Q0, DH_nm2_s=DH, DQ_nm2_s=DQ,
        kq_ref_s_inv=KQ, kloss_ref_s_inv=KLOSS, kdep_ref_s_inv=KDEP,
        temperature_c=T_REF_C, temperature_ref_c=T_REF_C,
        activation_energy_kj_mol=EA,
        t_end_s=T_END_S, dx_nm=DX_NM,
    )
    H_ref, Q_ref, P_ref = evolve_quencher_fd(
        H0, Q0_mol_dm3=Q0, DH_nm2_s=DH, DQ_nm2_s=DQ,
        kq_s_inv=KQ, kloss_s_inv=KLOSS, kdep_s_inv=KDEP,
        t_end_s=T_END_S, dx_nm=DX_NM,
    )
    err_h, err_p = linf(H_full, H_ref), linf(P_full, P_ref)
    rows.append(["T = T_ref", "Phase 7 quencher",
                 f"{err_h:.3e}", f"{err_p:.3e}", "1e-10",
                 "yes" if (err_h < 1e-10 and err_p < 1e-10) else "NO"])

    # 2. Ea = 0 at hot T -> Phase 7 quencher
    H_full, _, P_full = evolve_full_reaction_diffusion_fd_at_T(
        H0, Q0_mol_dm3=Q0, DH_nm2_s=DH, DQ_nm2_s=DQ,
        kq_ref_s_inv=KQ, kloss_ref_s_inv=KLOSS, kdep_ref_s_inv=KDEP,
        temperature_c=125.0, temperature_ref_c=T_REF_C,
        activation_energy_kj_mol=0.0,
        t_end_s=T_END_S, dx_nm=DX_NM,
    )
    err_h, err_p = linf(H_full, H_ref), linf(P_full, P_ref)
    rows.append(["Ea = 0  (T = 125 C)", "Phase 7 quencher",
                 f"{err_h:.3e}", f"{err_p:.3e}", "1e-10",
                 "yes" if (err_h < 1e-10 and err_p < 1e-10) else "NO"])

    # 3. kq = 0 at hot T -> Phase 6 Arrhenius (H, P)
    H_full, Q_full, P_full = evolve_full_reaction_diffusion_fd_at_T(
        H0, Q0_mol_dm3=Q0, DH_nm2_s=DH, DQ_nm2_s=DQ,
        kq_ref_s_inv=0.0, kloss_ref_s_inv=KLOSS, kdep_ref_s_inv=KDEP,
        temperature_c=110.0, temperature_ref_c=T_REF_C,
        activation_energy_kj_mol=EA,
        t_end_s=T_END_S, dx_nm=DX_NM,
    )
    H_p6, P_p6 = evolve_acid_loss_deprotection_fd_at_T(
        H0, DH_nm2_s=DH,
        kdep_ref_s_inv=KDEP, kloss_ref_s_inv=KLOSS,
        temperature_c=110.0, temperature_ref_c=T_REF_C,
        activation_energy_kj_mol=EA,
        t_end_s=T_END_S, dx_nm=DX_NM,
    )
    err_h, err_p = linf(H_full, H_p6), linf(P_full, P_p6)
    err_q = float((Q_full - Q0).abs().max().item())
    rows.append(["kq = 0  (T = 110 C)", "Phase 6 Arrhenius (H, P)",
                 f"{err_h:.3e}", f"{err_p:.3e}", "1e-5",
                 "yes" if (err_h < 1e-5 and err_p < 1e-5
                          and err_q < 1e-10) else "NO"])

    # 4. kq = 0, kdep = 0 -> Phase 4 acid loss (H only)
    H_full, _, P_full = evolve_full_reaction_diffusion_fd_at_T(
        H0, Q0_mol_dm3=Q0, DH_nm2_s=DH, DQ_nm2_s=DQ,
        kq_ref_s_inv=0.0, kloss_ref_s_inv=KLOSS, kdep_ref_s_inv=0.0,
        temperature_c=T_REF_C, temperature_ref_c=T_REF_C,
        activation_energy_kj_mol=EA,
        t_end_s=T_END_S, dx_nm=DX_NM,
    )
    H_p4 = diffuse_acid_loss_fd(
        H0, DH_nm2_s=DH, kloss_s_inv=KLOSS,
        t_end_s=T_END_S, dx_nm=DX_NM,
    )
    err_h, err_p = linf(H_full, H_p4), float(P_full.abs().max().item())
    rows.append(["kq = 0, kdep = 0", "Phase 4 acid loss",
                 f"{err_h:.3e}", f"{err_p:.3e}", "1e-5",
                 "yes" if (err_h < 1e-5 and err_p < 1e-10) else "NO"])

    # 5. kq = 0, kdep = 0, kloss = 0 -> Phase 2 pure diffusion
    H_full, _, P_full = evolve_full_reaction_diffusion_fd_at_T(
        H0, Q0_mol_dm3=Q0, DH_nm2_s=DH, DQ_nm2_s=DQ,
        kq_ref_s_inv=0.0, kloss_ref_s_inv=0.0, kdep_ref_s_inv=0.0,
        temperature_c=T_REF_C, temperature_ref_c=T_REF_C,
        activation_energy_kj_mol=EA,
        t_end_s=T_END_S, dx_nm=DX_NM,
    )
    H_p2 = diffuse_fd(H0, DH_nm2_s=DH, t_end_s=T_END_S, dx_nm=DX_NM)
    err_h, err_p = linf(H_full, H_p2), float(P_full.abs().max().item())
    rows.append(["kq = 0, kdep = 0, kloss = 0", "Phase 2 pure diffusion",
                 f"{err_h:.3e}", f"{err_p:.3e}", "1e-5",
                 "yes" if (err_h < 1e-5 and err_p < 1e-10) else "NO"])

    OUT_LOG.mkdir(parents=True, exist_ok=True)
    log = OUT_LOG / "peb_phase8_term_disable_check.csv"
    with open(log, "w", newline="") as f:
        csv.writer(f).writerows(rows)
    print(f"  wrote {log}")

    print()
    print("term-disable equivalence check:")
    widths = [max(len(str(r[c])) for r in rows) for c in range(len(rows[0]))]
    for r in rows:
        print("  " + "  ".join(str(v).ljust(w) for v, w in zip(r, widths)))


if __name__ == "__main__":
    main()
