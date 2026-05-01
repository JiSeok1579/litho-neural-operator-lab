# Stage 06G — strict-threshold selection rationale
Selection is **data-driven** -- we read `stage06F_threshold_sensitivity.csv` and pick the cell that retains a *nonzero but selective* survival rate.

## 06F survival table
| CD tol | LER cap | survivors / 100 (nominal) | survivors / 10 (MC strict) |
|---|---|---|---|
| ±1.00 nm | 3.0 nm | 69 / 100 | 8 / 10 |
| ±1.00 nm | 2.7 nm | 69 / 100 | 8 / 10 |
| ±1.00 nm | 2.5 nm | 9 / 100 | 0 / 10 |
| ±0.75 nm | 3.0 nm | 54 / 100 | 4 / 10 |
| ±0.75 nm | 2.7 nm | 54 / 100 | 4 / 10 |
| ±0.75 nm | 2.5 nm | 8 / 100 | 0 / 10 |
| ±0.50 nm | 3.0 nm | 36 / 100 | 3 / 10 |
| ±0.50 nm | 2.7 nm | 36 / 100 | 3 / 10 |
| ±0.50 nm | 2.5 nm | 4 / 100 | 0 / 10 |

## Rules applied
- Reject `≥ 60 %` survivors -- threshold is too permissive to discriminate (CD ±1.0 nm cells fall here).
- Reject `≤ 10 %` survivors -- threshold is so tight that the resulting optimisation has too small a feasible set (CD with LER ≤ 2.5 nm cells fall here).
- Among the remaining cells, pick the strictest CD as **primary** and the next-stricter as **backup**.
- LER cap dimension does not move the survival count in this dataset (every survivor already has LER far below 3.0 nm), so LER cap is fixed at 3.0 nm rather than tightened without data support.

## Selected configs
- **Primary**: `cd_tol_nm = 0.50` and `ler_cap_nm = 3.0` (36 / 100 nominal, 30 / 10 MC).
- **Backup**: `cd_tol_nm = 0.75` and `ler_cap_nm = 3.0` (54 / 100 nominal, 40 / 10 MC) -- usable if 06G primary yields too few feasible recipes after FD verification.

## Caveat
These thresholds are **internally derived** from the 06D / 06E FD distribution; they are *not* externally calibrated and *not* spec values. They are the smallest evidence-driven tightening that still gives the surrogate a meaningful gradient to climb.
