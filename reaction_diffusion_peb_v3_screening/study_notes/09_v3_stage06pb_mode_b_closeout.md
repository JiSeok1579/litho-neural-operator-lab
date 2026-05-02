# Stage 06P-B closeout — Mode B / G_4867 vs J_1453 magnitude-gap thread

> **Status:** CLOSED.
> **Preferred surrogate:** `stage06PB`.
> **Mode A primary recipe:** `G_4867`.
> **Mode B primary recipe:** `J_1453`.
> **Policy:** v2_OP_frozen = true · published_data_loaded = false · external_calibration = none.

This note closes the Mode B exploration thread that started with Stage
06J. It records the 06J → 06P-B arc, the diagnosis of the 06P
blindspot, the corrections that closed it, and the operating decision
that no further FD work is needed under this thread.

## TL;DR

| stage | what changed | headline metric |
|---|---|---|
| **06J**   | open-design Mode B Sobol exploration (5 000 candidates, 100-row top-list) | J_1453 wins Mode B strict_score (#1) |
| **06J-B** | FD verification of Mode B at scale (~2.9 k FD rows) | J_1453 FD MC strict_pass = 0.747 — matches G_4867 (0.68) |
| **06L**   | direct strict_score head | Mode B strict_score Spearman vs FD MC = +0.585 |
| **06M**   | G_4867 deterministic time deep MC (1 100 FD) | strict_pass_width 3 s |
| **06M-B** | J_1453 deterministic time deep MC + Gaussian time-smearing | strict_pass_width 5 s — wider than G_4867 |
| **06P**   | AL refresh fold-in of 06J-B + 06M-B FD | Mode B Spearman → +0.938; J_1453 in Mode B top-10 |
| **06Q**   | blindspot diagnosis | gap = G_4867 over-prediction at extreme offsets (residual mean +0.215); J_1453 already calibrated |
| **06R**   | feature-engineered surrogate (raw + 9 derived process-budget features) | relative J − G advantage residual −0.217 → −0.110 (halved) |
| **06P-B** | targeted AL on G_4867 extreme offsets (~702 FD) | residual −0.110 → **−0.005** (inside both 0.10 preferred and 0.05 stretch targets); Mode B top-10 overlap 10/10 |

## Diagnosis (Stage 06Q)

06P had Mode B strict_score Spearman = +0.938 against FD MC truth, but
the per-time-offset relative J − G strict_pass advantage residual was
biased: predicted mean = −0.217 vs FD mean ≈ +0.06. 06Q decomposed the
bias and found:

- J_1453 strict_pass residual mean was already calibrated (≈ −0.002).
- G_4867 strict_pass residual mean was **+0.215** — the model was
  over-predicting G_4867 at extreme time offsets.
- G_4867 CD_error residual at offset = −4 s was **−0.51 nm**: the
  model thought CD stayed close to 15 nm while FD showed it drifting
  away.

The gap was almost entirely about G_4867's CD-time sensitivity, not
about J_1453.

## Corrections

### Stage 06R — feature engineering (no new FD)

Added 9 derived process-budget features computed from the 11 raw
features:

```text
diffusion_length_nm = sqrt(2 * DH_nm2_s * time_s)
reaction_budget     = Hmax_mol_dm3 * kdep_s_inv * time_s
quencher_budget     = Q0_mol_dm3 * kq_s_inv * time_s
blur_to_pitch       = sigma_nm / pitch_nm
line_cd_nm_derived  = pitch_nm * line_cd_ratio
DH_x_time           = DH_nm2_s * time_s
Hmax_kdep_ratio     = Hmax_mol_dm3 / kdep_s_inv
Q0_kq_ratio         = Q0_mol_dm3 / kq_s_inv
dose_x_blur         = dose_mJ_cm2 * sigma_nm
```

The 4-head feature-engineered stack (06R) cut the relative-advantage
residual by ~50 % and lowered the aux CD_fixed MAE by ~33 % on the
04C 1 074 holdout. Derived features dominated importance: for the
4-target regressor `line_cd_nm_derived` carries 90 % of total
importance; for the strict_score head `blur_to_pitch` is 0.37 and
`line_cd_nm_derived` 0.17.

### Stage 06P-B — targeted AL (~702 new FD runs)

Three sub-phases concentrated on the offsets where 06R still
over-predicted G_4867:

1. **time_densification** — 7 offsets × 50 standard chemistry jitter = 350 FD
2. **residual_max_jitter** — 4 worst offsets (−4, −3, +4, +5) × 50 wider jitter = 200 FD
3. **boundary_candidates** — 4 intermediate offsets × ~38 standard jitter = 152 FD

Wall time ≈ 108 s. Output: `outputs/labels/stage06PB_targeted_fd_rows.csv`.

The refreshed surrogate (06PB) brings the relative J − G advantage
residual to **−0.005** (|.| 0.025) — inside both the **0.10 preferred**
and **0.05 stretch** targets specified for the thread — while leaving
Mode B ranking essentially unchanged (top-10 overlap 10/10).

## Final state — surrogate calibration vs FD MC truth

|  | 06P | 06R | **06PB** |
|---|---|---|---|
| Mode B Spearman vs FD MC strict_pass_prob | +0.938 (0.912 same-script) | +0.925 | **+0.925** |
| relative advantage residual mean | −0.217 | −0.110 | **−0.005** |
| abs relative advantage residual mean | 0.221 | 0.110 | **0.025** |
| G_4867 strict_pass residual mean | +0.215 | +0.109 | **+0.008** |
| J_1453 strict_pass residual mean | −0.002 | −0.001 | **+0.003** |
| G_4867 CD_error residual mean (nm) | −0.268 | −0.164 | **−0.032** |
| J_1453 in Mode B top-10 | True | True | **True** |
| G_4867 in Mode A top-10 | True | True | **True** |

## Final recipe roles

- **`G_4867` — Mode A default / CD-accurate fixed-design recipe.**
  pitch = 24, line_cd_ratio = 0.52, abs_len = 50. FD MC strict_pass =
  0.68, strict_pass_width = 3 s, FD nominal CD_error = 0.003 nm.
  Best overall on FD nominal CD accuracy at the Mode A geometry.

- **`J_1453` — Mode B production alternative / wider time-window.**
  pitch = 24, line_cd_ratio = 0.45, abs_len ≈ 90. FD MC strict_pass =
  0.747, strict_pass_width = 5 s. Open-design (different pitch / ratio /
  abs_len from the Mode A template); choose when wider time tolerance
  matters more than strict CD accuracy at nominal time.

Full recipe parameters, FD nominal / MC / time-window metrics, and
trade-off notes are in
`outputs/yield_optimization/stage06PB_final_recipe_manifest.yaml`.
The preferred-surrogate registry sits at
`outputs/models/preferred_surrogate.json`.

## Closed threads

- 06Q blindspot diagnosis
- 06R feature-engineered refresh
- 06PB targeted AL refresh
- G_4867 / J_1453 magnitude-gap thread

No further FD sweep is planned under this thread. Run new FD only if a
new objective family or a new failure mode appears that the current
06PB surrogate cannot already screen.

## What's next (out of scope for this thread)

The closeout deliberately does not pick a successor. Possible future
directions are listed in the README "Optional follow-ups" section
(autoencoder / inverse fit, deeper Mode B process-window map around
J_1453, anomaly detection, etc.). External calibration would require
real published or measured data, which the policy still excludes.
