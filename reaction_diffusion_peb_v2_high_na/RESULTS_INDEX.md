# PEB v2 — results index

Per-stage / per-phase pointer table → folder, main CSV, main figure
directory, one-line conclusion. All numbers are **internal-consistency
values**, not externally-calibrated predictions.

For full narrative see [`STUDY_SUMMARY.md`](./STUDY_SUMMARY.md). For per-stage
problem / decision journals see [`study_notes/`](./study_notes/).

---

## Core stages (first-pass)

| Stage | Folder | Main CSV | Main figures | One-line conclusion |
|---|---|---|---|---|
| **1** clean baseline | [`experiments/01_lspace_baseline/`](./experiments/01_lspace_baseline/) | [`outputs/logs/01_clean_geometry.csv`](./outputs/logs/01_clean_geometry.csv) | [`outputs/figures/01_clean_geometry/`](./outputs/figures/01_clean_geometry/) | σ=0 / t=30 baseline at pitch=24 produces CD ≈ 15, LER ≈ 2.65, all interior gates pass. |
| **1A** σ-budget calibration | [`experiments/01_lspace_baseline/`](./experiments/01_lspace_baseline/) (σ sweep + σ=5 search) | [`outputs/logs/01_sigma_sweep_t30.csv`](./outputs/logs/01_sigma_sweep_t30.csv), [`outputs/logs/01_calibration_sigma5_stage{A,B}.csv`](./outputs/logs/) | [`outputs/figures/01_sigma_sweep_t30/`](./outputs/figures/01_sigma_sweep_t30/), [`outputs/figures/01_calibration_sigma5_stage{A,B}/`](./outputs/figures/) | σ ∈ [0, 3] usable at kdep=0.5 / Hmax≤0.2; σ=5 has no compatible budget in the spec range. |
| **1B** over-budget reference | [`experiments/01_lspace_baseline/`](./experiments/01_lspace_baseline/) (overbudget config) | [`outputs/logs/01_lspace_baseline.csv`](./outputs/logs/01_lspace_baseline.csv) | [`outputs/figures/01_lspace_baseline/`](./outputs/figures/01_lspace_baseline/) | σ=5 / t=60 collapses all lines into a slab — kept as a stress-test reference. |
| **2** DH × time window | [`experiments/02_dh_time_sweep/`](./experiments/02_dh_time_sweep/) | [`outputs/logs/02_dh_time_sweep.csv`](./outputs/logs/02_dh_time_sweep.csv) | [`outputs/figures/02_dh_time_sweep/`](./outputs/figures/02_dh_time_sweep/) | 25-cell process window; algorithmic best DH=0.8/t=20 (margin 0.003) demoted in favour of robust DH=0.5/t=30 (margin 0.142). |
| **3** electron blur separation | [`experiments/03_electron_blur/`](./experiments/03_electron_blur/) | [`outputs/logs/03_electron_blur.csv`](./outputs/logs/03_electron_blur.csv) | [`outputs/figures/03_electron_blur/`](./outputs/figures/03_electron_blur/) | 3-stage LER measurement (design / e-blur / PEB); σ ↑ ⇒ e-blur LER reduction up but PEB LER reduction goes negative — needs CD-lock to disentangle. |
| **4** weak quencher | [`experiments/04_weak_quencher/`](./experiments/04_weak_quencher/) | [`outputs/logs/04_weak_quencher_summary.csv`](./outputs/logs/04_weak_quencher_summary.csv) | [`outputs/figures/04_weak_quencher_summary/`](./outputs/figures/04_weak_quencher_summary/), [`outputs/figures/04_weak_quencher_sigma2/`](./outputs/figures/04_weak_quencher_sigma2/) | 52 / 52 cells gate-pass; 51 / 52 robust; balanced OP Q0=0.02, kq=1; σ=3 LER recovery +29 pp. |
| **4B** CD-locked LER + 5B mini-sweep | [`experiments/04b_cd_locked/`](./experiments/04b_cd_locked/) | [`outputs/logs/04b_cd_locked_block_a.csv`](./outputs/logs/04b_cd_locked_block_a.csv), [`outputs/logs/04b_cd_locked_block_b.csv`](./outputs/logs/04b_cd_locked_block_b.csv) | [`outputs/figures/04b_cd_locked_block_a/`](./outputs/figures/04b_cd_locked_block_a/), [`outputs/figures/04b_cd_locked_block_b/`](./outputs/figures/04b_cd_locked_block_b/) | 1 displacement-artifact case isolated; pitch ≤ 20 LER worsening is **real**, not artifact; quencher tuning cannot recover it. |
| **5** pitch × dose process window | [`experiments/05_pitch_dose/`](./experiments/05_pitch_dose/) | [`outputs/logs/05_pitch_dose_summary.csv`](./outputs/logs/05_pitch_dose_summary.csv), [`outputs/logs/05_pitch_dose_recommendation.json`](./outputs/logs/05_pitch_dose_recommendation.json) | [`outputs/figures/05_pitch_dose/`](./outputs/figures/05_pitch_dose/) | 108-cell window; pitch=16 closed; dose=40 robust at pitch ≥ 24; quencher narrows small-pitch tolerance. |
| **6** x-z standing wave | [`experiments/06_xz_standing_wave/`](./experiments/06_xz_standing_wave/) | [`outputs/logs/06_xz_standing_wave_summary.csv`](./outputs/logs/06_xz_standing_wave_summary.csv) | [`outputs/figures/06_xz_standing_wave/`](./outputs/figures/06_xz_standing_wave/), [`outputs/figures/06_xz_standing_wave/summary/`](./outputs/figures/06_xz_standing_wave/summary/) | PEB absorbs standing-wave amplitude; 79 % at thick=15 → 60 % at thick=30; sidewall LER 1.32 → 3.87 nm thin → thick. |

---

## Calibration phases (all internal-only)

| Phase | Folder | Main CSV | Main figures | One-line conclusion |
|---|---|---|---|---|
| **Cal 1** Hmax × kdep × DH | [`calibration/experiments/cal01_hmax_kdep_dh/`](./calibration/experiments/cal01_hmax_kdep_dh/) | [`outputs/logs/cal01_hmax_kdep_dh_summary.csv`](./outputs/logs/cal01_hmax_kdep_dh_summary.csv) | [`outputs/figures/cal01_hmax_kdep_dh/`](./outputs/figures/cal01_hmax_kdep_dh/) | Internal-consistency PASS (best 0.0054). v2 OP at 0.0588 also PASS. DH=0.80 candidate kept on record only. |
| **Cal 2A** OAT controllability | [`calibration/experiments/cal02a_sensitivity/`](./calibration/experiments/cal02a_sensitivity/) | [`outputs/logs/cal02a_sensitivity_rows.csv`](./outputs/logs/cal02a_sensitivity_rows.csv), [`outputs/logs/cal02a_sensitivity_sensitivity.csv`](./outputs/logs/cal02a_sensitivity_sensitivity.csv) | [`outputs/figures/cal02a_sensitivity/`](./outputs/figures/cal02a_sensitivity/) | Dose dominates x-y CD/window; absorption_length dominates every x-z metric (484 % PSD-mid range). |
| **Cal 2B Part A** x-y atlas | [`calibration/experiments/cal03_atlas_xy/`](./calibration/experiments/cal03_atlas_xy/) | [`outputs/logs/cal03_atlas_xy_summary.csv`](./outputs/logs/cal03_atlas_xy_summary.csv) | [`outputs/figures/cal03_atlas_xy/`](./outputs/figures/cal03_atlas_xy/) | 141 cells (5 OAT + 5 pair sweeps). Reproduces Stage 1-5 trends and isolates Q0-margin trade-off. |
| **Cal 2B Part B** x-z atlas | [`calibration/experiments/cal04_atlas_xz/`](./calibration/experiments/cal04_atlas_xz/) | [`outputs/logs/cal04_atlas_xz_summary.csv`](./outputs/logs/cal04_atlas_xz_summary.csv) | [`outputs/figures/cal04_atlas_xz/`](./outputs/figures/cal04_atlas_xz/) | 144 / 144 bounds_ok 4-D grid. abs_len is the dominant z-knob; DH near-silent on standing-wave amplitude. |
| **Cal 2B Part C** small-pitch hypothesis | [`calibration/experiments/cal05_smallpitch/`](./calibration/experiments/cal05_smallpitch/) | [`outputs/logs/cal05_smallpitch_summary.csv`](./outputs/logs/cal05_smallpitch_summary.csv), [`outputs/logs/cal05_smallpitch_status_counts.json`](./outputs/logs/cal05_smallpitch_status_counts.json) | (counts in JSON; per-row CSV) | σ ↓ is the dominant small-pitch recovery knob (pitch=18 σ=0 → 3 robust vs σ=2 → 0). Quencher weakening has small or negative effect. |

---

## Per-stage study notes (problem / decision / result)

| | |
|---|---|
| Stage 1 + 1A | [`study_notes/01_stage1_clean_geometry.md`](./study_notes/01_stage1_clean_geometry.md) |
| Stage 2 | [`study_notes/02_stage2_dh_time_sweep.md`](./study_notes/02_stage2_dh_time_sweep.md) |
| Stage 3 | [`study_notes/03_stage3_electron_blur.md`](./study_notes/03_stage3_electron_blur.md) |
| Stage 4 | [`study_notes/04_stage4_weak_quencher.md`](./study_notes/04_stage4_weak_quencher.md) |
| Stage 5 | [`study_notes/05_stage5_pitch_dose.md`](./study_notes/05_stage5_pitch_dose.md) |
| Stage 4B + 5B | [`study_notes/06_stage4B_cd_locked.md`](./study_notes/06_stage4B_cd_locked.md) |
| Stage 6 + helper integration | [`study_notes/07_stage6_xz_standing_wave.md`](./study_notes/07_stage6_xz_standing_wave.md) |
| Calibration Phase 1 / 2A / 2B | [`calibration/calibration_plan.md`](./calibration/calibration_plan.md) |

---

## How to read a sweep CSV

Most calibration / sensitivity CSVs share a common subset of columns
(produced by `experiments/run_sigma_sweep_helpers.py`):

```text
status                           Stage-5 classifier (robust_valid / valid /
                                  under_exposed / merged / low_contrast / unstable)
P_space_center_mean              mean of P along an inter-line strip
P_line_center_mean               mean of P along a line-center strip
P_line_margin                    P_line_center_mean - 0.65   (Stage 3 gate)
contrast                         P_line_center_mean - P_space_center_mean
area_frac                        fraction of domain with P >= P_threshold
CD_initial_nm / CD_final_nm      design CD vs P=0.5 contour CD (= CD_fixed)
CD_locked_nm                     CD at the bisected threshold (≈ design CD)
P_threshold_locked               threshold that achieves CD_locked
LER_design_initial_nm            LER on the binary mask (σ-independent)
LER_after_eblur_H0_nm            LER on I_blurred contour
LER_after_PEB_P_nm               LER on P=0.5 contour (= LER_fixed)
LER_CD_locked_nm                 LER at the CD-locked threshold (intrinsic LER)
psd_locked_low/mid/high          PSD band powers of the CD-locked edge tracks
psd_mid_band_reduction_pct       100*(psd_design_mid - psd_PEB_mid)/psd_design_mid
total_LER_reduction_pct          fixed-threshold variant (Stage 1-4)
total_LER_reduction_locked_pct   CD-locked variant (Stage 4B onward)
```

x-z sweeps additionally report `H0_z_modulation_pct`,
`H0_z_modulation_sw_only_pct` (vs same-thickness A=0 baseline),
`P_final_z_modulation_pct`, `modulation_reduction_pct`,
`top_bottom_asymmetry`, `sidewall_x_displacement_std_nm`.

---

## Single-line x-z cross-sections (xz_companions)

x-y sweep figures (Stage 1–5, calibration atlases) show top-down views of multiple lines. To see the **side-wall depth profile** of one resist line, the renderer in [`experiments/render_xz_companions/`](./experiments/render_xz_companions/) re-runs the chemistry with the x-z solver and crops the field to a single line (middle line ± pitch/2).

For each representative configuration listed in [`configs/xz_companions.yaml`](./configs/xz_companions.yaml), the renderer saves four x-z panels: `I_xz.png`, `H0_xz.png`, `H_final_xz.png`, `P_final_xz.png`. The `P_final_xz.png` overlays the fixed-threshold (red) and CD-locked (white dashed) contours, with cyan dotted vertical lines marking the design line edges.

| Tag | Folder | What it shows |
|---|---|---|
| `stage1_clean_baseline` | [`outputs/figures/xz_companions/stage1_clean_baseline/`](./outputs/figures/xz_companions/stage1_clean_baseline/) | Stage 1 clean geometry (σ=0, t=30) — clean line side-wall, no standing wave |
| `stage1B_overbudget_sigma5_t60` | [`outputs/figures/xz_companions/stage1B_overbudget_sigma5_t60/`](./outputs/figures/xz_companions/stage1B_overbudget_sigma5_t60/) | Stage 1B over-budget (σ=5, t=60) — collapsed line: P uniform across the crop |
| `stage4_balanced_v2_OP` | [`outputs/figures/xz_companions/stage4_balanced_v2_OP/`](./outputs/figures/xz_companions/stage4_balanced_v2_OP/) | Stage 4 / v2 OP balanced — clean side-wall taper |
| `stage5_pitch24_dose40` | [`outputs/figures/xz_companions/stage5_pitch24_dose40/`](./outputs/figures/xz_companions/stage5_pitch24_dose40/) | Stage 5 recommended (pitch=24 dose=40 v2 OP) |
| `stage5_pitch18_dose28p4` | [`outputs/figures/xz_companions/stage5_pitch18_dose28p4/`](./outputs/figures/xz_companions/stage5_pitch18_dose28p4/) | Stage 5 small-pitch challenge — line still resolved, narrow contour |
| `stage5_pitch16_dose40_merged` | [`outputs/figures/xz_companions/stage5_pitch16_dose40_merged/`](./outputs/figures/xz_companions/stage5_pitch16_dose40_merged/) | Stage 5 closed window — P fills the full crop width (lines merged) |
| `stage6_thick20_A0p10_v2_OP` | [`outputs/figures/xz_companions/stage6_thick20_A0p10_v2_OP/`](./outputs/figures/xz_companions/stage6_thick20_A0p10_v2_OP/) | Stage 6 standing wave — H0 shows horizontal stripes; P after PEB mostly smooth |
| `stage6_thick20_A0p20_v2_OP` | [`outputs/figures/xz_companions/stage6_thick20_A0p20_v2_OP/`](./outputs/figures/xz_companions/stage6_thick20_A0p20_v2_OP/) | Stage 6 high-amplitude standing wave |
| `stage6_thick30_A0p20_v2_OP` | [`outputs/figures/xz_companions/stage6_thick30_A0p20_v2_OP/`](./outputs/figures/xz_companions/stage6_thick30_A0p20_v2_OP/) | Stage 6 thick film — top/bottom asymmetry from absorption envelope |
| `cal01_best_score` | [`outputs/figures/xz_companions/cal01_best_score/`](./outputs/figures/xz_companions/cal01_best_score/) | Calibration Phase 1 best score (Hmax=0.20, kdep=0.5, **DH=0.8**) — side-wall comparison vs v2 OP |
| `cal05_smallpitch_best` | [`outputs/figures/xz_companions/cal05_smallpitch_best/`](./outputs/figures/xz_companions/cal05_smallpitch_best/) | Phase 2B Part C small-pitch best (pitch=20, σ=0, DH=0.3, weak quencher) |

To regenerate:

```bash
python -m reaction_diffusion_peb_v2_high_na.experiments.render_xz_companions.run_render \
    --config reaction_diffusion_peb_v2_high_na/configs/xz_companions.yaml
```

Add a new entry to `configs/xz_companions.yaml > cases` to render an additional configuration.

---

## How to read a status heatmap

Status colours used across pair-sweep heatmaps in the atlas:

```text
unstable      black       NaN / Inf / bounds violation
merged        red         lines fused (P_space_mean ≥ 0.5 OR area ≥ 0.9 OR CD/p ≥ 0.85)
under_exposed blue        line center P < 0.65
low_contrast  purple      contrast ≤ 0.15 (rare)
valid         yellow      passes interior gate, margin < 0.05
robust_valid  green       passes interior gate AND P_line_margin ≥ 0.05
```

`robust_valid` cells are the recommended operating zones; `valid` cells
are workable with reduced margin.
