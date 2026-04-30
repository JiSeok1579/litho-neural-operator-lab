# PEB v2 — High-NA EUV inspired PEB simulator (first-pass closeout)

## Project goal (one paragraph)

Reframe the v1 PEB sandbox as a process-oriented simulator for **High-NA EUV line/space PEB**. Add real geometry (pitch / line-CD / edge roughness), Dill-style acid generation with electron blur, an operator-split spectral PEB solver in 2D x-y, a Neumann-z mirror-FFT solver in 2D x-z, weak-quencher chemistry, and CD-locked LER + PSD band metrics. Map out the chemistry / process knobs around an internally-consistent nominal operating point and quantify their dynamic range so that, when external published or measured CD / LER / process-window data lands, calibration is mechanical.

---

## Frozen v2 nominal OP

```yaml
pitch_nm:        24
line_cd_nm:      12.5
domain_x_nm:     120          # = pitch * 5 (FFT-seam-safe)
domain_y_nm:     120

dose_mJ_cm2:                40
reference_dose_mJ_cm2:      40
electron_blur_sigma_nm:      2
Hmax_mol_dm3:                0.20
eta:                         1.0

time_s:                     30
DH_nm2_s:                    0.5
kdep_s_inv:                  0.5
kloss_s_inv:                 0.005

quencher.enabled:           true
Q0_mol_dm3:                  0.02
kq_s_inv:                    1.0
DQ_nm2_s:                    0.0

P_threshold:                 0.5
```

This OP is the **frozen** nominal. Authoritative copy lives in [`calibration/calibration_targets.yaml > frozen_nominal_OP`](./calibration/calibration_targets.yaml). Any deviation must be labelled sensitivity / controllability / hypothesis study; "calibration" or "calibrated to real" wording is reserved for after `calibration_status.published_data_loaded` flips to `true`.

---

## Status (first-pass)

### Completed (✅)

| Stage | Topic | Notes |
|---|---|---|
| **1**   | clean geometry baseline | σ=0, t=30 contour-positive baseline at pitch=24 |
| **1A**  | σ-compatible budget calibration | σ ∈ [0, 3] usable at kdep=0.5 / Hmax≤0.2 |
| **1B**  | over-budget reference | σ=5/t=60 lines merge — kept as stress reference |
| **2**   | DH × time process window (25 cells) | algorithmic-best vs robust-alt; selected DH=0.5/t=30 robust alt |
| **3**   | electron blur separation | 3-stage LER measurement (design / e-blur / PEB) |
| **4**   | weak quencher (52 cells) | balanced OP Q0=0.02, kq=1.0; σ=3 LER recovery +29 pp |
| **4B**  | CD-locked LER + Stage 5B | one displacement-artifact case isolated; pitch ≤ 20 worsening is real, not artifact |
| **5**   | pitch × dose process window (108 cells) | pitch=16 closed, pitch ≥ 24 wide |
| **6**   | x-z standing wave (12 cells) | PEB modulation absorption 79–60 % thin → thick |
| **Cal Phase 1** | Hmax × kdep × DH internal-consistency check | gate PASS, **internal-only** |
| **Cal Phase 2A** | OAT controllability sweep | dose×σ×abs_len×DH spans recorded |
| **Cal Phase 2B** | sensitivity atlas (xy 141 + xz 144 + smallpitch 144) | absorption_length dominant z-knob; σ↓ reopens small pitch |

### Deferred (⏸)

| Stage | Reason |
|---|---|
| **3B** σ=5/8 compatibility | needs dose / kdep / Hmax search-space expansion; only triggered if downstream needs σ ≥ 5 |
| **5C** extended small-pitch work | Phase 2B Part C already characterised the σ-knob behaviour; further tuning waits for a target-specific need |
| **6B** full 3D x-y-z | needs external reference or a concrete y-roughness × z-modulation interaction hypothesis |

---

## Claims boundary (read before citing any number)

```text
✓ The model is internally consistent and physically plausible.
✓ Bounds, conservation, and sensitivity behaviour all match expectation
  for the equations and BCs implemented.
✓ The sensitivity atlas (Phase 2B) provides a controllability map that
  is ready to drive future calibration once external data lands.

✗ The model is NOT externally calibrated.
✗ Quantitative agreement with High-NA EUV measurements is NOT claimed.
✗ Absolute CD, LER, process-window-shape numbers must be treated as
  internally consistent values, not as predictions of any specific
  resist / scanner / dose.
```

The internal calibration targets (CD ≈ 15 nm, LER ≈ 2.6 nm) come from v2's own first-pass observations, so the Phase 1 PASS is an internal-consistency check, not a real calibration. To convert this into externally-calibrated work, see [`FUTURE_WORK.md`](./FUTURE_WORK.md) Gate A.

---

## Where things live

```text
reaction_diffusion_peb_v2_high_na/
├── README.md                       # this file
├── EXPERIMENT_PLAN.md               # full per-stage spec + verified results
├── STUDY_SUMMARY.md                 # first-pass closeout narrative
├── RESULTS_INDEX.md                 # per-stage table → folder / CSV / fig dir / 1-line conclusion
├── FUTURE_WORK.md                   # gated future work (A external / B physics / C small-pitch / D ref-search)
│
├── calibration/
│   ├── README.md
│   ├── calibration_targets.yaml     # frozen_nominal_OP, calibration_status, internal_best_score_candidate
│   └── calibration_plan.md          # Phase 1 / 2A / 2B log + decisions
│
├── configs/                         # YAML configs per stage / phase
├── src/                             # geometry, exposure, x-y solver, x-z solver, edge metrics, viz
├── experiments/                     # per-stage sweep runners
├── tests/                           # pytest unit tests (27 / 27 passing)
├── outputs/
│   ├── figures/                     # ~600 figures across stages + calibration
│   └── logs/                        # CSV + JSON metric summaries (all sweep cells)
├── study_notes/
│   └── 0[1-7]_*.md                  # per-stage journals (problem / decision / result / next)
└── calibration/experiments/         # cal01_, cal02a_, cal03_, cal04_, cal05_ sweep runners
```

---

## Reproducing the nominal OP

```bash
# 1. activate the project venv (see top-level README install steps)
source .venv/bin/activate

# 2. nominal Stage 1 baseline (single PEB run at v2 OP)
python -m reaction_diffusion_peb_v2_high_na.experiments.01_lspace_baseline.run_baseline_no_quencher \
    --config reaction_diffusion_peb_v2_high_na/configs/v2_stage1_clean_geometry.yaml

# 3. complete sensitivity atlas (≈ 6 minutes wall-clock)
python -m reaction_diffusion_peb_v2_high_na.calibration.experiments.cal03_atlas_xy.run_cal03 \
    --config reaction_diffusion_peb_v2_high_na/calibration/configs/cal03_atlas_xy.yaml
python -m reaction_diffusion_peb_v2_high_na.calibration.experiments.cal04_atlas_xz.run_cal04 \
    --config reaction_diffusion_peb_v2_high_na/calibration/configs/cal04_atlas_xz.yaml
python -m reaction_diffusion_peb_v2_high_na.calibration.experiments.cal05_smallpitch.run_cal05 \
    --config reaction_diffusion_peb_v2_high_na/calibration/configs/cal05_smallpitch.yaml

# 4. tests
pytest reaction_diffusion_peb_v2_high_na/tests -q
```

Per-stage runs are documented in each phase entry in [`EXPERIMENT_PLAN.md`](./EXPERIMENT_PLAN.md) and the corresponding [`study_notes/`](./study_notes/) journal.
