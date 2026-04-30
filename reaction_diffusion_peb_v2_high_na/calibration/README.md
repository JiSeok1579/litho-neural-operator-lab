# PEB v2 calibration

The calibration stage that follows v2 first-pass closeout, comparing against external reference / measured data.
Before adding any new chemistry / sweep, verify that v2 produces reasonable absolute values, classify any offsets that are found, and adjust the appropriate parameters (Hmax, kdep, DH, σ, abs_len, dose) accordingly.

## Policy (2026-04-30 freeze)

```text
External reference data is not available.
calibration_status: internal-consistency only.
published_data_loaded: false.
v2_OP_frozen: true.

frozen_nominal_OP (calibration_targets.yaml > frozen_nominal_OP):
  pitch=24, dose=40, σ=2, DH=0.5, time=30,
  kdep=0.5, Hmax=0.2, kloss=0.005, Q0=0.02, kq=1.0, DQ=0.0

All later sweeps must be labelled sensitivity / controllability / hypothesis study only.
The expressions "calibration" or "calibrated to real" are forbidden until published_data_loaded=true.
```

## Purpose (Phase 1)

```text
1. Does the frozen v2 OP (pitch=24, dose=40, σ=2, t=30, DH=0.5, kdep=0.5,
   Hmax=0.2, Q0=0.02, kq=1.0) match the internal target? (= internal-consistency check)
     CD ≈ 15 nm
     LER ≈ 2.5 ~ 2.7 nm
2. Does the Stage 5 process-window shape match the internal target?
3. If there is an offset, in which mechanism does it originate?
   - acid generation (Hmax)
   - reaction rate (kdep)
   - acid diffusion (DH)
   - electron blur (σ)
   - z absorption (abs_len)

Every answer in this phase concerns internal values that are independent of any
external reference.
External calibration is only possible once Gate A (FUTURE_WORK.md) has opened.
```

## Layout

```text
calibration/
  README.md                         # this file
  calibration_targets.yaml          # CD / LER / process-window targets + tolerance + sources
  calibration_plan.md               # Phase 1-4 plan + decision tree
  configs/
    cal01_hmax_kdep_dh.yaml         # Phase 1 base config
  experiments/
    cal01_hmax_kdep_dh/
      run_cal01.py                  # Phase 1 sweep runner

  # outputs are saved under v2's outputs/ tree with the cal01_* prefix
  outputs/figures/cal01_*           (in: reaction_diffusion_peb_v2_high_na/outputs/figures/)
  outputs/logs/cal01_*              (in: reaction_diffusion_peb_v2_high_na/outputs/logs/)
```

## Run

```bash
python -m reaction_diffusion_peb_v2_high_na.calibration.experiments.cal01_hmax_kdep_dh.run_cal01 \
    --config reaction_diffusion_peb_v2_high_na/calibration/configs/cal01_hmax_kdep_dh.yaml
```

## Per-phase gates

```text
Phase 1 (Hmax × kdep × DH):
  pass condition = at least one cell with "best score < 0.1"
  on failure → Phase 2A (extending dose / σ / abs_len)

Phase 2 (process-window re-verification):
  re-run the Stage 5 pitch × dose grid using the Phase 1 best cell
  pass condition = robust_valid region at pitch=20-32 matches or widens

Phase 3 (external-reference comparison):
  compare the OP up through Phase 2 against published / measured CD / LER / process-window
  pass condition = systematic offset < tolerance

Phase 4 (start of deferred stages):
  Stage 3B, 5C, 6B or new chemistry. Only starts after Phase 3 passes.
```

## Outputs

The results of every Phase are accumulated in `calibration_plan.md` (one-document policy: per-phase history + decision).
