# PEB v2 — future work (gated)

Future work is split into four **gates**. A gate must be opened (its required
artefact loaded / hypothesis specified) before any item below it is started.
Until the relevant gate opens, the v2 OP is frozen and the work below is
**out of scope**.

```text
A. External calibration gate    → required for any "calibrated to real" work
B. Physics extension gate       → only after A or after A is intentionally bypassed
C. Small-pitch hypothesis gate  → low-risk, target-specific only
D. External-reference search gate → housekeeping; populates A
```

---

## Gate A — external calibration

**Status**: closed.
`calibration_status.published_data_loaded` = `false`.
`calibration_status.v2_OP_frozen` = `true`.

### Required reference data

To open Gate A, [`calibration/calibration_targets.yaml`](./calibration/calibration_targets.yaml) must be replaced with externally-sourced values for at least:

```text
- pitch [nm]
- dose [mJ/cm²]
- final CD [nm]
- LER and/or LWR [nm]
- film thickness [nm]
- PEB temperature [°C] and time [s]
- resist type / chemistry family
```

Per-condition tolerances must come from the source (replace the placeholder
`tolerance_nm: 0.5` and `tolerance_nm: 0.3` defaults).

### Action when external data lands

1. Update `calibration/calibration_targets.yaml`:
   - flip `calibration_status.published_data_loaded` to `true`,
   - replace target values + tolerances,
   - record `reference.source` and any DOI / URL.
2. Re-run **Calibration Phase 1** (`cal01_hmax_kdep_dh`) to identify offset.
3. Re-run **Calibration Phase 2A** (`cal02a_sensitivity`) and
   **Phase 2B** (`cal03_atlas_xy`, `cal04_atlas_xz`, `cal05_smallpitch`) to
   verify the controllability map still holds at the new chemistry.
4. If a new operating point matches the external targets within tolerance,
   replace `frozen_nominal_OP` and re-flag `v2_OP_frozen` after a fresh
   freeze decision.
5. Only after step 4 is the model allowed to be referred to as
   "externally calibrated".

### Out of scope until A opens

- any rebranding of the current OP as "matched to literature" / "matched to measurement"
- any tuning that targets numbers from outside `calibration_targets.yaml`

---

## Gate B — physics extension

**Status**: closed. Open only after Gate A or after explicit decision to
develop physics independently of external calibration.

Items, in order of compute / scope cost:

```text
1. development model beyond P-threshold
   - real dissolution rate model (e.g., Mack 1992)
   - currently P_threshold contour is only a proxy

2. Stage 6B — full 3D x-y-z PEB
   - couples Stage 6 z-direction with Stage 1-5 y-roughness
   - 3D FFT solver, mirror-z extension as in fd_solver_xz
   - cost ~1 min/run, sweeps would explode

3. Dill ABC / depth-dependent acid generation
   - currently dose_norm * I + uniform η; Dill A,B,C separates exposure-induced
     bleach + absorption depth profile
   - prerequisite for accurate film thickness scans

4. multi-component PAG / quencher
   - currently single H + single Q; real resists carry multiple PAGs and
     quencher families
   - prerequisite for resist-family comparisons

5. arrhenius-coupled DH and kdep
   - currently DH and kdep are temperature-independent constants; PEB
     temperature only labels the run
   - cost trivial (constants → temperature-dependent expressions)
```

Each item must come with a hypothesis that names what new behaviour is
being unlocked, otherwise the v2 OP is sufficient.

---

## Gate C — small-pitch hypothesis

**Status**: open by Phase 2B Part C. Conclusion already documented; work
beyond the documentation is **target-specific only**.

### Phase 2B finding (recap)

```text
σ reduction is the dominant small-pitch process-window recovery knob.
  pitch=18, σ=0:           3 robust_valid / 12 cells
  pitch=18, σ=2 (v2 OP):   0–1 robust_valid / 12 cells
  pitch=20, σ=0/1:         4 robust_valid / 12 cells
  pitch=20, σ=2:           2 robust_valid / 12 cells

Quencher weakening has a small or negative effect on small pitch.
Best small-pitch cells: dose=28.4, σ=0, DH=0.3, t=30, weak quencher
                        → robust_valid (CD_shift ±0.25, LER_lock ≈ 3 nm)
```

### When to follow up

Open work in this gate **only** when there is a downstream task that
specifically requires pitch ≤ 20:

```text
- a target reference dataset includes pitch=18 or 20
- a Stage 6B 3D run needs a small-pitch configuration
- a closed-loop / inverse-design task targets sub-20 nm pitch
```

Otherwise the σ-reduction finding documented in [`calibration/calibration_plan.md`](./calibration/calibration_plan.md) Phase 2B is sufficient and no further runs are needed.

---

## Gate D — external-reference search

**Status**: open as housekeeping. Populates Gate A.

### Where to look

```text
- High-NA EUV PEB measurement papers (resist developers / industry consortia)
- 24 nm pitch / 12.5 nm CD line-space measurement reports
- standing-wave / film-thickness studies for EUV resists
- public datasets (if any) on resist CD vs PEB time / temperature
```

### Search keywords

```text
"high-NA EUV PEB" "post-exposure bake" "line edge roughness"
"acid diffusion length" "24 nm pitch line space"
"standing wave amplitude EUV resist"
"PAG quencher reaction-diffusion EUV"
```

### What to record

When a candidate reference is found, record under
`calibration/calibration_targets.yaml`:

```yaml
reference:
  source: "<author, year, DOI>"
  published_data_loaded: true   # only after numeric values copied in
  last_updated: "<ISO date>"
```

…and replace the placeholder `targets.cd_nm` / `targets.ler_nm` blocks
with the source's values + tolerances. Then proceed to Gate A actions.

---

## Non-goals (explicit)

- Rebranding current internal-consistency results as externally calibrated.
- Adding new chemistry or physics without a specific hypothesis.
- Closing pitch=16 process window without a target-specific need (it is
  closed at the current line_cd=12.5 / chemistry; that is a real lesson, not
  a bug).
- Introducing PINN / FNO surrogates back into v2 — this is decided in v1
  and documented in v2 plan §1.5; lift only after Gate A is open and
  there is a concrete acceleration target.
