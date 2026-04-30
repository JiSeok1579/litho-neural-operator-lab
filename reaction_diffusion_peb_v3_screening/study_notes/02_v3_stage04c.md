# v3 Stage 04C — `roughness_degraded` expansion + per-class regression acceptance

## TL;DR

Stage 04B left `roughness_degraded` with **3** examples and the global
LER MAE acceptance failing because the `merged` class drove the noise.
Stage 04C addresses both:

- a 3-OR-trigger label criterion grew `roughness_degraded` from 3 to
  **321** examples (re-labelling pulled in 184 from the seed; a
  focused 1 500-FD batch added 134 more);
- per-class regression acceptance replaces the global LER MAE band so
  `merged` and `roughness_degraded` get their own budgets;
- balanced accuracy lands at **0.934** and macro-F1 at **0.949** on a
  six-class held-out split (`roughness_degraded` recall **0.857**,
  precision **0.982**).

This stage is **screening / hypothesis testing on the v2 nominal
model** — not external calibration. `v2_OP_frozen` stays `true`,
`published_data_loaded` stays `false`.

## Goal

```text
1. Push roughness_degraded from 3 → ≥ 100 examples.
2. Lift roughness_degraded recall to ≥ 0.50 without regressing the
   other classes by more than 0.03 macro-F1.
3. Replace the global LER MAE acceptance with per-class
   bands so merged-class CD-lock noise no longer dominates the verdict.
4. Hold every policy flag — v2 OP is frozen; v3 is screening, not
   calibration.
```

## Steps taken

1. **Refined label** (`configs/label_schema.yaml`):
   - precedence reordered to `numerical_invalid > under_exposed > merged
     > roughness_degraded > margin_risk > robust_valid` — a failed-to-form
     line wins over a merged-line diagnosis when both signals fire;
   - `roughness_degraded` fires when **any** of three triggers hold
     (the cell must already pass bounds + interior gate):
     - `LER_CD_locked_nm > 3.0` nm                       (absolute)
     - LER excess vs design `> 5 %`                      (relative)
     - PSD mid-band excess vs design `> 20 %`            (PSD-domain).

2. **Roughness-focused sampler**
   (`configs/failure_seeking.yaml > target_roughness_degraded_v2`):
   the user-spec value lists for pitch / line_cd_ratio / dose / σ / DH
   / time / Q0 / kq, plus `prefilter_bypass: true` so near-failure
   roughness candidates survive (the standard prefilter would
   eliminate them).

3. **Re-label seed** then **Stage 04C runner** (`run_04c.py`):
   - re-label every prior CSV (Stage 02, 04, 04B-iter1, 04B-iter2)
     under the new criteria;
   - sample 3 000 candidates from `target_roughness_degraded_v2`;
   - bypass the prefilter (the bias preset declares it);
   - FD on the first 1 500 candidates;
   - retrain the RF classifier + regressor;
   - save `stage04C_classifier.joblib`,
     `stage04C_regressor.joblib`,
     `stage04C_training_dataset.csv`,
     `stage04C_summary.json`,
     plus confusion / reliability / per-class MAE figures.

4. **Per-class regression acceptance**:
   the runner reports `LER_CD_locked` MAE both as a global number
   **and** as `non-merged` / `merged-only` aggregates so the
   merged-class CD-lock noise no longer hides good behaviour on
   robust / margin / under_exposed cells.

## Problems and resolutions

### Problem 1 — `roughness_degraded` was barely reachable under the old criterion

The Stage 04B criterion (`LER_locked − LER_design > 1.5 nm`) is
absolute, so it only fires when the side-wall LER is already very
bad — which usually coincides with `merged` and gets the precedence
loss. Three out of 3 868 cells crossed the bar.

**Resolution**: switch to the three-OR-trigger criterion. The
relative trigger (`> 5 %`) and the PSD-domain trigger (`> 20 %`) fire
at much smaller excursions, which is what we want for "side-wall
roughness has a real signal" without requiring catastrophic line
collapse.

The re-label diff is informative:

```text
seed re-label (Stage 04B labels → Stage 04C labels):
  robust_valid       1 727 →  1 640  (-87)
  margin_risk          646 →    549  (-97)
  roughness_degraded     3 →    187  (+184)
  merged               604 →    602   (-2)
  under_exposed        675 →    677   (+2)
  numerical_invalid    213 →    213    (0)
```

184 cells moved out of clean labels and into `roughness_degraded`
purely because the criterion is now relative + PSD-aware.

### Problem 2 — global LER MAE acceptance kept tripping

By Stage 04B the global LER MAE was 0.038, just above the 0.03 band.
The per-class breakdown showed the FAIL was entirely the `merged`
class (CD-lock cannot converge cleanly when there is no inter-line
contour). The aggregate hides good behaviour on the dominant classes.

**Resolution**: replace the single-number band with three bands:

```text
CD_locked         global       ≤ 0.15 nm
LER_CD_locked     non-merged   ≤ 0.03 nm
LER_CD_locked     merged-only  ≤ 0.15 nm
```

Per-class numbers below show that the non-merged aggregate **still**
fails the 0.03 band (lands at 0.041), but the cause is now
`roughness_degraded` (0.071 MAE) and `under_exposed` (0.053). Both
classes are intrinsically noisy: a roughness-bad cell has a side-wall
that wanders by ~ 1 % of the line CD, and an under-exposed cell has
a contour that crosses a fast-varying P-field. The `robust_valid` MAE
is **0.026** (inside) and `margin_risk` is **0.015** (well inside).

A fairer "operational-zone" MAE would average only `robust_valid` and
`margin_risk` — that lands at ~0.020 nm and is what the screening
loop actually cares about. We expose this in the JSON summary; the
README and acceptance tables flag the FAIL but explain it.

### Problem 3 — bias preset wanted ranges outside the base candidate space

The user specified `line_cd_ratio = [0.50, 0.55, 0.60, 0.65]`, but the
base `candidate_space.yaml` only carried `[0.45, 0.52, 0.60]`. Same
issue for `DH_nm2_s = [..., 1.0]` outside the base `[0.3, 0.8]`.

**Resolution**: the bias preset declares the values explicitly with
`values: [...]` (uniform → choice conversion) or `choice: [...]`
(choice override). The sampler honours both and runs FD straight
through — the v2 helper does not enforce a parameter range, only the
documented OP convention.

## Decision log

| decision | choice | reason |
|---|---|---|
| roughness_degraded triggers | 3 OR (LER abs, LER rel, PSD rel) | absolute alone is too rare; PSD trigger captures band-specific roughness without requiring catastrophic LER |
| precedence reorder | under_exposed > merged | when both signals fire, "lines didn't form" is the more fundamental diagnosis |
| prefilter bypass for target_roughness_degraded_v2 | yes | the prefilter was rejecting near-failure candidates by design — bypassing keeps roughness cases in the FD batch |
| candidate batch size | 3 000 sampled, 1 500 FD | hit-rate on the bias preset is ~10 % rough → ≥ 100 with safety margin |
| re-label all prior rows | yes | keeps the training set internally consistent under the new criteria; old rows have FD outputs available so labelling is just a function call |
| `LER_CD_locked` acceptance | per-class split | global MAE conflates physically-noisy classes with the operational zone; per-class is honest |
| store `roughness_trigger` | yes | downstream runs need to know **which** trigger fired so we can investigate per-trigger precision |
| feature importance reporting | single shared RF | per-class importance would need shap or leave-one-out; defer to Stage 05 |

## Verified results

```text
seed + 04C cumulative dataset       : 5 368 rows
roughness_degraded total            : 321        (was 3)
balanced accuracy on held-out 80/20 : 0.934
macro F1                            : 0.949    (pre-04C: 0.968 — drop 0.019)

per-class P / R / F1 / support:
  robust_valid        0.946 / 0.985 / 0.965 / 340
  margin_risk         0.949 / 0.862 / 0.903 / 130
  roughness_degraded  0.982 / 0.857 / 0.915 /  63
  under_exposed       0.952 / 0.973 / 0.962 / 183
  merged              0.978 / 0.994 / 0.986 / 314
  numerical_invalid   1.000 / 0.932 / 0.965 /  44

regressor:
  CD_locked_nm        global MAE 0.104   non-merged 0.078   merged 0.188
  LER_CD_locked_nm    global MAE 0.057   non-merged 0.041   merged 0.119
  area_frac           global MAE 0.034
  P_line_margin       global MAE 0.021

acceptance:
  CD_locked global MAE ≤ 0.15        PASS  (0.104)
  LER_CD_locked non-merged ≤ 0.03    FAIL  (0.041 — driven by rough+under)
  LER_CD_locked merged ≤ 0.15        PASS  (0.119)
  macro-F1 drop ≤ 0.03               PASS  (-0.019)
  balanced accuracy ≥ 0.93           PASS  (0.934)
  roughness_degraded count ≥ 100     PASS  (321)
  roughness_degraded recall ≥ 0.50   PASS  (0.857)
  v2_OP_frozen unchanged             PASS  (true)
  published_data_loaded unchanged    PASS  (false)
```

## Follow-up work

- **Stage 04D — operational-zone MAE band**: report
  `LER_CD_locked` MAE on the `robust_valid + margin_risk` subset
  alongside the per-class split. The 0.03 band makes sense there
  (current value ~0.020).
- **Per-trigger precision**: `roughness_trigger` is now stored on every
  row. A small analysis script can compute precision per trigger
  (LER-abs vs LER-rel vs PSD) — useful before changing thresholds again.
- **Calibration**: at this dataset size the reliability diagram is
  smooth enough to motivate a probability-calibration step (Platt or
  isotonic). Defer to Stage 05.
- **PINN hold-out**: still deferred. v3 surrogates are fast enough to
  drive any closed-loop search; bring PINN in only if Stage 05
  (autoencoder / inverse fit) needs surrogate-free truth.
- **Policy line**: `published_data_loaded` is still `false`. Even with
  recall 0.857 on `roughness_degraded`, v3 must not be referred to as
  calibration.
