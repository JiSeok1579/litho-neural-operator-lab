# v3 overall study summary

> **Status:** v3 study complete as a personal-study nominal-simulator
> screening / yield-optimization workflow.
> **Preferred surrogate:** `stage06PB`.
> **Mode A primary recipe:** `G_4867`.
> **Mode B primary recipe:** `J_1453`.
> **Policy (unchanged across the whole study):**
> `v2_OP_frozen = true` · `published_data_loaded = false` ·
> `external_calibration = none`.

This note is a single high-level map of the v3 study. It links the
three closeouts (Stage 04C/04D first-pass classifier, Stage 06I Mode
A, Stage 06P-B Mode B) and points downstream readers at the final
manifests. It does not introduce new FD data, new models, or new
acceptance criteria.

## 1. Executive summary

- v3 is a **nominal-model candidate screening and yield-optimization**
  study built on top of the frozen v2 nominal physics. It is a
  *personal-study* workflow, not an external-calibration deliverable.
- Every CD / LER / margin / strict_pass number in the repo comes from
  the v2 nominal model. None is a measurement, none is calibrated to
  real fab yield, and no published process-window dataset has been
  loaded.
- **FD MC `strict_pass_prob` is the final ranking authority.** The
  `stage06PB` surrogate is the screening / candidate-proposal layer.
- The v3 study reaches a documented stopping point with two final
  recipes: `G_4867` (Mode A primary, CD-accurate fixed-design) and
  `J_1453` (Mode B primary, open-design wider-time-window).
- Policy preserved across the whole study: `v2_OP_frozen = true`,
  `published_data_loaded = false`, `external_calibration = none`.
  Closed Stage 04C / 04D / 06C / 06I artefacts were never modified
  after closure.

## 2. Study arc

```text
v3 first-pass screening
    Stage 02   10 000 Sobol -> prefilter -> 1 000 FD -> labels
    Stage 03   surrogate baseline (classifier + 4-target regressor)
    Stage 04   active-learning iteration on 316 uncertain candidates

class balancing and operational-zone evaluation
    Stage 04B  AL expansion + failure-seeking sampling
    Stage 04C  roughness_degraded class expansion + per-class
               regression acceptance (frozen first-pass surrogate)
    Stage 04D  operational-zone evaluation; v3 first-pass CLOSED

Mode A optimization (recipe-only, fixed pitch=24 / ratio=0.52)
    Stage 06A  surrogate-driven nominal-yield-proxy optimisation
    Stage 06B  FD verification + AL additions
    Stage 06C  surrogate refresh on 06B FD additions
    Stage 06D  second-pass surrogate optimisation
    Stage 06E  FD verification of 06D top recipes
    Stage 06F  Pareto ranking after yield_score saturation
    Stage 06G  data-driven strict_score (CD +/-0.5 nm, LER <= 3.0 nm)
               and Mode A re-optimisation
    Stage 06H  FD verification of 06G strict candidates (1,800 MC)
               + surrogate refresh
    Stage 06I  Mode A final manifest; primary recipe = G_4867 (CLOSED)

Mode B exploration and blindspot closure (open-design)
    Stage 06J   open-design Mode B Sobol exploration; J_1453 #1
    Stage 06J-B Mode B FD verification at scale (~2,900 FD rows)
    Stage 06L   direct strict_score head + Mode B AL additions
    Stage 06M   G_4867 deterministic time deep MC (1,100 + 300 FD)
    Stage 06M-B J_1453 deterministic time deep MC + Gaussian-time;
                outcome = j1453_wider_window
    Stage 06N   comparative deep MC (G_4867 vs G_4299) confirms
                G_4867 as Mode A primary
    Stage 06P   AL refresh fold-in of 06J-B + 06M-B FD;
                Mode B Spearman vs FD MC = +0.938
    Stage 06Q   blindspot diagnosis (no retraining); root cause =
                G_4867 over-prediction at extreme time offsets
    Stage 06R   feature-engineered surrogate (raw 11 + 9 derived
                process-budget features); residual halved
    Stage 06P-B targeted AL on G_4867 extreme offsets (~702 FD);
                relative J - G advantage residual = -0.005, inside
                both 0.10 preferred and 0.05 stretch targets
    Stage 06P-B closeout  promote 06PB; document Mode B thread CLOSED
```

The 04C → 04D arc closed the v3 first-pass surrogate. The 06A → 06I
arc closed Mode A. The 06J → 06P-B arc closed Mode B and documented
the surrogate calibration vs FD truth.

## 3. Final artifacts

Anything a downstream reader needs is reachable from one of these
files:

| artefact | location |
|---|---|
| Final recipe manifest (Mode A + Mode B)         | `outputs/yield_optimization/stage06PB_final_recipe_manifest.yaml` |
| Preferred-surrogate registry                    | `outputs/models/preferred_surrogate.json` |
| Stage 06P-B closeout note                       | `study_notes/09_v3_stage06pb_mode_b_closeout.md` |
| Stage 06I Mode A closeout                       | `study_notes/08_v3_stage06i_mode_a_final_selection.md` |
| Stage 06I Mode A manifest (closed)              | `outputs/yield_optimization/stage06I_mode_a_final_recipes.yaml` |
| Stage 04D first-pass closeout                   | `study_notes/03_v3_stage04d.md` |
| Stage 04C surrogate baseline note               | `study_notes/02_v3_stage04c.md` |
| Stage 06P-B README section (top-level summary)  | `README.md` ("Stages 06J – 06P-B — Mode B exploration and blindspot closeout — CLOSED") |
| 06PB blindspot vs 06R comparison                | `outputs/logs/stage06PB_blindspot_comparison.json` |
| 06PB Mode B ranking comparison                  | `outputs/logs/stage06PB_mode_b_ranking_comparison.json` |
| 06PB feature-list (raw + derived)               | `outputs/models/stage06R_feature_list.json` |

## 4. Final recipe roles

### `G_4867` — Mode A primary

- **Role:** CD-accurate fixed-design default recipe.
- **Geometry:** `pitch_nm = 24`, `line_cd_ratio = 0.52`, `abs_len_nm = 50` (Mode A fixed-design template).
- **FD nominal:** label `robust_valid`, `CD_final_nm = 15.003`, `CD_error_nm = 0.003`, `LER_CD_locked_nm = 2.51`, `P_line_margin = 0.176`.
- **FD MC (n = 100):** `strict_pass_prob = 0.68`, `robust_valid_prob = 1.0`, `defect_prob = 0.0`, `mean_CD_final_nm = 15.05`, `std_CD_final_nm = 0.485`.
- **Time window (06M deterministic offsets, 100 FD per cell):** `strict_pass_at_zero = 0.78`, `strict_pass_ge_0p5_band_s = [-2, +1]`, `strict_pass_width = 3 s`.
- **When to prefer:** the fixed-design comparison is required, or
  pitch / line_cd_ratio / abs_len must stay on the Mode A template.
  Best overall on FD nominal CD accuracy AND FD MC stability at the
  Mode A geometry — no single Mode A competitor dominates it on both.

### `J_1453` — Mode B primary

- **Role:** open-design production alternative — wider time-window candidate.
- **Geometry / chemistry (open-design):** `pitch_nm = 24`, `line_cd_ratio = 0.45`, `dose_mJ_cm2 = 52.49`, `sigma_nm = 0.342`, `time_s = 38.63`, `abs_len_nm = 90.4`.
- **FD nominal:** label `robust_valid`, `CD_final_nm ≈ 15.0`, `LER_CD_locked_nm = 2.557`, `P_line_margin = 0.241`.
- **FD MC (n = 100, Stage 06J-B at-scale):** `strict_pass_prob = 0.747`, `mean_CD_final_nm = 15.15`, `std_CD_final_nm = 0.40`.
- **Time window (06M-B deterministic offsets):** `strict_pass_at_zero = 0.91`, `strict_pass_ge_0p5_band_s = [-3, +2]`, `strict_pass_width = 5 s` — wider than `G_4867`'s 3 s.
- **When to prefer:** open-design flexibility is allowed and a wider
  time-window robustness is more important than strict CD accuracy at
  the nominal time. Pair with `G_4867` as the Mode A fallback.
- **Caveat (open-design):** uses different `pitch_nm`, `line_cd_ratio`
  and `abs_len_nm` from the Mode A fixed-design template, so it is
  NOT a recipe-only optimisation around Mode A.

## 5. Preferred surrogate

- **Preferred surrogate:** `stage06PB`.
- **Feature set:** 20 features = 11 raw + 9 derived process-budget
  features. The derived features were introduced in Stage 06R and are
  defined in `outputs/models/stage06R_feature_list.json`:

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

- **Models:**
  - `outputs/models/stage06PB_classifier.joblib`
  - `outputs/models/stage06PB_regressor.joblib`
  - `outputs/models/stage06PB_aux_cd_fixed_regressor.joblib`
  - `outputs/models/stage06PB_strict_score_regressor.joblib`

- **Calibration vs FD MC truth (06PB):**
  - Mode B Spearman vs FD MC `strict_pass_prob` = **+0.925**
  - Mode A Spearman vs FD MC `strict_pass_prob` = **+0.964**
  - relative J - G advantage residual mean = **-0.005** (|·| = **0.025** — inside both the 0.10 preferred and 0.05 stretch targets)
  - G_4867 strict_pass residual mean = **+0.008**
  - J_1453 strict_pass residual mean = **+0.003**

- **Use:** the 06PB stack is the screening / candidate-proposal
  layer — generate large candidate batches, send the top tier to FD,
  and use **FD MC `strict_pass_prob` as the final ranking authority**.
- **Previous baselines** remain on disk and discoverable via the
  registry: `stage06R` (feature-engineered, no targeted AL),
  `stage06P` (AL refresh, no derived features), `stage06L` (direct
  strict head), `stage06H` (used by 06I Mode A selection),
  `stage04C` (frozen first-pass surrogate, never modified).

## 6. Key lessons

- **The original `yield_score` saturated.** By Stage 06D the
  surrogate was returning `P(robust_valid) ≈ 1.0` for tens of
  candidates with mean CD inside ±1 nm and LER below 3 nm — every
  term either at its ceiling or zero. Stage 06E's FD confirmed the
  v2 frozen OP itself reached `yield_score = 1.0` under both nominal
  FD and 100× MC FD. "Beats yield_score" stopped being a ranking
  question.

- **Pareto / strict_score was needed for fine discrimination.**
  Stage 06F's multi-objective Pareto ranking and Stage 06G's
  data-driven `strict_score` (CD ±0.5 nm, LER ≤ 3.0 nm) reintroduced
  dynamic range at the top.

- **A direct `strict_score` head and derived process-budget features
  improved ranking.** Stage 06L's direct `strict_score` regressor
  beat the post-hoc surrogate composition (Spearman +0.967 vs +0.143
  for the 06G surrogate on the 06I diagnostic). Stage 06R's 9
  derived features (`blur_to_pitch`, `line_cd_nm_derived`,
  `reaction_budget`, …) raised classifier balanced accuracy from
  0.693 → 0.728, halved the relative-advantage residual, and cut
  aux CD MAE by ~33 %. Derived features dominated importance.

- **The G_4867 / J_1453 blindspot was about G_4867, not J_1453.**
  06P had Mode B Spearman = +0.938 but the per-offset relative
  advantage residual was biased: predicted mean = -0.217 vs FD
  +0.06. Stage 06Q decomposed it: J_1453 was already calibrated
  (residual ≈ 0); G_4867 was over-predicted at extreme time offsets
  (residual = +0.215; CD_error residual at offset = -4 s was
  -0.51 nm). The model was missing G_4867's CD-time sensitivity, not
  J_1453's time-window robustness.

- **Targeted AL closed the magnitude gap with a small budget.**
  Stage 06P-B added ~702 targeted G_4867 FD runs (~108 s wall time)
  in three sub-phases (time densification, residual-max wider
  chemistry jitter, intermediate boundary cells). The relative
  advantage residual collapsed from -0.110 → -0.005 (|·| 0.110 →
  0.025) while Mode B top-10 overlap stayed at 10/10.

- **Mode B can produce a meaningful production alternative, but it
  must be labelled open-design.** `J_1453` beats `G_4867` on
  time-window width and FD MC `strict_pass_prob` but uses different
  `pitch_nm`, `line_cd_ratio`, and `abs_len_nm`. It is NOT a
  recipe-only Mode A variant. The final recipe manifest tags it as
  open-design so future readers don't accidentally treat it as a
  Mode A optimisation result.

## 7. Claims boundary

These are the things v3 explicitly does **not** claim:

- ✗ External calibration — none.
- ✗ Real fab yield — none. Every "yield_score" / "strict_pass_prob"
  is a model-internal quantity on the v2 frozen nominal physics.
- ✗ Validation against published / measured CD / LER / process-window
  data — none. `published_data_loaded = false` from the first commit
  through this closeout.
- ✗ Modification of the v2 frozen nominal OP — none. v3 only *uses*
  the v2 helper.

What v3 does claim, inside the nominal model:

- ✓ A reproducible candidate-screening pipeline with a closed
  first-pass surrogate (`stage04C`) and an evaluable operational
  zone (`stage04D`).
- ✓ Two FD-verified recipes, `G_4867` (Mode A primary) and `J_1453`
  (Mode B primary), each with nominal FD, MC FD, and time-window
  metrics, and clear "when to prefer" guidance.
- ✓ A calibrated 06PB screening surrogate whose direct `strict_score`
  head matches FD MC `strict_pass_prob` at Mode B Spearman = +0.925
  and Mode A Spearman = +0.964, with relative-advantage residual
  inside the 0.05 stretch target.
- ✓ Documented blindspot diagnosis and correction (06Q -> 06R ->
  06P-B), so the magnitude bias that 06P shipped with is visible
  and reproducible.

## 8. Optional future work

These are explicitly **optional** and out of scope for this closeout:

- **Autoencoder / anomaly detector** — placeholder under Stage 05;
  would let the screening loop flag candidates the surrogate has
  never seen geometrically rather than just predict on them.
- **Inverse fitting** — given a target metric, search for compatible
  recipes; useful if downstream consumers want target-driven
  selection instead of top-k from a Sobol pool.
- **Deeper J_1453 process-window map** — Stage 06P-B addressed
  G_4867's extreme-offset over-prediction; a symmetric deep MC
  around J_1453 (other knobs, not just time) would tighten the Mode
  B side of the manifest. Not required because Mode B Spearman is
  already +0.925 and J_1453 stays calibrated.
- **External calibration** — only if real published or measured CD /
  LER / process-window data become available. This would change
  `published_data_loaded` and `external_calibration` policy values
  and would be a new milestone, not a continuation of v3.
- **New objective family** — e.g. a defect-rate or throughput
  objective in place of `strict_score`. Would also be a new
  milestone.

## 9. Recommended stopping point

The v3 study is complete as a personal-study nominal-simulator
screening / yield-optimization workflow. The first-pass surrogate
(04C/04D), the Mode A recipe (06I), the Mode B recipe (06P-B
closeout), the calibrated 06PB screening surrogate, and the final
manifest all exist as committed artefacts on `main`.

**Further work should start as a new explicit milestone**
(autoencoder, inverse fit, deeper J_1453 map, external calibration,
new objective family) **rather than continuing to add small patches
to Stage 06.** The 06J → 06P-B arc has already absorbed the patches
that the original blindspot motivated; adding more would mostly be
cosmetic.
