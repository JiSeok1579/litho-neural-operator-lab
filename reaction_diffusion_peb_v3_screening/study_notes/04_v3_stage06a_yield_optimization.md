# v3 Stage 06A — surrogate-driven nominal-yield-proxy optimisation

## Scope

Stage 06A starts a new milestone on top of the closed v3 first pass. The
goal is to find process recipes that maximise a *nominal-model yield
proxy* under independent process variation, using only the closed
Stage 04C surrogate (classifier + 4-target regressor) and an auxiliary
CD_fixed regressor trained on the same Stage 04C labelled pool.

This is **not** real fab-yield prediction. The surrogate is treated as
a fast oracle for the v2 nominal physics, not for a real lithography
toolset.

```text
v2_OP_frozen           = true
published_data_loaded  = false
external_calibration   = none
```

## Pipeline

1. **Auxiliary CD_fixed regressor** (`outputs/models/06_yield_optimization_cd_fixed_aux.joblib`).
   Random-Forest, 300 trees, target = `CD_final_nm` (= v2 alias `CD_fixed_nm`).
   Train MAE on the 04C 80/20 split = 1.083 nm. The closed Stage 04C
   classifier and 4-target regressor are not retrained; only this
   single auxiliary model is added.

2. **Sobol candidate sampling** over the v3 candidate space — 5,000
   candidates per mode, fixed seed 1011.

3. **Bounded uniform process variation** — for each candidate, draw 200
   independent variations using widths `dose ±5 %`, `time ±2 s`,
   `sigma ±0.2 nm`, `DH ±10 %`, `Q0 ±10 %`, `Hmax ±5 %`,
   `line_cd ±0.5 nm`. Knobs not listed (kdep, kq, abs_len, pitch,
   line_cd_ratio when not the line_cd channel) get zero variation.
   Every varied feature is clipped back to candidate-space bounds.

4. **Surrogate evaluation** — for each candidate × variation row, run
   the closed 04C classifier (`predict_proba`), the 04C 4-target
   regressor, and the auxiliary CD_fixed regressor. Per-recipe means
   over the 200 variations feed the score.

5. **Yield score**:

       yield_score = + 1.0  · P(robust_valid)
                     - 0.5  · P(margin_risk)
                     - 2.0  · P(merged)
                     - 2.0  · P(under_exposed)
                     - 1.5  · P(roughness_degraded)
                     - 3.0  · P(numerical_invalid)
                     - 1.0  · CD_error_penalty
                     - 1.0  · LER_penalty

   with

       CD_error_penalty = max(0, |mean(CD_fixed) − 15.0| − 1.0) / 1.0
       LER_penalty      = max(0, mean(LER_CD_locked) − 3.0) / 1.0

6. **Two modes**:
   - **Mode A — fixed-design recipe optimisation (primary)**.
     `pitch_nm = 24`, `line_cd_ratio = 0.52`, `abs_len_nm = 50`. Sobol
     samples only the 8 recipe / material-effective knobs.
   - **Mode B — open design-space exploration (secondary)**. All
     candidate-space axes free, including `pitch_nm`, `line_cd_ratio`,
     `abs_len_nm`. Reported separately and not compared against Mode A
     directly.

7. **Baseline** — the v2 frozen nominal OP runs through the *same*
   200-variation pipeline so the score is on the same footing as every
   candidate.

## Headline numbers (n_candidates=5000, n_variations=200, seed=1011)

| metric                      | v2 frozen OP | Mode A best | Mode B best |
|---|---:|---:|---:|
| `yield_score`               | **0.6828**   | 0.9522      | 0.9386      |
| P(robust_valid)             | 0.848        | 0.979       | 0.973       |
| P(margin_risk)              | 0.088        | 0.010       | 0.011       |
| mean CD_fixed (nm)          | 14.81        | 15.62       | 14.97       |
| mean LER_CD_locked (nm)     | 2.52         | 2.52        | 2.55        |
| CD_error_penalty            | 0.0          | 0.0         | 0.0         |
| LER_penalty                 | 0.0          | 0.0         | 0.0         |

Mode A: **190 / 5,000** recipes (3.8 %) improve over the v2 baseline.
Mode B: **87 / 5,000** recipes (1.7 %) improve, in spite of the larger
search volume — Sobol spreads samples across pitch × ratio combos that
sit far from CD_target=15 nm and incur the CD-error penalty, lowering
the median yield_score (Mode A median −1.41; Mode B median −2.28).

## Recipe-knob sensitivity (Mode A, Spearman ρ vs yield_score)

```text
dose_mJ_cm2  +0.40    Hmax_mol_dm3  +0.13
kdep_s_inv   +0.21    Q0_mol_dm3    +0.00
sigma_nm     −0.16    kq_s_inv      +0.00
time_s       −0.15    DH_nm2_s      −0.11
```

Inside the Mode A region, dose dominates and the quencher knobs have
near-zero direct effect on the score — the v2 OP already sits in a
weak-quencher regime where Q0 / kq do not move the surrogate's class
boundaries.

## Outputs

```text
outputs/labels/
  06_yield_optimization_summary.csv               all 10,000 rows + baseline
  06_top_recipes_fixed_design_surrogate.csv       Mode A top-100
  06_top_recipes_open_design_surrogate.csv        Mode B top-100
outputs/models/
  06_yield_optimization_cd_fixed_aux.joblib       aux CD_fixed regressor
outputs/logs/
  06_yield_optimization_summary.json              run-level summary
outputs/figures/06_yield_optimization/
  pareto_front_fixed_design.png
  pareto_front_open_design.png
  defect_breakdown_heatmaps.png
  recipe_sensitivity.png
```

## Limitations

1. **Yield is a nominal-model proxy.** The 04C surrogate's labels come
   from v2 nominal FD, not measurement. A *recipe that the surrogate
   thinks is robust* is a recipe that survives the v2 model's labeller
   under MC, not a recipe that produces yield in any fab.
2. **Surrogate failure modes propagate.** `numerical_invalid` is
   penalised at −3.0 to discourage the optimiser from steering into
   regimes the surrogate has never seen, but a high-confidence
   `robust_valid` prediction in an extrapolation region is still
   possible.
3. **Process variation is independent and uniform.** Real fab variation
   is correlated and tail-heavy. The width is also uniform across the
   space, which means the score gives the same weight to a 5 % dose
   excursion at 25 mJ as at 55 mJ.
4. **Only the recipe knobs the user listed get MC variation.** kdep,
   kq, abs_len, and (in Mode A) pitch / line_cd_ratio see zero MC
   spread by design.
5. **The auxiliary CD_fixed regressor MAE is 1.08 nm**, which is not
   tighter than the ±1 nm CD tolerance. The CD-error penalty term
   therefore has a noise floor of about half a nm — recipes whose
   true CD lands exactly at 15.0 nm and recipes whose true CD lands at
   14.0 nm look nearly identical to this scoring channel.
6. **No FD verification yet.** Stage 06A reports surrogate-only
   rankings. Stage 06B will run the top-100 nominal recipes and the
   top-10 with 100 MC variations under FD to compare ranking
   (Spearman) and specifically harvest *false-PASS* cases for the
   active-learning extension.

## What Stage 06B will do

- top-100 fixed-design and top-100 open-design FD verification
  (single-shot per recipe);
- top-10 × 100 FD Monte-Carlo variations;
- Spearman ρ of surrogate yield_score vs FD-derived per-recipe
  yield_score;
- collect surrogate-says-PASS-but-FD-says-defect rows into
  `outputs/labels/06_yield_optimization_al_additions.csv`;
- the closed Stage 04C training dataset is not mutated.
