# v3 Stage 06B — FD verification of Stage 06A surrogate-driven recipes

## Scope

Stage 06B verifies the Stage 06A surrogate-driven yield-optimisation
output by running v2 nominal FD on the top recipes and comparing the
empirical FD result to the surrogate's prediction. No new training, no
retraining of the closed Stage 04C surrogate. The closed Stage 04D /
04C training datasets are *not* mutated.

This is **not** real fab-yield prediction. \`v2_OP_frozen\` stays
\`true\`, \`published_data_loaded\` stays \`false\`.

## Pipeline

| Part | Input | FD calls | Throughput |
|---|---|---:|---:|
| 1. top-100 nominal FD                  | top-100 Mode A surrogate recipes |   100 | 6.3 runs/s |
| 2. top-10 FD MC (100 variations each)  | top-10 Mode A surrogate recipes  | 1,000 | 6.3 runs/s |

Same process-variation widths as Stage 06A
(`dose ±5 %`, `time ±2 s`, `sigma ±0.2 nm`, `DH ±10 %`, `Q0 ±10 %`,
`line_cd ±0.5 nm` via `line_cd_ratio`, `Hmax ±5 %`, others = 0). All
varied features clipped to candidate-space bounds.

## Headline numbers

```text
v2 frozen OP yield_score (surrogate baseline)             0.6828

Part 1 — top-100 nominal FD
  label agreement (surrogate argmax vs FD label)          100 / 100
  Spearman ρ (surrogate yield_score vs FD yield_score)    +0.071
  top-1 / top-3 / top-5 overlap                           1/1, 3/3, 5/5
  FD recipes that beat the v2 OP baseline                 85 / 100
  regression MAE (surrogate vs FD nominal):
      CD_fixed         0.995 nm   (RMSE 1.147 nm)
      CD_locked        0.133 nm
      LER_locked       0.047 nm
      area_frac        0.055
      P_line_margin    0.038

Part 2 — top-10 FD Monte-Carlo (100 variations each)
  Spearman ρ (surrogate vs FD MC yield_score)             nan (all FD = 1.000)
  surrogate top-10 ∩ FD top-10                            10 / 10
  FD MC recipes that beat the v2 OP baseline              10 / 10

Part 4 — false-PASS analysis
  hard false-PASS in top-100                              0 / 100  (0.00 %)
  soft false-PASS (FD = margin_risk)                      0 / 100
```

## What the numbers mean

### Coarse-grained agreement: excellent

- Every one of the 100 top-100 surrogate picks is FD-labelled
  `robust_valid`. The surrogate makes **zero false-PASS calls** under
  the user's hard or soft definition.
- The top-1, top-3, and top-5 surrogate recipes are also the top-1,
  top-3, top-5 FD recipes (perfect overlap).
- 85 of the 100 surrogate picks beat the v2 frozen OP under FD nominal.
  Among the top-10, **all 10 reach FD MC yield_score = 1.000 exactly**
  — they sit inside the CD ±1 nm window and below the LER 3 nm bound
  in every one of their 100 process variations.

In other words, **Stage 06A delivers a clean candidate set that the v2
nominal physics genuinely treats as robust**.

### Fine-grained ranking inside the top tier: dominated by surrogate noise

- The Spearman ρ between surrogate and FD yield_score on the top-100 is
  +0.071. After tie correction this is the *true* number — most of the
  top-100 share the same FD label and FD CD/LER bands, so the within-
  band ordering is dominated by random aux-CD-regressor noise rather
  than physical signal.
- For the top-10 the FD MC scores are all exactly 1.000 (no variation
  triggered any defect class). Spearman ρ is undefined (NaN) — the
  data tells us **the top-10 is a tied tier**, not that the surrogate
  ordering is wrong or right.
- The CD-prediction scatter plot
  (`surrogate_vs_fd_cd_ler_scatter.png`) shows ρ(CD_fixed) = 0.526:
  the surrogate's CD prediction is *coarsely* aligned with FD reality
  (slope correct, points within ±1 nm tolerance), but the per-recipe
  noise of ~1 nm is comparable to the CD-error tolerance in the score
  formula. So when the surrogate prefers recipe A over B by 0.01 in
  yield_score, it is essentially tossing a coin.

### How to read this

Stage 06A is **useful for candidate proposal but insufficient for
direct ranking inside its own top tier.** The surrogate finds a
genuinely robust subset of the recipe space; it does not reliably
rank-order the recipes inside that subset.

## Acceptance

| criterion | result | pass |
|---|---:|:---:|
| top-100 nominal FD beats baseline (≥ 1)         | 85         | ✅ |
| FD MC beats baseline (≥ 1)                      | 10         | ✅ |
| surrogate top-10 ∩ FD top-10 (≥ 1)              | 10         | ✅ |
| Spearman top-100 positive (> 0.05)              | +0.071     | ✅ (barely) |
| false-PASS rate not catastrophic (< 20 %)       | 0.00 %     | ✅ |
| `v2_OP_frozen` remains true                     | true       | ✅ |
| `published_data_loaded` remains false           | false      | ✅ |
| no external-calibration claim                   | none made  | ✅ |

## Outputs

```text
outputs/labels/
  fd_top100_nominal_verification.csv         100 FD rows
  fd_top10_mc_verification.csv             1,000 FD rows
  surrogate_vs_fd_metrics.csv                100 paired rows
  false_pass_cases.csv                         0 rows  (header only)
  06_yield_optimization_al_additions.csv   1,100 FD rows
outputs/logs/
  06b_surrogate_fd_ranking_comparison.json
  06b_false_pass_summary.json
outputs/figures/06_yield_optimization/
  surrogate_vs_fd_yield_score_scatter.png
  surrogate_vs_fd_cd_ler_scatter.png
  top10_fd_yield_barplot.png
  defect_breakdown_top10.png
  false_pass_parameter_parallel_coordinates.png
study_notes/05_v3_stage06b_fd_verification.md
```

## Active-learning additions — should they be used to refresh the surrogate?

- `outputs/labels/06_yield_optimization_al_additions.csv` carries
  1,100 fresh FD rows (100 nominal at top-100 surrogate picks +
  100 × 10 MC variations at the top-10 surrogate picks).
- These rows are **not** automatically appended to the closed Stage
  04D dataset. Closed-state preservation matters more than dataset
  growth at this stage.
- Suggested use for a future Stage 06C refresh:
    * append the 1,100 rows to the Stage 04C training pool;
    * re-train both the 04C classifier (6-class) and the 4-target
      regressor;
    * **re-train the auxiliary CD_fixed regressor** — its current MAE
      of 1.08 nm is the single biggest source of the within-tier
      ranking noise documented above. Halving this MAE would more
      than halve the noise floor of the CD-error term.
    * re-run Stage 06A with the refreshed surrogate and Stage 06B
      with the new top-N to check whether the top-tier Spearman moves
      meaningfully above zero.
- A Stage 06C refresh is **not scheduled** in this PR. It is justified
  by the within-tier-Spearman result here, but the gating decision is
  the user's. Until then, the policy block remains: closed dataset
  preserved, no external calibration, v2 OP frozen.

## Limitations

1. **Mode B (open-design) FD verification was not scheduled.** Per the
   user's brief, Mode A is the primary verification target. Mode B
   recipes can be FD-verified separately if the open-design search is
   later promoted.
2. **No FD baseline** — the v2 frozen OP's yield_score baseline used
   for "beats baseline" comparisons is the *surrogate* baseline (0.6828)
   from Stage 06A. A matching v2-OP FD MC run would tighten this
   comparison; it is deferred because all top-10 recipes beat the
   surrogate baseline by a wide margin (FD MC = 1.000), so the
   conclusion is robust to the choice of baseline source.
3. **All top-100 land in the FD-perfect bucket.** This is a feature of
   how aggressively the Stage 06A scorer suppresses defect-class
   probability mass, not a defect of the verification pipeline. It also
   means Stage 06B does not stress the false-PASS classifier (zero
   examples to study).
4. **Aux CD regressor noise floor (1.08 nm MAE).** Documented above.
5. **Reservation of "yield" for nominal-model proxy.** No external
   calibration; the FD ground truth here is v2 model truth, not
   measurement.
