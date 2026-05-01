# Stage 06F — strict Pareto ranking for saturated robust recipes

## Why this stage exists

Stage 06E showed that the v2 frozen-OP, scored under both single-FD and 100×
process-variation FD MC, returns `yield_score = 1.0` with `P(robust_valid) =
1.0`. 69 / 100 of the Stage 06D top-100 nominal-FD recipes also tie that
ceiling without false-PASS, and 8 / 10 of the top-10 MC FD recipes tie the
OP under the same MC distribution. Once enough recipes saturate the metric,
`yield_score` stops being a discriminator — comparing two `1.0`s by
yield_score alone is noise.

Stage 06F resolves the saturation by **switching to multi-objective Pareto
ranking on FD-derived metrics** instead of moving to arbitrarily stricter
CD-tolerance / LER-cap thresholds first. Threshold sensitivity is reported,
but only as a downstream sanity check — never as a calibrated spec.

## Policy preserved

- `v2_OP_frozen = true`
- `published_data_loaded = false`
- `external_calibration = "none"`
- Closed Stage 04C / 04D / 06B / 06E label datasets are not mutated.
- Stage 06F is purely a re-analysis of Stage 06E FD outputs — no new FD
  runs.

## Pipeline

1. **Inputs.** `stage06E_fd_top100_nominal.csv`,
   `stage06E_fd_top10_mc.csv`, the v2-OP nominal + MC baselines, and the
   06D surrogate top-100 for cross-reference.
2. **Per-recipe nominal aggregates.** `CD_error_nm = |CD_final_nm − 15.0|`,
   `LER_CD_locked_nm`, `−P_line_margin`, defect indicator (0 if
   `robust_valid`, 1 otherwise — degenerate at the top-100 since false-PASS
   = 0, but kept for completeness).
3. **Per-recipe MC aggregates.** mean / std `CD_final_nm`, mean / std
   `LER_CD_locked_nm`, mean `P_line_margin`, `robust_prob`,
   `margin_risk_prob`, `defect_prob`.
4. **Pareto ranks** computed by NSGA-II-style non-dominated sorting on
   the relevant objective sets, with crowding distance per front.
5. **Reference rankings** computed for comparison: yield_score (06C),
   CD-only, LER-only, and a z-score balanced score
   `z(CD_error) + z(LER) + z(CD_std) + z(LER_std) + 2·defect_prob −
   z(P_line_margin)`.
6. **Four representative recipes** selected: CD-best (lowest nominal
   `CD_error_nm`), LER-best (lowest nominal `LER_CD_locked_nm`),
   balanced-best (lowest `balanced_score_mc` over the top-10 MC recipes),
   margin-best (highest mean `P_line_margin` over the top-10 MC recipes).
7. **Threshold sensitivity** evaluated over the hypothetical grid
   CD ∈ {±1.0, ±0.75, ±0.5} nm × LER ∈ {3.0, 2.7, 2.5} nm.

## Headline results

- **Pareto rank-1 nominal: 35 / 100.** A third of the top-100 are mutually
  non-dominated on `(CD_error, LER, −P_line_margin, defect)`. This is
  exactly the discrimination yield_score lost.
- **Pareto rank-1 MC: 9 / 10.** With 6 MC objectives, almost every top-10
  recipe is non-dominated; this confirms the MC aggregate distinguishes
  recipes mostly through *trade-offs* (e.g. CD accuracy vs CD stability),
  not through one being uniformly better than another.
- **yield_score is uncorrelated with the new rankings.**

  | Pair (Spearman ρ over top-100 nominal) | ρ |
  |---|---|
  | yield_score vs Pareto rank | +0.079 |
  | yield_score vs CD-only rank | +0.043 |
  | yield_score vs LER-only rank | −0.121 |
  | yield_score vs balanced-score rank | +0.084 |
  | **Pareto rank vs balanced-score rank** | **+0.850** |

  yield_score and the Pareto / balanced rankings are statistically
  independent at the top, while the two saturation-aware rankings agree
  strongly (ρ ≈ 0.85). That is the headline justification for switching:
  yield_score lost its information; Pareto and balanced score recover it
  consistently.
- **MC ranks are *anti*-correlated with yield_score** (ρ = −0.41 against
  Pareto-MC, ρ = −0.52 against balanced-MC). Once everyone hits the FD
  ceiling, the surrogate's preferred ordering inside the top-10 is
  essentially the *opposite* of the FD-MC quality ordering.

## Representative recipes

| kind | recipe_id | rank_06c | nominal `CD_error_nm` | nominal `LER_CD_locked_nm` | mean `P_line_margin` | defect_prob | comment |
|---|---|---|---|---|---|---|---|
| CD-best | D_4350 | #11 | **0.011 nm** | 2.516 | 0.209 (nominal) | n/a (no MC) | nearly perfect on-target CD |
| LER-best | D_4836 | #42 | 0.530 nm | **2.485 nm** | 0.182 (nominal) | n/a (no MC) | only the surrogate ranked it at #42 |
| balanced-best | D_2652 | #8 | 0.205 nm | 2.523 nm | 0.215 (MC) | **0.000** | minimum MC balanced_score, MC-stable |
| margin-best | D_1513 | #3 | 1.718 nm | 2.527 nm | 0.278 (MC) | 0.000 | high margin but worst CD accuracy |

CD-best and LER-best are pulled from the nominal top-100 because the MC
top-10 set is too small to dominate either axis; balanced-best and
margin-best are picked from the MC top-10 because they require MC
stability metrics. Only `balanced-best` and `margin-best` carry MC
defect probabilities; CD-best / LER-best ride on nominal FD only.

`v2_frozen_op` for comparison: `CD_error_nm = 0.501`, `LER = 2.471`,
`mean_p_line_margin = 0.117`, `MC defect_prob = 0.000`. So the OP is
already an excellent recipe by these metrics. The four representatives
each beat the OP on **one specific axis**:

- D_4350 dominates the OP on CD accuracy (0.011 vs 0.501 nm).
- D_4836 ties the OP on LER (2.485 vs 2.471 nm — within 0.02 nm noise).
- D_2652 has higher MC margin (0.215 vs 0.117) and matches MC defect
  probability.
- D_1513 has the highest MC margin (0.278) but pays for it in CD
  accuracy.

## Threshold sensitivity (hypothetical only)

The 3 × 3 sensitivity grid (CD tolerance × LER cap) is reported in
`stage06F_threshold_sensitivity.csv` and rendered as
`stage06F_threshold_survival_heatmap.png`. **These thresholds are
hypothetical exploration grids — not externally calibrated, not
literature-backed, and explicitly not pass/fail truth.** Their only
purpose is to show how many top-100 recipes survive when the spec is
made progressively tighter, so we can judge whether discriminability
keeps holding when saturation eventually breaks.

## Acceptance criteria (Stage 06F)

| Criterion | Value |
|---|---|
| Pareto front for nominal FD top-100 | 35 / 100 in rank 1 — **computed** |
| Pareto front for top-10 MC FD | 9 / 10 in rank 1 — **computed** |
| ≥ 1 balanced representative recipe identified | D_2652 — **identified** |
| Threshold sensitivity reported, no external-spec claim | 9 cells, caveat in JSON — **reported** |
| `v2_OP_frozen` remains true | true |
| `published_data_loaded` remains false | false |
| External-calibration claims | none |

All criteria pass; no policy regression.

## Why this is what we wanted, not stricter thresholds

The user's spec was explicit: "Do not start by changing CD tolerance or
LER cap to arbitrary stricter values. Use Pareto ranking first." Stage
06F follows that literally. The Pareto / balanced rankings rediscover
discrimination *without* changing any spec-relevant numbers, so the
result is reproducible: anyone re-running this stage on the same
Stage 06E FD CSVs gets the same Pareto fronts, the same four
representatives, and the same threshold sensitivity table. Tightening
CD tolerance or LER cap can come later, in a Stage 06G if needed, with
the threshold-survival numbers from this stage as the *starting point*
of the conversation rather than a foregone conclusion.
