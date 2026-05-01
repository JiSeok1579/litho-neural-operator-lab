# Stage 06I — Mode A final recipe selection and strict-ranking closeout

## Why this stage exists

Stages 06A → 06H walked through the full surrogate-driven Mode A
pipeline twice (06A then 06D), built two saturation-aware ranking
strategies (06F Pareto, 06G strict_score), and verified each with FD
(06B and 06H). Stage 06H's interpretation block mandates using FD —
not the surrogate — for **final** ranking among saturated robust
recipes, because the surrogate refresh did not improve strict ranking
on the 06G top recipes (only 1 of 6 06G representatives stayed in the
06H top-20).

Stage 06I closes Mode A by writing a manifest of FD-verified
representative recipes and a decision table that orders them on
strict_pass_prob, CD_error, LER, margin, and FD MC robustness. The
manifest is what downstream consumers should read, not the surrogate
rankings.

## Why the original yield_score saturated

`yield_score` is

> `+1.0·P(robust_valid) − 0.5·P(margin_risk) − 2.0·P(under_exposed) − 2.0·P(merged) − 1.5·P(roughness_degraded) − 3.0·P(numerical_invalid) − 1.0·CD_pen − 1.0·LER_pen`

with CD penalty active outside ±1 nm of the target and LER penalty
active above 3 nm. By Stage 06D the surrogate was already returning
`P(robust_valid) ≈ 1.0` for tens of candidates, with mean CD inside
±1 nm and mean LER below 3 nm — every term either at its ceiling or
zero. The metric ran out of dynamic range at the top.

Stage 06E's FD verification confirmed the saturation: the v2 frozen
OP itself returned `yield_score = 1.0` under both single nominal FD
and 100× MC FD. Once a metric saturates at the OP, "this candidate
beats the OP by yield_score" becomes a definitionally impossible
question, not a ranking question.

## Why strict / Pareto ranking was introduced

Stage 06F replaced the saturated yield_score with multi-objective
Pareto ranking on FD-derived metrics — CD_error, LER, std_CD, std_LER,
defect_prob, P_line_margin. Pareto rank-1 captured 35 of 100 06D
nominal recipes and 9 of 10 MC recipes, recovering the discrimination
that yield_score had lost.

Stage 06G then promoted Pareto's exploration findings into a single
scalar `strict_score`:

> `+1.0·P(robust_valid) − 0.5·P(margin_risk) − 2.0·P(under_exposed) − 2.0·P(merged) − 1.5·P(roughness_degraded) − 3.0·P(numerical_invalid) − 1.5·CD_strict_pen − 1.5·LER_strict_pen − 0.5·CD_std_pen − 0.5·LER_std_pen + 0.25·mean(P_line_margin)`

with `CD_strict_pen` active outside `cd_tol_nm = 0.5` and
`LER_strict_pen` active above `ler_cap_nm = 3.0`. Both thresholds were
chosen from `stage06F_threshold_sensitivity.csv`, not invented — the
±0.5 nm cell retained 36 / 100 of the 06D top-100 (selective but
non-empty), and lowering `ler_cap_nm` did not change the survivor
count in the 06D distribution, so 3.0 nm is the natural non-arbitrary
choice.

Stage 06G's surrogate-side optimisation produced a clean spread:
`strict_score` for the v2 frozen OP under the same 06C surrogate is
0.323; the surrogate's 06G best recipe G_3691 is 0.771. The metric
discriminates again.

## Why FD is used for final ranking

The 06H surrogate refresh added the 17 06E disagreement rows and
2,900 new FD rows to the 06C training pool, then re-scored the 06G
representatives. 5 of the 6 representatives moved out of the 06H
top-20 (only margin-best G_4299 stayed — at rank #12). The interpretation
block in the 06H spec explicitly covers this case:

> If refresh does not improve ranking: keep 06H FD rows as AL
> additions but continue using FD for final ranking.

Stage 06I follows that rule. The manifest's primary recommended
recipe is selected by **FD MC strict_pass_prob**, not by any surrogate
score. Surrogate scores are kept in the manifest as columns for
diagnostic comparison, but they are not the ranking authority.

## What the FD truth says about each role

| role (06G surrogate name) | recipe_id | FD MC `strict_pass_prob` | FD nominal CD_error | FD nominal LER | comment |
|---|---|---|---|---|---|
| **fd_stability_best** *(new in 06I)* | **G_4867** | **0.680** | **0.003 nm** | 2.511 nm | dominant on FD MC stability AND nominal CD accuracy at the same time |
| fd_stability_co_winner *(new in 06I)* | G_1096 | 0.680 | 0.123 nm | 2.522 nm | tied on stability, slightly less on nominal CD |
| fd_stability_runner_up *(new in 06I)* | G_715 | 0.630 | 0.049 nm | 2.509 nm | next-best on stability |
| strict_best (06G surrogate) | G_3691 | 0.430 | 0.630 nm | 2.522 nm | surrogate's #1 by strict_score; FD's #8 |
| cd_best (06G surrogate) | G_2311 | 0.287 | **0.793 nm** | 2.522 nm | surrogate predicted CD_error 0.008 nm; FD says 0.793 nm — single largest surrogate-vs-FD discrepancy in the manifest |
| ler_best (06G surrogate) | G_829 | 0.090 | 1.200 nm | 2.479 nm | best LER, worst CD_error and worst MC stability |
| margin_best (06G surrogate) | G_4299 | 0.467 | 0.491 nm | 2.539 nm | secondary stable candidate |
| balanced (06G surrogate) | G_1226 | 0.297 | 0.762 nm | 2.513 nm | 06G surrogate's z-score balanced winner |

The "fd_stability_*" roles are FD discoveries — these recipes were in
the 06G top-10 but were *not* picked as 06G representatives because
the 06G representative selection used surrogate criteria (lowest
predicted CD_error, lowest predicted LER, max predicted margin,
balanced z-score, novelty distance) that did not include an FD-stability
proxy. The 06H Part 2 MC FD revealed them.

The earlier spec note saying "if G_4299 is the current primary winner"
was written before the 06H FD MC results were in. Once we have FD
truth, G_4299's `strict_pass_prob = 0.467` is clearly below the three
FD-stability leaders (0.680 / 0.680 / 0.630), so the 06I primary is
**G_4867**, not G_4299. The manifest is explicit about this.

## Why no single recipe is universally best

Even with FD truth, the manifest contains *trade-offs*:

- **G_4867** dominates on FD MC stability and FD nominal CD accuracy,
  but its FD MC std_CD is mid-range (0.485). G_4299 has the lowest
  FD MC std_CD (0.395) — if you specifically want the *smallest CD
  variance*, G_4299 wins. If you want highest *probability of clearing
  CD ±0.5 nm under variation*, G_4867 wins.
- **G_829** has the lowest FD nominal LER (2.479 nm), but its MC
  `strict_pass_prob` is 0.090 — a recipe that is excellent at the
  nominal point and falls apart under variation. Useful if you have a
  tight LER spec but a deterministic process; not useful for variation
  robustness.
- **G_2311** is the surrogate's CD-best pick (predicted CD_error 0.008
  nm); FD measured 0.793 nm. Including it documents the surrogate
  failure honestly, not as a recommended recipe.

So the manifest exposes: one default (G_4867), two stability backups
(G_1096, G_715), one secondary (G_4299), and four diagnostic /
use-case-dependent recipes.

## Recommended default Mode A recipe

`primary_recommended_recipe: G_4867`

with G_1096 as the explicit backup primary and G_715 as the third
option. Anything else in the manifest is either use-case-specific or
diagnostic.

## Optional diagnostic — direct strict_score regressor

To answer the open question from 06H — *did the 4-target regressor
loss cause the strict-ranking weakness?* — Stage 06I trained a single
RF regressor directly on per-row strict_score (one-hot, no std
penalties). The regressor was scored on the 06G top-100 with the same
200-variation MC pipeline used elsewhere, then compared to FD MC
truth on the 13 recipes that have FD MC.

| ranker | Spearman ρ vs FD `strict_pass_prob` | top-3 overlap | top-5 overlap |
|---|---|---|---|
| 06G surrogate strict_score | +0.143 | 1 / 3 | 1 / 5 |
| 06H surrogate strict_score | +0.890 | 1 / 3 | 5 / 5 |
| **06I direct strict_score regressor** | **+0.967** | **3 / 3** | **5 / 5** |

Manifest recipe ranks under the 06I direct regressor (over the full
06G top-100):

| recipe | role | rank under 06I |
|---|---|---|
| G_4867 | fd_stability_best | **#1** |
| G_1096 | fd_stability_co_winner | **#2** |
| G_715  | fd_stability_runner_up | **#3** |
| G_4299 | margin_best | **#5** |
| G_3691 | strict_best | #20 |
| G_1226 | balanced | #40 |
| G_2311 | cd_best | #44 |
| G_829  | ler_best | #89 |

This answers 06H's open question: yes, the 4-target loss was the
bottleneck for strict ranking. A direct `strict_score` head trained on
the 06H training pool recovers FD-aligned ordering of the FD MC
winners almost exactly (top-3 perfect overlap, +0.97 Spearman).

**However**, per the spec the direct regressor is treated as
**diagnostic only** for this stage — its per-row strict_score MAE on
the 1,074-row 04C holdout is 4.3 (the strict_score range is wide,
[-49.9, +1.07]), which means absolute predictions are noisy even
though their rank order is very good. The manifest's primary
recommendation stays FD-truth based; the diagnostic only recommends
that a future stage (06J or 06K) consider adding a `strict_score`
head to the surrogate stack alongside the existing 4-target regressor
+ aux CD_fixed regressor.

## Limitations

- **Nominal model only.** Every FD label here is internal model truth
  on the v2 nominal physics; not measurement, not externally
  calibrated. Do not interpret `strict_pass_prob` as wafer-level
  yield.
- `published_data_loaded` stays false. `external_calibration` stays
  `"none"`.
- **Mode A only.** pitch = 24 nm, line_cd_ratio = 0.52,
  abs_len_nm = 50 nm. Mode B open-design exploration is deferred.
- **Personal / study setting.** This repo is a research notebook; the
  manifest is a research conclusion, not an engineering release.

## Acceptance vs Stage 06I spec

| criterion | result |
|---|---|
| final recipe manifest exists | `outputs/yield_optimization/stage06I_mode_a_final_recipes.yaml` |
| ≥ 4 representative FD-verified recipes documented | 8 (3 FD-stability + 5 surrogate-derived roles) |
| each representative has rationale + weakness + when-to-prefer | yes, in the YAML manifest |
| one default Mode A recipe selected | `G_4867` (with G_1096 / G_715 explicit backups) |
| optional strict_score regressor reported or deferred | reported; per-recipe Spearman vs FD MC = +0.967 |
| no policy regression (`v2_OP_frozen`, `published_data_loaded`, no external calibration) | yes |

All criteria pass.
