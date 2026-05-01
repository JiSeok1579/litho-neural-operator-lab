"""Stage 06I -- Mode A final recipe manifest + decision table.

Closes Mode A by writing a manifest of FD-verified recipes plus a
decision table that ranks them across strict_score, CD_error, LER,
margin, and FD MC robustness. FD is the final ranking authority --
the surrogate is treated as candidate proposal / screening only.

Key FD-truth observation that drives the manifest:
    The Stage 06G surrogate strict_score top-10 disagrees with FD MC
    strict_pass_prob (06H Spearman = -0.37). Under FD MC truth the
    primary winner is **G_4867** (strict_pass_prob = 0.680), not
    G_4299 (the 06G margin-best representative, 0.467).

Inputs:
    outputs/yield_optimization/stage06G_top_recipes.csv
    outputs/yield_optimization/stage06G_representative_recipes.csv
    outputs/yield_optimization/stage06G_strict_score_config.yaml
    outputs/labels/stage06H_fd_top100_nominal.csv
    outputs/logs/stage06H_fd_verification_summary.json

Outputs:
    outputs/yield_optimization/stage06I_mode_a_final_recipes.yaml
    outputs/yield_optimization/stage06I_final_decision_table.csv

Policy
    v2_OP_frozen           = true
    published_data_loaded  = false
    No external calibration. Stage 04C / 04D / 06C closed artefacts
    untouched.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reaction_diffusion_peb_v3_screening.src.metrics_io import (
    FEATURE_KEYS,
    read_labels_csv,
)


V3_DIR = ROOT / "reaction_diffusion_peb_v3_screening"
CD_TARGET_NM = 15.0


def _safe_float(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


# Manifest universe: 8 FD-verified Mode A recipes covering all roles.
# Each tuple = (recipe_id, role).
# Roles cover the 4 spec-required (strict_best, cd_best, ler_best, balanced/margin)
# plus 2 FD-discovered roles found by 06H Part 2 MC.
MANIFEST_RECIPES = [
    ("G_4867", "fd_stability_best"),       # primary -- FD truth winner
    ("G_1096", "fd_stability_co_winner"),  # tied at 0.680 strict_pass_prob
    ("G_715",  "fd_stability_runner_up"),  # 0.630 strict_pass_prob
    ("G_3691", "strict_best"),             # 06G surrogate #1
    ("G_2311", "cd_best"),                 # nominal CD accuracy 0.008 nm
    ("G_829",  "ler_best"),                # nominal LER 2.532 nm
    ("G_4299", "margin_best"),             # 06G surrogate margin-best
    ("G_1226", "balanced"),                # 06G surrogate balanced-best
]
PRIMARY_RECOMMENDED = "G_4867"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--top_06g_csv", type=str,
                   default=str(V3_DIR / "outputs" / "yield_optimization"
                               / "stage06G_top_recipes.csv"))
    p.add_argument("--reps_06g_csv", type=str,
                   default=str(V3_DIR / "outputs" / "yield_optimization"
                               / "stage06G_representative_recipes.csv"))
    p.add_argument("--strict_cfg_yaml", type=str,
                   default=str(V3_DIR / "outputs" / "yield_optimization"
                               / "stage06G_strict_score_config.yaml"))
    p.add_argument("--fd_top100_csv", type=str,
                   default=str(V3_DIR / "outputs" / "labels"
                               / "stage06H_fd_top100_nominal.csv"))
    p.add_argument("--fd_summary_json", type=str,
                   default=str(V3_DIR / "outputs" / "logs"
                               / "stage06H_fd_verification_summary.json"))
    p.add_argument("--config", type=str,
                   default=str(V3_DIR / "configs" / "yield_optimization.yaml"))
    args = p.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    strict_yaml = yaml.safe_load(Path(args.strict_cfg_yaml).read_text())
    cd_tol  = float(strict_yaml["thresholds"]["cd_tol_nm"])
    ler_cap = float(strict_yaml["thresholds"]["ler_cap_nm"])

    # 06G top-100 surrogate metrics.
    rows_06g = read_labels_csv(args.top_06g_csv)
    num_06g = ["rank_strict", "strict_score", "yield_score",
                "p_robust_valid", "p_margin_risk", "p_under_exposed",
                "p_merged", "p_roughness_degraded", "p_numerical_invalid",
                "mean_cd_fixed", "std_cd_fixed",
                "mean_ler_locked", "std_ler_locked",
                "mean_p_line_margin", "std_p_line_margin"] + FEATURE_KEYS
    for r in rows_06g:
        for k in num_06g:
            if k in r: r[k] = _safe_float(r[k])
    sur_lookup = {r["recipe_id"]: r for r in rows_06g}

    # 06G representatives (kind annotation).
    rep_06g_kind = {}
    for r in read_labels_csv(args.reps_06g_csv):
        rep_06g_kind[r["recipe_id"]] = str(r.get("kind", ""))

    # 06H FD top-100 nominal (single FD row per recipe).
    fd_nominal = read_labels_csv(args.fd_top100_csv)
    for r in fd_nominal:
        for k in ["CD_final_nm", "CD_locked_nm", "LER_CD_locked_nm",
                   "area_frac", "P_line_margin", "contrast",
                   "psd_locked_mid", "rank_strict"]:
            if k in r: r[k] = _safe_float(r[k])
    fd_nom_by_id = {r.get("source_recipe_id", ""): r for r in fd_nominal}

    # 06H FD MC aggregates: top-10 by strict_score and 6 representatives.
    fd_summary = json.loads(Path(args.fd_summary_json).read_text())
    mc_lookup: dict[str, dict] = {}
    for r in fd_summary.get("fd_top10_mc_aggr", []):
        mc_lookup[r["recipe_id"]] = {**r, "_source": "top10_mc"}
    for r in fd_summary.get("fd_rep_mc_aggr", []):
        # rep MC has 300 variations vs top-10 MC's 100, so prefer rep
        # numbers when both exist.
        existing = mc_lookup.get(r["recipe_id"], {})
        merged = {**r,
                   "_source": "representative_mc_300var",
                   "rep_kind": r.get("rep_kind", "")}
        if existing.get("_source") == "top10_mc":
            merged["_top10_mc_strict_pass_100var"] = float(existing.get("p_strict_pass", float("nan")))
        mc_lookup[r["recipe_id"]] = merged

    # Build manifest entries.
    manifest_entries: list[dict] = []
    decision_rows: list[dict] = []
    for rid, role in MANIFEST_RECIPES:
        sur = sur_lookup.get(rid, {})
        nom = fd_nom_by_id.get(rid, {})
        mc = mc_lookup.get(rid, {})
        cd_err_nom = abs(_safe_float(nom.get("CD_final_nm")) - CD_TARGET_NM) \
            if nom else float("nan")
        # FD MC strict_pass_prob -- prefer the rep_mc 300-var number; else top10_mc.
        strict_pass_mc = float(mc.get("p_strict_pass",
                                         mc.get("strict_pass_prob", float("nan"))))
        std_cd_mc = float(mc.get("std_cd_final",
                                    mc.get("std_CD_final_nm", float("nan"))))
        std_ler_mc = float(mc.get("std_ler_locked",
                                     mc.get("std_LER_CD_locked_nm", float("nan"))))
        mean_cd_mc = float(mc.get("mean_cd_final",
                                     mc.get("mean_CD_final_nm", float("nan"))))
        mean_ler_mc = float(mc.get("mean_ler_locked",
                                      mc.get("mean_LER_CD_locked_nm", float("nan"))))
        mean_margin_mc = float(mc.get("mean_p_line_margin",
                                          mc.get("mean_P_line_margin", float("nan"))))

        # Selection rationale, weakness, when-to-prefer per role.
        rationale_table = {
            "fd_stability_best": (
                "FD MC strict_pass_prob = 0.680 -- highest probability of "
                "clearing the strict CD ±0.5 nm + LER ≤ 3.0 nm thresholds "
                "under 100x process variation. Also dominant on FD nominal "
                "CD accuracy (CD_error = 0.003 nm). Was 06G surrogate rank "
                "#3 by strict_score; FD ground-truth promoted it above the "
                "surrogate's #1 (G_3691, FD strict_pass = 0.430).",
                "Mid-range MC std on CD (0.485) and LER (0.021) -- not the "
                "absolute lowest variability, but still well within the "
                "strict thresholds. The surrogate did not flag this as the "
                "winner; the 06H ranking flip is purely an FD discovery.",
                "Default Mode A recipe. Best overall on FD MC stability and "
                "FD nominal CD accuracy at the same time -- no single "
                "competing recipe dominates it on both axes.",
            ),
            "fd_stability_co_winner": (
                "FD MC strict_pass_prob = 0.680, tied with G_4867. Was 06G "
                "surrogate rank #9 -- another FD discovery the surrogate "
                "ranked far below its actual MC quality.",
                "Slightly worse FD nominal CD accuracy than G_4867 "
                "(CD_error 0.123 nm vs 0.003 nm). LER very close (2.522 "
                "vs 2.511 nm).",
                "Backup primary if G_4867 fails downstream verification or "
                "if hardware constraints rule out G_4867's specific "
                "parameter values.",
            ),
            "fd_stability_runner_up": (
                "FD MC strict_pass_prob = 0.630 -- next-best MC stability "
                "after the two co-winners. FD nominal CD_error = 0.049 nm "
                "and FD nominal LER = 2.509 nm -- among the best on both "
                "axes.",
                "Slightly less stable than the two leaders.",
                "Third option for MC-robust selection. A reasonable "
                "alternate when both G_4867 and G_1096 are unavailable.",
            ),
            "strict_best": (
                "06G surrogate strict_score #1. Indicative of where the "
                "surrogate believed the strict optimum lives.",
                "FD ground truth disagrees: nominal CD_error = 0.630 nm "
                "(20x larger than the FD CD-best G_4867's 0.003 nm) and "
                "MC strict_pass_prob = 0.430 -- only #8 by FD. The "
                "surrogate's #1 was incorrect.",
                "Reference for studying surrogate-vs-FD discrepancy; not "
                "for production deployment.",
            ),
            "cd_best": (
                "06G surrogate's CD-best pick. Surrogate predicted nominal "
                "CD_error = 0.008 nm -- the smallest in the 06G top-100.",
                "FD ground truth disagrees by ~100x: actual nominal "
                "CD_error = 0.793 nm. The aux CD_fixed regressor was off "
                "by 0.79 nm on this specific recipe -- this is the largest "
                "single-recipe surrogate-vs-FD discrepancy in the manifest. "
                "MC strict_pass_prob = 0.287 confirms it is not stable. "
                "The actual FD CD-best in this manifest is G_4867 at "
                "0.003 nm.",
                "Reference only -- the surrogate prediction failed for "
                "this recipe. For real CD targeting use G_4867.",
            ),
            "ler_best": (
                "Lowest FD nominal LER among the manifest (2.479 nm). "
                "Was 06G surrogate's LER-best pick.",
                "FD nominal CD_error = 1.200 nm -- worst CD accuracy in "
                "the manifest. MC strict_pass_prob = 0.090 -- lowest of "
                "all manifest recipes; the nominal LER advantage does not "
                "survive process variation.",
                "Reference only -- low LER is offset by poor CD accuracy "
                "and very poor MC stability. Not suitable as a production "
                "candidate.",
            ),
            "margin_best": (
                "Highest 06G surrogate-predicted P_line_margin. FD MC "
                "strict_pass_prob = 0.467 -- fifth-best in the manifest.",
                "FD nominal CD_error = 0.491 nm and LER = 2.539 nm "
                "(highest LER in the manifest, just above the 06G "
                "surrogate's nominal LER ranking).",
                "Solid alternate when margin is the leading metric and "
                "MC stability up to ~47% is acceptable. Less robust than "
                "the three FD-stability leaders.",
            ),
            "balanced": (
                "06G balanced z-score winner across nominal CD_error, LER, "
                "MC std, defect_prob and (negative) margin.",
                "FD MC strict_pass_prob = 0.297 -- nominal-balanced by 06G "
                "weights did not translate to MC-stability winner. FD "
                "nominal CD_error = 0.762 nm.",
                "Reference for surrogate balanced ranking; for FD-stable "
                "production use prefer G_4867 / G_1096 / G_715.",
            ),
        }
        rationale, weakness, when_to_prefer = rationale_table[role]

        entry = {
            "recipe_id":       rid,
            "role":            role,
            "is_primary":      bool(rid == PRIMARY_RECOMMENDED),
            "parameters": {k: float(_safe_float(sur.get(k))) for k in FEATURE_KEYS},
            "nominal_fd": {
                "label":            str(nom.get("label", "")) if nom else "",
                "CD_final_nm":      _safe_float(nom.get("CD_final_nm")) if nom else float("nan"),
                "CD_error_nm":      float(cd_err_nom),
                "CD_locked_nm":     _safe_float(nom.get("CD_locked_nm")) if nom else float("nan"),
                "LER_CD_locked_nm": _safe_float(nom.get("LER_CD_locked_nm")) if nom else float("nan"),
                "area_frac":        _safe_float(nom.get("area_frac")) if nom else float("nan"),
                "P_line_margin":    _safe_float(nom.get("P_line_margin")) if nom else float("nan"),
                "contrast":         _safe_float(nom.get("contrast")) if nom else float("nan"),
                "psd_locked_mid":   _safe_float(nom.get("psd_locked_mid")) if nom else float("nan"),
                "strict_score_06g_surrogate":   _safe_float(sur.get("strict_score")),
                "yield_score_06g_surrogate":    _safe_float(sur.get("yield_score")),
                "rank_06g_surrogate_strict":    int(_safe_float(sur.get("rank_strict", 0))) or None,
            },
            "mc_fd": {
                "n_mc":               int(_safe_float(mc.get("n_mc", 0))),
                "robust_valid_prob":  float(mc.get("robust_prob", float("nan"))),
                "margin_risk_prob":   float(mc.get("margin_risk_prob", float("nan"))),
                "defect_prob":        float(mc.get("defect_prob", float("nan"))),
                "strict_pass_prob":   strict_pass_mc,
                "mean_CD_final_nm":   mean_cd_mc,
                "std_CD_final_nm":    std_cd_mc,
                "mean_LER_CD_locked_nm": mean_ler_mc,
                "std_LER_CD_locked_nm":  std_ler_mc,
                "mean_P_line_margin": mean_margin_mc,
                "source":             str(mc.get("_source", "")),
            },
            "selection_rationale": rationale,
            "known_weakness":      weakness,
            "when_to_prefer":      when_to_prefer,
        }
        manifest_entries.append(entry)

        decision_rows.append({
            "recipe_id":          rid,
            "role":                role,
            "is_primary":          bool(rid == PRIMARY_RECOMMENDED),
            "fd_mc_strict_pass_prob": strict_pass_mc,
            "fd_nominal_CD_error_nm": cd_err_nom,
            "fd_nominal_LER":         _safe_float(nom.get("LER_CD_locked_nm")) if nom else float("nan"),
            "fd_nominal_P_line_margin": _safe_float(nom.get("P_line_margin")) if nom else float("nan"),
            "fd_mc_robust_prob":      float(mc.get("robust_prob", float("nan"))),
            "fd_mc_std_CD":           std_cd_mc,
            "fd_mc_std_LER":          std_ler_mc,
            "rank_06g_surrogate":     int(_safe_float(sur.get("rank_strict", 0))) or None,
        })

    # ----- Compute within-manifest ranks (FD truth) -----
    # Lower rank = better; ties get the same rank.
    def _rank_by(key: str, *, reverse: bool) -> dict:
        vals = [(r["recipe_id"], r[key]) for r in decision_rows]
        # stable sort: best first.
        sign = -1.0 if reverse else 1.0
        vals.sort(key=lambda kv: sign * (kv[1] if kv[1] == kv[1] else 0.0))
        rank: dict = {}
        last = None; cur = 0
        for i, (rid, v) in enumerate(vals, start=1):
            if v != last:
                cur = i; last = v
            rank[rid] = cur
        return rank

    rank_strict   = _rank_by("fd_mc_strict_pass_prob",       reverse=True)
    rank_cd       = _rank_by("fd_nominal_CD_error_nm",       reverse=False)
    rank_ler      = _rank_by("fd_nominal_LER",                 reverse=False)
    rank_margin   = _rank_by("fd_nominal_P_line_margin",       reverse=True)
    rank_robust   = _rank_by("fd_mc_robust_prob",              reverse=True)

    use_case_table = {
        "fd_stability_best":      "default Mode A recipe (FD MC stability winner)",
        "fd_stability_co_winner": "alternate primary if G_4867 unavailable",
        "fd_stability_runner_up": "third option for MC-robust selection",
        "strict_best":             "diagnostic: surrogate-vs-FD discrepancy reference",
        "cd_best":                 "best nominal CD accuracy (frozen process only)",
        "ler_best":                "diagnostic only -- MC stability is poor",
        "margin_best":             "secondary candidate when margin is the leading metric",
        "balanced":                "diagnostic: surrogate balanced-best reference",
    }
    final_rec_table = {
        "fd_stability_best":      "RECOMMENDED",
        "fd_stability_co_winner": "RECOMMENDED (backup primary)",
        "fd_stability_runner_up": "RECOMMENDED (third option)",
        "strict_best":             "DIAGNOSTIC ONLY",
        "cd_best":                 "USE CASE-DEPENDENT",
        "ler_best":                "DIAGNOSTIC ONLY",
        "margin_best":             "RECOMMENDED (secondary)",
        "balanced":                "DIAGNOSTIC ONLY",
    }
    for r in decision_rows:
        rid = r["recipe_id"]
        r["strict_score_rank"] = rank_strict[rid]
        r["CD_error_rank"]     = rank_cd[rid]
        r["LER_rank"]          = rank_ler[rid]
        r["margin_rank"]       = rank_margin[rid]
        r["robustness_rank"]   = rank_robust[rid]
        r["recommended_use_case"] = use_case_table.get(r["role"], "")
        r["final_recommendation"]   = final_rec_table.get(r["role"], "")

    # ----- Write manifest YAML -----
    manifest_payload = {
        "stage": "06I",
        "policy": {
            **cfg["policy"],
            "external_calibration": "none",
        },
        "scope": {
            "mode_a_fixed_design": {
                "pitch_nm":       cfg["mode_a_fixed_design"]["fixed"]["pitch_nm"],
                "line_cd_ratio":  cfg["mode_a_fixed_design"]["fixed"]["line_cd_ratio"],
                "abs_len_nm":     cfg["mode_a_fixed_design"]["fixed"]["abs_len_nm"],
            },
            "mode_b_open_design": "out of scope -- deferred to a later stage",
        },
        "strict_thresholds": {
            "cd_tolerance_nm": cd_tol,
            "ler_cap_nm":      ler_cap,
            "rationale_source": "stage06G_threshold_selection.md",
        },
        "ranking_authority": (
            "FD MC strict_pass_prob is the final ranking authority for "
            "Mode A recipe selection. The 06G surrogate's strict_score "
            "is treated as candidate proposal / screening only. The 06H "
            "surrogate refresh did not improve strict ranking (only 1 of "
            "6 06G representatives stayed in 06H top-20), so this manifest "
            "uses FD truth, not surrogate ranks."
        ),
        "primary_recommended_recipe":      PRIMARY_RECOMMENDED,
        "primary_recommended_recipe_kind": "fd_stability_best",
        "primary_selection_note": (
            f"FD MC truth places G_4867 (strict_pass_prob = 0.680) above "
            f"the 06G surrogate's strict-score #1 G_3691 (0.430) and "
            f"above the 06G surrogate's margin-best G_4299 (0.467). The "
            f"earlier note in the spec saying 'if G_4299 is the current "
            f"primary winner' was written before 06H FD MC ran -- 06H "
            f"FD truth supersedes that placeholder."
        ),
        "representatives": manifest_entries,
        "limitations": [
            "Nominal v2 physics only -- not externally calibrated.",
            "FD ranking authority is internal model truth, not real-fab "
            "yield. Do not interpret strict_pass_prob as wafer-level yield.",
            "Mode A only -- pitch=24 nm, line_cd_ratio=0.52 fixed.",
            "Mode B open-design exploration is deferred to a later stage.",
            "The 06H surrogate was retrained at n_estimators=200 on the "
            "4-target regressor to keep the joblib under 100 MB, vs "
            "n_estimators=300 on classifier and aux. Test-set metrics "
            "are unchanged at 3-decimal precision.",
        ],
    }
    yopt_dir = V3_DIR / "outputs" / "yield_optimization"
    yopt_dir.mkdir(parents=True, exist_ok=True)
    (yopt_dir / "stage06I_mode_a_final_recipes.yaml").write_text(
        yaml.safe_dump(manifest_payload, sort_keys=False, default_flow_style=False))

    # ----- Write decision table CSV -----
    cols = [
        "recipe_id", "role", "is_primary",
        "strict_score_rank", "CD_error_rank", "LER_rank",
        "margin_rank", "robustness_rank",
        "fd_mc_strict_pass_prob", "fd_mc_robust_prob",
        "fd_nominal_CD_error_nm", "fd_nominal_LER",
        "fd_nominal_P_line_margin",
        "fd_mc_std_CD", "fd_mc_std_LER",
        "rank_06g_surrogate",
        "recommended_use_case", "final_recommendation",
    ]
    with (yopt_dir / "stage06I_final_decision_table.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in decision_rows:
            w.writerow(r)

    # ----- Console summary -----
    print(f"\nStage 06I -- Mode A final recipe manifest")
    print(f"  primary recommended: {PRIMARY_RECOMMENDED}  "
          f"(strict_pass_prob = "
          f"{[r['fd_mc_strict_pass_prob'] for r in decision_rows if r['recipe_id'] == PRIMARY_RECOMMENDED][0]:.3f})")
    print(f"  manifest entries: {len(manifest_entries)}")
    print(f"  decision table:")
    print(f"  {'recipe':>10} {'role':>26} {'strict_pass':>12} {'CD_err':>7} {'LER':>7} "
           f"{'std_CD':>7} {'final_rec':>30}")
    for r in decision_rows:
        print(f"  {r['recipe_id']:>10} {r['role']:>26} "
              f"{r['fd_mc_strict_pass_prob']:>12.3f} "
              f"{r['fd_nominal_CD_error_nm']:>7.3f} "
              f"{r['fd_nominal_LER']:>7.3f} "
              f"{r['fd_mc_std_CD']:>7.3f} "
              f"{r['final_recommendation']:>30}")
    print(f"  manifest -> {yopt_dir / 'stage06I_mode_a_final_recipes.yaml'}")
    print(f"  decision -> {yopt_dir / 'stage06I_final_decision_table.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
