"""Stage 06J -- Mode B open-design exploration.

Re-runs the surrogate-driven optimisation pipeline (06G shape) but
with pitch_nm and line_cd_ratio also varying. Tests whether opening
the design space surfaces recipes that combine G_4867-like CD
accuracy with a wider time / process-window budget than the closed
Mode A solution offers.

Sampling / scoring:
    1. Sobol 5,000 candidates over the full mode_b_open_design space
       (11 axes including pitch + line_cd_ratio).
    2. 200 process variations per candidate, 06H surrogate + Stage
       06G strict_score formula. Adds a Gaussian strict_pass_prob
       proxy from the regressor mean/std for ranking.
    3. Build six top-100 lists: strict_score, CD_error, LER,
       balanced z-score, P_line_margin, geometry-novelty (far from
       Mode A pitch=24 / line_cd_ratio=0.52).
    4. Light nominal FD sanity check on the deduplicated union of
       top-20 strict + top-10 CD + top-10 balanced. Then FD MC (100
       variations each) on 5 selected candidates.

Mode A is NOT modified -- this stage uses outputs/yield_optimization
as a separate Stage 06J namespace and reads 06H joblibs read-only.

Policy
    v2_OP_frozen           = true
    published_data_loaded  = false
    No external calibration. Closed Stage 04C / 04D / 06C joblibs and
    Mode A artefacts are not mutated.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reaction_diffusion_peb_v3_screening.src.candidate_sampler import (
    CandidateSpace,
    sample_candidates,
)
from reaction_diffusion_peb_v3_screening.src.fd_batch_runner import (
    run_one_candidate,
)
from reaction_diffusion_peb_v3_screening.src.fd_yield_score import (
    nominal_yield_score,
)
from reaction_diffusion_peb_v3_screening.src.labeler import LabelThresholds
from reaction_diffusion_peb_v3_screening.src.metrics_io import (
    FEATURE_KEYS,
    LABEL_CSV_COLUMNS,
    load_model,
)
from reaction_diffusion_peb_v3_screening.src.process_variation import (
    VariationSpec,
    sample_variations,
)
from reaction_diffusion_peb_v3_screening.src.yield_optimizer import (
    YieldScoreConfig,
    evaluate_recipes,
    evaluate_single_recipe,
)

# Reuse Stage 06G strict_score formula.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_stage06g_strict_optimization import (  # noqa: E402
    StrictScoreConfig,
    compute_strict_score,
)


V3_DIR = ROOT / "reaction_diffusion_peb_v3_screening"
CD_TARGET_NM = 15.0


def _safe_float(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def _zscore(x: np.ndarray) -> np.ndarray:
    finite = np.isfinite(x)
    if finite.sum() == 0:
        return np.zeros_like(x)
    mu = float(np.nanmean(x[finite]))
    sd = float(np.nanstd(x[finite]))
    if sd <= 1e-12:
        return np.zeros_like(x)
    return np.where(finite, (x - mu) / sd, 0.0)


# --------------------------------------------------------------------------
# Surrogate strict_pass_prob proxy (Gaussian).
# --------------------------------------------------------------------------
def _gauss_p_in(mu: float, sd: float, lo: float, hi: float) -> float:
    if not np.isfinite(mu) or not np.isfinite(sd):
        return float("nan")
    if sd <= 1e-12:
        return float(1.0 if lo <= mu <= hi else 0.0)
    from math import erf, sqrt
    z_hi = (hi - mu) / (sd * sqrt(2)); z_lo = (lo - mu) / (sd * sqrt(2))
    return float(0.5 * (erf(z_hi) - erf(z_lo)))


def attach_strict_pass_proxy(rows: list[dict],
                                cd_tol: float, ler_cap: float) -> None:
    for r in rows:
        cd_mu = _safe_float(r.get("mean_cd_fixed"))
        cd_sd = _safe_float(r.get("std_cd_fixed"))
        ler_mu = _safe_float(r.get("mean_ler_locked"))
        ler_sd = _safe_float(r.get("std_ler_locked"))
        p_rob = _safe_float(r.get("p_robust_valid"))
        p_cd = _gauss_p_in(cd_mu, cd_sd,
                              CD_TARGET_NM - cd_tol, CD_TARGET_NM + cd_tol)
        p_ler = _gauss_p_in(ler_mu, ler_sd, -np.inf, ler_cap)
        proxy = float(p_rob * p_cd * p_ler) \
            if all(np.isfinite([p_rob, p_cd, p_ler])) else float("nan")
        r["strict_pass_prob_proxy"] = proxy
        r["mean_cd_error"] = abs(cd_mu - CD_TARGET_NM) if np.isfinite(cd_mu) else float("nan")


# --------------------------------------------------------------------------
# Top-N selection helpers.
# --------------------------------------------------------------------------
def _topk_by(rows: list[dict], key: str, n: int, *, reverse: bool) -> list[dict]:
    sign = -1.0 if reverse else 1.0
    finite = [r for r in rows if np.isfinite(_safe_float(r.get(key)))]
    finite.sort(key=lambda r: sign * float(r[key]))
    return finite[:n]


def attach_balanced_score(rows: list[dict]) -> None:
    cd_err = np.array([_safe_float(r.get("mean_cd_error"))      for r in rows])
    ler    = np.array([_safe_float(r.get("mean_ler_locked"))    for r in rows])
    cd_std = np.array([_safe_float(r.get("std_cd_fixed"))        for r in rows])
    ler_std= np.array([_safe_float(r.get("std_ler_locked"))      for r in rows])
    margin = np.array([_safe_float(r.get("mean_p_line_margin")) for r in rows])
    p_rob  = np.array([_safe_float(r.get("p_robust_valid"))      for r in rows])
    defect = np.array([
        _safe_float(r.get("p_under_exposed", 0.0))
        + _safe_float(r.get("p_merged", 0.0))
        + _safe_float(r.get("p_roughness_degraded", 0.0))
        + _safe_float(r.get("p_numerical_invalid", 0.0))
        for r in rows
    ])
    z = (
        _zscore(cd_err) + _zscore(ler) + _zscore(cd_std) + _zscore(ler_std)
        + 2.0 * defect - _zscore(margin)
    )
    for i, r in enumerate(rows):
        r["balanced_score_06j"] = float(z[i])


def _norm_geometry_distance(r: dict, mode_a_pitch: float = 24.0,
                              mode_a_ratio: float = 0.52) -> float:
    """Normalised distance from the Mode A (pitch, line_cd_ratio) point."""
    p = _safe_float(r.get("pitch_nm")); rt = _safe_float(r.get("line_cd_ratio"))
    pitch_lo, pitch_hi = 18.0, 32.0
    ratio_lo, ratio_hi = 0.45, 0.60
    pn = (p - pitch_lo) / (pitch_hi - pitch_lo)
    p0 = (mode_a_pitch - pitch_lo) / (pitch_hi - pitch_lo)
    rn = (rt - ratio_lo) / (ratio_hi - ratio_lo)
    r0 = (mode_a_ratio - ratio_lo) / (ratio_hi - ratio_lo)
    return float(np.sqrt((pn - p0) ** 2 + (rn - r0) ** 2))


# --------------------------------------------------------------------------
# Recipe row -> FD candidate.
# --------------------------------------------------------------------------
def _row_to_fd_candidate(row: dict, space: CandidateSpace, _id: str) -> dict:
    out = {}
    for k in FEATURE_KEYS:
        out[k] = float(row[k])
    out["pitch_nm"]    = float(out["pitch_nm"])
    out["line_cd_nm"]  = out["pitch_nm"] * out["line_cd_ratio"]
    out["domain_x_nm"] = out["pitch_nm"] * 5.0
    out["dose_norm"]   = out["dose_mJ_cm2"] / float(space.fixed["reference_dose_mJ_cm2"])
    for fk, fv in space.fixed.items():
        out.setdefault(fk, fv)
    out["_id"] = _id
    return out


def _v2_baseline_recipe(space: CandidateSpace, op: dict) -> dict:
    base = {p["name"]: (p["values"][0] if p["type"] == "choice" else p["low"])
            for p in space.parameters}
    base.update({k: float(v) if not isinstance(v, int) else int(v)
                 for k, v in op.items()})
    base["pitch_nm"]    = float(base["pitch_nm"])
    base["line_cd_nm"]  = base["pitch_nm"] * float(base["line_cd_ratio"])
    base["domain_x_nm"] = base["pitch_nm"] * 5.0
    base["dose_norm"]   = base["dose_mJ_cm2"] / float(space.fixed["reference_dose_mJ_cm2"])
    for fk, fv in space.fixed.items():
        base.setdefault(fk, fv)
    base["_id"] = "v2_frozen_op"
    return base


# --------------------------------------------------------------------------
# CSV writers.
# --------------------------------------------------------------------------
SUMMARY_OUT_COLS = [
    "recipe_id", "rank_strict",
    "strict_score", "yield_score",
    "strict_pass_prob_proxy",
    "p_robust_valid", "p_margin_risk",
    "p_under_exposed", "p_merged",
    "p_roughness_degraded", "p_numerical_invalid",
    "mean_cd_fixed", "std_cd_fixed",
    "mean_cd_error",
    "mean_cd_locked", "std_cd_locked",
    "mean_ler_locked", "std_ler_locked",
    "mean_p_line_margin", "std_p_line_margin",
    "balanced_score_06j",
    "geometry_distance_to_modeA",
    "strict_cd_pen", "strict_ler_pen",
    "strict_cd_std_pen", "strict_ler_std_pen", "strict_margin_bonus",
    "cd_error_penalty", "ler_penalty",
] + FEATURE_KEYS


def _write_csv(rows: list[dict], path: Path,
                column_order: list[str] | None = None) -> None:
    if not rows:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as f:
            csv.DictWriter(f, fieldnames=column_order or []).writeheader()
        return
    cols = column_order if column_order else list(rows[0].keys())
    extras = []
    for r in rows:
        for k in r.keys():
            if k not in cols and k not in extras:
                extras.append(k)
    cols = cols + extras
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _open_fd_csv(path: Path):
    extra = ["source_recipe_id", "role", "phase", "variation_idx",
              "strict_score_surrogate", "yield_score_surrogate",
              "strict_pass_prob_proxy_surrogate"]
    cols = LABEL_CSV_COLUMNS + extra
    path.parent.mkdir(parents=True, exist_ok=True)
    f = path.open("w", newline="")
    w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
    w.writeheader()
    return f, w


def _write_fd_row(writer, row, *, source_recipe_id, role, phase,
                    variation_idx, sur):
    out = dict(row)
    out["source_recipe_id"]                = source_recipe_id
    out["role"]                             = role
    out["phase"]                            = phase
    out["variation_idx"]                    = int(variation_idx)
    out["strict_score_surrogate"]          = float(sur.get("strict_score", float("nan")))
    out["yield_score_surrogate"]           = float(sur.get("yield_score", float("nan")))
    out["strict_pass_prob_proxy_surrogate"] = float(sur.get("strict_pass_prob_proxy", float("nan")))
    writer.writerow(out)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str,
                   default=str(V3_DIR / "configs" / "yield_optimization.yaml"))
    p.add_argument("--clf", type=str,
                   default=str(V3_DIR / "outputs" / "models"
                               / "stage06H_classifier.joblib"))
    p.add_argument("--reg", type=str,
                   default=str(V3_DIR / "outputs" / "models"
                               / "stage06H_regressor.joblib"))
    p.add_argument("--aux", type=str,
                   default=str(V3_DIR / "outputs" / "models"
                               / "stage06H_aux_cd_fixed_regressor.joblib"))
    p.add_argument("--strict_cfg_yaml", type=str,
                   default=str(V3_DIR / "outputs" / "yield_optimization"
                               / "stage06G_strict_score_config.yaml"))
    p.add_argument("--n_candidates", type=int, default=5000)
    p.add_argument("--n_var",        type=int, default=200)
    p.add_argument("--seed",         type=int, default=7373,
                   help="Sobol seed -- distinct from prior Mode A seeds.")
    p.add_argument("--n_top_strict",   type=int, default=20)
    p.add_argument("--n_top_cd",       type=int, default=10)
    p.add_argument("--n_top_balanced", type=int, default=10)
    p.add_argument("--n_mc_per_recipe", type=int, default=100)
    p.add_argument("--label_schema",
                   default=str(V3_DIR / "configs" / "label_schema.yaml"))
    p.add_argument("--skip_fd_sanity", action="store_true")
    p.add_argument("--skip_fd_mc",     action="store_true")
    args = p.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    space = CandidateSpace.from_yaml(cfg["candidate_space_yaml"])
    var_spec = VariationSpec.from_yaml_dict(cfg["process_variation"])
    score_cfg = YieldScoreConfig.from_yaml_dict(cfg["yield_score"])

    strict_yaml = yaml.safe_load(Path(args.strict_cfg_yaml).read_text())
    cd_tol  = float(strict_yaml["thresholds"]["cd_tol_nm"])
    ler_cap = float(strict_yaml["thresholds"]["ler_cap_nm"])
    strict_cfg = StrictScoreConfig(cd_tol_nm=cd_tol, ler_cap_nm=ler_cap)

    print(f"Stage 06J -- Mode B open-design exploration")
    print(f"  policy: v2_OP_frozen={cfg['policy']['v2_OP_frozen']}, "
          f"published_data_loaded={cfg['policy']['published_data_loaded']}")
    print(f"  Sobol seed: {args.seed}")
    print(f"  n_candidates={args.n_candidates}  n_var={args.n_var}")
    print(f"  strict thresholds: cd_tol={cd_tol}, ler_cap={ler_cap}")

    # ----- Sample Mode B candidates (open design, no fixed-design wrapper) -----
    candidates = sample_candidates(
        space, n=args.n_candidates,
        method=cfg["sampling"]["method"], seed=args.seed,
    )
    for c in candidates:
        c["_id"] = f"J_{c['_id']}"
    pitch_count = {}
    ratio_count = {}
    for c in candidates:
        pitch_count[c["pitch_nm"]] = pitch_count.get(c["pitch_nm"], 0) + 1
        ratio_count[c["line_cd_ratio"]] = ratio_count.get(c["line_cd_ratio"], 0) + 1
    print(f"  pitch distribution in Sobol pool: {pitch_count}")
    print(f"  ratio distribution in Sobol pool: {ratio_count}")

    # ----- Surrogate scoring -----
    clf, _ = load_model(args.clf)
    reg, _ = load_model(args.reg)
    aux, _ = load_model(args.aux)

    op_base = _v2_baseline_recipe(space, cfg["v2_frozen_op"])
    baseline_row = evaluate_single_recipe(
        op_base, clf, reg, aux,
        var_spec, args.n_var, space, score_cfg, seed=args.seed,
    )
    baseline_row.update(compute_strict_score(baseline_row, strict_cfg))
    attach_strict_pass_proxy([baseline_row], cd_tol, ler_cap)
    print(f"  v2 frozen OP under 06H surrogate: yield={baseline_row['yield_score']:.4f} "
          f"strict={baseline_row['strict_score']:.4f} proxy={baseline_row['strict_pass_prob_proxy']:.3f}")

    print(f"\n  scoring with 06H surrogate (n_var={args.n_var}) ...")
    t0 = time.time()
    rows = evaluate_recipes(
        candidates, clf, reg, aux,
        var_spec, args.n_var, space, score_cfg, seed=args.seed + 1,
    )
    for r in rows:
        r.update(compute_strict_score(r, strict_cfg))
    attach_strict_pass_proxy(rows, cd_tol, ler_cap)
    for r in rows:
        r["geometry_distance_to_modeA"] = _norm_geometry_distance(r)
    attach_balanced_score(rows)
    rows.sort(key=lambda r: -float(r["strict_score"]))
    for i, r in enumerate(rows, start=1):
        r["rank_strict"] = i
    print(f"  scored {len(rows)} recipes in {time.time() - t0:.1f}s")

    # ----- Top lists -----
    top_strict   = rows[:100]
    top_cd       = _topk_by(rows, "mean_cd_error",       100, reverse=False)
    top_ler      = _topk_by(rows, "mean_ler_locked",     100, reverse=False)
    top_balanced = _topk_by(rows, "balanced_score_06j",  100, reverse=False)
    top_margin   = _topk_by(rows, "mean_p_line_margin",  100, reverse=True)
    # Novelty: high strict_score AND large geometry distance from Mode A.
    novelty = sorted(
        [r for r in rows[:200] if r["geometry_distance_to_modeA"] > 0.30],
        key=lambda r: -float(r["strict_score"]),
    )[:100]

    yopt_dir = V3_DIR / "outputs" / "yield_optimization"
    labels_dir = V3_DIR / "outputs" / "labels"
    logs_dir = V3_DIR / "outputs" / "logs"

    _write_csv(rows, yopt_dir / "stage06J_mode_b_recipe_summary.csv",
                column_order=SUMMARY_OUT_COLS)
    _write_csv(top_strict,   yopt_dir / "stage06J_mode_b_top_recipes.csv",
                column_order=SUMMARY_OUT_COLS)

    # Tagged top-list table.
    role_lists = [
        ("strict_top",   top_strict),
        ("cd_top",       top_cd),
        ("ler_top",      top_ler),
        ("balanced_top", top_balanced),
        ("margin_top",   top_margin),
        ("novelty_top",  novelty),
    ]
    tagged_rows = []
    for tag, rs in role_lists:
        for rank_within, r in enumerate(rs, start=1):
            tagged_rows.append({**r, "list": tag, "rank_within_list": rank_within})
    _write_csv(tagged_rows, yopt_dir / "stage06J_mode_b_top_lists_tagged.csv",
                column_order=["list", "rank_within_list"] + SUMMARY_OUT_COLS)

    # ----- Mode A vs Mode B comparison row table -----
    cmp_rows = []
    cmp_rows.append({
        "stage":         "v2_frozen_op",
        "recipe_id":     "v2_frozen_op",
        "strict_score":  float(baseline_row["strict_score"]),
        "yield_score":   float(baseline_row["yield_score"]),
        "mean_cd_error": float(baseline_row["mean_cd_error"]),
        "mean_ler_locked": float(baseline_row["mean_ler_locked"]),
        "p_robust_valid": float(baseline_row["p_robust_valid"]),
        "mean_p_line_margin": float(baseline_row["mean_p_line_margin"]),
        "strict_pass_prob_proxy": float(baseline_row["strict_pass_prob_proxy"]),
        "pitch_nm":      24.0,
        "line_cd_ratio": 0.52,
    })
    # Mode A representatives: read parameters + best surrogate metrics from
    # the 06I manifest, but evaluate them under 06H+06J pipeline so the
    # numbers are apples-to-apples.
    manifest = yaml.safe_load(
        (V3_DIR / "outputs" / "yield_optimization" / "stage06I_mode_a_final_recipes.yaml").read_text())
    mode_a_eval_pool = []
    for rep in manifest["representatives"]:
        rid = rep["recipe_id"]
        params = {k: float(v) for k, v in rep["parameters"].items()}
        cand = {k: params[k] for k in FEATURE_KEYS}
        cand["pitch_nm"] = float(cand["pitch_nm"])
        cand["line_cd_nm"] = cand["pitch_nm"] * cand["line_cd_ratio"]
        cand["domain_x_nm"] = cand["pitch_nm"] * 5.0
        cand["dose_norm"] = cand["dose_mJ_cm2"] / float(space.fixed["reference_dose_mJ_cm2"])
        for fk, fv in space.fixed.items():
            cand.setdefault(fk, fv)
        cand["_id"] = f"modeA_{rid}"
        mode_a_eval_pool.append((rid, rep["role"], cand))
    if mode_a_eval_pool:
        mode_a_rows = evaluate_recipes(
            [c for _, _, c in mode_a_eval_pool], clf, reg, aux,
            var_spec, args.n_var, space, score_cfg, seed=args.seed + 2,
        )
        for r in mode_a_rows:
            r.update(compute_strict_score(r, strict_cfg))
        attach_strict_pass_proxy(mode_a_rows, cd_tol, ler_cap)
        for (rid, role, cand), mr in zip(mode_a_eval_pool, mode_a_rows):
            cmp_rows.append({
                "stage":          f"mode_a_{role}",
                "recipe_id":      rid,
                "strict_score":   float(mr["strict_score"]),
                "yield_score":    float(mr["yield_score"]),
                "mean_cd_error":  float(mr["mean_cd_error"]),
                "mean_ler_locked": float(mr["mean_ler_locked"]),
                "p_robust_valid": float(mr["p_robust_valid"]),
                "mean_p_line_margin": float(mr["mean_p_line_margin"]),
                "strict_pass_prob_proxy": float(mr["strict_pass_prob_proxy"]),
                "pitch_nm":       float(cand["pitch_nm"]),
                "line_cd_ratio":  float(cand["line_cd_ratio"]),
            })

    def _row_for(stage_name: str, r: dict) -> dict:
        return {
            "stage": stage_name,
            "recipe_id": r["recipe_id"],
            "strict_score": float(r["strict_score"]),
            "yield_score":  float(r["yield_score"]),
            "mean_cd_error": float(r["mean_cd_error"]),
            "mean_ler_locked": float(r["mean_ler_locked"]),
            "p_robust_valid": float(r["p_robust_valid"]),
            "mean_p_line_margin": float(r["mean_p_line_margin"]),
            "strict_pass_prob_proxy": float(r["strict_pass_prob_proxy"]),
            "pitch_nm": float(r["pitch_nm"]),
            "line_cd_ratio": float(r["line_cd_ratio"]),
        }

    if rows:
        cmp_rows.append(_row_for("mode_b_strict_best",   rows[0]))
        cmp_rows.append(_row_for("mode_b_cd_best",       top_cd[0]))
        cmp_rows.append(_row_for("mode_b_ler_best",      top_ler[0]))
        cmp_rows.append(_row_for("mode_b_balanced_best", top_balanced[0]))
        cmp_rows.append(_row_for("mode_b_margin_best",   top_margin[0]))
        if novelty:
            cmp_rows.append(_row_for("mode_b_novelty_best", novelty[0]))

    cmp_cols = ["stage", "recipe_id", "strict_score", "yield_score",
                "mean_cd_error", "mean_ler_locked", "p_robust_valid",
                "mean_p_line_margin", "strict_pass_prob_proxy",
                "pitch_nm", "line_cd_ratio"]
    _write_csv(cmp_rows, yopt_dir / "stage06J_mode_b_vs_mode_a_comparison.csv",
                column_order=cmp_cols)

    # ----- Light FD sanity -----
    fd_sanity_path = labels_dir / "stage06J_mode_b_fd_sanity.csv"
    fd_mc_path     = labels_dir / "stage06J_mode_b_fd_mc_optional.csv"

    sanity_pool: list[tuple[dict, str]] = []
    seen_ids: set[str] = set()
    for r in top_strict[:args.n_top_strict]:
        if r["recipe_id"] in seen_ids:
            continue
        sanity_pool.append((r, "strict_top"))
        seen_ids.add(r["recipe_id"])
    for r in top_cd[:args.n_top_cd]:
        if r["recipe_id"] in seen_ids:
            continue
        sanity_pool.append((r, "cd_top"))
        seen_ids.add(r["recipe_id"])
    for r in top_balanced[:args.n_top_balanced]:
        if r["recipe_id"] in seen_ids:
            continue
        sanity_pool.append((r, "balanced_top"))
        seen_ids.add(r["recipe_id"])

    fd_sanity_rows: list[dict] = []
    if not args.skip_fd_sanity:
        thresholds = LabelThresholds.from_yaml(args.label_schema)
        f, writer = _open_fd_csv(fd_sanity_path)
        t0 = time.time()
        try:
            for i, (r, role) in enumerate(sanity_pool, start=1):
                cand = _row_to_fd_candidate(r, space, _id=f"{r['recipe_id']}__nom")
                fd_row = run_one_candidate(cand, thresholds)
                _write_fd_row(writer, fd_row,
                                 source_recipe_id=str(r["recipe_id"]),
                                 role=role, phase="nominal", variation_idx=0, sur=r)
                f.flush()
                fd_sanity_rows.append({
                    **fd_row,
                    "source_recipe_id": r["recipe_id"],
                    "role": role,
                    "strict_score_surrogate": float(r["strict_score"]),
                })
                if i % 10 == 0:
                    rate = i / max(time.time() - t0, 1e-9)
                    print(f"    FD sanity: {i}/{len(sanity_pool)} done ({rate:.1f} runs/s)")
        finally:
            f.close()
        print(f"  FD sanity -> {fd_sanity_path} ({len(sanity_pool)} runs in {time.time() - t0:.1f}s)")

    # ----- Optional FD MC subset -----
    fd_mc_rows: list[dict] = []
    if not args.skip_fd_mc and fd_sanity_rows:
        # Pick targets:
        #   1. mode_b strict_best
        #   2. mode_b balanced_best
        #   3. mode_b cd_best
        #   4. closest-to-G_4867 mode_b candidate among top-200 strict
        #   5. most novel high-scoring candidate (largest geometry_distance among top-50)
        g4867 = next(rep for rep in manifest["representatives"]
                       if rep["recipe_id"] == "G_4867")
        g4867_params = {k: float(v) for k, v in g4867["parameters"].items()}

        def _knob_dist_to_g4867(r):
            d = 0.0; n = 0
            bounds = {"dose_mJ_cm2": (21.0, 60.0), "sigma_nm": (0.0, 3.0),
                       "DH_nm2_s": (0.3, 0.8), "time_s": (20.0, 45.0),
                       "Hmax_mol_dm3": (0.15, 0.22), "kdep_s_inv": (0.35, 0.65),
                       "Q0_mol_dm3": (0.0, 0.03), "kq_s_inv": (0.5, 2.0),
                       "abs_len_nm": (15.0, 100.0), "pitch_nm": (18.0, 32.0),
                       "line_cd_ratio": (0.45, 0.60)}
            for k in FEATURE_KEYS:
                lo, hi = bounds.get(k, (0.0, 1.0))
                w = max(hi - lo, 1e-12)
                a = float(r.get(k, np.nan)); b = float(g4867_params.get(k, np.nan))
                if np.isfinite(a) and np.isfinite(b):
                    d += ((a - b) / w) ** 2; n += 1
            return float(np.sqrt(d / max(n, 1)))

        for r in top_strict[:200]:
            r["_dist_g4867"] = _knob_dist_to_g4867(r)

        targets = []
        seen = set()
        for tag, src in [
            ("mode_b_strict_best",   rows[0] if rows else None),
            ("mode_b_balanced_best", top_balanced[0] if top_balanced else None),
            ("mode_b_cd_best",       top_cd[0] if top_cd else None),
            ("mode_b_closest_to_G_4867",
                min(top_strict[:200], key=lambda r: r["_dist_g4867"]) if top_strict else None),
            ("mode_b_most_novel",
                max(top_strict[:50], key=lambda r: r["geometry_distance_to_modeA"])
                    if top_strict else None),
        ]:
            if src is None:
                continue
            if src["recipe_id"] in seen:
                continue
            targets.append((src, tag))
            seen.add(src["recipe_id"])

        thresholds = LabelThresholds.from_yaml(args.label_schema)
        f, writer = _open_fd_csv(fd_mc_path)
        t0 = time.time()
        base_rng = np.random.default_rng(args.seed + 13)
        try:
            for k_idx, (r, tag) in enumerate(targets, start=1):
                cand = _row_to_fd_candidate(r, space, _id=f"{r['recipe_id']}__{tag}_base")
                sub_rng = np.random.default_rng(int(base_rng.integers(0, 2**31 - 1)))
                variations = sample_variations(cand, var_spec, args.n_mc_per_recipe,
                                                  space, rng=sub_rng)
                for j, v in enumerate(variations, start=1):
                    v["_id"] = f"{r['recipe_id']}__{tag}_mc{j:03d}"
                    for fk, fv in space.fixed.items():
                        v.setdefault(fk, fv)
                    v["domain_x_nm"] = float(v["pitch_nm"]) * 5.0
                    v["dose_norm"]   = float(v["dose_mJ_cm2"]) / float(space.fixed["reference_dose_mJ_cm2"])
                    fd_row = run_one_candidate(v, thresholds)
                    _write_fd_row(writer, fd_row,
                                     source_recipe_id=str(r["recipe_id"]),
                                     role=tag, phase="mc", variation_idx=j, sur=r)
                    f.flush()
                    fd_mc_rows.append({**fd_row, "source_recipe_id": r["recipe_id"], "role": tag})
                print(f"    FD MC: target {k_idx}/{len(targets)} ({tag}: "
                      f"{r['recipe_id']}) done ({args.n_mc_per_recipe} variations)")
        finally:
            f.close()
        print(f"  FD MC -> {fd_mc_path} ({len(fd_mc_rows)} runs in {time.time() - t0:.1f}s)")

    # ----- JSON summary -----
    pitch_top100 = [_safe_float(r.get("pitch_nm")) for r in top_strict]
    ratio_top100 = [_safe_float(r.get("line_cd_ratio")) for r in top_strict]
    pitch_dist_top100 = {p: int(sum(1 for x in pitch_top100 if x == p))
                            for p in sorted(set(pitch_top100))}
    ratio_dist_top100 = {r: int(sum(1 for x in ratio_top100 if x == r))
                            for r in sorted(set(ratio_top100))}

    n_at_modeA_geom = int(sum(
        1 for r in top_strict
        if _safe_float(r.get("pitch_nm")) == 24.0
            and _safe_float(r.get("line_cd_ratio")) == 0.52
    ))

    HARD_FAIL = {"under_exposed", "merged",
                  "roughness_degraded", "numerical_invalid"}
    n_sanity = len(fd_sanity_rows)
    n_sanity_robust = sum(1 for r in fd_sanity_rows if str(r.get("label", "")) == "robust_valid")
    n_sanity_strict_pass = sum(
        1 for r in fd_sanity_rows
        if str(r.get("label", "")) == "robust_valid"
            and abs(_safe_float(r.get("CD_final_nm")) - CD_TARGET_NM) <= cd_tol
            and _safe_float(r.get("LER_CD_locked_nm")) <= ler_cap
    )
    n_sanity_false_pass = sum(
        1 for r in fd_sanity_rows if str(r.get("label", "")) in HARD_FAIL
    )

    payload = {
        "stage":  "06J",
        "policy": cfg["policy"],
        "scope":  "mode_b_open_design (pitch + line_cd_ratio variable)",
        "seed":   int(args.seed),
        "n_candidates": int(len(rows)),
        "n_var_per_candidate": int(args.n_var),
        "strict_thresholds": {"cd_tol_nm": cd_tol, "ler_cap_nm": ler_cap},
        "v2_frozen_op_under_06h": {
            "yield_score":  float(baseline_row["yield_score"]),
            "strict_score": float(baseline_row["strict_score"]),
            "strict_pass_proxy": float(baseline_row["strict_pass_prob_proxy"]),
            "mean_cd_error":   float(baseline_row["mean_cd_error"]),
            "mean_ler_locked": float(baseline_row["mean_ler_locked"]),
            "p_robust_valid":  float(baseline_row["p_robust_valid"]),
        },
        "best_mode_b": {
            "strict_best":     {"recipe_id": rows[0]["recipe_id"],
                                  "strict_score": float(rows[0]["strict_score"]),
                                  "pitch_nm":   float(rows[0]["pitch_nm"]),
                                  "line_cd_ratio": float(rows[0]["line_cd_ratio"])},
            "cd_best":         {"recipe_id": top_cd[0]["recipe_id"],
                                  "mean_cd_error": float(top_cd[0]["mean_cd_error"]),
                                  "pitch_nm":   float(top_cd[0]["pitch_nm"]),
                                  "line_cd_ratio": float(top_cd[0]["line_cd_ratio"])},
            "ler_best":        {"recipe_id": top_ler[0]["recipe_id"],
                                  "mean_ler_locked": float(top_ler[0]["mean_ler_locked"]),
                                  "pitch_nm":   float(top_ler[0]["pitch_nm"]),
                                  "line_cd_ratio": float(top_ler[0]["line_cd_ratio"])},
            "balanced_best":   {"recipe_id": top_balanced[0]["recipe_id"],
                                  "balanced_score": float(top_balanced[0]["balanced_score_06j"]),
                                  "pitch_nm":   float(top_balanced[0]["pitch_nm"]),
                                  "line_cd_ratio": float(top_balanced[0]["line_cd_ratio"])},
            "margin_best":     {"recipe_id": top_margin[0]["recipe_id"],
                                  "mean_p_line_margin": float(top_margin[0]["mean_p_line_margin"]),
                                  "pitch_nm":   float(top_margin[0]["pitch_nm"]),
                                  "line_cd_ratio": float(top_margin[0]["line_cd_ratio"])},
            "novelty_best":    ({"recipe_id": novelty[0]["recipe_id"],
                                   "geometry_distance_to_modeA": float(novelty[0]["geometry_distance_to_modeA"]),
                                   "strict_score": float(novelty[0]["strict_score"]),
                                   "pitch_nm":   float(novelty[0]["pitch_nm"]),
                                   "line_cd_ratio": float(novelty[0]["line_cd_ratio"])}
                                  if novelty else {}),
        },
        "geometry_clustering_top_strict_100": {
            "n_at_modeA_geometry":  int(n_at_modeA_geom),
            "pitch_distribution":   {str(k): v for k, v in pitch_dist_top100.items()},
            "line_cd_ratio_distribution": {str(k): v for k, v in ratio_dist_top100.items()},
        },
        "fd_sanity_check": {
            "n_runs":          int(n_sanity),
            "n_robust_valid":  int(n_sanity_robust),
            "n_strict_pass":   int(n_sanity_strict_pass),
            "n_false_pass":    int(n_sanity_false_pass),
            "n_targets_dedup": int(len(sanity_pool)),
        },
        "fd_mc_optional": {
            "n_targets":       int(len(fd_mc_rows) // max(args.n_mc_per_recipe, 1)),
            "n_runs":          int(len(fd_mc_rows)),
        },
        "acceptance": {
            "mode_b_search_run":               True,
            "top_candidates_identified":       int(min(len(top_strict), 100)),
            "compared_against_modeA_recipes":  True,
            "fd_sanity_check_run":             not args.skip_fd_sanity,
            "false_pass_count":                int(n_sanity_false_pass),
            "policy_v2_OP_frozen":             bool(cfg["policy"].get("v2_OP_frozen", True)),
            "policy_published_data_loaded":    bool(cfg["policy"].get("published_data_loaded", False)),
            "policy_external_calibration":     "none",
        },
    }
    (logs_dir / "stage06J_summary.json").write_text(
        json.dumps(payload, indent=2, default=float))

    # ----- Console summary -----
    print(f"\nStage 06J summary")
    print(f"  v2 frozen OP (06H surrogate, Mode B re-eval): "
          f"strict = {baseline_row['strict_score']:.4f}, "
          f"proxy = {baseline_row['strict_pass_prob_proxy']:.3f}")
    print(f"  06J best (Mode B strict): {rows[0]['recipe_id']}  "
          f"strict = {rows[0]['strict_score']:.4f}, "
          f"pitch = {rows[0]['pitch_nm']:.0f}, ratio = {rows[0]['line_cd_ratio']:.2f}")
    print(f"  best CD: {top_cd[0]['recipe_id']}  cd_err = {top_cd[0]['mean_cd_error']:.4f}, "
          f"pitch = {top_cd[0]['pitch_nm']:.0f}, ratio = {top_cd[0]['line_cd_ratio']:.2f}")
    print(f"  geometry clustering of strict top-100:")
    print(f"    pitch distribution:  {pitch_dist_top100}")
    print(f"    ratio distribution:  {ratio_dist_top100}")
    print(f"    n at Mode A geometry (24, 0.52): {n_at_modeA_geom}")
    print(f"  FD sanity: {n_sanity_robust}/{n_sanity} robust, "
          f"{n_sanity_strict_pass}/{n_sanity} strict_pass, "
          f"{n_sanity_false_pass} false-PASS")
    if fd_mc_rows:
        print(f"  FD MC subset: {len(fd_mc_rows)} runs across "
              f"{len(fd_mc_rows) // max(args.n_mc_per_recipe, 1)} targets")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
