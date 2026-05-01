"""Stage 06K -- local process-window robustness map around G_4867.

Three sampling layers around the G_4867 base recipe:
    A. OAT (one-at-a-time) sweeps: 11 points per knob x 8 knobs.
    B. Local Latin Hypercube: 2,000 candidates inside the OAT box.
    C. Pairwise grids (21 x 21 each) for 6 expected-interaction pairs:
         dose x sigma, dose x Q0, DH x time, Hmax x kdep, Q0 x kq, sigma x DH.

All samples are clipped to candidate_space bounds. We score each
candidate with the 06H surrogate (200 MC variations per recipe), then
add the strict_score formula from Stage 06G. A small FD verification
subset is run on top, on the worst, and on boundary-band candidates.

Policy
    v2_OP_frozen           = true
    published_data_loaded  = false
    No external calibration. Closed Stage 04C / 04D / 06C joblibs are
    read-only. New FD rows go to Stage 06K-only CSVs.
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
    load_model,
)
from reaction_diffusion_peb_v3_screening.src.process_variation import (
    VariationSpec,
    sample_variations,
)
from reaction_diffusion_peb_v3_screening.src.yield_optimizer import (
    YieldScoreConfig,
    evaluate_recipes,
)

# Reuse strict_score formula from Stage 06G.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_stage06g_strict_optimization import (  # noqa: E402
    StrictScoreConfig,
    compute_strict_score,
)


V3_DIR = ROOT / "reaction_diffusion_peb_v3_screening"
CD_TARGET_NM = 15.0


# --------------------------------------------------------------------------
# Knob-perturbation specification.
# Knobs that vary; absolute or relative half-width per knob.
# --------------------------------------------------------------------------
KNOBS = [
    ("dose_mJ_cm2",  "rel", 0.10),
    ("sigma_nm",     "abs", 0.30),
    ("DH_nm2_s",     "rel", 0.20),
    ("time_s",       "abs", 5.00),
    ("Hmax_mol_dm3", "rel", 0.10),
    ("kdep_s_inv",   "rel", 0.15),
    ("Q0_mol_dm3",   "rel", 0.30),
    ("kq_s_inv",     "rel", 0.30),
]
PAIRS = [
    ("dose_mJ_cm2", "sigma_nm"),
    ("dose_mJ_cm2", "Q0_mol_dm3"),
    ("DH_nm2_s",    "time_s"),
    ("Hmax_mol_dm3","kdep_s_inv"),
    ("Q0_mol_dm3",  "kq_s_inv"),
    ("sigma_nm",    "DH_nm2_s"),
]


def _knob_bounds(space: CandidateSpace) -> dict[str, tuple[float, float]]:
    out = {}
    for p in space.parameters:
        if p["type"] == "uniform":
            out[p["name"]] = (float(p["low"]), float(p["high"]))
        elif p["type"] == "choice":
            vs = [float(v) for v in p["values"]]
            out[p["name"]] = (min(vs), max(vs))
    return out


def _half_width(base: float, mode: str, w: float) -> float:
    if mode == "abs":
        return float(w)
    return float(base) * float(w)


def _local_range(name: str, base: float, mode: str, w: float,
                   bounds: dict[str, tuple[float, float]]) -> tuple[float, float]:
    hw = _half_width(base, mode, w)
    lo = base - hw; hi = base + hw
    blo, bhi = bounds.get(name, (lo, hi))
    return float(max(lo, blo)), float(min(hi, bhi))


def _clip(name: str, v: float, bounds: dict[str, tuple[float, float]]) -> float:
    blo, bhi = bounds.get(name, (-np.inf, np.inf))
    return float(np.clip(v, blo, bhi))


def _make_candidate(base: dict, overrides: dict, space: CandidateSpace,
                     bounds: dict, _id: str) -> dict:
    out = dict(base)
    for k, v in overrides.items():
        out[k] = _clip(k, float(v), bounds)
    out["pitch_nm"]    = float(out["pitch_nm"])
    out["line_cd_nm"]  = out["pitch_nm"] * out["line_cd_ratio"]
    out["domain_x_nm"] = out["pitch_nm"] * 5.0
    out["dose_norm"]   = out["dose_mJ_cm2"] / float(space.fixed["reference_dose_mJ_cm2"])
    for fk, fv in space.fixed.items():
        out.setdefault(fk, fv)
    out["_id"] = _id
    return out


# --------------------------------------------------------------------------
# Sampling
# --------------------------------------------------------------------------
def sample_oat(base: dict, n_pts: int, space: CandidateSpace,
                bounds: dict) -> list[dict]:
    out: list[dict] = []
    # Include the base point (all knobs at baseline) once.
    out.append(_make_candidate(base, {}, space, bounds, "K_base"))
    for kname, mode, w in KNOBS:
        lo, hi = _local_range(kname, base[kname], mode, w, bounds)
        # n_pts including endpoints, skipping the centre value (already in base).
        vals = np.linspace(lo, hi, n_pts)
        for j, v in enumerate(vals):
            cand = _make_candidate(
                base, {kname: float(v)}, space, bounds,
                _id=f"K_oat_{kname}_{j:02d}",
            )
            cand["_oat_knob"] = kname
            cand["_oat_step"] = int(j)
            cand["_oat_value"] = float(v)
            out.append(cand)
    return out


def sample_local_lh(base: dict, n: int, space: CandidateSpace,
                      bounds: dict, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    out: list[dict] = []
    # Build per-knob ranges.
    ranges = {kname: _local_range(kname, base[kname], mode, w, bounds)
                for kname, mode, w in KNOBS}
    # Latin Hypercube via shuffled strata.
    knob_names = list(ranges.keys())
    n_knobs = len(knob_names)
    strata = np.zeros((n, n_knobs), dtype=np.float64)
    for j in range(n_knobs):
        u = (np.arange(n) + rng.uniform(size=n)) / n
        rng.shuffle(u)
        strata[:, j] = u
    for i in range(n):
        overrides = {}
        for j, kname in enumerate(knob_names):
            lo, hi = ranges[kname]
            overrides[kname] = lo + (hi - lo) * strata[i, j]
        out.append(_make_candidate(base, overrides, space, bounds,
                                      _id=f"K_lh_{i:04d}"))
    return out


def sample_pairwise(base: dict, pairs: list[tuple[str, str]], n_grid: int,
                      space: CandidateSpace, bounds: dict) -> list[dict]:
    out: list[dict] = []
    for k1, k2 in pairs:
        m1, w1 = next((m, w) for k, m, w in KNOBS if k == k1)
        m2, w2 = next((m, w) for k, m, w in KNOBS if k == k2)
        lo1, hi1 = _local_range(k1, base[k1], m1, w1, bounds)
        lo2, hi2 = _local_range(k2, base[k2], m2, w2, bounds)
        v1 = np.linspace(lo1, hi1, n_grid)
        v2 = np.linspace(lo2, hi2, n_grid)
        for i, x1 in enumerate(v1):
            for j, x2 in enumerate(v2):
                cand = _make_candidate(
                    base, {k1: float(x1), k2: float(x2)}, space, bounds,
                    _id=f"K_pair_{k1}_{k2}_{i:02d}_{j:02d}",
                )
                cand["_pair_k1"] = k1; cand["_pair_k2"] = k2
                cand["_pair_i"] = int(i); cand["_pair_j"] = int(j)
                cand["_pair_v1"] = float(x1); cand["_pair_v2"] = float(x2)
                out.append(cand)
    return out


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------
def score_local(candidates: list[dict], clf, reg, aux,
                  var_spec: VariationSpec, n_var: int,
                  space: CandidateSpace, score_cfg: YieldScoreConfig,
                  strict_cfg: StrictScoreConfig, seed: int) -> list[dict]:
    rows = evaluate_recipes(
        candidates, clf, reg, aux,
        var_spec, n_var, space, score_cfg, seed=seed,
    )
    # evaluate_recipes returns plain dicts keyed by recipe_id == cand['_id'].
    # Match back to source candidates so we keep _oat_* and _pair_* tags.
    by_id = {r["recipe_id"]: r for r in rows}
    out: list[dict] = []
    for c in candidates:
        rid = c["_id"]
        r = by_id.get(rid)
        if r is None:
            continue
        r.update(compute_strict_score(r, strict_cfg))
        # Carry tags + the actual feature values so the OAT axis is reconstructable.
        for tag in ("_oat_knob", "_oat_step", "_oat_value",
                     "_pair_k1", "_pair_k2", "_pair_i", "_pair_j",
                     "_pair_v1", "_pair_v2"):
            if tag in c:
                r[tag] = c[tag]
        out.append(r)
    return out


# --------------------------------------------------------------------------
# FD verification subset
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


def select_fd_subset(scored: list[dict], strict_cfg: StrictScoreConfig,
                       n_top: int = 50, n_worst: int = 30,
                       n_boundary: int = 20) -> list[tuple[dict, str]]:
    """Pick (candidate row, role) pairs:
      - 'top'      : best surrogate strict_score
      - 'worst'    : worst surrogate strict_score
      - 'boundary' : surrogate strict_score nearest the OP-tier threshold
                      (we use median of top half as a soft boundary)
    Plus 'baseline' for the K_base candidate.
    """
    rows_sorted = sorted(scored, key=lambda r: -float(r["strict_score"]))
    base_row = next((r for r in scored if r["recipe_id"] == "K_base"), None)
    selected: list[tuple[dict, str]] = []
    seen_ids: set[str] = set()
    if base_row is not None:
        selected.append((base_row, "baseline"))
        seen_ids.add(base_row["recipe_id"])
    for r in rows_sorted[:n_top]:
        if r["recipe_id"] in seen_ids:
            continue
        selected.append((r, "top"))
        seen_ids.add(r["recipe_id"])
    for r in rows_sorted[-n_worst:]:
        if r["recipe_id"] in seen_ids:
            continue
        selected.append((r, "worst"))
        seen_ids.add(r["recipe_id"])
    # Boundary band: rows whose strict_score is closest to the median of the top half.
    upper_half = rows_sorted[: max(len(rows_sorted) // 2, 1)]
    boundary_target = float(np.median([r["strict_score"] for r in upper_half]))
    rows_by_proximity = sorted(scored,
                                  key=lambda r: abs(float(r["strict_score"]) - boundary_target))
    for r in rows_by_proximity:
        if len([s for s, role in selected if role == "boundary"]) >= n_boundary:
            break
        if r["recipe_id"] in seen_ids:
            continue
        selected.append((r, "boundary"))
        seen_ids.add(r["recipe_id"])
    return selected


def _open_fd_csv(path: Path):
    from reaction_diffusion_peb_v3_screening.src.metrics_io import LABEL_CSV_COLUMNS
    extra = ["source_recipe_id", "role", "phase", "variation_idx",
              "strict_score_surrogate", "yield_score_surrogate"]
    cols = LABEL_CSV_COLUMNS + extra
    path.parent.mkdir(parents=True, exist_ok=True)
    f = path.open("w", newline="")
    w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
    w.writeheader()
    return f, w, cols


def _write_fd_row(writer, row: dict, *, source_recipe_id: str, role: str,
                    phase: str, variation_idx: int,
                    strict_score_surrogate: float,
                    yield_score_surrogate: float) -> None:
    out = dict(row)
    out["source_recipe_id"]      = source_recipe_id
    out["role"]                   = role
    out["phase"]                  = phase
    out["variation_idx"]          = int(variation_idx)
    out["strict_score_surrogate"] = float(strict_score_surrogate)
    out["yield_score_surrogate"]  = float(yield_score_surrogate)
    writer.writerow(out)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str,
                   default=str(V3_DIR / "configs" / "yield_optimization.yaml"))
    p.add_argument("--manifest_yaml", type=str,
                   default=str(V3_DIR / "outputs" / "yield_optimization"
                               / "stage06I_mode_a_final_recipes.yaml"))
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
    p.add_argument("--n_oat",     type=int, default=11)
    p.add_argument("--n_lh",      type=int, default=2000)
    p.add_argument("--n_pair",    type=int, default=21)
    p.add_argument("--n_var",     type=int, default=200)
    p.add_argument("--n_top_fd",       type=int, default=50)
    p.add_argument("--n_worst_fd",     type=int, default=30)
    p.add_argument("--n_boundary_fd",  type=int, default=20)
    p.add_argument("--n_mc_per_recipe", type=int, default=100)
    p.add_argument("--n_mc_recipes",    type=int, default=6,
                   help="G_4867 baseline + 5 worst-OAT axes by default.")
    p.add_argument("--seed_score", type=int, default=8181)
    p.add_argument("--seed_lh",     type=int, default=8282)
    p.add_argument("--seed_mc",     type=int, default=8383)
    p.add_argument("--label_schema",
                   default=str(V3_DIR / "configs" / "label_schema.yaml"))
    p.add_argument("--skip_fd", action="store_true")
    args = p.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    space = CandidateSpace.from_yaml(cfg["candidate_space_yaml"])
    var_spec = VariationSpec.from_yaml_dict(cfg["process_variation"])
    score_cfg = YieldScoreConfig.from_yaml_dict(cfg["yield_score"])

    strict_yaml = yaml.safe_load(Path(args.strict_cfg_yaml).read_text())
    cd_tol  = float(strict_yaml["thresholds"]["cd_tol_nm"])
    ler_cap = float(strict_yaml["thresholds"]["ler_cap_nm"])
    strict_cfg = StrictScoreConfig(cd_tol_nm=cd_tol, ler_cap_nm=ler_cap)

    # ----- Read G_4867 baseline from the 06I manifest -----
    manifest = yaml.safe_load(Path(args.manifest_yaml).read_text())
    base_recipe = next(r for r in manifest["representatives"]
                         if r["recipe_id"] == manifest["primary_recommended_recipe"])
    base_params = {k: float(v) for k, v in base_recipe["parameters"].items()}
    print(f"Stage 06K -- local process-window map around {base_recipe['recipe_id']}")
    print(f"  policy: v2_OP_frozen={cfg['policy']['v2_OP_frozen']}, "
          f"published_data_loaded={cfg['policy']['published_data_loaded']}")
    print(f"  base parameters:")
    for kname, mode, w in KNOBS:
        print(f"    {kname:<14} = {base_params[kname]:.4f}  "
              f"(perturbation: {'+/-' + str(w) + ' nm' if mode == 'abs' else f'+/-{w*100:.0f}%'})")

    # ----- Sampling -----
    bounds = _knob_bounds(space)
    cands_oat = sample_oat(base_params, args.n_oat, space, bounds)
    cands_lh  = sample_local_lh(base_params, args.n_lh, space, bounds, args.seed_lh)
    cands_pair = sample_pairwise(base_params, PAIRS, args.n_pair, space, bounds)
    print(f"  sampling: OAT {len(cands_oat)}  LH {len(cands_lh)}  "
          f"pairwise {len(cands_pair)}  total {len(cands_oat) + len(cands_lh) + len(cands_pair)}")

    # ----- Surrogate scoring (single batched call) -----
    clf, _ = load_model(args.clf)
    reg, _ = load_model(args.reg)
    aux, _ = load_model(args.aux)

    print(f"\n  scoring with 06H surrogate (n_var={args.n_var}) ...")
    t0 = time.time()
    rows_oat  = score_local(cands_oat,  clf, reg, aux, var_spec, args.n_var,
                              space, score_cfg, strict_cfg, args.seed_score)
    rows_lh   = score_local(cands_lh,   clf, reg, aux, var_spec, args.n_var,
                              space, score_cfg, strict_cfg, args.seed_score + 1)
    rows_pair = score_local(cands_pair, clf, reg, aux, var_spec, args.n_var,
                              space, score_cfg, strict_cfg, args.seed_score + 2)
    print(f"  scored {len(rows_oat) + len(rows_lh) + len(rows_pair)} recipes "
          f"in {time.time() - t0:.1f}s")

    yopt_dir = V3_DIR / "outputs" / "yield_optimization"
    labels_dir = V3_DIR / "outputs" / "labels"

    # Combined local_candidates CSV (everything).
    all_rows = rows_oat + rows_lh + rows_pair
    summary_cols = [
        "recipe_id", "strict_score", "yield_score",
        "p_robust_valid", "p_margin_risk", "p_under_exposed",
        "p_merged", "p_roughness_degraded", "p_numerical_invalid",
        "mean_cd_fixed", "std_cd_fixed",
        "mean_cd_locked", "std_cd_locked",
        "mean_ler_locked", "std_ler_locked",
        "mean_p_line_margin", "std_p_line_margin",
        "strict_cd_pen", "strict_ler_pen",
        "strict_cd_std_pen", "strict_ler_std_pen", "strict_margin_bonus",
        "cd_error_penalty", "ler_penalty",
    ] + FEATURE_KEYS + [
        "_oat_knob", "_oat_step", "_oat_value",
        "_pair_k1", "_pair_k2", "_pair_i", "_pair_j",
        "_pair_v1", "_pair_v2",
    ]
    yopt_dir.mkdir(parents=True, exist_ok=True)
    with (yopt_dir / "stage06K_local_candidates.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=summary_cols, extrasaction="ignore")
        w.writeheader()
        for r in all_rows:
            w.writerow(r)
    print(f"  local candidates CSV -> {yopt_dir / 'stage06K_local_candidates.csv'}")

    # Save OAT and pairwise as separate CSVs for plot scripts to slice cleanly.
    with (yopt_dir / "stage06K_oat_sensitivity.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=summary_cols, extrasaction="ignore")
        w.writeheader()
        for r in rows_oat:
            w.writerow(r)
    with (yopt_dir / "stage06K_pairwise_maps.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=summary_cols, extrasaction="ignore")
        w.writeheader()
        for r in rows_pair:
            w.writerow(r)

    # ----- FD verification subset selection -----
    if args.skip_fd:
        print("  FD verification: skipped (--skip_fd)")
        return 0

    selected = select_fd_subset(all_rows, strict_cfg,
                                  n_top=args.n_top_fd,
                                  n_worst=args.n_worst_fd,
                                  n_boundary=args.n_boundary_fd)
    print(f"\n  FD verification subset: {len(selected)} recipes")

    # ----- FD nominal on subset + FD MC on n_mc_recipes -----
    thresholds = LabelThresholds.from_yaml(args.label_schema)
    fd_path = labels_dir / "stage06K_fd_verification.csv"
    f, writer, _ = _open_fd_csv(fd_path)
    t0 = time.time()
    try:
        for i, (row, role) in enumerate(selected, start=1):
            cand = _row_to_fd_candidate(row, space, _id=f"{row['recipe_id']}__nom")
            fd_row = run_one_candidate(cand, thresholds)
            _write_fd_row(
                writer, fd_row,
                source_recipe_id=str(row["recipe_id"]),
                role=role, phase="nominal", variation_idx=0,
                strict_score_surrogate=float(row["strict_score"]),
                yield_score_surrogate=float(row["yield_score"]),
            )
            f.flush()
            if i % 20 == 0:
                rate = i / max(time.time() - t0, 1e-9)
                print(f"    FD nominal: {i}/{len(selected)} done ({rate:.2f} runs/s)")
    finally:
        # Keep file open for MC writes below.
        pass
    print(f"    FD nominal: {len(selected)} runs in {time.time() - t0:.1f}s")

    # MC FD recipes:
    #   - K_base
    #   - worst OAT step from each of the 5 most-fragile knobs
    #     (chosen by largest |Delta strict_score| across that knob's OAT steps).
    oat_drop = {}
    base_strict = next((r["strict_score"] for r in rows_oat
                          if r["recipe_id"] == "K_base"), float("nan"))
    for r in rows_oat:
        kname = r.get("_oat_knob")
        if not kname:
            continue
        d = abs(float(r["strict_score"]) - float(base_strict))
        if d > oat_drop.get(kname, (-1.0, None))[0]:
            oat_drop[kname] = (d, r)
    fragile_ranked = sorted(oat_drop.items(), key=lambda kv: -kv[1][0])
    mc_targets = []
    base_row = next((r for r in rows_oat if r["recipe_id"] == "K_base"), None)
    if base_row is not None:
        mc_targets.append((base_row, "baseline_mc"))
    for kname, (d, worst) in fragile_ranked[: args.n_mc_recipes - 1]:
        mc_targets.append((worst, f"worst_oat_{kname}"))

    print(f"\n  FD MC: {len(mc_targets)} recipes x {args.n_mc_per_recipe} variations")
    base_rng = np.random.default_rng(args.seed_mc)
    total = 0; t0 = time.time()
    for k_idx, (row, role) in enumerate(mc_targets, start=1):
        base_cand = {k: float(row[k]) for k in FEATURE_KEYS}
        base_cand["pitch_nm"]    = float(base_cand["pitch_nm"])
        base_cand["line_cd_nm"]  = base_cand["pitch_nm"] * base_cand["line_cd_ratio"]
        base_cand["domain_x_nm"] = base_cand["pitch_nm"] * 5.0
        base_cand["dose_norm"]   = base_cand["dose_mJ_cm2"] / float(space.fixed["reference_dose_mJ_cm2"])
        for fk, fv in space.fixed.items():
            base_cand.setdefault(fk, fv)
        sub_rng = np.random.default_rng(int(base_rng.integers(0, 2**31 - 1)))
        variations = sample_variations(base_cand, var_spec, args.n_mc_per_recipe,
                                          space, rng=sub_rng)
        for j, v in enumerate(variations, start=1):
            v["_id"] = f"{row['recipe_id']}__{role}_mc{j:03d}"
            for fk, fv in space.fixed.items():
                v.setdefault(fk, fv)
            v["domain_x_nm"] = v["pitch_nm"] * 5.0
            v["dose_norm"]   = v["dose_mJ_cm2"] / float(space.fixed["reference_dose_mJ_cm2"])
            fd_row = run_one_candidate(v, thresholds)
            _write_fd_row(
                writer, fd_row,
                source_recipe_id=str(row["recipe_id"]),
                role=role, phase="mc", variation_idx=j,
                strict_score_surrogate=float(row["strict_score"]),
                yield_score_surrogate=float(row["yield_score"]),
            )
            f.flush()
            total += 1
        print(f"    MC: recipe {k_idx}/{len(mc_targets)} ({role}: "
              f"{row['recipe_id']}) done")
    f.close()
    print(f"  FD MC: {total} runs in {time.time() - t0:.1f}s")
    print(f"  FD verification CSV -> {fd_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
