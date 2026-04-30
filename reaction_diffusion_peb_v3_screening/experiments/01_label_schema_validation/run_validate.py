"""Tiny FD batch that exercises the labeler and confirms every label
shows up at least once on the screening parameter space.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reaction_diffusion_peb_v3_screening.src.budget_prefilter import score_all
from reaction_diffusion_peb_v3_screening.src.candidate_sampler import (
    CandidateSpace, sample_candidates,
)
from reaction_diffusion_peb_v3_screening.src.fd_batch_runner import run_batch
from reaction_diffusion_peb_v3_screening.src.labeler import LabelThresholds


V3_DIR = ROOT / "reaction_diffusion_peb_v3_screening"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--n_candidates", type=int, default=128)
    p.add_argument("--fd_budget", type=int, default=64)
    p.add_argument("--tag", type=str, default="01_label_schema_validation")
    args = p.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    space = CandidateSpace.from_yaml(cfg["candidate_space_yaml"])
    thresholds = LabelThresholds.from_yaml(cfg["label_schema_yaml"])

    cs = sample_candidates(space, n=args.n_candidates,
                            method=cfg["sampling"]["method"],
                            seed=cfg["run"]["seed"])
    cs = score_all(cs)
    cs.sort(key=lambda c: -c["prefilter_score"])
    selected = cs[: args.fd_budget]

    print(f"=== Stage 01 — label schema validation ===")
    print(f"  sampling : {cfg['sampling']['method']} × {args.n_candidates}")
    print(f"  fd_budget: {args.fd_budget} (top scoring)")

    out_csv = V3_DIR / "outputs" / "labels" / f"{args.tag}.csv"
    rows = run_batch(selected, thresholds=thresholds, out_csv=out_csv)

    counts = Counter(r["label"] for r in rows)
    print("\nLabel histogram:")
    for k in ["robust_valid", "margin_risk", "roughness_degraded",
                "under_exposed", "merged", "numerical_invalid"]:
        print(f"  {k:<22} {counts.get(k, 0)}")

    summary = {
        "n_runs": len(rows),
        "label_counts": dict(counts),
        "out_csv": str(out_csv),
    }
    log = V3_DIR / "outputs" / "logs" / f"{args.tag}_summary.json"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
