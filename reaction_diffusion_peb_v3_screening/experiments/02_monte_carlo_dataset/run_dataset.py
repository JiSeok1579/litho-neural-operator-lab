"""Monte-Carlo dataset stage:
   sample N → prefilter → FD on top K → label → save.
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

from reaction_diffusion_peb_v3_screening.src.budget_prefilter import score_all, select_top_n
from reaction_diffusion_peb_v3_screening.src.candidate_sampler import (
    CandidateSpace, sample_candidates,
)
from reaction_diffusion_peb_v3_screening.src.fd_batch_runner import run_batch
from reaction_diffusion_peb_v3_screening.src.labeler import LabelThresholds
from reaction_diffusion_peb_v3_screening.src.metrics_io import (
    write_candidates_jsonl,
)


V3_DIR = ROOT / "reaction_diffusion_peb_v3_screening"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--tag", type=str, default="02_monte_carlo_dataset")
    args = p.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    space = CandidateSpace.from_yaml(cfg["candidate_space_yaml"])
    thresholds = LabelThresholds.from_yaml(cfg["label_schema_yaml"])

    n_candidates = int(cfg["sampling"]["n_candidates"])
    retain_top = int(cfg["prefilter"]["retain_top_n"])
    fd_budget = int(cfg["fd_run"]["budget"])

    print(f"=== Stage 02 — monte-carlo dataset ===")
    print(f"  sampling     : {cfg['sampling']['method']} × {n_candidates}")
    print(f"  prefilter    : retain top {retain_top}")
    print(f"  FD budget    : {fd_budget}")

    cs = sample_candidates(space, n=n_candidates,
                            method=cfg["sampling"]["method"],
                            seed=cfg["run"]["seed"])
    write_candidates_jsonl(cs, V3_DIR / "outputs" / "candidates" / f"{args.tag}_all.jsonl")

    if cfg["prefilter"]["enable"]:
        cs_scored = score_all(cs)
        retained = select_top_n(cs_scored, retain_top)
    else:
        retained = cs

    write_candidates_jsonl(retained, V3_DIR / "outputs" / "candidates" / f"{args.tag}_retained.jsonl")
    selected = retained[:fd_budget]

    out_csv = V3_DIR / "outputs" / "labels" / f"{args.tag}.csv"
    rows = run_batch(selected, thresholds=thresholds, out_csv=out_csv)

    counts = Counter(r["label"] for r in rows)
    print("\nLabel histogram:")
    for k in ["robust_valid", "margin_risk", "roughness_degraded",
                "under_exposed", "merged", "numerical_invalid"]:
        print(f"  {k:<22} {counts.get(k, 0)}")

    summary = {
        "n_candidates_sampled": n_candidates,
        "n_retained_after_prefilter": len(retained),
        "n_fd_runs": len(rows),
        "label_counts": dict(counts),
        "out_csv": str(out_csv),
    }
    log = V3_DIR / "outputs" / "logs" / f"{args.tag}_summary.json"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
