# Phase 0 — Environment and project scaffold

**Status:** ✅ done
**PR:** #1

## Goal

Stand up the project skeleton, install the scientific stack, and lay
down the documentation conventions that every later phase will reuse.

## What landed

- `src/` package tree for the 9 study-plan phases
  (`common`, `mask`, `optics`, `inverse`, `resist`, `pinn`,
  `neural_operator`, `closed_loop`).
- `tests/` skeleton with project-root `conftest.py` so pytest picks up
  the `src/` packages without an editable install.
- `outputs/{figures,checkpoints,logs,datasets}/` (each `.gitkeep`'d).
- `configs/` placeholder for Hydra / OmegaConf YAML.
- `requirements.txt`, `.gitignore`.
- `README.md` (env / install / phase roadmap / working rules) and
  `PROGRESS.md` (running What / How / Why / Next log).

## Verified

- `pip install -r requirements.txt` resolves clean.
- `torch.cuda.is_available()` returns True when a CUDA build matches
  the local GPU.
- 2048² fp32 matmul, complex64 FFT2, and an autograd round-trip all
  succeed on the GPU.

## Key takeaway

The src layout with empty `__init__.py` files keeps imports clean and
makes it easy to switch to `pip install -e .` later. The matching
PROGRESS.md + per-phase doc structure means anyone (or future-me) can
answer "where are we and why" in one read.

## How to run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip wheel setuptools
pip install torch torchvision
pip install -r requirements.txt
pytest tests/ -q
```

## See also

- [PROGRESS.md §A.1 / A.2 / A.3](../PROGRESS.md)
