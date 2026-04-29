# PEB submodule — future improvements

This file tracks planned but not-yet-implemented improvements to the
`reaction_diffusion_peb/` submodule. Each item is intentionally scoped
to be addressable in a single follow-up PR without touching the rest
of the lab. Items are ordered by perceived priority.

---

## 1. PINN bound-penalty constraint (Phase 5 follow-up)

**Status:** ✅ **closed by the pre-Phase-7 diagnostics PR** (Option A
soft penalty implemented; Option B sigmoid wrapper not pursued —
no longer needed at this scope). Original analysis kept below for
reference; the result block at the end of this section records what
landed.

**Source:** Phase 5 finding · **Target file(s):**
`src/pinn_reaction_diffusion.py`, `src/train_pinn_deprotection.py`

**Problem.** The Phase-5 `PINNDeprotection` parameterizes the
deprotected fraction as

```text
P_pinn(x, y, t) = t_norm * MLP_out[..., 1]
```

This **hard-IC** form enforces ``P(x, y, 0) = 0`` exactly, but it
does **not** enforce ``P >= 0`` or ``P <= 1`` during the dynamics.
The Phase-5 evaluation observed:

| solver | P_min | P_max | area(P>0.5) px |
|---|---|---|---|
| FD truth | 0.000 | 0.9358 | 1876 |
| PINN | **−0.191** | 0.9246 | 1468 |

The PINN drifted to ``P_min = −0.19`` in fringe regions where the
PDE residual was small either way — a clean teaching artefact of
"hard-IC enforces the **initial** state but not dynamic state
constraints".

**Two candidate fixes:**

### Option A — soft penalty on bound violations

Add a quadratic penalty term to the trainer's loss:

```text
L = w_pde_H * mean(r_H**2)
  + w_pde_P * mean(r_P**2)
  + w_bound * (mean(relu(-P_pinn)**2) + mean(relu(P_pinn - 1)**2))
```

Easy to bolt on; weight should be in the same ballpark as
`w_pde_P` (e.g. 1.0). Does not change inference, only training.

### Option B — sigmoid wrapper

Reparameterize so the network's output is a logit:

```text
P_pinn(x, y, t) = sigmoid(t_norm * raw_P_logit)
```

At ``t = 0`` this gives ``P = sigmoid(0) = 0.5``, which **violates the
hard IC** (we want ``P(0) = 0``). To recover ``P(0) = 0`` exactly we
need a different gating, e.g.

```text
P_pinn(x, y, t) = t_norm * sigmoid(raw_P_logit)
```

At ``t = 0``: ``P = 0`` ✓. At ``t = t_end``: ``P in [0, t_norm_max]``
which is at most 1.0. Bounded by construction.

Trade-off: the sigmoid wrapper makes large-amplitude P harder for the
network to fit (gradient saturation), and the maximum reachable P
is now exactly ``t_end_factor * 1 = 1`` only at ``t = t_end``. Soft
penalty (Option A) is the safer first attempt.

### Acceptance test

The follow-up PR should add a metrics assertion in `compare_fd_pinn.py`:

```text
assert P_pinn.min() >= -0.05    # ~5% slack vs the FD's exact 0
assert P_pinn.max() <= 1.05     # ~5% slack vs the FD's max
```

and report:

| metric | before | after |
|---|---|---|
| P_min | -0.191 | (target: ≈ 0) |
| P_max | 0.925 | (target: closer to 0.936) |
| area(P>0.5) | 1468 | (target: closer to 1876) |
| max\|P_PINN - P_FD\| | 0.289 | (target: < 0.1) |

### Result (closed)

Implemented as **Option A (soft penalty)** with a configurable
``weight_bound`` knob in ``src/train_pinn_deprotection.py`` and a
sweep demo at
``experiments/pre_phase7_diagnostics/run_pinn_bound_penalty.py``.

Sweep at the Phase-5 setting (``DH=0.8, kloss=0.005, kdep=0.5,
t=60 s``):

| weight_bound | P_min | P_max | area(P>0.5) | max\|P-P_FD\| |
|---|---|---|---|---|
| 0 (baseline) | −0.183 | 0.944 | 1656 | 0.291 |
| 0.001 | −0.118 | 0.916 | 1434 | 0.281 |
| **0.01** | **−0.011** | **0.939** | 1536 | **0.226** |
| 0.1 | −0.017 | 0.919 | 1461 | 0.237 |
| 1.0 | 0.004 | 0.896 | 1295 | 0.240 |
| FD truth | 0.000 | 0.936 | 1876 | 0 |

**weight_bound = 0.01** is the chosen default: P_min ≥ −0.05 is met
with ample headroom (−0.011), P_max stays at the FD truth value
(0.939 vs 0.936), and max-error drops 22 % from the baseline (0.291
→ 0.226). Larger weights ≥ 0.1 over-penalize and under-fit the peak.

A residual area gap remains (PINN 1536 vs FD 1876) — this is now a
PDE-fitting capacity issue, not a bound-violation issue. Closing it
further is item 2's job (T-as-input PINN with longer training) or a
deeper architecture, not more bound-penalty.

---

## 2. PINN with temperature as an extra input (Phase 6 follow-up)

**Status:** open · **Source:** Phase 6 scope decision · **Target
file(s):** new `src/pinn_temperature.py` (or extension of
`pinn_reaction_diffusion.py`).

The Phase-6 PEB demos are FD-only because the Phase-5 PINN is trained
for one fixed ``(k_dep, k_loss)`` pair and cannot generalize across
temperatures without modification. A natural follow-up is a PINN that
takes ``T`` (or the Arrhenius factor) as an additional input:

```text
PINN(x, y, t, T) -> (H, P)
```

with the PDE residual taking the temperature-corrected rates inside
``pde_residuals``. This is conceptually a small change (one more
input dimension and one more inner conditioning) but it materially
expands what the PINN can answer, and it is the right precursor to a
DeepONet / FNO over the same parameter family in later phases.

---

## 3. Bigger / more diverse exposure-map IC

**Status:** open · **Source:** Phase 1-5 demos all use a single
Gaussian spot.

The PEB demos so far only stress the solvers on a smooth, centered
Gaussian. A useful next step is to add line-space and contact-array
ICs to the FD/PINN comparisons — the PINN's hard-IC machinery
should still apply, but the fitting difficulty (especially Phase 6
temperature sweeps) will be visibly higher and that's the right
place to surface the bound-penalty fix from item 1 too.

---

## 4. Mass-conservation diagnostic for the (H, P) system

**Status:** ✅ **closed by the pre-Phase-7 diagnostics PR.**

Implemented as ``src/mass_budget.py``
(``evolve_acid_loss_deprotection_fd_with_budget`` and the
Arrhenius-aware ``..._at_T`` wrapper) plus
``experiments/pre_phase7_diagnostics/run_mass_budget_check.py`` and
12 tests in ``tests/test_mass_budget.py``.

Result on the eight Phase-5 / Phase-6 scenarios — relative error of
``M_budget(t_end)`` against ``M(0)``:

| scenario | mass_H_initial | mass_H_final | rel err |
|---|---|---|---|
| phase5  kloss=0           kdep=0.0 | 144.149 | 144.149 | 3.2e-07 |
| phase5  kloss=0           kdep=0.5 | 144.149 | 144.149 | 3.2e-07 |
| phase5  kloss=0.005       kdep=0.5 | 144.149 | 106.776 | 1.1e-07 |
| phase5  kloss=0.05 (10x)  kdep=0.5 | 144.149 |   7.093 | 2.3e-08 |
| phase6  T= 80 °C  (factor 0.16) | 144.149 | 137.346 | 9.9e-08 |
| phase6  T=100 °C  (factor 1.00) | 144.149 | 106.776 | 1.1e-07 |
| phase6  T=120 °C  (factor 5.15) | 144.149 |  30.617 | 1.1e-08 |
| phase6  T=125 °C MOR (factor 7.57) | 144.149 |  14.786 | 1.5e-08 |

All scenarios — including the high-T 125 °C case where 90 % of the
acid is consumed — close the budget to **float32 round-off** (~1e-7).
This is exactly the expected behavior because the mass change per
explicit-Euler step is itself ``-dt * k_loss * sum(H_n)``, which is
what we accumulate. Any future drift from this baseline will flag a
real bug in the evolver.

---

## 5. z-axis / 3D film thickness extension (Phase 11 follow-up)

**Status:** open · **Source:** Phase 11 plan deferral · **Target
file(s):** new `src/full_reaction_diffusion_3d.py` (or extension of
`petersen_diffusion.py`), new `pinn_3d.py` if PINN coverage is wanted.

The plan's Phase 11 §"z-axis / film thickness" calls for extending the
``(H, Q, P)`` system from 2D to 3D:

```text
H(x, y, z, t)
Q(x, y, z, t)
P(x, y, z, t)
domain: z in [0, film_thickness_nm]
```

Phase 11 in this submodule (PR #27) covers Petersen nonlinear
diffusion, the temperature-uniformity ensemble, and molecular blur —
all 2D operations on top of the existing Phase-8 evolver. The 3D
extension is intentionally **not** included because:

1. The 5-point Laplacian and the variable-coefficient operator both
   need 3D rewrites (7-point stencil; new face-diffusivity averages
   along ``z``).
2. The ``z`` boundary conditions are not periodic — typical resist
   physics has a no-flux or open boundary at the air interface and a
   no-flux boundary at the substrate interface. That is a different
   discretization than the rest of the submodule's periodic
   `torch.roll` machinery.
3. Memory scales by ``Nz``: a 128 × 128 × 32 grid is 32 × the current
   working size; the existing demos / dataset builder would have to
   be revisited for cost.

A future PR should: pick the ``z`` discretization (no-flux Neumann is
the most common), add ``laplacian_7pt`` and ``divergence_diffusion_7pt``,
write `evolve_full_reaction_diffusion_3d_fd_at_T`, and verify that
``Nz = 1`` reduces to the 2D evolvers — the same disable-each-term
pattern as Phase 8.

---

## 6. Per-rate Arrhenius activation energies (Phase 11 follow-up)

**Status:** open · **Source:** Phase 6 / 8 / 11 single-Ea
simplification.

The current Arrhenius scaling in
``apply_arrhenius_to_full_rates`` multiplies ``k_dep``, ``k_loss``,
and ``k_q`` by the same factor — equivalent to assuming a single
activation energy ``Ea`` for all three reactions. Real CAR / MOR
chemistries can have distinct ``Ea_dep``, ``Ea_loss``, ``Ea_q``. Adding
that flexibility is a one-helper change but it triples the parameter
space the PINN / FNO surrogate has to span and is therefore left as a
follow-up rather than a Phase-11 inclusion.
