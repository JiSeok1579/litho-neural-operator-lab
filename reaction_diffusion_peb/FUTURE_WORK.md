# PEB submodule — future improvements

This file tracks planned but not-yet-implemented improvements to the
`reaction_diffusion_peb/` submodule. Each item is intentionally scoped
to be addressable in a single follow-up PR without touching the rest
of the lab. Items are ordered by perceived priority.

---

## 1. PINN bound-penalty constraint (Phase 5 follow-up)

**Status:** open · **Source:** Phase 5 finding · **Target file(s):**
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

**Status:** open · **Source:** Phase 5/6 metrics CSVs already log
``P_max`` / ``P_mean`` / threshold area; total mass of acid + losses
is not yet aggregated.

In the absence of quencher (out of scope until Phase 7), the only
acid sink is the loss term ``- k_loss * H``. A clean conservation
identity:

```text
M(t) = total_mass(H(t)) + integral_0^t k_loss * total_mass(H(tau)) dtau
     = total_mass(H_0)                      (for the closed system)
```

Adding this as a column in the metrics CSV gives a global sanity
check that the FD evolver does not drift, especially at large
Arrhenius-corrected ``k_loss`` (e.g. 125 °C runs reach ``k_loss_eff``
of 0.038 — still small but worth tracking).
