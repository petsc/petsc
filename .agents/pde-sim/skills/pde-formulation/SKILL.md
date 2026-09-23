---
name: pde-formulation
description: >-
  Knowledge for formulating well-posed PDE/ODE models from physical
  descriptions: PDE classification (elliptic/parabolic/hyperbolic/mixed) and the
  boundary/initial conditions each class admits, conservation-law structure,
  nondimensionalization and scaling, constant-vs-field parameters, and
  closure/subgrid modeling for unresolved scales. Load when deriving governing
  equations, choosing boundary/initial conditions, or deciding how parameters
  and scales enter a model.
---

# PDE/ODE model formulation

Domain knowledge for the `pde-modeling` agent. Concerned with the CONTINUOUS
model only — discretization lives in `numerical-methods`.

## PDE classification and well-posedness
- **Elliptic** (e.g. Poisson, steady Stokes): boundary-value problems; need
  Dirichlet, Neumann, or Robin data on the whole boundary. Pure-Neumann problems
  need a compatibility/gauge condition.
- **Parabolic** (e.g. heat, diffusion-reaction): initial data + boundary data
  for all time; smoothing in time.
- **Hyperbolic** (e.g. advection, wave, Euler): characteristics and finite
  propagation speed; boundary conditions must respect inflow/outflow; initial
  data required.
- **Mixed / coupled systems** (e.g. Navier–Stokes, poroelasticity): classify
  term-by-term; watch for saddle-point structure and inf-sup requirements.
- State the **well-posedness** rationale for the BC/IC set you choose.

## Conservation structure
- Prefer conservation-law form when the physics is conservative; identify the
  conserved quantities (mass, momentum, energy) up front — they become
  `invariants` the numerical-analysis agent will check.
- Note flux functions and source terms explicitly.

## Geometry and boundary description
- Give a concise, complete description of the domain and name each boundary
  region so BCs can be keyed to it.
- Keep geometry continuous and mesh-agnostic; do not commit to structured vs
  unstructured here.

## Parameters: constant vs field
- Decide whether each coefficient is a **constant** or a **field**. Variable
  viscosity, heterogeneous conductivity, and spatially varying sources must be
  fields.
- Field parameters imply extra coefficient functions that `code-generation` will
  need to implement — flag them.

## Scaling and nondimensionalization
- Nondimensionalize to expose the governing dimensionless groups (Reynolds,
  Péclet, Courant, etc.); these reveal which terms dominate and whether a
  parameter is **scale dependent**.
- Record characteristic scales and the resulting nondimensional parameters.

## Unresolved scales and closure
- If the phenomenon has scales too fine to resolve (turbulence, subgrid
  heterogeneity, microstructure), a **closure/subgrid model** is required.
- State the closure assumption and its regime of validity, or record the need
  for one as an open question.

## Output discipline
The model must be emitted as a **Problem Spec JSON** (see
`.agents/pde-sim/contracts/problem-spec.schema.json`), not prose — it is a contract the
downstream agents parse.
