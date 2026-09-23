---
name: petsc-solvers
description: >-
  High-level PETSc solver and DM composition: choosing and combining KSP
  (Krylov) methods, PC preconditioners (incl. GAMG multigrid, fieldsplit for
  saddle-point/multiphysics, ASM/domain decomposition), SNES nonlinear solvers
  with globalization, TS time integrators with adaptive control, and the DM
  layer (DMDA/DMPlex/DMSwarm, and DMForest/AMR). Load when selecting an
  algorithmic solver stack (numerical-analysis, high level) or configuring it in
  code (code-generation, implementation).
---

# PETSc solver & DM composition (high level)

Shared knowledge base. `numerical-analysis` uses it to CHOOSE a solver stack;
`code-generation` uses it to CONFIGURE one. Deep API mechanics live in
`petsc-codegen`.

## KSP — Krylov solvers
- **CG** for SPD systems; **MINRES** for symmetric indefinite; **GMRES/FGMRES**
  for nonsymmetric (FGMRES when the preconditioner varies); **BiCGStab** as a
  short-recurrence alternative.
- Iteration count depends on conditioning and spectrum — report and monitor it.

## PC — preconditioners
- Simple: `jacobi`, `sor`, `ilu`/`icc` (serial-ish; block variants in parallel).
- **`gamg`** (algebraic multigrid) and geometric MG (`mg` via DM) for
  mesh-independent iteration counts on elliptic/parabolic operators.
- **`asm`/`gasm`** (additive Schwarz / domain decomposition) for scalable
  parallel preconditioning.
- **`fieldsplit`** for block/saddle-point systems (Stokes, Navier–Stokes,
  poroelasticity); compose with Schur complements and nest recursively for
  multiphysics.

## SNES — nonlinear solvers
- `newtonls` (line search) and `newtontr` (trust region) with quadratic
  convergence near the root; `ngmres`, `ncg`, `nrichardson`, FAS for specific
  structures.
- Matrix-free Jacobians (`-snes_mf`, `-snes_mf_operator`) with a cheaper PC;
  supply an analytic Jacobian when possible, else color/FD it.

## TS — time integration
- `beuler`, `cn` (Crank–Nicolson), `bdf` for stiff/implicit; `rk`, `ssp` for
  explicit; `arkimex`/`rosw` (IMEX) for stiff+nonstiff splits.
- Use TS adaptivity (embedded error control) and choose tolerances from the
  target global error.

## DM — discretization management
- **`DMDA`** structured grids; **`DMPlex`** unstructured meshes/FEM/FVM;
  **`DMSwarm`** particles (PIC, tracing), often coupled to a background DM.
- **`DMForest`/p4est** for adaptive mesh refinement.
- Prefer DM-driven geometric multigrid and matrix assembly for scalability.

## Composition principles
- Match PC to operator structure (elliptic→MG, saddle-point→fieldsplit,
  parallel→DD).
- Aim for mesh-independent and process-count-independent iteration counts; treat
  growth in either as a red flag.
- Keep algorithmic choices here; expose everything as runtime `-options` so
  configuration can change without recompiling.
