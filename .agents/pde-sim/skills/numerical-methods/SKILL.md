---
name: numerical-methods
description: >-
  Deep knowledge for the numerical solution of ODEs/PDEs: discretization
  methods (FD/FVM/FEM), time integration and stability, iterative-solver and
  preconditioner convergence theory, error sources and control (grid
  refinement, adaptive time stepping, tolerance selection), the method of
  manufactured solutions, convergence studies, and conservation/constraint
  checking. Load when choosing or evaluating a discretization or solver, when
  designing MMS verification, or when analyzing simulation results for
  convergence and conservation.
---

# Numerical methods for ODE/PDE simulation

This skill holds the domain knowledge used by the `numerical-analysis` agent (and
any other agent reasoning about accuracy or convergence). It is intentionally
implementation-agnostic — PETSc-specific realization lives in `petsc-solvers`
and `petsc-codegen`.

## Discretization
- **Finite difference (FD)** — best on structured grids (`DMDA`). Note stencil
  width, order, and boundary treatment.
- **Finite volume (FVM)** — locally conservative; preferred for hyperbolic /
  conservation-law problems. Track fluxes and discrete conservation.
- **Finite element (FEM)** — flexible geometry (`DMPlex`), natural for weak
  forms and higher order; mind quadrature order and inf-sup (LBB) stability for
  mixed problems.
- Record for every choice: **order of accuracy**, stability region, and the
  assumptions it places on the mesh.

## Time integration
- Explicit (RK, AB) vs implicit (BDF, Radau, θ-methods); IMEX for stiff/nonstiff
  splits.
- Stability: CFL for explicit hyperbolic; A-/L-stability for stiff parabolic.
- **Adaptive time stepping** via embedded error estimators; choose tolerances
  from the target global error, not just local error.

## Solver convergence (high level)
- Krylov methods (CG for SPD, GMRES/FGMRES for nonsymmetric) and their
  dependence on conditioning and spectrum.
- Preconditioning: Jacobi/SOR, ILU, domain decomposition (ASM), and multigrid
  (GMG/AMG) — the last for mesh-independent iteration counts.
- Nonlinear: Newton and its globalization (line search, trust region);
  quadratic vs linear convergence and how to detect the difference.
- Always monitor residual histories; distinguish algorithmic stagnation from
  ill-conditioning.

## Error sources and control
- Truncation/discretization error → reduce via refinement or higher order.
- Iteration error → controlled by solver tolerances; keep it below
  discretization error so it doesn't pollute convergence studies.
- Roundoff → watch near machine precision and in ill-conditioned operators.
- Use **local and global error analysis** to set grid and time-step tolerances
  consistently.

## Method of manufactured solutions (MMS)
The primary verification tool. Procedure:
1. Choose a smooth manufactured solution `u_ms(x[,t])` that exercises the
   operators (avoid solutions that make terms vanish trivially).
2. Substitute into the governing PDE to derive the **forcing function** `f` and
   the boundary/initial data it implies.
3. Run on a sequence of refined grids (and time steps); measure the error in
   appropriate norms (L2, L∞, energy).
4. Estimate the **observed order** `p ≈ log(e_h/e_{h/2}) / log(2)` and compare
   with the theoretical order of the scheme.
5. **Interpret:** matching order ⇒ implementation consistent with the method;
   order reduction ⇒ a bug, a BC/quadrature error, or an unmet smoothness
   assumption — flag for revision.

## Convergence studies
- Refine grid and time step together or separately (as appropriate) to isolate
  spatial vs temporal error.
- Report a table of `h`, error, and observed rate; expected slopes are part of
  the acceptance criteria, not an afterthought.

## Conservation and constraints
- For conservation laws, verify discrete conservation of the relevant invariants
  (mass, momentum, energy) to the expected tolerance.
- Check any problem-specific constraints (positivity, incompressibility, entropy)
  and treat violations as evidence the plan needs revision.
