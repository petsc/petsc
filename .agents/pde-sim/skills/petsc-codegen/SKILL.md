---
name: petsc-codegen
description: >-
  Implementation knowledge for writing, building, and running PETSc programs:
  project/main() structure, PetscCall error handling, DMDA/DMPlex/DMSwarm setup,
  residual/Jacobian assembly, wiring KSP/SNES/TS, the runtime -options database,
  MPI and GPU (CUDA/Kokkos) execution, parallel I/O for solution output, and an
  MMS/convergence-study harness. Load when generating or debugging PETSc code.
---

# PETSc code generation & execution

Implementation know-how for the `code-generation` agent. Algorithmic choices come
from `petsc-solvers`; this skill is about correct, parallel, buildable code.

## Program structure
- `PetscInitialize`/`PetscFinalize` bracket `main()`; read all tunables from the
  options database so runs are reconfigurable without recompiling.
- One repo-level main per executable (simulation; separate viz post-processors).

## Error handling (non-negotiable)
- Wrap every PETSc call: `PetscCall(...)`, `PetscCallMPI(...)`,
  `PetscCall(PetscMalloc1(...))`, etc. Functions return `PetscErrorCode` and
  begin with `PetscFunctionBeginUser;` and end with
  `PetscFunctionReturn(PETSC_SUCCESS);` (not `PetscFunctionReturn(0)`).
- Never ignore a return code.

## Source style — let the formatter own layout
PETSc's layout is defined by `.clang-format` (and checked in CI by
`make checkclangformat`). Do not try to reproduce its many rules by hand — the
formatter is authoritative and you run it as a final step (see Build & run).
Just avoid the habits that turn that pass into a huge reflow diff:
- **One statement per line** — never `row.i = i; row.j = j;`; the formatter
  splits them anyway.
- **Don't hand-align** `=` or declarations with padding spaces, and don't indent
  continuation lines yourself — clang-format recomputes all alignment.
- **Don't manually wrap long calls/signatures** — the column limit is wide (250),
  so a long `DMDACreate2d(...)`/`PetscPrintf(...)` stays on one line; manual
  wrapping just gets reflowed.
- **Short `if`/`else` bodies stay on one line without braces:**
  `if (i == 1) val += ...;`.

Grammar-level rules that clang-format does NOT fix are still on you (see
AGENTS.md): contiguous top-of-function declaration block, `PetscFunctionBeginUser`
after declarations, `//` for short and `/* ... */` for multiline comments, `()`
on function names in comments.

## Discretization scaffolding
- **DMDA**: `DMDACreate1d`/`DMDACreate2d`/`DMDACreate3d`, `DMDAVecGetArray` for
  stencil loops; use local/global scatters and ghost points correctly.
- **DMPlex**: build/read the mesh, set up `PetscSection`/`PetscFE`, assemble via
  `DMPlexSNESComputeResidualFEM`/`DMPlexSNESComputeJacobianFEM` callbacks.
- **DMSwarm**: register particle fields, migrate particles across ranks, couple
  to a background DM for deposition/interpolation (PIC, particle tracing).
- Provide user callbacks for coefficient functions (forcing, variable viscosity,
  sources, BC/IC) with clean signatures; keep them branch-light for GPU.

## Solvers, wired
- KSP: `KSPCreate/SetOperators/SetFromOptions/Solve`; check
  `KSPGetConvergedReason`.
- SNES: `SNESSetFunction`/`SNESSetJacobian` (or matrix-free/coloring); check the
  converged reason.
- TS: `TSSetRHSFunction`/`TSSetIFunction`/`IJacobian`, enable adaptivity.

## Parallelism & GPU
- Correct on 1 and N ranks; verify with `-np` sweeps. Avoid rank-0-only logic
  except I/O.
- GPU via `-dm_vec_type (cuda|kokkos) -dm_mat_type (aijcusparse|aijkokkos)`; keep
  kernels data-parallel and avoid host/device thrashing.

## Output (large data)
- Write solutions with `PetscViewer`: VTK (`.vtu/.vts`) or, preferred at scale,
  **HDF5 + XDMF** for parallel, ParaView-readable output.
- Emit convergence-study data (h, dof, errors, iterations) as CSV/JSON for the
  Results Manifest. Reference big files by path.

## MMS / convergence harness
- Install the manufactured forcing and exact solution; loop over the refinement
  sequence; compute error norms; report observed order. Keep iteration tolerance
  below discretization error so it doesn't pollute the study.

## Build & run
- Assume `PETSC_DIR` and `PETSC_ARCH` are set and `mpiexec` is available (when
  MPI is needed). Never configure/build PETSc yourself or hardcode absolute
  paths — rely on these variables.
- Generate a **makefile** that includes PETSc's config
  (`include ${PETSC_DIR}/lib/petsc/conf/variables` and `.../rules`), links with
  `${PETSC_LIB}`, and exposes build + `run` targets. A minimal pattern:
  ```makefile
  include ${PETSC_DIR}/lib/petsc/conf/variables
  include ${PETSC_DIR}/lib/petsc/conf/rules
  app: app.o
  	${CLINKER} -o app app.o ${PETSC_LIB}
  run: app
  	mpiexec -n 4 ./app -options...
  ```
  (CMake via pkg-config is an alternative, but a makefile is the default.)
- Build and run THROUGH the makefile; capture logs; iterate on compile/runtime
  errors until clean.

## Final cleanup (once, after the code is correct)
Formatting is orthogonal to correctness, so do NOT run it inside the
compile/run/fix loop — code you are about to rewrite will just be reflowed. Only
after the program compiles, runs, and passes MMS, run `clang-format -i` once on
the files you wrote, using PETSc's style file (`$PETSC_DIR/.clang-format`; PETSc
pins clang-format major version 23).

Do NOT use `make clangformat` or `make checkbadSource` for this. Both act only on
git-tracked files, but generated studies live in the git-ignored
`artifacts/<study-id>/`: `make clangformat` would reflow the whole PETSc tree (or
fail) while still skipping your code, and `make checkbadSource` matches nothing
and exits 0 — reporting a pass that verified nothing. The source-style rules
`checkbadSource` enforces (the grammar-level rules above) stay your
responsibility, since it cannot see generated files.
