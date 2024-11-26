#pragma once

/* SUBMANSEC = SNES */

/*S
  SNES - Abstract PETSc object that manages nonlinear solves

  Level: beginner

  Notes:
  The most commonly used `SNESType` is `SNESNEWTONLS` which uses Newton's method with a line search. For all the Newton based `SNES` nonlinear
  solvers, `KSP`, the PETSc abstract linear solver object, is used to (approximately) solve the required linear systems.

  See `SNESType` for a list of all the nonlinear solver algorithms provided by PETSc.

  Some of the `SNES` solvers support nonlinear preconditioners, which themselves are also `SNES` objects managed with `SNESGetNPC()`

.seealso: [](doc_nonlinsolve), [](ch_snes), `SNESCreate()`, `SNESSolve()`, `SNESSetType()`, `SNESType`, `TS`, `SNESType`, `KSP`, `PC`, `SNESDestroy()`
S*/
typedef struct _p_SNES *SNES;
