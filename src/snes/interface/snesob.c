#include <petsc/private/snesimpl.h> /*I    "petscsnes.h"   I*/

/*@C
  SNESSetObjective - Sets the objective function minimized by some of the `SNES` linesearch methods, used instead of the 2-norm of the residual in the line search

  Logically Collective

  Input Parameters:
+ snes - the `SNES` context
. obj  - objective evaluation routine; see `SNESObjectiveFn` for the calling sequence
- ctx  - [optional] user-defined context for private data for the objective evaluation routine (may be `NULL`)

  Level: intermediate

  Note:
  Some of the `SNESLineSearch` methods attempt to minimize a given objective provided by this function to determine a step length.

  If not provided then this defaults to the two-norm of the function evaluation (set with `SNESSetFunction()`)

  This is not used in the `SNESLINESEARCHCP` line search.

.seealso: [](ch_snes), `SNES`, `SNESLineSearch()`, `SNESGetObjective()`, `SNESComputeObjective()`, `SNESSetFunction()`, `SNESSetJacobian()`,
          `SNESObjectiveFn`
@*/
PetscErrorCode SNESSetObjective(SNES snes, SNESObjectiveFn *obj, void *ctx)
{
  DM dm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes, SNES_CLASSID, 1);
  PetscCall(SNESGetDM(snes, &dm));
  PetscCall(DMSNESSetObjective(dm, obj, ctx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  SNESGetObjective - Returns the objective function set with `SNESSetObjective()`

  Not Collective

  Input Parameter:
. snes - the `SNES` context

  Output Parameters:
+ obj - objective evaluation routine (or `NULL`); see `SNESObjectiveFn` for the calling sequence
- ctx - the function context (or `NULL`)

  Level: advanced

.seealso: [](ch_snes), `SNES`, `SNESSetObjective()`, `SNESGetSolution()`, `SNESObjectiveFn`
@*/
PetscErrorCode SNESGetObjective(SNES snes, SNESObjectiveFn **obj, void **ctx)
{
  DM dm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes, SNES_CLASSID, 1);
  PetscCall(SNESGetDM(snes, &dm));
  PetscCall(DMSNESGetObjective(dm, obj, ctx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  SNESComputeObjective - Computes the objective function that has been provided by `SNESSetObjective()`

  Collective

  Input Parameters:
+ snes - the `SNES` context
- X    - the state vector

  Output Parameter:
. ob - the objective value

  Level: developer

  Notes:
  `SNESComputeObjective()` is typically used within line-search routines,
  so users would not generally call this routine themselves.

  When solving for $F(x) = b$, this routine computes $objective(x) - x^T b$ where $objective(x)$ is the function provided with `SNESSetObjective()`

.seealso: [](ch_snes), `SNESLineSearch`, `SNES`, `SNESSetObjective()`, `SNESGetSolution()`
@*/
PetscErrorCode SNESComputeObjective(SNES snes, Vec X, PetscReal *ob)
{
  DM     dm;
  DMSNES sdm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes, SNES_CLASSID, 1);
  PetscValidHeaderSpecific(X, VEC_CLASSID, 2);
  PetscAssertPointer(ob, 3);
  PetscCall(SNESGetDM(snes, &dm));
  PetscCall(DMGetDMSNES(dm, &sdm));
  PetscCheck(sdm->ops->computeobjective, PetscObjectComm((PetscObject)snes), PETSC_ERR_ARG_WRONGSTATE, "Must call SNESSetObjective() before SNESComputeObjective().");
  PetscCall(PetscLogEventBegin(SNES_ObjectiveEval, snes, X, 0, 0));
  PetscCall(sdm->ops->computeobjective(snes, X, ob, sdm->objectivectx));
  PetscCall(PetscLogEventEnd(SNES_ObjectiveEval, snes, X, 0, 0));
  if (snes->vec_rhs) {
    PetscScalar dot;

    PetscCall(VecDot(snes->vec_rhs, X, &dot));
    *ob -= PetscRealPart(dot);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  SNESObjectiveComputeFunctionDefaultFD - Computes the gradient of a user provided objective function

  Collective

  Input Parameters:
+ snes - the `SNES` context
. X    - the state vector
- ctx  - the (ignored) function context

  Output Parameter:
. F - the function value

  Options Database Keys:
+ -snes_fd_function_eps - Tolerance for including non-zero entries into the gradient, default is 1.e-6
- -snes_fd_function     - Computes function from user provided objective function (set with `SNESSetObjective()`) with finite difference

  Level: advanced

  Notes:
  This function can be used with `SNESSetFunction()` to have the nonlinear function solved for with `SNES` defined by the gradient of an objective function
  `SNESObjectiveComputeFunctionDefaultFD()` is similar in character to `SNESComputeJacobianDefault()`.
  Therefore, it should be used for debugging purposes only.  Using it in conjunction with
  `SNESComputeJacobianDefault()` is excessively costly and produces a Jacobian that is quite
  noisy.  This is often necessary, but should be done with care, even when debugging
  small problems.

  This uses quadratic interpolation of the objective to form each value in the function.

.seealso: [](ch_snes), `SNESSetObjective()`, `SNESSetFunction()`, `SNESComputeObjective()`, `SNESComputeJacobianDefault()`, `SNESObjectiveFn`
@*/
PetscErrorCode SNESObjectiveComputeFunctionDefaultFD(SNES snes, Vec X, Vec F, void *ctx)
{
  Vec         Xh;
  PetscInt    i, N, start, end;
  PetscReal   ob, ob1, ob2, ob3, fob, dx, eps = 1e-6;
  PetscScalar fv, xv;

  PetscFunctionBegin;
  PetscCall(VecDuplicate(X, &Xh));
  PetscOptionsBegin(PetscObjectComm((PetscObject)snes), ((PetscObject)snes)->prefix, "Differencing parameters", "SNES");
  PetscCall(PetscOptionsReal("-snes_fd_function_eps", "Tolerance for nonzero entries in fd function", "None", eps, &eps, NULL));
  PetscOptionsEnd();
  PetscCall(VecSet(F, 0.));

  PetscCall(VecNorm(X, NORM_2, &fob));

  PetscCall(VecGetSize(X, &N));
  PetscCall(VecGetOwnershipRange(X, &start, &end));
  PetscCall(SNESComputeObjective(snes, X, &ob));

  if (fob > 0.) dx = 1e-6 * fob;
  else dx = 1e-6;

  for (i = 0; i < N; i++) {
    /* compute the 1st value */
    PetscCall(VecCopy(X, Xh));
    if (i >= start && i < end) {
      xv = dx;
      PetscCall(VecSetValues(Xh, 1, &i, &xv, ADD_VALUES));
    }
    PetscCall(VecAssemblyBegin(Xh));
    PetscCall(VecAssemblyEnd(Xh));
    PetscCall(SNESComputeObjective(snes, Xh, &ob1));

    /* compute the 2nd value */
    PetscCall(VecCopy(X, Xh));
    if (i >= start && i < end) {
      xv = 2. * dx;
      PetscCall(VecSetValues(Xh, 1, &i, &xv, ADD_VALUES));
    }
    PetscCall(VecAssemblyBegin(Xh));
    PetscCall(VecAssemblyEnd(Xh));
    PetscCall(SNESComputeObjective(snes, Xh, &ob2));

    /* compute the 3rd value */
    PetscCall(VecCopy(X, Xh));
    if (i >= start && i < end) {
      xv = -dx;
      PetscCall(VecSetValues(Xh, 1, &i, &xv, ADD_VALUES));
    }
    PetscCall(VecAssemblyBegin(Xh));
    PetscCall(VecAssemblyEnd(Xh));
    PetscCall(SNESComputeObjective(snes, Xh, &ob3));

    if (i >= start && i < end) {
      /* set this entry to be the gradient of the objective */
      fv = (-ob2 + 6. * ob1 - 3. * ob - 2. * ob3) / (6. * dx);
      if (PetscAbsScalar(fv) > eps) {
        PetscCall(VecSetValues(F, 1, &i, &fv, INSERT_VALUES));
      } else {
        fv = 0.;
        PetscCall(VecSetValues(F, 1, &i, &fv, INSERT_VALUES));
      }
    }
  }
  PetscCall(VecDestroy(&Xh));
  PetscCall(VecAssemblyBegin(F));
  PetscCall(VecAssemblyEnd(F));
  PetscFunctionReturn(PETSC_SUCCESS);
}
