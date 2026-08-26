#include <../src/ksp/ksp/impls/rich/richardsonimpl.h> /*I "petscksp.h" I*/

/*@
  KSPRichardsonSetScale - Set the damping factor; if this routine is not called, the factor defaults to 1.0.

  Logically Collective

  Input Parameters:
+ ksp   - the iterative context
- scale - the damping factor

  Options Database Key:
. -ksp_richardson_scale scale - Set the scale factor

  Level: intermediate

.seealso: [](ch_ksp), `KSPRICHARDSON`, `KSPRichardsonSetSelfScale()`
@*/
PetscErrorCode KSPRichardsonSetScale(KSP ksp, PetscReal scale)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp, KSP_CLASSID, 1);
  PetscValidLogicalCollectiveReal(ksp, scale, 2);
  PetscTryMethod(ksp, "KSPRichardsonSetScale_C", (KSP, PetscReal), (ksp, scale));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  KSPRichardsonSetSelfScale - Sets Richardson to automatically determine optimal scaling at each iteration to minimize the 2-norm of the
  preconditioned residual

  Logically Collective

  Input Parameters:
+ ksp   - the iterative context
- scale - `PETSC_TRUE` or the default of `PETSC_FALSE`

  Options Database Key:
. -ksp_richardson_self_scale - Use self-scaling

  Level: intermediate

  Notes:
  Requires two extra work vectors. Uses an extra `VecAXPY()` and `VecDotNorm2()` per iteration.

  Changing this flag runs `KSPSetUp()` again, since the number of work vectors depends on it. The self-scaled variant has no block analog, so turning it on makes
  `KSPMatSolve()` solve the right-hand sides one at a time instead of iterating on the whole block at once, and releases the work space that block iteration cached

  Developer Note:
  Could also minimize the 2-norm of the true residual with one less work vector

.seealso: [](ch_ksp), `KSPRICHARDSON`, `KSPRichardsonSetScale()`, `KSPMatSolve()`
@*/
PetscErrorCode KSPRichardsonSetSelfScale(KSP ksp, PetscBool scale)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp, KSP_CLASSID, 1);
  PetscValidLogicalCollectiveBool(ksp, scale, 2);
  PetscTryMethod(ksp, "KSPRichardsonSetSelfScale_C", (KSP, PetscBool), (ksp, scale));
  PetscFunctionReturn(PETSC_SUCCESS);
}
