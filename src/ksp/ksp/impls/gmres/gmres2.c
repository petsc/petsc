#include <../src/ksp/ksp/impls/gmres/gmresimpl.h> /*I  "petscksp.h"  I*/

/*@C
  KSPGMRESSetOrthogonalization - Sets the orthogonalization routine used by `KSPGMRES` and `KSPFGMRES`.

  Logically Collective

  Input Parameters:
+ ksp - iterative context obtained from `KSPCreate()`
- fcn - orthogonalization function

  Calling sequence of `fcn`:
+ ksp - the solver context
- it  - the current iteration

  Options Database Keys:
+ -ksp_gmres_classicalgramschmidt - Activates KSPGMRESClassicalGramSchmidtOrthogonalization() (default)
- -ksp_gmres_modifiedgramschmidt  - Activates KSPGMRESModifiedGramSchmidtOrthogonalization()

  Level: intermediate

  Notes:
  Two orthogonalization routines are predefined, `KSPGMRESModifiedGramSchmidtOrthogonalization()` and the default
  `KSPGMRESClassicalGramSchmidtOrthogonalization()`.

  Use `KSPGMRESSetCGSRefinementType()` to determine if iterative refinement is used to increase stability.

.seealso: [](ch_ksp), `KSPGMRESSetRestart()`, `KSPGMRESSetPreAllocateVectors()`,
`KSPGMRESSetCGSRefinementType()`, `KSPGMRESModifiedGramSchmidtOrthogonalization()`,
`KSPGMRESClassicalGramSchmidtOrthogonalization()`, `KSPGMRESGetCGSRefinementType()`
@*/
PetscErrorCode KSPGMRESSetOrthogonalization(KSP ksp, PetscErrorCode (*fcn)(KSP ksp, PetscInt it))
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp, KSP_CLASSID, 1);
  PetscTryMethod(ksp, "KSPGMRESSetOrthogonalization_C", (KSP, PetscErrorCode (*)(KSP, PetscInt)), (ksp, fcn));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  KSPGMRESGetOrthogonalization - Gets the orthogonalization routine used by `KSPGMRES` and `KSPFGMRES`.

  Not Collective

  Input Parameter:
. ksp - iterative context obtained from `KSPCreate()`

  Output Parameter:
. fcn - orthogonalization function

  Calling sequence of `fcn`:
+ ksp - the solver context
- it  - the current iteration

  Level: intermediate

  Notes:
  Two orthogonalization routines are predefined,  `KSPGMRESModifiedGramSchmidtOrthogonalization()`, and the default
  `KSPGMRESClassicalGramSchmidtOrthogonalization()`

  Use `KSPGMRESSetCGSRefinementType()` to determine if iterative refinement is used to increase stability.

.seealso: [](ch_ksp), `KSPGMRESSetRestart()`, `KSPGMRESSetPreAllocateVectors()`, `KSPGMRESSetCGSRefinementType()`, `KSPGMRESSetOrthogonalization()`,
          `KSPGMRESModifiedGramSchmidtOrthogonalization()`, `KSPGMRESClassicalGramSchmidtOrthogonalization()`, `KSPGMRESGetCGSRefinementType()`
@*/
PetscErrorCode KSPGMRESGetOrthogonalization(KSP ksp, PetscErrorCode (**fcn)(KSP ksp, PetscInt it))
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp, KSP_CLASSID, 1);
  PetscUseMethod(ksp, "KSPGMRESGetOrthogonalization_C", (KSP, PetscErrorCode (**)(KSP, PetscInt)), (ksp, fcn));
  PetscFunctionReturn(PETSC_SUCCESS);
}
