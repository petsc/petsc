
#include <../src/ksp/ksp/impls/gmres/gmresimpl.h> /*I  "petscksp.h"  I*/

/*@C
   KSPGMRESSetOrthogonalization - Sets the orthogonalization routine used by `KSPGMRES` and `KSPFGMRES`.

   Logically Collective on ksp

   Input Parameters:
+  ksp - iterative context obtained from `KSPCreate()`
-  fcn - orthogonalization function

   Calling Sequence of function:
$   errorcode = PetscErrorCode fcn(KSP ksp,PetscInt it);
$   it is one minus the number of GMRES iterations since last restart;
$    i.e. the size of Krylov space minus one

   Options Database Keys:
+  -ksp_gmres_classicalgramschmidt - Activates KSPGMRESClassicalGramSchmidtOrthogonalization() (default)
-  -ksp_gmres_modifiedgramschmidt - Activates KSPGMRESModifiedGramSchmidtOrthogonalization()

   Level: intermediate

   Notes:
   Two orthogonalization routines are predefined, including `KSPGMRESModifiedGramSchmidtOrthogonalization()` and the default
   `KSPGMRESClassicalGramSchmidtOrthogonalization()`.

   Use `KSPGMRESSetCGSRefinementType()` to determine if iterative refinement is used to increase stability.

.seealso: [](chapter_ksp), `KSPGMRESSetRestart()`, `KSPGMRESSetPreAllocateVectors()`, `KSPGMRESSetCGSRefinementType()`, `KSPGMRESSetOrthogonalization()`,
          `KSPGMRESModifiedGramSchmidtOrthogonalization()`, `KSPGMRESClassicalGramSchmidtOrthogonalization()`, `KSPGMRESGetCGSRefinementType()`
@*/
PetscErrorCode KSPGMRESSetOrthogonalization(KSP ksp, PetscErrorCode (*fcn)(KSP, PetscInt))
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp, KSP_CLASSID, 1);
  PetscTryMethod(ksp, "KSPGMRESSetOrthogonalization_C", (KSP, PetscErrorCode(*)(KSP, PetscInt)), (ksp, fcn));
  PetscFunctionReturn(0);
}

/*@C
   KSPGMRESGetOrthogonalization - Gets the orthogonalization routine used by `KSPGMRES` and `KSPFGMRES`.

   Not Collective

   Input Parameter:
.  ksp - iterative context obtained from `KSPCreate()`

   Output Parameter:
.  fcn - orthogonalization function

   Calling Sequence of function:
.vb
   errorcode = PetscErrorCode fcn(KSP ksp,PetscInt it);
   it is one minus the number of GMRES iterations since last restart; i.e. the size of Krylov space minus one
.ve

   Options Database Keys:
+  -ksp_gmres_classicalgramschmidt - Activates KSPGMRESClassicalGramSchmidtOrthogonalization() (default)
-  -ksp_gmres_modifiedgramschmidt - Activates KSPGMRESModifiedGramSchmidtOrthogonalization()

   Level: intermediate

   Notes:
   Two orthogonalization routines are predefined, including `KSPGMRESModifiedGramSchmidtOrthogonalization()`, and the default
   `KSPGMRESClassicalGramSchmidtOrthogonalization()`

   Use `KSPGMRESSetCGSRefinementType()` to determine if iterative refinement is used to increase stability.

.seealso: [](chapter_ksp), `KSPGMRESSetRestart()`, `KSPGMRESSetPreAllocateVectors()`, `KSPGMRESSetCGSRefinementType()`, `KSPGMRESSetOrthogonalization()`,
          `KSPGMRESModifiedGramSchmidtOrthogonalization()`, `KSPGMRESClassicalGramSchmidtOrthogonalization()`, `KSPGMRESGetCGSRefinementType()`
@*/
PetscErrorCode KSPGMRESGetOrthogonalization(KSP ksp, PetscErrorCode (**fcn)(KSP, PetscInt))
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp, KSP_CLASSID, 1);
  PetscUseMethod(ksp, "KSPGMRESGetOrthogonalization_C", (KSP, PetscErrorCode(**)(KSP, PetscInt)), (ksp, fcn));
  PetscFunctionReturn(0);
}
