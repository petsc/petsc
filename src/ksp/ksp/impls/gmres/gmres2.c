
#include <../src/ksp/ksp/impls/gmres/gmresimpl.h>       /*I  "petscksp.h"  I*/

#undef __FUNCT__
#define __FUNCT__ "KSPGMRESSetOrthogonalization"
/*@C
   KSPGMRESSetOrthogonalization - Sets the orthogonalization routine used by GMRES and FGMRES.

   Logically Collective on KSP

   Input Parameters:
+  ksp - iterative context obtained from KSPCreate
-  fcn - orthogonalization function

   Calling Sequence of function:
$   errorcode = int fcn(KSP ksp,int it);
$   it is one minus the number of GMRES iterations since last restart;
$    i.e. the size of Krylov space minus one

   Notes:
   Two orthogonalization routines are predefined, including

   KSPGMRESModifiedGramSchmidtOrthogonalization()

   KSPGMRESClassicalGramSchmidtOrthogonalization() - Default. Use KSPGMRESSetCGSRefinementType() to determine if
     iterative refinement is used to increase stability.


   Options Database Keys:

+  -ksp_gmres_classicalgramschmidt - Activates KSPGMRESClassicalGramSchmidtOrthogonalization() (default)
-  -ksp_gmres_modifiedgramschmidt - Activates KSPGMRESModifiedGramSchmidtOrthogonalization()

   Level: intermediate

.keywords: KSP, GMRES, set, orthogonalization, Gram-Schmidt, iterative refinement

.seealso: KSPGMRESSetRestart(), KSPGMRESSetPreAllocateVectors(), KSPGMRESSetCGSRefinementType(), KSPGMRESSetOrthogonalization(),
          KSPGMRESModifiedGramSchmidtOrthogonalization(), KSPGMRESClassicalGramSchmidtOrthogonalization(), KSPGMRESGetCGSRefinementType()
@*/
PetscErrorCode  KSPGMRESSetOrthogonalization(KSP ksp,PetscErrorCode (*fcn)(KSP,PetscInt))
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  ierr = PetscTryMethod(ksp,"KSPGMRESSetOrthogonalization_C",(KSP,PetscErrorCode (*)(KSP,PetscInt)),(ksp,fcn));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "KSPGMRESGetOrthogonalization"
/*@C
   KSPGMRESGetOrthogonalization - Gets the orthogonalization routine used by GMRES and FGMRES.

   Not Collective

   Input Parameter:
.  ksp - iterative context obtained from KSPCreate

   Output Parameter:
.  fcn - orthogonalization function

   Calling Sequence of function:
$   errorcode = int fcn(KSP ksp,int it);
$   it is one minus the number of GMRES iterations since last restart;
$    i.e. the size of Krylov space minus one

   Notes:
   Two orthogonalization routines are predefined, including

   KSPGMRESModifiedGramSchmidtOrthogonalization()

   KSPGMRESClassicalGramSchmidtOrthogonalization() - Default. Use KSPGMRESSetCGSRefinementType() to determine if
     iterative refinement is used to increase stability.


   Options Database Keys:

+  -ksp_gmres_classicalgramschmidt - Activates KSPGMRESClassicalGramSchmidtOrthogonalization() (default)
-  -ksp_gmres_modifiedgramschmidt - Activates KSPGMRESModifiedGramSchmidtOrthogonalization()

   Level: intermediate

.keywords: KSP, GMRES, set, orthogonalization, Gram-Schmidt, iterative refinement

.seealso: KSPGMRESSetRestart(), KSPGMRESSetPreAllocateVectors(), KSPGMRESSetCGSRefinementType(), KSPGMRESSetOrthogonalization(),
          KSPGMRESModifiedGramSchmidtOrthogonalization(), KSPGMRESClassicalGramSchmidtOrthogonalization(), KSPGMRESGetCGSRefinementType()
@*/
PetscErrorCode  KSPGMRESGetOrthogonalization(KSP ksp,PetscErrorCode (**fcn)(KSP,PetscInt))
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  ierr = PetscUseMethod(ksp,"KSPGMRESGetOrthogonalization_C",(KSP,PetscErrorCode (**)(KSP,PetscInt)),(ksp,fcn));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
