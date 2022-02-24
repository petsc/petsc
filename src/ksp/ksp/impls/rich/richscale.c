
#include <../src/ksp/ksp/impls/rich/richardsonimpl.h>     /*I "petscksp.h" I*/

/*@
    KSPRichardsonSetScale - Set the damping factor; if this routine is not called, the factor
    defaults to 1.0.

    Logically Collective on ksp

    Input Parameters:
+   ksp - the iterative context
-   scale - the relaxation factor

    Options Database Keys:
. -ksp_richardson_self <scale> - Set the scale factor

    Level: intermediate

    .seealso: KSPRICHARDSON, KSPRichardsonSetSelfScale()
@*/
PetscErrorCode  KSPRichardsonSetScale(KSP ksp,PetscReal scale)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  PetscValidLogicalCollectiveReal(ksp,scale,2);
  CHKERRQ(PetscTryMethod(ksp,"KSPRichardsonSetScale_C",(KSP,PetscReal),(ksp,scale)));
  PetscFunctionReturn(0);
}

/*@
    KSPRichardsonSetSelfScale - Sets Richardson to automatically determine optimal scaling at each iteration to minimize the 2-norm of the
       preconditioned residual

    Logically Collective on ksp

    Input Parameters:
+   ksp - the iterative context
-   scale - PETSC_TRUE or the default of PETSC_FALSE

    Options Database Keys:
. -ksp_richardson_self_scale - Use self-scaling

    Level: intermediate

    Notes:
    Requires two extra work vectors. Uses an extra VecAXPY() and VecDotNorm2() per iteration.

    Developer Notes:
    Could also minimize the 2-norm of the true residual with one less work vector

    .seealso: KSPRICHARDSON, KSPRichardsonSetScale()
@*/
PetscErrorCode  KSPRichardsonSetSelfScale(KSP ksp,PetscBool scale)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  PetscValidLogicalCollectiveBool(ksp,scale,2);
  CHKERRQ(PetscTryMethod(ksp,"KSPRichardsonSetSelfScale_C",(KSP,PetscBool),(ksp,scale)));
  PetscFunctionReturn(0);
}
