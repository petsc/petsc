
#include <petsc-private/kspimpl.h>         /*I "petscksp.h" I*/
#include <../src/ksp/ksp/impls/rich/richardsonimpl.h>


#undef __FUNCT__  
#define __FUNCT__ "KSPRichardsonSetScale"
/*@
    KSPRichardsonSetScale - Set the damping factor; if this routine is not called, the factor 
    defaults to 1.0.

    Logically Collective on KSP

    Input Parameters:
+   ksp - the iterative context
-   scale - the relaxation factor

    Level: intermediate

.keywords: KSP, Richardson, set, scale
@*/
PetscErrorCode  KSPRichardsonSetScale(KSP ksp,PetscReal scale)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  PetscValidLogicalCollectiveReal(ksp,scale,2);
  ierr = PetscTryMethod(ksp,"KSPRichardsonSetScale_C",(KSP,PetscReal),(ksp,scale));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "KSPRichardsonSetSelfScale"
/*@
    KSPRichardsonSetSelfScale - Sets Richardson to automatically determine optimal scaling at each iteration to minimize the 2-norm of the 
       preconditioned residual

    Logically Collective on KSP

    Input Parameters:
+   ksp - the iterative context
-   scale - PETSC_TRUE or the default of PETSC_FALSE

    Level: intermediate

    Notes: Requires two extra work vectors. Uses an extra axpy() and VecDotNorm2() per iteration.

    Developer Notes: Could also minimize the 2-norm of the true residual with one less work vector


.keywords: KSP, Richardson, set, scale
@*/
PetscErrorCode  KSPRichardsonSetSelfScale(KSP ksp,PetscBool  scale)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  PetscValidLogicalCollectiveBool(ksp,scale,2);
  ierr = PetscTryMethod(ksp,"KSPRichardsonSetSelfScale_C",(KSP,PetscBool),(ksp,scale));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
