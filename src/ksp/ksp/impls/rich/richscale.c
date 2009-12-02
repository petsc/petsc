#define PETSCKSP_DLL

#include "private/kspimpl.h"         /*I "petscksp.h" I*/
#include "../src/ksp/ksp/impls/rich/richardsonimpl.h"


#undef __FUNCT__  
#define __FUNCT__ "KSPRichardsonSetScale"
/*@
    KSPRichardsonSetScale - Set the damping factor; if this routine is not called, the factor 
    defaults to 1.0.

    Collective on KSP

    Input Parameters:
+   ksp - the iterative context
-   scale - the relaxation factor

    Level: intermediate

.keywords: KSP, Richardson, set, scale
@*/
PetscErrorCode PETSCKSP_DLLEXPORT KSPRichardsonSetScale(KSP ksp,PetscReal scale)
{
  PetscErrorCode ierr,(*f)(KSP,PetscReal);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_COOKIE,1);
  ierr = PetscObjectQueryFunction((PetscObject)ksp,"KSPRichardsonSetScale_C",(void (**)(void))&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(ksp,scale);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
