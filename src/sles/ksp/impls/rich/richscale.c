/*$Id: richscale.c,v 1.16 2000/01/11 21:02:08 bsmith Exp bsmith $*/

#include "src/sles/ksp/kspimpl.h"         /*I "ksp.h" I*/
#include "src/sles/ksp/impls/rich/richctx.h"


#undef __FUNC__  
#define  __FUNC__ /*<a name=""></a>*/"KSPRichardsonSetScale"
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
int KSPRichardsonSetScale(KSP ksp,PetscReal scale)
{
  int ierr,(*f)(KSP,PetscReal);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_COOKIE);
  ierr = PetscObjectQueryFunction((PetscObject)ksp,"KSPRichardsonSetScale_C",(void **)&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(ksp,scale);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
