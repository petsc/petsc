#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: richscale.c,v 1.10 1998/05/29 20:35:58 bsmith Exp bsmith $";
#endif

#include "src/ksp/kspimpl.h"         /*I "ksp.h" I*/
#include "src/ksp/impls/rich/richctx.h"


#undef __FUNC__  
#define __FUNC__ "KSPRichardsonSetScale"
/*@
    KSPRichardsonSetScale - Set the damping factor; if this routine is not called, the factor 
    defaults to 1.0.

    Collective on KSP

    Input Parameters:
+   ksp - the iterative context
-   scale - the relaxation factor

.keywords: KSP, Richardson, set, scale
@*/
int KSPRichardsonSetScale(KSP ksp,double scale)
{
  int ierr, (*f)(KSP,double);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_COOKIE);
  ierr = PetscObjectQueryFunction((PetscObject)ksp,"KSPRichardsonSetScale_C",(void **)&f); CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(ksp,scale);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
