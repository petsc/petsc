#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: richscale.c,v 1.9 1998/04/25 03:09:04 curfman Exp bsmith $";
#endif
/*          
            This implements Richardson Iteration.       
*/
#include <math.h>
#include "petsc.h"
#include "src/ksp/kspimpl.h"         /*I "ksp.h" I*/
#include "src/ksp/impls/rich/richctx.h"
#include "pinclude/pviewer.h"


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
