#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: richscale.c,v 1.4 1997/10/19 03:23:32 bsmith Exp bsmith $";
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

    Input Parameters:
.   ksp - the iterative context
.   scale - the relaxation factor

.keywords: KSP, Richardson, set, scale
@*/
int KSPRichardsonSetScale(KSP ksp,double scale)
{
  int ierr, (*f)(KSP,double);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_COOKIE);
  ierr = DLRegisterFind(ksp->qlist,"KSPRichardsonSetScale",(int (**)(void *))&f); CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(ksp,scale);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
