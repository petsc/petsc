#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: richscale.c,v 1.3 1997/07/09 20:50:50 balay Exp bsmith $";
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
    KSPRichardsonSetScale - Call after KSPCreate(KSPRICHARDSON) to set
    the damping factor; if this routine is not called, the factor 
    defaults to 1.0.

    Input Parameters:
.   ksp - the iterative context
.   scale - the relaxation factor

.keywords: KSP, Richardson, set, scale
@*/
int KSPRichardsonSetScale(KSP ksp,double scale)
{
  KSP_Richardson *richardsonP;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_COOKIE);
  if (ksp->type != KSPRICHARDSON) PetscFunctionReturn(0);
  richardsonP = (KSP_Richardson *) ksp->data;
  richardsonP->scale = scale;
  PetscFunctionReturn(0);
}
