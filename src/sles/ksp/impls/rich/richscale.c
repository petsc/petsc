#ifndef lint
static char vcid[] = "$Id: richscale.c,v 1.1 1997/01/25 04:22:47 bsmith Exp bsmith $";
#endif
/*          
            This implements Richardson Iteration.       
*/
#include <stdio.h>
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
  PetscValidHeaderSpecific(ksp,KSP_COOKIE);
  if (ksp->type != KSPRICHARDSON) return 0;
  richardsonP = (KSP_Richardson *) ksp->data;
  richardsonP->scale = scale;
  return 0;
}
