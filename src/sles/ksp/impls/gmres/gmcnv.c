#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: gmcnv.c,v 1.5 1997/10/19 03:23:21 bsmith Exp bsmith $";
#endif

#include "src/sles/ksp/impls/gmres/gmresp.h"       /*I  "ksp.h"  I*/

#undef __FUNC__  
#define __FUNC__ "KSPDefaultConverged_GMRES"
int KSPDefaultConverged_GMRES(KSP ksp,int n,double rnorm,void *dummy)
{
  PetscFunctionBegin;
  if ( rnorm <= ksp->ttol || rnorm != rnorm) PetscFunctionReturn(1);
  PetscFunctionReturn(0);
}
