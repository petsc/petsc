/*$Id: gmcnv.c,v 1.6 1999/01/31 16:08:49 bsmith Exp bsmith $*/

#include "src/sles/ksp/impls/gmres/gmresp.h"       /*I  "ksp.h"  I*/

#undef __FUNC__  
#define __FUNC__ "KSPDefaultConverged_GMRES"
int KSPDefaultConverged_GMRES(KSP ksp,int n,double rnorm,void *dummy)
{
  PetscFunctionBegin;
  if ( rnorm <= ksp->ttol || rnorm != rnorm) PetscFunctionReturn(1);
  PetscFunctionReturn(0);
}
