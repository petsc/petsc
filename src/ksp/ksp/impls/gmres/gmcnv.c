/*$Id: gmcnv.c,v 1.7 1999/10/24 14:03:14 bsmith Exp bsmith $*/

#include "src/sles/ksp/impls/gmres/gmresp.h"       /*I  "ksp.h"  I*/

#undef __FUNC__  
#define __FUNC__ "KSPDefaultConverged_GMRES"
int KSPDefaultConverged_GMRES(KSP ksp,int n,double rnorm,KSPConvergedReason *reason,void *dummy)
{
  PetscFunctionBegin;
  if ( rnorm <= ksp->ttol || rnorm != rnorm) {
    *reason = KSP_CONVERGED_RTOL;
  }
  PetscFunctionReturn(0);
}
