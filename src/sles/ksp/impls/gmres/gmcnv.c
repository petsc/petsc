#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: gmcnv.c,v 1.3 1997/02/15 19:13:38 curfman Exp balay $";
#endif

#include "src/ksp/impls/gmres/gmresp.h"       /*I  "ksp.h"  I*/

#undef __FUNC__  
#define __FUNC__ "KSPDefaultConverged_GMRES"
int KSPDefaultConverged_GMRES(KSP ksp,int n,double rnorm,void *dummy)
{
  if ( rnorm <= ksp->ttol || rnorm != rnorm) return(1);
  else return(0);
}
