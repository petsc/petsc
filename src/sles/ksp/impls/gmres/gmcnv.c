
#ifndef lint
static char vcid[] = "$Id: gmcnv.c,v 1.2 1997/01/27 18:15:31 bsmith Exp curfman $";
#endif

#include "src/ksp/impls/gmres/gmresp.h"       /*I  "ksp.h"  I*/

#undef __FUNC__  
#define __FUNC__ "KSPDefaultConverged_GMRES"
int KSPDefaultConverged_GMRES(KSP ksp,int n,double rnorm,void *dummy)
{
  if ( rnorm <= ksp->ttol || rnorm != rnorm) return(1);
  else return(0);
}
