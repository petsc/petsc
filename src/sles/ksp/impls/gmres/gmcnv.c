
#ifndef lint
static char vcid[] = "$Id: gmcnv.c,v 1.1 1997/01/25 16:58:32 bsmith Exp bsmith $";
#endif

#include "src/ksp/impls/gmres/gmresp.h"       /*I  "ksp.h"  I*/

#undef __FUNC__  
#define __FUNC__ "KSPDefaultConverged_GMRES"
int KSPDefaultConverged_GMRES(KSP ksp,int n,double rnorm,void *dummy)
{
  if ( rnorm <= ksp->ttol ) return(1);
  else return(0);
}
