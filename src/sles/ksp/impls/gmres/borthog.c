#ifndef lint
static char vcid[] = "$Id: borthog.c,v 1.36 1997/01/27 18:15:31 bsmith Exp bsmith $";
#endif
/*
    Routines used for the orthogonalization of the Hessenberg matrix.

    Note that for the complex numbers version, the VecDot() and
    VecMDot() arguments within the code MUST remain in the order
    given for correct computation of inner products.
*/
#include "src/ksp/impls/gmres/gmresp.h"
#include <math.h>

/*
    This is the basic orthogonalization routine using modified Gram-Schmidt.
 */
#undef __FUNC__  
#define __FUNC__ "KSPGMRESModifiedGramSchmidtOrthogonalization"
int KSPGMRESModifiedGramSchmidtOrthogonalization( KSP ksp,int it )
{
  KSP_GMRES *gmres = (KSP_GMRES *)(ksp->data);
  int       j;
  Scalar    *hh, *hes, tmp;

  PLogEventBegin(KSP_GMRESOrthogonalization,ksp,0,0,0);
  /* update Hessenberg matrix and do Gram-Schmidt */
  hh  = HH(0,it);
  hes = HES(0,it);
  for (j=0; j<=it; j++) {
    /* ( vv(it+1), vv(j) ) */
    VecDot( VEC_VV(it+1), VEC_VV(j), hh );
    *hes++   = *hh;
    /* vv(it+1) <- vv(it+1) - hh[it+1][j] vv(j) */
    tmp = - (*hh++);  VecAXPY(&tmp , VEC_VV(j), VEC_VV(it+1) );
  }
  PLogEventEnd(KSP_GMRESOrthogonalization,ksp,0,0,0);
  return 0;
}



