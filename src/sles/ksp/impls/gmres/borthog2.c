#ifndef lint
static char vcid[] = "$Id: borthog2.c,v 1.2 1997/01/27 18:15:31 bsmith Exp bsmith $";
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
  This version uses UNMODIFIED Gram-Schmidt.  It is NOT recommended, 
  but it can give better performance when running in a parallel 
  environment

  Multiple applications of this can be used to provide a better 
  orthogonalization (but be careful of the HH and HES values).
 */
#undef __FUNC__  
#define __FUNC__ "KSPGMRESUnmodifiedGramSchmidtOrthogonalization"
int KSPGMRESUnmodifiedGramSchmidtOrthogonalization(KSP  ksp,int it )
{
  KSP_GMRES *gmres = (KSP_GMRES *)(ksp->data);
  int       j;
  Scalar    *hh, *hes;

  PLogEventBegin(KSP_GMRESOrthogonalization,ksp,0,0,0);
  /* update hessenberg matrix and do unmodified Gram-Schmidt */
  hh  = HH(0,it);
  hes = HES(0,it);

  /* 
   This is really a matrix-vector product, with the matrix stored
   as pointer to rows 
  */
  VecMDot( it+1, VEC_VV(it+1), &(VEC_VV(0)), hes );

  /*
    This is really a matrix-vector product: 
        [h[0],h[1],...]*[ v[0]; v[1]; ...] subtracted from v[it+1].
  */
  for (j=0; j<=it; j++) hh[j] = -hes[j];
  VecMAXPY(it+1, hh, VEC_VV(it+1),&VEC_VV(0) );
  for (j=0; j<=it; j++) hh[j] = -hh[j];
  PLogEventEnd(KSP_GMRESOrthogonalization,ksp,0,0,0);
  return 0;
}

