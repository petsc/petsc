#ifndef lint
static char vcid[] = "$Id: borthog.c,v 1.6 1995/04/26 18:22:06 bsmith Exp bsmith $";
#endif

#define RERROR  gmres_error
#include "gmresp.h"

/*
  This is the basic version and does not assume anything. 
 */
int GMRESBasicOrthog( KSP itP,int it )
{
  KSP_GMRES *gmresP = (KSP_GMRES *)(itP->MethodPrivate);
  int    j;
  Scalar *hh, *hes, tmp;

  PLogEventBegin(KSP_GMRESOrthogonalization,itP,0,0,0);
  /* update hessenberg matrix and do Gram-Schmidt */
  hh  = HH(0,it);
  hes = HES(0,it);
  for (j=0; j<=it; j++) {
    /* vv(j) . vv(it+1) */
    VecDot( VEC_VV(j), VEC_VV(it+1), hh );
    *hes++   = *hh;
    /* vv(j) <- vv(j) - hh[j][it] vv(it) */
    tmp = - (*hh++);  VecAXPY(&tmp , VEC_VV(j), VEC_VV(it+1) );
  }
  PLogEventEnd(KSP_GMRESOrthogonalization,itP,0,0,0);
  return 0;
}

/*
  This version uses UNMODIFIED Gram-Schmidt.  It is NOT recommended, 
  but it can give better performance when running in a parallel 
  environment

  Multiple applications of this can be used to provide a better 
  orthogonalization (but be careful of the HH and HES values).
 */
int GMRESUnmodifiedOrthog(KSP  itP,int it )
{
  KSP_GMRES *gmresP = (KSP_GMRES *)(itP->MethodPrivate);
  int    j;
  Scalar *hh, *hes;

  PLogEventBegin(KSP_GMRESOrthogonalization,itP,0,0,0);
  /* update hessenberg matrix and do unmodified Gram-Schmidt */
  hh  = HH(0,it);
  hes = HES(0,it);

  /* 
   This is really a matrix-vector product, with the matrix stored
   as pointer to rows 
  */
  VecMDot( it+1, VEC_VV(it+1), &(VEC_VV(0)), hes );

  /*
    This is really a matrix vector product: 
              [h[0],h[1],...]*[ v[0]; v[1]; ...] subtracted from v[it].
  */
  for (j=0; j<=it; j++) hh[j] = -hes[j];
  VecMAXPY(it+1, hh, VEC_VV(it+1),&VEC_VV(0) );
  for (j=0; j<=it; j++) hh[j] = -hh[j];
  PLogEventEnd(KSP_GMRESOrthogonalization,itP,0,0,0);
  return 0;
}
