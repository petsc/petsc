
#define RERROR  gmres_error
#include "gmresp.h"

/*
  This is the basic version and does not assume anything. 
 */
int GMRESBasicOrthog( KSP itP,int it )
{
  KSPiGMRESCntx *gmresP = (KSPiGMRESCntx *)(itP->MethodPrivate);
  int    j;
  double *hh, *hes, tmp;

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
  return 0;
}

/*
  This version uses UNMODIFIED Gram-Schmidt.  It is NOT recommended, 
  but it can give better performance when running in a parallel environment

  Multiple applications of this can be used to provide a better 
  orthogonalization (but be careful of the HH and HES values).
 */
int GMRESUnmodifiedOrthog(KSP  itP,int it )
{
  KSPiGMRESCntx *gmresP = (KSPiGMRESCntx *)(itP->MethodPrivate);
  int    j;
  double *hh, *hes, tmp;

  /* update hessenberg matrix and do unmodified Gram-Schmidt */
  hh  = HH(0,it);
  hes = HES(0,it);
  /* 
   This is really a matrix-vector product, with the matrix stored
   as pointer to rows 
  */
#ifdef USE_BASIC_VECTOR
  for (j=0; j<=it; j++) {
    /* vv(j) . vv(it+1) */
    VecDot( VEC_VV(j), VEC_VV(it+1), (hh+j) );
    *hes++   = hh[j];
  }
#else
  /* This is a multiple dot product */
  VecMDot( it+1, VEC_VV(it+1), &(VEC_VV(0)), hes );
  for (j=0; j<=it; j++) hh[j] = hes[j];
#endif
  /*
    This is really a matrix vector product: [h[0],h[1],...]*[ v[0]; v[1]; ...]
    subtracted from v[it].
  */
  for (j=0; j<=it; j++) {
    tmp = - hh[j]; VecAXPY(&tmp , VEC_VV(j), VEC_VV(it+1) );
  }
  return 0;
}
