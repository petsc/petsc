#ifndef lint
static char vcid[] = "$Id: borthog.c,v 1.9 1995/10/17 14:06:22 gropp Exp bsmith $";
#endif

#define RERROR  gmres_error
#include "gmresp.h"

/*
  This is the basic version and does not assume anything. 
 */
int GMRESBasicOrthog( KSP itP,int it )
{
  KSP_GMRES *gmresP = (KSP_GMRES *)(itP->data);
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
  KSP_GMRES *gmresP = (KSP_GMRES *)(itP->data);
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

/*
  This version uses iterative refinement of UNMODIFIED Gram-Schmidt.  
  It can give better performance when running in a parallel 
  environment and in some cases even in a sequential environment (because
  MAXPY has more data reuse).

  Care is taken to accumulate the updated HH/HES values.
 */
int GMRESOrthogIR(KSP  itP,int it )
{
  KSP_GMRES *gmresP = (KSP_GMRES *)(itP->data);
  int       j,ncnt;
  Scalar    *hh, *hes,shh[20], *lhh;
  double    dnorm;

  PLogEventBegin(KSP_GMRESOrthogonalization,itP,0,0,0);
  /* Don't allocate small arrays */
  if (it < 20) lhh = shh;
  else {
    lhh = (Scalar *)PETSCMALLOC((it+1) * sizeof(Scalar));
  }
  
  /* update hessenberg matrix and do unmodified Gram-Schmidt */
  hh  = HH(0,it);
  hes = HES(0,it);

  /* Clear hh and hes since we will accumulate values into them */
  for (j=0; j<=it; j++) {
    hh[j] = 0.0;
    hes[j] = 0.0;
  }

  ncnt = 0;
  do {
    /* 
	 This is really a matrix-vector product, with the matrix stored
	 as pointer to rows 
    */
    VecMDot( it+1, VEC_VV(it+1), &(VEC_VV(0)), lhh ); /* <v,vnew> */

    /*
	 This is really a matrix vector product: 
	 [h[0],h[1],...]*[ v[0]; v[1]; ...] subtracted from v[it].
    */
    for (j=0; j<=it; j++) lhh[j] = - lhh[j];
    VecMAXPY(it+1, lhh, VEC_VV(it+1),&VEC_VV(0) );
    for (j=0; j<=it; j++) {
      hh[j]  -= lhh[j];     /* hh += <v,vnew> */
      hes[j] += lhh[j];     /* hes += - <v,vnew> */
    }

    /* Note that dnorm = (norm(d))**2 */
    dnorm = 0.0;
#if defined(PETSC_COMPLEX)
    for (j=0; j<=it; j++) dnorm += real(lhh[j] * conj(lhh[j]));
#else
    for (j=0; j<=it; j++) dnorm += lhh[j] * lhh[j];
#endif

    /* Continue until either we have only small corrections or we've done
	 as much work as a full orthogonalization (in terms of Mdots) */
  } while (dnorm > 1.0e-16 && ncnt++ < it);

  /* It would be nice to put ncnt somewhere.... */

  if (it >= 20) PETSCFREE( lhh );
  PLogEventEnd(KSP_GMRESOrthogonalization,itP,0,0,0);
  return 0;
}

