#ifndef lint
static char vcid[] = "$Id: borthog.c,v 1.27 1996/06/17 20:54:02 bsmith Exp bsmith $";
#endif
/*
    Routines used for the orthogonalization of the Hessenberg matrix.

    Note that for the complex numbers version, the VecDot() and
    VecMDot() arguments within the code MUST remain in the order
    given for correct computation of inner products.
*/
#include "gmresp.h"
#include <math.h>

/*
    This is the basic orthogonalization routine using modified Gram-Schmidt.
 */
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
    /* vv(j) <- vv(j) - hh[j][it] vv(it) */
    tmp = - (*hh++);  VecAXPY(&tmp , VEC_VV(j), VEC_VV(it+1) );
  }
  PLogEventEnd(KSP_GMRESOrthogonalization,ksp,0,0,0);
  return 0;
}

/*
  This version uses UNMODIFIED Gram-Schmidt.  It is NOT recommended, 
  but it can give better performance when running in a parallel 
  environment

  Multiple applications of this can be used to provide a better 
  orthogonalization (but be careful of the HH and HES values).
 */
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
        [h[0],h[1],...]*[ v[0]; v[1]; ...] subtracted from v[it].
  */
  for (j=0; j<=it; j++) hh[j] = -hes[j];
  VecMAXPY(it+1, hh, VEC_VV(it+1),&VEC_VV(0) );
  for (j=0; j<=it; j++) hh[j] = -hh[j];
  PLogEventEnd(KSP_GMRESOrthogonalization,ksp,0,0,0);
  return 0;
}

/*
  This version uses iterative refinement of UNMODIFIED Gram-Schmidt.  
  It can give better performance when running in a parallel 
  environment and in some cases even in a sequential environment (because
  MAXPY has more data reuse).

  Care is taken to accumulate the updated HH/HES values.
 */
int KSPGMRESIROrthogonalization(KSP  ksp,int it )
{
  KSP_GMRES *gmres = (KSP_GMRES *)(ksp->data);
  int       j,ncnt;
  Scalar    *hh, *hes,shh[20], *lhh;
  double    dnorm;

  PLogEventBegin(KSP_GMRESOrthogonalization,ksp,0,0,0);
  /* Don't allocate small arrays */
  if (it < 20) lhh = shh;
  else {
    lhh = (Scalar *)PetscMalloc((it+1) * sizeof(Scalar)); CHKPTRQ(lhh);
  }
  
  /* update Hessenberg matrix and do unmodified Gram-Schmidt */
  hh  = HH(0,it);
  hes = HES(0,it);

  /* Clear hh and hes since we will accumulate values into them */
  for (j=0; j<=it; j++) {
    hh[j]  = 0.0;
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

  if (it >= 20) PetscFree( lhh );
  PLogEventEnd(KSP_GMRESOrthogonalization,ksp,0,0,0);
  return 0;
}

#if !defined(PETSC_COMPLEX)

#include "pinclude/plapack.h"
/*
    The Hessenberg matrix is factored at H = QR by the 
  GMRESUpdateHessenberg() routine. This routine computes the 
  singular values of R as estimates for the singular values of 
  the preconditioned operator.
*/
int KSPComputeExtremeSingularValues_GMRES(KSP ksp,Scalar *emax,Scalar *emin)
{
  KSP_GMRES *gmres = (KSP_GMRES *) ksp->data;
  int       n = gmres->it + 1, N = gmres->max_k + 2, ierr, lwork = 5*N;
  int       idummy = N, i;
  Scalar    *R = gmres->Rsvd;
  Scalar    *realpart = R + N*N, *work = realpart + N;
  Scalar    sdummy;

  if (n == 0) {
    *emax = *emin = 1.0;
    return 0;
  }
  /* copy R matrix to work space */
  PetscMemcpy(R,gmres->hh_origin,N*N*sizeof(Scalar));

  /* zero below diagonal garbage */
  for ( i=0; i<n; i++ ) {
    R[i*N+i+1] = 0.0;
  }
  
  /* compute Singular Values */
#if defined(PARCH_t3d)
  SETERRQ(1,"KSPComputeExtremeSingularValues_GMRES:DGESVD not found on Cray T3D\n\
             Therefore not able to provide singular value estimates.");
#else
  LAgesvd_("N","N",&n,&n,R,&N,realpart,&sdummy,&idummy,&sdummy,
           &idummy,work,&lwork,&ierr);
  if (ierr) SETERRQ(1,"KSPComputeExtremeSingularValues_GMRES:Error in SVD");
#endif

  *emin = realpart[n-1];
  *emax = realpart[0];

  return 0;
}

#else

int KSPComputeExtremeSingularValues_GMRES(KSP ksp,Scalar *emax,Scalar *emin)
{
  SETERRQ(1,"KSPComputeExtremeSingularValues_GMRES:No code for complex");
}

#endif


