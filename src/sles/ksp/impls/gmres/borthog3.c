/*$Id: borthog3.c,v 1.22 2001/01/16 18:19:33 balay Exp balay $*/
/*
    Routines used for the orthogonalization of the Hessenberg matrix.

    Note that for the complex numbers version, the VecDot() and
    VecMDot() arguments within the code MUST remain in the order
    given for correct computation of inner products.
*/
#include "src/sles/ksp/impls/gmres/gmresp.h"

/*
  This version uses 1 iteration of iterative refinement of UNMODIFIED Gram-Schmidt.  
  It can give better performance when running in a parallel 
  environment and in some cases even in a sequential environment (because
  MAXPY has more data reuse).

  Care is taken to accumulate the updated HH/HES values.
 */
#undef __FUNCT__  
#define __FUNCT__ "KSPGMRESIROrthogonalization"
int KSPGMRESIROrthogonalization(KSP  ksp,int it)
{
  KSP_GMRES *gmres = (KSP_GMRES *)(ksp->data);
  int       j,ncnt,ierr;
  Scalar    *hh,*hes,shh[100],*lhh;

  PetscFunctionBegin;
  ierr = PetscLogEventBegin(KSP_GMRESOrthogonalization,ksp,0,0,0);CHKERRQ(ierr);
  /* Don't allocate small arrays */
  if (it < 100) lhh = shh;
  else {
    ierr = PetscMalloc((it+1) * sizeof(Scalar),&lhh);CHKERRQ(ierr);
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
    VecMDot(it+1,VEC_VV(it+1),&(VEC_VV(0)),lhh); /* <v,vnew> */

    /*
	 This is really a matrix vector product: 
	 [h[0],h[1],...]*[ v[0]; v[1]; ...] subtracted from v[it+1].
    */
    for (j=0; j<=it; j++) lhh[j] = - lhh[j];
    VecMAXPY(it+1,lhh,VEC_VV(it+1),&VEC_VV(0));
    for (j=0; j<=it; j++) {
      hh[j]  -= lhh[j];     /* hh += <v,vnew> */
      hes[j] += lhh[j];     /* hes += - <v,vnew> */
    }
  } while (ncnt++ < 2);

  if (it >= 100) {ierr = PetscFree(lhh);CHKERRQ(ierr);}
  ierr = PetscLogEventEnd(KSP_GMRESOrthogonalization,ksp,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}



