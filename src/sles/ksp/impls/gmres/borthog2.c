/*$Id: borthog2.c,v 1.20 2001/08/07 03:03:51 balay Exp $*/
/*
    Routines used for the orthogonalization of the Hessenberg matrix.

    Note that for the complex numbers version, the VecDot() and
    VecMDot() arguments within the code MUST remain in the order
    given for correct computation of inner products.
*/
#include "src/sles/ksp/impls/gmres/gmresp.h"

/*
  This version uses UNMODIFIED Gram-Schmidt.  It is NOT always recommended, 
  but it can give MUCH better performance than the default modified form
  when running in a parallel environment.
 */
#undef __FUNCT__  
#define __FUNCT__ "KSPGMRESUnmodifiedGramSchmidtOrthogonalization"
int KSPGMRESUnmodifiedGramSchmidtOrthogonalization(KSP  ksp,int it)
{
  KSP_GMRES *gmres = (KSP_GMRES *)(ksp->data);
  int       j,ierr;
  PetscScalar    *hh,*hes;

  PetscFunctionBegin;
  ierr = KSPLogEventBegin(KSP_GMRESOrthogonalization,ksp,0,0,0);CHKERRQ(ierr);
  /* update Hessenberg matrix and do unmodified Gram-Schmidt */
  hh  = HH(0,it);
  hes = HES(0,it);

  /* 
   This is really a matrix-vector product, with the matrix stored
   as pointer to rows 
  */
  ierr = VecMDot(it+1,VEC_VV(it+1),&(VEC_VV(0)),hes);CHKERRQ(ierr);

  /*
    This is really a matrix-vector product: 
        [h[0],h[1],...]*[ v[0]; v[1]; ...] subtracted from v[it+1].
  */
  for (j=0; j<=it; j++) hh[j] = -hes[j];
  ierr = VecMAXPY(it+1,hh,VEC_VV(it+1),&VEC_VV(0));CHKERRQ(ierr);
  for (j=0; j<=it; j++) hh[j] = -hh[j];
  ierr = KSPLogEventEnd(KSP_GMRESOrthogonalization,ksp,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}








