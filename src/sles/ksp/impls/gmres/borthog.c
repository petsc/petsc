/*$Id: borthog.c,v 1.51 1999/10/24 14:03:14 bsmith Exp bsmith $*/
/*
    Routines used for the orthogonalization of the Hessenberg matrix.

    Note that for the complex numbers version, the VecDot() and
    VecMDot() arguments within the code MUST remain in the order
    given for correct computation of inner products.
*/
#include "src/sles/ksp/impls/gmres/gmresp.h"

/*
    This is the basic orthogonalization routine using modified Gram-Schmidt.
 */
#undef __FUNC__  
#define __FUNC__ "KSPGMRESModifiedGramSchmidtOrthogonalization"
int KSPGMRESModifiedGramSchmidtOrthogonalization(KSP ksp,int it)
{
  KSP_GMRES *gmres = (KSP_GMRES *)(ksp->data);
  int       ierr,j;
  Scalar    *hh,*hes,tmp;

  PetscFunctionBegin;
  PLogEventBegin(KSP_GMRESOrthogonalization,ksp,0,0,0);
  /* update Hessenberg matrix and do Gram-Schmidt */
  hh  = HH(0,it);
  hes = HES(0,it);
  for (j=0; j<=it; j++) {
    /* (vv(it+1), vv(j)) */
    ierr   = VecDot(VEC_VV(it+1),VEC_VV(j),hh);CHKERRQ(ierr);
    *hes++ = *hh;
    /* vv(it+1) <- vv(it+1) - hh[it+1][j] vv(j) */
    tmp    = - (*hh++);  
    ierr   = VecAXPY(&tmp,VEC_VV(j),VEC_VV(it+1));CHKERRQ(ierr);
  }
  PLogEventEnd(KSP_GMRESOrthogonalization,ksp,0,0,0);
  PetscFunctionReturn(0);
}



