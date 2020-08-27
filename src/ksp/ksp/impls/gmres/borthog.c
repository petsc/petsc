
/*
    Routines used for the orthogonalization of the Hessenberg matrix.

    Note that for the complex numbers version, the VecDot() and
    VecMDot() arguments within the code MUST remain in the order
    given for correct computation of inner products.
*/
#include <../src/ksp/ksp/impls/gmres/gmresimpl.h>

/*@C
     KSPGMRESModifiedGramSchmidtOrthogonalization -  This is the basic orthogonalization routine
                using modified Gram-Schmidt.

     Collective on ksp

  Input Parameters:
+   ksp - KSP object, must be associated with GMRES, FGMRES, or LGMRES Krylov method
-   its - one less then the current GMRES restart iteration, i.e. the size of the Krylov space

   Options Database Keys:
.  -ksp_gmres_modifiedgramschmidt - Activates KSPGMRESModifiedGramSchmidtOrthogonalization()

   Notes:
     In general this is much slower than KSPGMRESClassicalGramSchmidtOrthogonalization() but has better stability properties.

   Level: intermediate

.seealso:  KSPGMRESSetOrthogonalization(), KSPGMRESClassicalGramSchmidtOrthogonalization(), KSPGMRESGetOrthogonalization()

@*/
PetscErrorCode  KSPGMRESModifiedGramSchmidtOrthogonalization(KSP ksp,PetscInt it)
{
  KSP_GMRES      *gmres = (KSP_GMRES*)(ksp->data);
  PetscErrorCode ierr;
  PetscInt       j;
  PetscScalar    *hh,*hes;

  PetscFunctionBegin;
  ierr = PetscLogEventBegin(KSP_GMRESOrthogonalization,ksp,0,0,0);CHKERRQ(ierr);
  /* update Hessenberg matrix and do Gram-Schmidt */
  hh  = HH(0,it);
  hes = HES(0,it);
  for (j=0; j<=it; j++) {
    /* (vv(it+1), vv(j)) */
    ierr   = VecDot(VEC_VV(it+1),VEC_VV(j),hh);CHKERRQ(ierr);
    KSPCheckDot(ksp,*hh);
    if (ksp->reason) break;
    *hes++ = *hh;
    /* vv(it+1) <- vv(it+1) - hh[it+1][j] vv(j) */
    ierr = VecAXPY(VEC_VV(it+1),-(*hh++),VEC_VV(j));CHKERRQ(ierr);
  }
  ierr = PetscLogEventEnd(KSP_GMRESOrthogonalization,ksp,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


