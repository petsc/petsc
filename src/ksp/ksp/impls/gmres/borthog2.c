
/*
    Routines used for the orthogonalization of the Hessenberg matrix.

    Note that for the complex numbers version, the VecDot() and
    VecMDot() arguments within the code MUST remain in the order
    given for correct computation of inner products.
*/
#include <../src/ksp/ksp/impls/gmres/gmresimpl.h>

/*@C
     KSPGMRESClassicalGramSchmidtOrthogonalization -  This is the basic orthogonalization routine
                using classical Gram-Schmidt with possible iterative refinement to improve the stability

     Collective on ksp

  Input Parameters:
+   ksp - KSP object, must be associated with GMRES, FGMRES, or LGMRES Krylov method
-   its - one less then the current GMRES restart iteration, i.e. the size of the Krylov space

   Options Database Keys:
+   -ksp_gmres_classicalgramschmidt - Activates KSPGMRESClassicalGramSchmidtOrthogonalization()
-   -ksp_gmres_cgs_refinement_type <refine_never,refine_ifneeded,refine_always> - determine if iterative refinement is
                                   used to increase the stability of the classical Gram-Schmidt  orthogonalization.

    Notes:
    Use KSPGMRESSetCGSRefinementType() to determine if iterative refinement is to be used.
    This is much faster than KSPGMRESModifiedGramSchmidtOrthogonalization() but has the small possibility of stability issues
    that can usually be handled by using a a single step of iterative refinement with KSPGMRESSetCGSRefinementType()

   Level: intermediate

.seelalso:  `KSPGMRESSetOrthogonalization()`, `KSPGMRESClassicalGramSchmidtOrthogonalization()`, `KSPGMRESSetCGSRefinementType()`,
            `KSPGMRESGetCGSRefinementType()`, `KSPGMRESGetOrthogonalization(), `KSPGMRESModifiedGramSchmidtOrthogonalization()`

@*/
PetscErrorCode  KSPGMRESClassicalGramSchmidtOrthogonalization(KSP ksp,PetscInt it)
{
  KSP_GMRES      *gmres = (KSP_GMRES*)(ksp->data);
  PetscInt       j;
  PetscScalar    *hh,*hes,*lhh;
  PetscReal      hnrm, wnrm;
  PetscBool      refine = (PetscBool)(gmres->cgstype == KSP_GMRES_CGS_REFINE_ALWAYS);

  PetscFunctionBegin;
  PetscCall(PetscLogEventBegin(KSP_GMRESOrthogonalization,ksp,0,0,0));
  if (!gmres->orthogwork) {
    PetscCall(PetscMalloc1(gmres->max_k + 2,&gmres->orthogwork));
  }
  lhh = gmres->orthogwork;

  /* update Hessenberg matrix and do unmodified Gram-Schmidt */
  hh  = HH(0,it);
  hes = HES(0,it);

  /* Clear hh and hes since we will accumulate values into them */
  for (j=0; j<=it; j++) {
    hh[j]  = 0.0;
    hes[j] = 0.0;
  }

  /*
     This is really a matrix-vector product, with the matrix stored
     as pointer to rows
  */
  PetscCall(VecMDot(VEC_VV(it+1),it+1,&(VEC_VV(0)),lhh)); /* <v,vnew> */
  for (j=0; j<=it; j++) {
    KSPCheckDot(ksp,lhh[j]);
    if (ksp->reason) goto done;
    lhh[j] = -lhh[j];
  }

  /*
         This is really a matrix vector product:
         [h[0],h[1],...]*[ v[0]; v[1]; ...] subtracted from v[it+1].
  */
  PetscCall(VecMAXPY(VEC_VV(it+1),it+1,lhh,&VEC_VV(0)));
  /* note lhh[j] is -<v,vnew> , hence the subtraction */
  for (j=0; j<=it; j++) {
    hh[j]  -= lhh[j];     /* hh += <v,vnew> */
    hes[j] -= lhh[j];     /* hes += <v,vnew> */
  }

  /*
     the second step classical Gram-Schmidt is only necessary
     when a simple test criteria is not passed
  */
  if (gmres->cgstype == KSP_GMRES_CGS_REFINE_IFNEEDED) {
    hnrm = 0.0;
    for (j=0; j<=it; j++) hnrm +=  PetscRealPart(lhh[j] * PetscConj(lhh[j]));

    hnrm = PetscSqrtReal(hnrm);
    PetscCall(VecNorm(VEC_VV(it+1),NORM_2, &wnrm));
    KSPCheckNorm(ksp,wnrm);
    if (ksp->reason) goto done;
    if (wnrm < hnrm) {
      refine = PETSC_TRUE;
      PetscCall(PetscInfo(ksp,"Performing iterative refinement wnorm %g hnorm %g\n",(double)wnrm,(double)hnrm));
    }
  }

  if (refine) {
    PetscCall(VecMDot(VEC_VV(it+1),it+1,&(VEC_VV(0)),lhh)); /* <v,vnew> */
    for (j=0; j<=it; j++) {
       KSPCheckDot(ksp,lhh[j]);
       if (ksp->reason) goto done;
       lhh[j] = -lhh[j];
    }
    PetscCall(VecMAXPY(VEC_VV(it+1),it+1,lhh,&VEC_VV(0)));
    /* note lhh[j] is -<v,vnew> , hence the subtraction */
    for (j=0; j<=it; j++) {
      hh[j]  -= lhh[j];     /* hh += <v,vnew> */
      hes[j] -= lhh[j];     /* hes += <v,vnew> */
    }
  }
done:
  PetscCall(PetscLogEventEnd(KSP_GMRESOrthogonalization,ksp,0,0,0));
  PetscFunctionReturn(0);
}
