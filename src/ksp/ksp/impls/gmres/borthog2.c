#define PETSCKSP_DLL

/*
    Routines used for the orthogonalization of the Hessenberg matrix.

    Note that for the complex numbers version, the VecDot() and
    VecMDot() arguments within the code MUST remain in the order
    given for correct computation of inner products.
*/
#include "../src/ksp/ksp/impls/gmres/gmresimpl.h"

/*@C
     KSPGMRESClassicalGramSchmidtOrthogonalization -  This is the basic orthogonalization routine 
                using classical Gram-Schmidt with possible iterative refinement to improve the stability

     Collective on KSP

  Input Parameters:
+   ksp - KSP object, must be associated with GMRES, FGMRES, or LGMRES Krylov method
-   its - one less then the current GMRES restart iteration, i.e. the size of the Krylov space

   Options Database Keys:
+   -ksp_gmres_classicalgramschmidt - Activates KSPGMRESClassicalGramSchmidtOrthogonalization()
-   -ksp_gmres_cgs_refinement_type <refine_never,refine_ifneeded,refine_always> - determine if iterative refinement is 
                                   used to increase the stability of the classical Gram-Schmidt  orthogonalization.

    Notes: Use KSPGMRESSetCGSRefinementType() to determine if iterative refinement is to be used

   Level: intermediate

.seelaso:  KSPGMRESSetOrthogonalization(), KSPGMRESClassicalGramSchmidtOrthogonalization(), KSPGMRESSetCGSRefinementType()

@*/
#undef __FUNCT__  
#define __FUNCT__ "KSPGMRESClassicalGramSchmidtOrthogonalization"
PetscErrorCode PETSCKSP_DLLEXPORT KSPGMRESClassicalGramSchmidtOrthogonalization(KSP  ksp,PetscInt it)
{
  KSP_GMRES      *gmres = (KSP_GMRES *)(ksp->data);
  PetscErrorCode ierr;
  PetscInt       j;
  PetscScalar    *hh,*hes,*lhh;
  PetscReal      hnrm, wnrm;
  PetscTruth     refine = (PetscTruth)(gmres->cgstype == KSP_GMRES_CGS_REFINE_ALWAYS);

  PetscFunctionBegin;
  ierr = PetscLogEventBegin(KSP_GMRESOrthogonalization,ksp,0,0,0);CHKERRQ(ierr);
  if (!gmres->orthogwork) {
    ierr = PetscMalloc((gmres->max_k + 2)*sizeof(PetscScalar),&gmres->orthogwork);CHKERRQ(ierr);
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
  ierr = VecMDot(VEC_VV(it+1),it+1,&(VEC_VV(0)),lhh);CHKERRQ(ierr); /* <v,vnew> */
  for (j=0; j<=it; j++) {
    lhh[j] = - lhh[j];
  }

  /*
         This is really a matrix vector product: 
         [h[0],h[1],...]*[ v[0]; v[1]; ...] subtracted from v[it+1].
  */
  ierr = VecMAXPY(VEC_VV(it+1),it+1,lhh,&VEC_VV(0));CHKERRQ(ierr);
  /* note lhh[j] is -<v,vnew> , hence the subtraction */
  for (j=0; j<=it; j++) {
    hh[j]  -= lhh[j];     /* hh += <v,vnew> */
    hes[j] -= lhh[j];     /* hes += <v,vnew> */
  }

  /*
   *  the second step classical Gram-Schmidt is only necessary
   *  when a simple test criteria is not passed
   */
  if (gmres->cgstype == KSP_GMRES_CGS_REFINE_IFNEEDED) {
    hnrm = 0.0;
    for (j=0; j<=it; j++) {
      hnrm  +=  PetscRealPart(lhh[j] * PetscConj(lhh[j]));
    }
    hnrm = sqrt(hnrm);
    ierr = VecNorm(VEC_VV(it+1),NORM_2, &wnrm);CHKERRQ(ierr);
    if (wnrm < 1.0286 * hnrm) {
      refine = PETSC_TRUE;
      ierr = PetscInfo2(ksp,"Performing iterative refinement wnorm %G hnorm %G\n",wnrm,hnrm);CHKERRQ(ierr);
    }
  }

  if (refine) {
    ierr = VecMDot(VEC_VV(it+1),it+1,&(VEC_VV(0)),lhh);CHKERRQ(ierr); /* <v,vnew> */
    for (j=0; j<=it; j++) lhh[j] = - lhh[j];
    ierr = VecMAXPY(VEC_VV(it+1),it+1,lhh,&VEC_VV(0));CHKERRQ(ierr);
    /* note lhh[j] is -<v,vnew> , hence the subtraction */
    for (j=0; j<=it; j++) {
      hh[j]  -= lhh[j];     /* hh += <v,vnew> */
      hes[j] -= lhh[j];     /* hes += <v,vnew> */
    }
  }
  ierr = PetscLogEventEnd(KSP_GMRESOrthogonalization,ksp,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}








