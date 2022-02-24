
/*

    Note that for the complex numbers version, the VecDot() arguments
    within the code MUST remain in the order given for correct computation
    of inner products.
*/
#include <petsc/private/kspimpl.h>

static PetscErrorCode KSPSetUp_CGS(KSP ksp)
{
  PetscFunctionBegin;
  CHKERRQ(KSPSetWorkVecs(ksp,7));
  PetscFunctionReturn(0);
}

static PetscErrorCode  KSPSolve_CGS(KSP ksp)
{
  PetscInt       i;
  PetscScalar    rho,rhoold,a,s,b;
  Vec            X,B,V,P,R,RP,T,Q,U,AUQ;
  PetscReal      dp = 0.0;
  PetscBool      diagonalscale;

  PetscFunctionBegin;
  /* not sure what residual norm it does use, should use for right preconditioning */

  CHKERRQ(PCGetDiagonalScale(ksp->pc,&diagonalscale));
  PetscCheckFalse(diagonalscale,PetscObjectComm((PetscObject)ksp),PETSC_ERR_SUP,"Krylov method %s does not support diagonal scaling",((PetscObject)ksp)->type_name);

  X   = ksp->vec_sol;
  B   = ksp->vec_rhs;
  R   = ksp->work[0];
  RP  = ksp->work[1];
  V   = ksp->work[2];
  T   = ksp->work[3];
  Q   = ksp->work[4];
  P   = ksp->work[5];
  U   = ksp->work[6];
  AUQ = V;

  /* Compute initial preconditioned residual */
  CHKERRQ(KSPInitialResidual(ksp,X,V,T,R,B));

  /* Test for nothing to do */
  if (ksp->normtype != KSP_NORM_NONE) {
    CHKERRQ(VecNorm(R,NORM_2,&dp));
    KSPCheckNorm(ksp,dp);
    if (ksp->normtype == KSP_NORM_NATURAL) dp *= dp;
  } else dp = 0.0;

  CHKERRQ(PetscObjectSAWsTakeAccess((PetscObject)ksp));
  ksp->its   = 0;
  ksp->rnorm = dp;
  CHKERRQ(PetscObjectSAWsGrantAccess((PetscObject)ksp));
  CHKERRQ(KSPLogResidualHistory(ksp,dp));
  CHKERRQ(KSPMonitor(ksp,0,dp));
  CHKERRQ((*ksp->converged)(ksp,0,dp,&ksp->reason,ksp->cnvP));
  if (ksp->reason) PetscFunctionReturn(0);

  /* Make the initial Rp == R */
  CHKERRQ(VecCopy(R,RP));
  /*  added for Fidap */
  /* Penalize Startup - Isaac Hasbani Trick for CGS
     Since most initial conditions result in a mostly 0 residual,
     we change all the 0 values in the vector RP to the maximum.
  */
  if (ksp->normtype == KSP_NORM_NATURAL) {
    PetscReal   vr0max;
    PetscScalar *tmp_RP=NULL;
    PetscInt    numnp  =0, *max_pos=NULL;
    CHKERRQ(VecMax(RP, max_pos, &vr0max));
    CHKERRQ(VecGetArray(RP, &tmp_RP));
    CHKERRQ(VecGetLocalSize(RP, &numnp));
    for (i=0; i<numnp; i++) {
      if (tmp_RP[i] == 0.0) tmp_RP[i] = vr0max;
    }
    CHKERRQ(VecRestoreArray(RP, &tmp_RP));
  }
  /*  end of addition for Fidap */

  /* Set the initial conditions */
  CHKERRQ(VecDot(R,RP,&rhoold));        /* rhoold = (r,rp)      */
  CHKERRQ(VecCopy(R,U));
  CHKERRQ(VecCopy(R,P));
  CHKERRQ(KSP_PCApplyBAorAB(ksp,P,V,T));

  i = 0;
  do {

    CHKERRQ(VecDot(V,RP,&s));           /* s <- (v,rp)          */
    KSPCheckDot(ksp,s);
    a    = rhoold / s;                               /* a <- rho / s         */
    CHKERRQ(VecWAXPY(Q,-a,V,U));      /* q <- u - a v         */
    CHKERRQ(VecWAXPY(T,1.0,U,Q));      /* t <- u + q           */
    CHKERRQ(VecAXPY(X,a,T));           /* x <- x + a (u + q)   */
    CHKERRQ(KSP_PCApplyBAorAB(ksp,T,AUQ,U));
    CHKERRQ(VecAXPY(R,-a,AUQ));       /* r <- r - a K (u + q) */
    CHKERRQ(VecDot(R,RP,&rho));         /* rho <- (r,rp)        */
    KSPCheckDot(ksp,rho);
    if (ksp->normtype == KSP_NORM_NATURAL) {
      dp = PetscAbsScalar(rho);
    } else if (ksp->normtype != KSP_NORM_NONE) {
      CHKERRQ(VecNorm(R,NORM_2,&dp));
      KSPCheckNorm(ksp,dp);
    } else dp = 0.0;

    CHKERRQ(PetscObjectSAWsTakeAccess((PetscObject)ksp));
    ksp->its++;
    ksp->rnorm = dp;
    CHKERRQ(PetscObjectSAWsGrantAccess((PetscObject)ksp));
    CHKERRQ(KSPLogResidualHistory(ksp,dp));
    CHKERRQ(KSPMonitor(ksp,i+1,dp));
    CHKERRQ((*ksp->converged)(ksp,i+1,dp,&ksp->reason,ksp->cnvP));
    if (ksp->reason) break;

    b      = rho / rhoold;                           /* b <- rho / rhoold    */
    CHKERRQ(VecWAXPY(U,b,Q,R));       /* u <- r + b q         */
    CHKERRQ(VecAXPY(Q,b,P));
    CHKERRQ(VecWAXPY(P,b,Q,U));       /* p <- u + b(q + b p)  */
    CHKERRQ(KSP_PCApplyBAorAB(ksp,P,V,Q));    /* v <- K p    */
    rhoold = rho;
    i++;
  } while (i<ksp->max_it);
  if (i >= ksp->max_it) ksp->reason = KSP_DIVERGED_ITS;

  CHKERRQ(KSPUnwindPreconditioner(ksp,X,T));
  PetscFunctionReturn(0);
}

/*MC
     KSPCGS - This code implements the CGS (Conjugate Gradient Squared) method.

   Options Database Keys:
    see KSPSolve()

   Level: beginner

   References:
.  * - Sonneveld, 1989.

   Notes:
    Does not require a symmetric matrix. Does not apply transpose of the matrix.
        Supports left and right preconditioning, but not symmetric.

   Developer Notes:
    Has this weird support for doing the convergence test with the natural norm, I assume this works only with
      no preconditioning and symmetric positive definite operator.

.seealso: KSPCreate(), KSPSetType(), KSPType (for list of available types), KSP, KSPBCGS, KSPSetPCSide()
M*/
PETSC_EXTERN PetscErrorCode KSPCreate_CGS(KSP ksp)
{
  PetscFunctionBegin;
  ksp->data = (void*)0;

  CHKERRQ(KSPSetSupportedNorm(ksp,KSP_NORM_PRECONDITIONED,PC_LEFT,3));
  CHKERRQ(KSPSetSupportedNorm(ksp,KSP_NORM_UNPRECONDITIONED,PC_RIGHT,2));
  CHKERRQ(KSPSetSupportedNorm(ksp,KSP_NORM_NATURAL,PC_LEFT,2));
  CHKERRQ(KSPSetSupportedNorm(ksp,KSP_NORM_NATURAL,PC_RIGHT,2));
  CHKERRQ(KSPSetSupportedNorm(ksp,KSP_NORM_NONE,PC_LEFT,1));
  CHKERRQ(KSPSetSupportedNorm(ksp,KSP_NORM_NONE,PC_RIGHT,1));

  ksp->ops->setup          = KSPSetUp_CGS;
  ksp->ops->solve          = KSPSolve_CGS;
  ksp->ops->destroy        = KSPDestroyDefault;
  ksp->ops->buildsolution  = KSPBuildSolutionDefault;
  ksp->ops->buildresidual  = KSPBuildResidualDefault;
  ksp->ops->setfromoptions = NULL;
  ksp->ops->view           = NULL;
  PetscFunctionReturn(0);
}
