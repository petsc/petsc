
#include <petsc/private/kspimpl.h>

static PetscErrorCode KSPSetUp_TFQMR(KSP ksp)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscCheckFalse(ksp->pc_side == PC_SYMMETRIC,PetscObjectComm((PetscObject)ksp),PETSC_ERR_SUP,"no symmetric preconditioning for KSPTFQMR");
  ierr = KSPSetWorkVecs(ksp,9);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode  KSPSolve_TFQMR(KSP ksp)
{
  PetscErrorCode ierr;
  PetscInt       i,m;
  PetscScalar    rho,rhoold,a,s,b,eta,etaold,psiold,cf;
  PetscReal      dp,dpold,w,dpest,tau,psi,cm;
  Vec            X,B,V,P,R,RP,T,T1,Q,U,D,AUQ;

  PetscFunctionBegin;
  X   = ksp->vec_sol;
  B   = ksp->vec_rhs;
  R   = ksp->work[0];
  RP  = ksp->work[1];
  V   = ksp->work[2];
  T   = ksp->work[3];
  Q   = ksp->work[4];
  P   = ksp->work[5];
  U   = ksp->work[6];
  D   = ksp->work[7];
  T1  = ksp->work[8];
  AUQ = V;

  /* Compute initial preconditioned residual */
  ierr = KSPInitialResidual(ksp,X,V,T,R,B);CHKERRQ(ierr);

  /* Test for nothing to do */
  ierr = VecNorm(R,NORM_2,&dp);CHKERRQ(ierr);
  KSPCheckNorm(ksp,dp);
  ierr = PetscObjectSAWsTakeAccess((PetscObject)ksp);CHKERRQ(ierr);
  if (ksp->normtype != KSP_NORM_NONE) ksp->rnorm = dp;
  else ksp->rnorm = 0.0;
  ksp->its = 0;
  ierr     = PetscObjectSAWsGrantAccess((PetscObject)ksp);CHKERRQ(ierr);
  ierr     = KSPMonitor(ksp,0,ksp->rnorm);CHKERRQ(ierr);
  ierr     = (*ksp->converged)(ksp,0,ksp->rnorm,&ksp->reason,ksp->cnvP);CHKERRQ(ierr);
  if (ksp->reason) PetscFunctionReturn(0);

  /* Make the initial Rp == R */
  ierr = VecCopy(R,RP);CHKERRQ(ierr);

  /* Set the initial conditions */
  etaold = 0.0;
  psiold = 0.0;
  tau    = dp;
  dpold  = dp;

  ierr = VecDot(R,RP,&rhoold);CHKERRQ(ierr);       /* rhoold = (r,rp)     */
  ierr = VecCopy(R,U);CHKERRQ(ierr);
  ierr = VecCopy(R,P);CHKERRQ(ierr);
  ierr = KSP_PCApplyBAorAB(ksp,P,V,T);CHKERRQ(ierr);
  ierr = VecSet(D,0.0);CHKERRQ(ierr);

  i=0;
  do {
    ierr = PetscObjectSAWsTakeAccess((PetscObject)ksp);CHKERRQ(ierr);
    ksp->its++;
    ierr = PetscObjectSAWsGrantAccess((PetscObject)ksp);CHKERRQ(ierr);
    ierr = VecDot(V,RP,&s);CHKERRQ(ierr);          /* s <- (v,rp)          */
    KSPCheckDot(ksp,s);
    a    = rhoold / s;                              /* a <- rho / s         */
    ierr = VecWAXPY(Q,-a,V,U);CHKERRQ(ierr);  /* q <- u - a v         */
    ierr = VecWAXPY(T,1.0,U,Q);CHKERRQ(ierr);     /* t <- u + q           */
    ierr = KSP_PCApplyBAorAB(ksp,T,AUQ,T1);CHKERRQ(ierr);
    ierr = VecAXPY(R,-a,AUQ);CHKERRQ(ierr);      /* r <- r - a K (u + q) */
    ierr = VecNorm(R,NORM_2,&dp);CHKERRQ(ierr);
    KSPCheckNorm(ksp,dp);
    for (m=0; m<2; m++) {
      if (!m) w = PetscSqrtReal(dp*dpold);
      else w = dp;
      psi = w / tau;
      cm  = 1.0 / PetscSqrtReal(1.0 + psi * psi);
      tau = tau * psi * cm;
      eta = cm * cm * a;
      cf  = psiold * psiold * etaold / a;
      if (!m) {
        ierr = VecAYPX(D,cf,U);CHKERRQ(ierr);
      } else {
        ierr = VecAYPX(D,cf,Q);CHKERRQ(ierr);
      }
      ierr = VecAXPY(X,eta,D);CHKERRQ(ierr);

      dpest = PetscSqrtReal(2*i + m + 2.0) * tau;
      ierr  = PetscObjectSAWsTakeAccess((PetscObject)ksp);CHKERRQ(ierr);
      if (ksp->normtype != KSP_NORM_NONE) ksp->rnorm = dpest;
      else ksp->rnorm = 0.0;
      ierr = PetscObjectSAWsGrantAccess((PetscObject)ksp);CHKERRQ(ierr);
      ierr = KSPLogResidualHistory(ksp,ksp->rnorm);CHKERRQ(ierr);
      ierr = KSPMonitor(ksp,i+1,ksp->rnorm);CHKERRQ(ierr);
      ierr = (*ksp->converged)(ksp,i+1,ksp->rnorm,&ksp->reason,ksp->cnvP);CHKERRQ(ierr);
      if (ksp->reason) break;

      etaold = eta;
      psiold = psi;
    }
    if (ksp->reason) break;

    ierr = VecDot(R,RP,&rho);CHKERRQ(ierr);        /* rho <- (r,rp)       */
    b    = rho / rhoold;                            /* b <- rho / rhoold   */
    ierr = VecWAXPY(U,b,Q,R);CHKERRQ(ierr);       /* u <- r + b q        */
    ierr = VecAXPY(Q,b,P);CHKERRQ(ierr);
    ierr = VecWAXPY(P,b,Q,U);CHKERRQ(ierr);       /* p <- u + b(q + b p) */
    ierr = KSP_PCApplyBAorAB(ksp,P,V,Q);CHKERRQ(ierr); /* v <- K p  */

    rhoold = rho;
    dpold  = dp;

    i++;
  } while (i<ksp->max_it);
  if (i >= ksp->max_it) ksp->reason = KSP_DIVERGED_ITS;

  ierr = KSPUnwindPreconditioner(ksp,X,T);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*MC
     KSPTFQMR - A transpose free QMR (quasi minimal residual),

   Options Database Keys:
.   see KSPSolve()

   Level: beginner

   Notes:
    Supports left and right preconditioning, but not symmetric

          The "residual norm" computed in this algorithm is actually just an upper bound on the actual residual norm.
          That is for left preconditioning it is a bound on the preconditioned residual and for right preconditioning
          it is a bound on the true residual.

   References:
.   1. -  Freund, 1993

.seealso: KSPCreate(), KSPSetType(), KSPType (for list of available types), KSP, KSPTCQMR
M*/
PETSC_EXTERN PetscErrorCode KSPCreate_TFQMR(KSP ksp)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = KSPSetSupportedNorm(ksp,KSP_NORM_PRECONDITIONED,PC_LEFT,3);CHKERRQ(ierr);
  ierr = KSPSetSupportedNorm(ksp,KSP_NORM_UNPRECONDITIONED,PC_RIGHT,2);CHKERRQ(ierr);
  ierr = KSPSetSupportedNorm(ksp,KSP_NORM_NONE,PC_LEFT,1);CHKERRQ(ierr);
  ierr = KSPSetSupportedNorm(ksp,KSP_NORM_NONE,PC_RIGHT,1);CHKERRQ(ierr);

  ksp->data                = (void*)0;
  ksp->ops->setup          = KSPSetUp_TFQMR;
  ksp->ops->solve          = KSPSolve_TFQMR;
  ksp->ops->destroy        = KSPDestroyDefault;
  ksp->ops->buildsolution  = KSPBuildSolutionDefault;
  ksp->ops->buildresidual  = KSPBuildResidualDefault;
  ksp->ops->setfromoptions = NULL;
  ksp->ops->view           = NULL;
  PetscFunctionReturn(0);
}
