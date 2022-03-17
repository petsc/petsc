
#include <../src/ksp/ksp/impls/bcgs/bcgsimpl.h>       /*I  "petscksp.h"  I*/

PetscErrorCode KSPSetFromOptions_BCGS(PetscOptionItems *PetscOptionsObject,KSP ksp)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscOptionsHead(PetscOptionsObject,"KSP BCGS Options");CHKERRQ(ierr);
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode KSPSetUp_BCGS(KSP ksp)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = KSPSetWorkVecs(ksp,6);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode KSPSolve_BCGS(KSP ksp)
{
  PetscErrorCode ierr;
  PetscInt       i;
  PetscScalar    rho,rhoold,alpha,beta,omega,omegaold,d1;
  Vec            X,B,V,P,R,RP,T,S;
  PetscReal      dp    = 0.0,d2;
  KSP_BCGS       *bcgs = (KSP_BCGS*)ksp->data;

  PetscFunctionBegin;
  X  = ksp->vec_sol;
  B  = ksp->vec_rhs;
  R  = ksp->work[0];
  RP = ksp->work[1];
  V  = ksp->work[2];
  T  = ksp->work[3];
  S  = ksp->work[4];
  P  = ksp->work[5];

  /* Compute initial preconditioned residual */
  ierr = KSPInitialResidual(ksp,X,V,T,R,B);CHKERRQ(ierr);

  /* with right preconditioning need to save initial guess to add to final solution */
  if (ksp->pc_side == PC_RIGHT && !ksp->guess_zero) {
    if (!bcgs->guess) {
      ierr = VecDuplicate(X,&bcgs->guess);CHKERRQ(ierr);
    }
    ierr = VecCopy(X,bcgs->guess);CHKERRQ(ierr);
    ierr = VecSet(X,0.0);CHKERRQ(ierr);
  }

  /* Test for nothing to do */
  if (ksp->normtype != KSP_NORM_NONE) {
    ierr = VecNorm(R,NORM_2,&dp);CHKERRQ(ierr);
    KSPCheckNorm(ksp,dp);
  }
  ierr       = PetscObjectSAWsTakeAccess((PetscObject)ksp);CHKERRQ(ierr);
  ksp->its   = 0;
  ksp->rnorm = dp;
  ierr       = PetscObjectSAWsGrantAccess((PetscObject)ksp);CHKERRQ(ierr);
  ierr = KSPLogResidualHistory(ksp,dp);CHKERRQ(ierr);
  ierr = KSPMonitor(ksp,0,dp);CHKERRQ(ierr);
  ierr = (*ksp->converged)(ksp,0,dp,&ksp->reason,ksp->cnvP);CHKERRQ(ierr);
  if (ksp->reason) {
    if (bcgs->guess) {
      ierr = VecAXPY(X,1.0,bcgs->guess);CHKERRQ(ierr);
    }
    PetscFunctionReturn(0);
  }

  /* Make the initial Rp == R */
  ierr = VecCopy(R,RP);CHKERRQ(ierr);

  rhoold   = 1.0;
  alpha    = 1.0;
  omegaold = 1.0;
  ierr     = VecSet(P,0.0);CHKERRQ(ierr);
  ierr     = VecSet(V,0.0);CHKERRQ(ierr);

  i = 0;
  do {
    ierr = VecDot(R,RP,&rho);CHKERRQ(ierr);       /*   rho <- (r,rp)      */
    beta = (rho/rhoold) * (alpha/omegaold);
    ierr = VecAXPBYPCZ(P,1.0,-omegaold*beta,beta,R,V);CHKERRQ(ierr);  /* p <- r - omega * beta* v + beta * p */
    ierr = KSP_PCApplyBAorAB(ksp,P,V,T);CHKERRQ(ierr);  /*   v <- K p           */
    ierr = VecDot(V,RP,&d1);CHKERRQ(ierr);
    KSPCheckDot(ksp,d1);
    if (d1 == 0.0) {
      PetscCheckFalse(ksp->errorifnotconverged,PetscObjectComm((PetscObject)ksp),PETSC_ERR_NOT_CONVERGED,"KSPSolve breakdown due to zero inner product");
      else ksp->reason = KSP_DIVERGED_BREAKDOWN;
      ierr  = PetscInfo(ksp,"Breakdown due to zero inner product\n");CHKERRQ(ierr);
      break;
    }
    alpha = rho / d1;                 /*   a <- rho / (v,rp)  */
    ierr  = VecWAXPY(S,-alpha,V,R);CHKERRQ(ierr);     /*   s <- r - a v       */
    ierr  = KSP_PCApplyBAorAB(ksp,S,T,R);CHKERRQ(ierr); /*   t <- K s    */
    ierr  = VecDotNorm2(S,T,&d1,&d2);CHKERRQ(ierr);
    if (d2 == 0.0) {
      /* t is 0.  if s is 0, then alpha v == r, and hence alpha p
         may be our solution.  Give it a try? */
      ierr = VecDot(S,S,&d1);CHKERRQ(ierr);
      if (d1 != 0.0) {
        PetscCheckFalse(ksp->errorifnotconverged,PetscObjectComm((PetscObject)ksp),PETSC_ERR_NOT_CONVERGED,"KSPSolve has failed due to singular preconditioned operator");
        else ksp->reason = KSP_DIVERGED_BREAKDOWN;
        ierr  = PetscInfo(ksp,"Failed due to singular preconditioned operator\n");CHKERRQ(ierr);
        break;
      }
      ierr = VecAXPY(X,alpha,P);CHKERRQ(ierr);   /*   x <- x + a p       */
      ierr = PetscObjectSAWsTakeAccess((PetscObject)ksp);CHKERRQ(ierr);
      ksp->its++;
      ksp->rnorm  = 0.0;
      ksp->reason = KSP_CONVERGED_RTOL;
      ierr = PetscObjectSAWsGrantAccess((PetscObject)ksp);CHKERRQ(ierr);
      ierr = KSPLogResidualHistory(ksp,dp);CHKERRQ(ierr);
      ierr = KSPMonitor(ksp,i+1,0.0);CHKERRQ(ierr);
      break;
    }
    omega = d1 / d2;                               /*   w <- (t's) / (t't) */
    ierr  = VecAXPBYPCZ(X,alpha,omega,1.0,P,S);CHKERRQ(ierr); /* x <- alpha * p + omega * s + x */
    ierr  = VecWAXPY(R,-omega,T,S);CHKERRQ(ierr);     /*   r <- s - w t       */
    if (ksp->normtype != KSP_NORM_NONE && ksp->chknorm < i+2) {
      ierr = VecNorm(R,NORM_2,&dp);CHKERRQ(ierr);
      KSPCheckNorm(ksp,dp);
    }

    rhoold   = rho;
    omegaold = omega;

    ierr = PetscObjectSAWsTakeAccess((PetscObject)ksp);CHKERRQ(ierr);
    ksp->its++;
    ksp->rnorm = dp;
    ierr = PetscObjectSAWsGrantAccess((PetscObject)ksp);CHKERRQ(ierr);
    ierr = KSPLogResidualHistory(ksp,dp);CHKERRQ(ierr);
    ierr = KSPMonitor(ksp,i+1,dp);CHKERRQ(ierr);
    ierr = (*ksp->converged)(ksp,i+1,dp,&ksp->reason,ksp->cnvP);CHKERRQ(ierr);
    if (ksp->reason) break;
    if (rho == 0.0) {
      PetscCheckFalse(ksp->errorifnotconverged,PetscObjectComm((PetscObject)ksp),PETSC_ERR_NOT_CONVERGED,"KSPSolve breakdown due to zero inner product");
      else ksp->reason = KSP_DIVERGED_BREAKDOWN;
      ierr  = PetscInfo(ksp,"Breakdown due to zero rho inner product\n");CHKERRQ(ierr);
      break;
    }
    i++;
  } while (i<ksp->max_it);

  if (i >= ksp->max_it) ksp->reason = KSP_DIVERGED_ITS;

  ierr = KSPUnwindPreconditioner(ksp,X,T);CHKERRQ(ierr);
  if (bcgs->guess) {
    ierr = VecAXPY(X,1.0,bcgs->guess);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode KSPBuildSolution_BCGS(KSP ksp,Vec v,Vec *V)
{
  PetscErrorCode ierr;
  KSP_BCGS       *bcgs = (KSP_BCGS*)ksp->data;

  PetscFunctionBegin;
  if (ksp->pc_side == PC_RIGHT) {
    if (v) {
      ierr = KSP_PCApply(ksp,ksp->vec_sol,v);CHKERRQ(ierr);
      if (bcgs->guess) {
        ierr = VecAXPY(v,1.0,bcgs->guess);CHKERRQ(ierr);
      }
      *V = v;
    } else SETERRQ(PetscObjectComm((PetscObject)ksp),PETSC_ERR_SUP,"Not working with right preconditioner");
  } else {
    if (v) {
      ierr = VecCopy(ksp->vec_sol,v);CHKERRQ(ierr); *V = v;
    } else *V = ksp->vec_sol;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode KSPReset_BCGS(KSP ksp)
{
  KSP_BCGS       *cg = (KSP_BCGS*)ksp->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecDestroy(&cg->guess);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode KSPDestroy_BCGS(KSP ksp)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = KSPReset_BCGS(ksp);CHKERRQ(ierr);
  ierr = KSPDestroyDefault(ksp);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*MC
     KSPBCGS - Implements the BiCGStab (Stabilized version of BiConjugate Gradient) method.

   Options Database Keys:
.   see KSPSolve()

   Level: beginner

   Notes:
    See KSPBCGSL for additional stabilization
          Supports left and right preconditioning but not symmetric

   References:
.  * - van der Vorst, SIAM J. Sci. Stat. Comput., 1992.

.seealso:  KSPCreate(), KSPSetType(), KSPType (for list of available types), KSP, KSPBICG, KSPBCGSL, KSPFBICG, KSPQMRCGS, KSPSetPCSide()
M*/
PETSC_EXTERN PetscErrorCode KSPCreate_BCGS(KSP ksp)
{
  PetscErrorCode ierr;
  KSP_BCGS       *bcgs;

  PetscFunctionBegin;
  ierr = PetscNewLog(ksp,&bcgs);CHKERRQ(ierr);

  ksp->data                = bcgs;
  ksp->ops->setup          = KSPSetUp_BCGS;
  ksp->ops->solve          = KSPSolve_BCGS;
  ksp->ops->destroy        = KSPDestroy_BCGS;
  ksp->ops->reset          = KSPReset_BCGS;
  ksp->ops->buildsolution  = KSPBuildSolution_BCGS;
  ksp->ops->buildresidual  = KSPBuildResidualDefault;
  ksp->ops->setfromoptions = KSPSetFromOptions_BCGS;

  ierr = KSPSetSupportedNorm(ksp,KSP_NORM_PRECONDITIONED,PC_LEFT,3);CHKERRQ(ierr);
  ierr = KSPSetSupportedNorm(ksp,KSP_NORM_UNPRECONDITIONED,PC_RIGHT,2);CHKERRQ(ierr);
  ierr = KSPSetSupportedNorm(ksp,KSP_NORM_NONE,PC_LEFT,1);CHKERRQ(ierr);
  ierr = KSPSetSupportedNorm(ksp,KSP_NORM_NONE,PC_RIGHT,1);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
