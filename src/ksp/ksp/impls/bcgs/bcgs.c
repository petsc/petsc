
#include <../src/ksp/ksp/impls/bcgs/bcgsimpl.h>       /*I  "petscksp.h"  I*/

PetscErrorCode KSPSetFromOptions_BCGS(PetscOptionItems *PetscOptionsObject,KSP ksp)
{
  PetscFunctionBegin;
  CHKERRQ(PetscOptionsHead(PetscOptionsObject,"KSP BCGS Options"));
  CHKERRQ(PetscOptionsTail());
  PetscFunctionReturn(0);
}

PetscErrorCode KSPSetUp_BCGS(KSP ksp)
{
  PetscFunctionBegin;
  CHKERRQ(KSPSetWorkVecs(ksp,6));
  PetscFunctionReturn(0);
}

PetscErrorCode KSPSolve_BCGS(KSP ksp)
{
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
  CHKERRQ(KSPInitialResidual(ksp,X,V,T,R,B));

  /* with right preconditioning need to save initial guess to add to final solution */
  if (ksp->pc_side == PC_RIGHT && !ksp->guess_zero) {
    if (!bcgs->guess) {
      CHKERRQ(VecDuplicate(X,&bcgs->guess));
    }
    CHKERRQ(VecCopy(X,bcgs->guess));
    CHKERRQ(VecSet(X,0.0));
  }

  /* Test for nothing to do */
  if (ksp->normtype != KSP_NORM_NONE) {
    CHKERRQ(VecNorm(R,NORM_2,&dp));
    KSPCheckNorm(ksp,dp);
  }
  CHKERRQ(PetscObjectSAWsTakeAccess((PetscObject)ksp));
  ksp->its   = 0;
  ksp->rnorm = dp;
  CHKERRQ(PetscObjectSAWsGrantAccess((PetscObject)ksp));
  CHKERRQ(KSPLogResidualHistory(ksp,dp));
  CHKERRQ(KSPMonitor(ksp,0,dp));
  CHKERRQ((*ksp->converged)(ksp,0,dp,&ksp->reason,ksp->cnvP));
  if (ksp->reason) {
    if (bcgs->guess) {
      CHKERRQ(VecAXPY(X,1.0,bcgs->guess));
    }
    PetscFunctionReturn(0);
  }

  /* Make the initial Rp == R */
  CHKERRQ(VecCopy(R,RP));

  rhoold   = 1.0;
  alpha    = 1.0;
  omegaold = 1.0;
  CHKERRQ(VecSet(P,0.0));
  CHKERRQ(VecSet(V,0.0));

  i = 0;
  do {
    CHKERRQ(VecDot(R,RP,&rho));       /*   rho <- (r,rp)      */
    beta = (rho/rhoold) * (alpha/omegaold);
    CHKERRQ(VecAXPBYPCZ(P,1.0,-omegaold*beta,beta,R,V));  /* p <- r - omega * beta* v + beta * p */
    CHKERRQ(KSP_PCApplyBAorAB(ksp,P,V,T));  /*   v <- K p           */
    CHKERRQ(VecDot(V,RP,&d1));
    KSPCheckDot(ksp,d1);
    if (d1 == 0.0) {
      PetscCheckFalse(ksp->errorifnotconverged,PetscObjectComm((PetscObject)ksp),PETSC_ERR_NOT_CONVERGED,"KSPSolve breakdown due to zero inner product");
      else ksp->reason = KSP_DIVERGED_BREAKDOWN;
      CHKERRQ(PetscInfo(ksp,"Breakdown due to zero inner product\n"));
      break;
    }
    alpha = rho / d1;                 /*   a <- rho / (v,rp)  */
    CHKERRQ(VecWAXPY(S,-alpha,V,R));     /*   s <- r - a v       */
    CHKERRQ(KSP_PCApplyBAorAB(ksp,S,T,R)); /*   t <- K s    */
    CHKERRQ(VecDotNorm2(S,T,&d1,&d2));
    if (d2 == 0.0) {
      /* t is 0.  if s is 0, then alpha v == r, and hence alpha p
         may be our solution.  Give it a try? */
      CHKERRQ(VecDot(S,S,&d1));
      if (d1 != 0.0) {
        PetscCheckFalse(ksp->errorifnotconverged,PetscObjectComm((PetscObject)ksp),PETSC_ERR_NOT_CONVERGED,"KSPSolve has failed due to singular preconditioned operator");
        else ksp->reason = KSP_DIVERGED_BREAKDOWN;
        CHKERRQ(PetscInfo(ksp,"Failed due to singular preconditioned operator\n"));
        break;
      }
      CHKERRQ(VecAXPY(X,alpha,P));   /*   x <- x + a p       */
      CHKERRQ(PetscObjectSAWsTakeAccess((PetscObject)ksp));
      ksp->its++;
      ksp->rnorm  = 0.0;
      ksp->reason = KSP_CONVERGED_RTOL;
      CHKERRQ(PetscObjectSAWsGrantAccess((PetscObject)ksp));
      CHKERRQ(KSPLogResidualHistory(ksp,dp));
      CHKERRQ(KSPMonitor(ksp,i+1,0.0));
      break;
    }
    omega = d1 / d2;                               /*   w <- (t's) / (t't) */
    CHKERRQ(VecAXPBYPCZ(X,alpha,omega,1.0,P,S)); /* x <- alpha * p + omega * s + x */
    CHKERRQ(VecWAXPY(R,-omega,T,S));     /*   r <- s - w t       */
    if (ksp->normtype != KSP_NORM_NONE && ksp->chknorm < i+2) {
      CHKERRQ(VecNorm(R,NORM_2,&dp));
      KSPCheckNorm(ksp,dp);
    }

    rhoold   = rho;
    omegaold = omega;

    CHKERRQ(PetscObjectSAWsTakeAccess((PetscObject)ksp));
    ksp->its++;
    ksp->rnorm = dp;
    CHKERRQ(PetscObjectSAWsGrantAccess((PetscObject)ksp));
    CHKERRQ(KSPLogResidualHistory(ksp,dp));
    CHKERRQ(KSPMonitor(ksp,i+1,dp));
    CHKERRQ((*ksp->converged)(ksp,i+1,dp,&ksp->reason,ksp->cnvP));
    if (ksp->reason) break;
    if (rho == 0.0) {
      PetscCheckFalse(ksp->errorifnotconverged,PetscObjectComm((PetscObject)ksp),PETSC_ERR_NOT_CONVERGED,"KSPSolve breakdown due to zero inner product");
      else ksp->reason = KSP_DIVERGED_BREAKDOWN;
      CHKERRQ(PetscInfo(ksp,"Breakdown due to zero rho inner product\n"));
      break;
    }
    i++;
  } while (i<ksp->max_it);

  if (i >= ksp->max_it) ksp->reason = KSP_DIVERGED_ITS;

  CHKERRQ(KSPUnwindPreconditioner(ksp,X,T));
  if (bcgs->guess) {
    CHKERRQ(VecAXPY(X,1.0,bcgs->guess));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode KSPBuildSolution_BCGS(KSP ksp,Vec v,Vec *V)
{
  KSP_BCGS       *bcgs = (KSP_BCGS*)ksp->data;

  PetscFunctionBegin;
  if (ksp->pc_side == PC_RIGHT) {
    if (v) {
      CHKERRQ(KSP_PCApply(ksp,ksp->vec_sol,v));
      if (bcgs->guess) {
        CHKERRQ(VecAXPY(v,1.0,bcgs->guess));
      }
      *V = v;
    } else SETERRQ(PetscObjectComm((PetscObject)ksp),PETSC_ERR_SUP,"Not working with right preconditioner");
  } else {
    if (v) {
      CHKERRQ(VecCopy(ksp->vec_sol,v)); *V = v;
    } else *V = ksp->vec_sol;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode KSPReset_BCGS(KSP ksp)
{
  KSP_BCGS       *cg = (KSP_BCGS*)ksp->data;

  PetscFunctionBegin;
  CHKERRQ(VecDestroy(&cg->guess));
  PetscFunctionReturn(0);
}

PetscErrorCode KSPDestroy_BCGS(KSP ksp)
{
  PetscFunctionBegin;
  CHKERRQ(KSPReset_BCGS(ksp));
  CHKERRQ(KSPDestroyDefault(ksp));
  PetscFunctionReturn(0);
}

/*MC
     KSPBCGS - Implements the BiCGStab (Stabilized version of BiConjugate Gradient) method.

   Options Database Keys:
    see KSPSolve()

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
  KSP_BCGS       *bcgs;

  PetscFunctionBegin;
  CHKERRQ(PetscNewLog(ksp,&bcgs));

  ksp->data                = bcgs;
  ksp->ops->setup          = KSPSetUp_BCGS;
  ksp->ops->solve          = KSPSolve_BCGS;
  ksp->ops->destroy        = KSPDestroy_BCGS;
  ksp->ops->reset          = KSPReset_BCGS;
  ksp->ops->buildsolution  = KSPBuildSolution_BCGS;
  ksp->ops->buildresidual  = KSPBuildResidualDefault;
  ksp->ops->setfromoptions = KSPSetFromOptions_BCGS;

  CHKERRQ(KSPSetSupportedNorm(ksp,KSP_NORM_PRECONDITIONED,PC_LEFT,3));
  CHKERRQ(KSPSetSupportedNorm(ksp,KSP_NORM_UNPRECONDITIONED,PC_RIGHT,2));
  CHKERRQ(KSPSetSupportedNorm(ksp,KSP_NORM_NONE,PC_LEFT,1));
  CHKERRQ(KSPSetSupportedNorm(ksp,KSP_NORM_NONE,PC_RIGHT,1));
  PetscFunctionReturn(0);
}
