
/*
    This file implements flexible BiCGStab (FBiCGStab).
    Only allow right preconditioning.
*/
#include <../src/ksp/ksp/impls/bcgs/bcgsimpl.h>       /*I  "petscksp.h"  I*/

static PetscErrorCode KSPSetUp_FBCGS(KSP ksp)
{
  PetscFunctionBegin;
  CHKERRQ(KSPSetWorkVecs(ksp,8));
  PetscFunctionReturn(0);
}

/* Only need a few hacks from KSPSolve_BCGS */

static PetscErrorCode  KSPSolve_FBCGS(KSP ksp)
{
  PetscInt       i;
  PetscScalar    rho,rhoold,alpha,beta,omega,omegaold,d1;
  Vec            X,B,V,P,R,RP,T,S,P2,S2;
  PetscReal      dp    = 0.0,d2;
  KSP_BCGS       *bcgs = (KSP_BCGS*)ksp->data;
  PC             pc;
  Mat            mat;

  PetscFunctionBegin;
  X  = ksp->vec_sol;
  B  = ksp->vec_rhs;
  R  = ksp->work[0];
  RP = ksp->work[1];
  V  = ksp->work[2];
  T  = ksp->work[3];
  S  = ksp->work[4];
  P  = ksp->work[5];
  S2 = ksp->work[6];
  P2 = ksp->work[7];

  /* Only supports right preconditioning */
  PetscCheckFalse(ksp->pc_side != PC_RIGHT,PetscObjectComm((PetscObject)ksp),PETSC_ERR_SUP,"KSP fbcgs does not support %s",PCSides[ksp->pc_side]);
  if (!ksp->guess_zero) {
    if (!bcgs->guess) {
      CHKERRQ(VecDuplicate(X,&bcgs->guess));
    }
    CHKERRQ(VecCopy(X,bcgs->guess));
  } else {
    CHKERRQ(VecSet(X,0.0));
  }

  /* Compute initial residual */
  CHKERRQ(KSPGetPC(ksp,&pc));
  CHKERRQ(PCSetUp(pc));
  CHKERRQ(PCGetOperators(pc,&mat,NULL));
  if (!ksp->guess_zero) {
    CHKERRQ(KSP_MatMult(ksp,mat,X,S2));
    CHKERRQ(VecCopy(B,R));
    CHKERRQ(VecAXPY(R,-1.0,S2));
  } else {
    CHKERRQ(VecCopy(B,R));
  }

  /* Test for nothing to do */
  if (ksp->normtype != KSP_NORM_NONE) {
    CHKERRQ(VecNorm(R,NORM_2,&dp));
  }
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

  rhoold   = 1.0;
  alpha    = 1.0;
  omegaold = 1.0;
  CHKERRQ(VecSet(P,0.0));
  CHKERRQ(VecSet(V,0.0));

  i=0;
  do {
    CHKERRQ(VecDot(R,RP,&rho)); /* rho <- (r,rp) */
    beta = (rho/rhoold) * (alpha/omegaold);
    CHKERRQ(VecAXPBYPCZ(P,1.0,-omegaold*beta,beta,R,V)); /* p <- r - omega * beta* v + beta * p */

    CHKERRQ(KSP_PCApply(ksp,P,P2)); /* p2 <- K p */
    CHKERRQ(KSP_MatMult(ksp,mat,P2,V)); /* v <- A p2 */

    CHKERRQ(VecDot(V,RP,&d1));
    if (d1 == 0.0) {
      PetscCheck(!ksp->errorifnotconverged,PetscObjectComm((PetscObject)ksp),PETSC_ERR_NOT_CONVERGED,"KSPSolve breakdown due to zero inner product");
      else ksp->reason = KSP_DIVERGED_BREAKDOWN;
      CHKERRQ(PetscInfo(ksp,"Breakdown due to zero inner product\n"));
      break;
    }
    alpha = rho / d1; /* alpha <- rho / (v,rp) */
    CHKERRQ(VecWAXPY(S,-alpha,V,R)); /* s <- r - alpha v */

    CHKERRQ(KSP_PCApply(ksp,S,S2)); /* s2 <- K s */
    CHKERRQ(KSP_MatMult(ksp,mat,S2,T)); /* t <- A s2 */

    CHKERRQ(VecDotNorm2(S,T,&d1,&d2));
    if (d2 == 0.0) {
      /* t is 0. if s is 0, then alpha v == r, and hence alpha p may be our solution. Give it a try? */
      CHKERRQ(VecDot(S,S,&d1));
      if (d1 != 0.0) {
        PetscCheck(!ksp->errorifnotconverged,PetscObjectComm((PetscObject)ksp),PETSC_ERR_NOT_CONVERGED,"KSPSolve has failed due to singular preconditioned operator");
        else ksp->reason = KSP_DIVERGED_BREAKDOWN;
        CHKERRQ(PetscInfo(ksp,"Failed due to singular preconditioned operator\n"));
        break;
      }
      CHKERRQ(VecAXPY(X,alpha,P2));   /* x <- x + alpha p2 */
      CHKERRQ(PetscObjectSAWsTakeAccess((PetscObject)ksp));
      ksp->its++;
      ksp->rnorm  = 0.0;
      ksp->reason = KSP_CONVERGED_RTOL;
      CHKERRQ(PetscObjectSAWsGrantAccess((PetscObject)ksp));
      CHKERRQ(KSPLogResidualHistory(ksp,dp));
      CHKERRQ(KSPMonitor(ksp,i+1,0.0));
      break;
    }
    omega = d1 / d2; /* omega <- (t's) / (t't) */
    CHKERRQ(VecAXPBYPCZ(X,alpha,omega,1.0,P2,S2)); /* x <- alpha * p2 + omega * s2 + x */

    CHKERRQ(VecWAXPY(R,-omega,T,S));  /* r <- s - omega t */
    if (ksp->normtype != KSP_NORM_NONE && ksp->chknorm < i+2) {
      CHKERRQ(VecNorm(R,NORM_2,&dp));
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
     PetscCheck(!ksp->errorifnotconverged,PetscObjectComm((PetscObject)ksp),PETSC_ERR_NOT_CONVERGED,"KSPSolve breakdown due to zero rho inner product");
      else ksp->reason = KSP_DIVERGED_BREAKDOWN;
      CHKERRQ(PetscInfo(ksp,"Breakdown due to zero rho inner product\n"));
      break;
    }
    i++;
  } while (i<ksp->max_it);

  if (i >= ksp->max_it) ksp->reason = KSP_DIVERGED_ITS;
  PetscFunctionReturn(0);
}

/*MC
     KSPFBCGS - Implements flexible BiCGStab method.

   Options Database Keys:
    see KSPSolve()

   Level: beginner

   Notes:
    Only allow right preconditioning

.seealso:  KSPCreate(), KSPSetType(), KSPType (for list of available types), KSP, KSPBICG, KSPFBCGSL, KSPSetPCSide()
M*/
PETSC_EXTERN PetscErrorCode KSPCreate_FBCGS(KSP ksp)
{
  KSP_BCGS       *bcgs;

  PetscFunctionBegin;
  CHKERRQ(PetscNewLog(ksp,&bcgs));

  ksp->data                = bcgs;
  ksp->ops->setup          = KSPSetUp_FBCGS;
  ksp->ops->solve          = KSPSolve_FBCGS;
  ksp->ops->destroy        = KSPDestroy_BCGS;
  ksp->ops->reset          = KSPReset_BCGS;
  ksp->ops->buildresidual  = KSPBuildResidualDefault;
  ksp->ops->setfromoptions = KSPSetFromOptions_BCGS;
  ksp->pc_side             = PC_RIGHT;  /* set default PC side */

  CHKERRQ(KSPSetSupportedNorm(ksp,KSP_NORM_PRECONDITIONED,PC_LEFT,3));
  CHKERRQ(KSPSetSupportedNorm(ksp,KSP_NORM_UNPRECONDITIONED,PC_RIGHT,2));
  CHKERRQ(KSPSetSupportedNorm(ksp,KSP_NORM_NONE,PC_LEFT,1));
  CHKERRQ(KSPSetSupportedNorm(ksp,KSP_NORM_NONE,PC_RIGHT,1));
  PetscFunctionReturn(0);
}
