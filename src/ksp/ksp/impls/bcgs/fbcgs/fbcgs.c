
/*
    This file implements flexible BiCGStab contributed by Jie Chen.
    Only right preconditioning is supported.
*/
#include <../src/ksp/ksp/impls/bcgs/bcgsimpl.h>       /*I  "petscksp.h"  I*/

/* copied from KSPBuildSolution_GCR() */
#undef __FUNCT__
#define __FUNCT__ "KSPBuildSolution_FBCGS"
PetscErrorCode  KSPBuildSolution_FBCGS(KSP ksp, Vec v, Vec *V)
{
  PetscErrorCode ierr;
  Vec            x;

  PetscFunctionBegin;
  x = ksp->vec_sol;
  if (v) {
    ierr = VecCopy(x, v);CHKERRQ(ierr);
    if (V) *V = v;
  } else if (V) {
    *V = ksp->vec_sol;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "KSPSetUp_FBCGS"
static PetscErrorCode KSPSetUp_FBCGS(KSP ksp)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = KSPDefaultGetWork(ksp,8);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* Only need a few hacks from KSPSolve_BCGS */
#include <petsc-private/pcimpl.h>            /*I "petscksp.h" I*/
#undef __FUNCT__
#define __FUNCT__ "KSPSolve_FBCGS"
static PetscErrorCode  KSPSolve_FBCGS(KSP ksp)
{
  PetscErrorCode ierr;
  PetscInt       i;
  PetscScalar    rho,rhoold,alpha,beta,omega,omegaold,d1,d2;
  Vec            X,B,V,P,R,RP,T,S,P2,S2;
  PetscReal      dp = 0.0;
  KSP_BCGS       *bcgs = (KSP_BCGS*)ksp->data;
  PC             pc;

  PetscFunctionBegin;
  X       = ksp->vec_sol;
  B       = ksp->vec_rhs;
  R       = ksp->work[0];
  RP      = ksp->work[1];
  V       = ksp->work[2];
  T       = ksp->work[3];
  S       = ksp->work[4];
  P       = ksp->work[5];
  S2      = ksp->work[6];
  P2      = ksp->work[7];

  /* Only supports right preconditioning */
  if (ksp->pc_side != PC_RIGHT)
    SETERRQ1(((PetscObject)ksp)->comm,PETSC_ERR_SUP,"KSP fbcgs does not support %s",PCSides[ksp->pc_side]);
  if (!ksp->guess_zero) {
    if (!bcgs->guess) {
      ierr = VecDuplicate(X,&bcgs->guess);CHKERRQ(ierr);
    }
    ierr = VecCopy(X,bcgs->guess);CHKERRQ(ierr);
  } else {
    ierr = VecSet(X,0.0);CHKERRQ(ierr);
  }

  /* Compute initial residual */
  ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
  ierr = PCSetUp(pc);CHKERRQ(ierr);
  if (!ksp->guess_zero) {
    ierr = MatMult(pc->mat,X,S2);CHKERRQ(ierr);
    ierr = VecCopy(B,R);CHKERRQ(ierr);
    ierr = VecAXPY(R,-1.0,S2);CHKERRQ(ierr);
  } else {
    ierr = VecCopy(B,R);CHKERRQ(ierr);
  }

  /* Test for nothing to do */
  if (ksp->normtype != KSP_NORM_NONE) {
    ierr = VecNorm(R,NORM_2,&dp);CHKERRQ(ierr);
  }
  ierr = PetscObjectTakeAccess(ksp);CHKERRQ(ierr);
  ksp->its   = 0;
  ksp->rnorm = dp;
  ierr = PetscObjectGrantAccess(ksp);CHKERRQ(ierr);
  KSPLogResidualHistory(ksp,dp);
  ierr = KSPMonitor(ksp,0,dp);CHKERRQ(ierr);
  ierr = (*ksp->converged)(ksp,0,dp,&ksp->reason,ksp->cnvP);CHKERRQ(ierr);
  if (ksp->reason) PetscFunctionReturn(0);

  /* Make the initial Rp == R */
  ierr = VecCopy(R,RP);CHKERRQ(ierr);

  rhoold   = 1.0;
  alpha    = 1.0;
  omegaold = 1.0;
  ierr = VecSet(P,0.0);CHKERRQ(ierr);
  ierr = VecSet(V,0.0);CHKERRQ(ierr);

  i=0;
  do {
    ierr = VecDot(R,RP,&rho);CHKERRQ(ierr); /* rho <- (r,rp) */
    beta = (rho/rhoold) * (alpha/omegaold);
    ierr = VecAXPBYPCZ(P,1.0,-omegaold*beta,beta,R,V);CHKERRQ(ierr); /* p <- r - omega * beta* v + beta * p */

    ierr = PCApply(pc,P,P2);CHKERRQ(ierr); /* p2 <- K p */
    ierr = MatMult(pc->mat,P2,V);CHKERRQ(ierr); /* v <- A p2 */

    ierr = VecDot(V,RP,&d1);CHKERRQ(ierr);
    if (d1 == 0.0) SETERRQ(((PetscObject)ksp)->comm,PETSC_ERR_PLIB,"Divide by zero");
    alpha = rho / d1; /* alpha <- rho / (v,rp) */
    ierr = VecWAXPY(S,-alpha,V,R);CHKERRQ(ierr);  /* s <- r - alpha v */

    ierr = PCApply(pc,S,S2);CHKERRQ(ierr); /* s2 <- K s */
    ierr = MatMult(pc->mat,S2,T);CHKERRQ(ierr); /* t <- A s2 */

    ierr = VecDotNorm2(S,T,&d1,&d2);CHKERRQ(ierr);
    if (d2 == 0.0) {
      /* t is 0. if s is 0, then alpha v == r, and hence alpha p may be our solution. Give it a try? */
      ierr = VecDot(S,S,&d1);CHKERRQ(ierr);
      if (d1 != 0.0) {
        ksp->reason = KSP_DIVERGED_BREAKDOWN;
        break;
      }
      ierr = VecAXPY(X,alpha,P2);CHKERRQ(ierr);   /* x <- x + alpha p2 */
      ierr = PetscObjectTakeAccess(ksp);CHKERRQ(ierr);
      ksp->its++;
      ksp->rnorm  = 0.0;
      ksp->reason = KSP_CONVERGED_RTOL;
      ierr = PetscObjectGrantAccess(ksp);CHKERRQ(ierr);
      KSPLogResidualHistory(ksp,dp);
      ierr = KSPMonitor(ksp,i+1,0.0);CHKERRQ(ierr);
      break;
    }
    omega = d1 / d2; /* omega <- (t's) / (t't) */
    ierr = VecAXPBYPCZ(X,alpha,omega,1.0,P2,S2);CHKERRQ(ierr); /* x <- alpha * p2 + omega * s2 + x */

    ierr  = VecWAXPY(R,-omega,T,S);CHKERRQ(ierr); /* r <- s - omega t */
    if (ksp->normtype != KSP_NORM_NONE && ksp->chknorm < i+2) {
      ierr = VecNorm(R,NORM_2,&dp);CHKERRQ(ierr);
    }

    rhoold   = rho;
    omegaold = omega;

    ierr = PetscObjectTakeAccess(ksp);CHKERRQ(ierr);
    ksp->its++;
    ksp->rnorm = dp;
    ierr = PetscObjectGrantAccess(ksp);CHKERRQ(ierr);
    KSPLogResidualHistory(ksp,dp);
    ierr = KSPMonitor(ksp,i+1,dp);CHKERRQ(ierr);
    ierr = (*ksp->converged)(ksp,i+1,dp,&ksp->reason,ksp->cnvP);CHKERRQ(ierr);
    if (ksp->reason) break;
    if (rho == 0.0) {
      ksp->reason = KSP_DIVERGED_BREAKDOWN;
      break;
    }
    i++;
  } while (i<ksp->max_it);

  if (i >= ksp->max_it) {
    ksp->reason = KSP_DIVERGED_ITS;
  }
  PetscFunctionReturn(0);
}

/*MC
     KSPFBCGS - Implements flexible BiCGStab (Stabilized version of BiConjugate Gradient Squared) method.

   Options Database Keys:
.   see KSPSolve()

   Level: beginner

   Notes: Only supports right preconditioning

.seealso:  KSPCreate(), KSPSetType(), KSPType (for list of available types), KSP, KSPBICG, KSPFBCGSL, KSPSetPCSide()
M*/
EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "KSPCreate_FBCGS"
PetscErrorCode  KSPCreate_FBCGS(KSP ksp)
{
  PetscErrorCode ierr;
  KSP_BCGS       *bcgs;

  PetscFunctionBegin;
  ierr = PetscNewLog(ksp,KSP_BCGS,&bcgs);CHKERRQ(ierr);
  ksp->data                 = bcgs;
  ksp->ops->setup           = KSPSetUp_FBCGS;
  ksp->ops->solve           = KSPSolve_FBCGS;
  ksp->ops->destroy         = KSPDestroy_BCGS;
  ksp->ops->reset           = KSPReset_BCGS;
  ksp->ops->buildsolution   = KSPBuildSolution_FBCGS;
  ksp->ops->buildresidual   = KSPDefaultBuildResidual;
  ksp->ops->setfromoptions  = KSPSetFromOptions_BCGS;
  ksp->ops->view            = KSPView_BCGS;
  ksp->pc_side              = PC_RIGHT; /* set default PC side */

  ierr = KSPSetSupportedNorm(ksp,KSP_NORM_PRECONDITIONED,PC_LEFT,2);CHKERRQ(ierr);
  ierr = KSPSetSupportedNorm(ksp,KSP_NORM_UNPRECONDITIONED,PC_RIGHT,1);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END
