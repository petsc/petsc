
/*
    This file implements pipelined BiCGStab (pipe-BiCGStab).
    Only allow right preconditioning.
*/
#include <../src/ksp/ksp/impls/bcgs/bcgsimpl.h>       /*I  "petscksp.h"  I*/

static PetscErrorCode KSPSetUp_PIPEBCGS(KSP ksp)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = KSPSetWorkVecs(ksp,15);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* Only need a few hacks from KSPSolve_BCGS */
#include <petsc/private/pcimpl.h>            /*I "petscksp.h" I*/
static PetscErrorCode  KSPSolve_PIPEBCGS(KSP ksp)
{
  PetscErrorCode ierr;
  PetscInt       i;
  PetscScalar    rho,rhoold,alpha,beta,omega=0.0,d1,d2,d3;
  Vec            X,B,S,R,RP,Y,Q,P2,Q2,R2,S2,W,Z,W2,Z2,T,V;
  PetscReal      dp    = 0.0;
  KSP_BCGS       *bcgs = (KSP_BCGS*)ksp->data;
  PC             pc;

  PetscFunctionBegin;
  X  = ksp->vec_sol;
  B  = ksp->vec_rhs;
  R  = ksp->work[0];
  RP = ksp->work[1];
  S  = ksp->work[2];
  Y  = ksp->work[3];
  Q  = ksp->work[4];
  Q2 = ksp->work[5];
  P2 = ksp->work[6];
  R2 = ksp->work[7];
  S2 = ksp->work[8];
  W  = ksp->work[9];
  Z  = ksp->work[10];
  W2 = ksp->work[11];
  Z2 = ksp->work[12];
  T  = ksp->work[13];
  V  = ksp->work[14];

  /* Only supports right preconditioning */
  if (ksp->pc_side != PC_RIGHT) SETERRQ1(PetscObjectComm((PetscObject)ksp),PETSC_ERR_SUP,"KSP pipebcgs does not support %s",PCSides[ksp->pc_side]);
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
    ierr = KSP_MatMult(ksp,pc->mat,X,Q2);CHKERRQ(ierr);
    ierr = VecCopy(B,R);CHKERRQ(ierr);
    ierr = VecAXPY(R,-1.0,Q2);CHKERRQ(ierr);
  } else {
    ierr = VecCopy(B,R);CHKERRQ(ierr);
  }

  /* Test for nothing to do */
  if (ksp->normtype != KSP_NORM_NONE) {
    ierr = VecNorm(R,NORM_2,&dp);CHKERRQ(ierr);
  } else dp = 0.0;
  ierr       = PetscObjectSAWsTakeAccess((PetscObject)ksp);CHKERRQ(ierr);
  ksp->its   = 0;
  ksp->rnorm = dp;
  ierr = PetscObjectSAWsGrantAccess((PetscObject)ksp);CHKERRQ(ierr);
  ierr = KSPLogResidualHistory(ksp,dp);CHKERRQ(ierr);
  ierr = KSPMonitor(ksp,0,dp);CHKERRQ(ierr);
  ierr = (*ksp->converged)(ksp,0,dp,&ksp->reason,ksp->cnvP);CHKERRQ(ierr);
  if (ksp->reason) PetscFunctionReturn(0);

  /* Initialize */
  ierr = VecCopy(R,RP);CHKERRQ(ierr); /* rp <- r */

  ierr = VecDotBegin(R,RP,&rho);CHKERRQ(ierr); /* rho <- (r,rp) */
  ierr = PetscCommSplitReductionBegin(PetscObjectComm((PetscObject)R));CHKERRQ(ierr);
  ierr = KSP_PCApply(ksp,R,R2);CHKERRQ(ierr); /* r2 <- K r */
  ierr = KSP_MatMult(ksp,pc->mat,R2,W);CHKERRQ(ierr); /* w <- A r2 */
  ierr = VecDotEnd(R,RP,&rho);CHKERRQ(ierr);

  ierr = VecDotBegin(W,RP,&d2);CHKERRQ(ierr); /* d2 <- (w,rp) */
  ierr = PetscCommSplitReductionBegin(PetscObjectComm((PetscObject)W));CHKERRQ(ierr);
  ierr = KSP_PCApply(ksp,W,W2);CHKERRQ(ierr); /* w2 <- K w */
  ierr = KSP_MatMult(ksp,pc->mat,W2,T);CHKERRQ(ierr); /* t <- A w2 */
  ierr = VecDotEnd(W,RP,&d2);CHKERRQ(ierr);

  alpha = rho/d2;
  beta = 0.0;

  /* Main loop */
  i=0;
  do {
    if (i == 0) {
      ierr  = VecCopy(R2,P2);CHKERRQ(ierr); /* p2 <- r2 */
      ierr  = VecCopy(W,S);CHKERRQ(ierr);   /* s  <- w  */
      ierr  = VecCopy(W2,S2);CHKERRQ(ierr); /* s2 <- w2 */
      ierr  = VecCopy(T,Z);CHKERRQ(ierr);   /* z  <- t  */
    } else {
      ierr  = VecAXPBYPCZ(P2,1.0,-beta*omega,beta,R2,S2);CHKERRQ(ierr); /* p2 <- beta * p2 + r2 - beta * omega * s2 */
      ierr  = VecAXPBYPCZ(S,1.0,-beta*omega,beta,W,Z);CHKERRQ(ierr);    /* s  <- beta * s  + w  - beta * omega * z  */
      ierr  = VecAXPBYPCZ(S2,1.0,-beta*omega,beta,W2,Z2);CHKERRQ(ierr); /* s2 <- beta * s2 + w2 - beta * omega * z2 */
      ierr  = VecAXPBYPCZ(Z,1.0,-beta*omega,beta,T,V);CHKERRQ(ierr);    /* z  <- beta * z  + t  - beta * omega * v  */
    }
    ierr  = VecWAXPY(Q,-alpha,S,R);CHKERRQ(ierr);    /* q  <- r  - alpha s  */
    ierr  = VecWAXPY(Q2,-alpha,S2,R2);CHKERRQ(ierr); /* q2 <- r2 - alpha s2 */
    ierr  = VecWAXPY(Y,-alpha,Z,W);CHKERRQ(ierr);    /* y  <- w  - alpha z  */

    ierr = VecDotBegin(Q,Y,&d1);CHKERRQ(ierr); /* d1 <- (q,y) */
    ierr = VecDotBegin(Y,Y,&d2);CHKERRQ(ierr); /* d2 <- (y,y) */

    ierr = PetscCommSplitReductionBegin(PetscObjectComm((PetscObject)Q));CHKERRQ(ierr);
    ierr = KSP_PCApply(ksp,Z,Z2);CHKERRQ(ierr); /* z2 <- K z */
    ierr = KSP_MatMult(ksp,pc->mat,Z2,V);CHKERRQ(ierr); /* v <- A z2 */

    ierr = VecDotEnd(Q,Y,&d1);CHKERRQ(ierr);
    ierr = VecDotEnd(Y,Y,&d2);CHKERRQ(ierr);

    if (d2 == 0.0) {
      /* y is 0. if q is 0, then alpha s == r, and hence alpha p may be our solution. Give it a try? */
      ierr = VecDot(Q,Q,&d1);CHKERRQ(ierr);
      if (d1 != 0.0) {
        ksp->reason = KSP_DIVERGED_BREAKDOWN;
        break;
      }
      ierr = VecAXPY(X,alpha,P2);CHKERRQ(ierr);   /* x <- x + alpha p2 */
      ierr = PetscObjectSAWsTakeAccess((PetscObject)ksp);CHKERRQ(ierr);
      ksp->its++;
      ksp->rnorm  = 0.0;
      ksp->reason = KSP_CONVERGED_RTOL;
      ierr = PetscObjectSAWsGrantAccess((PetscObject)ksp);CHKERRQ(ierr);
      ierr = KSPLogResidualHistory(ksp,dp);CHKERRQ(ierr);
      ierr = KSPMonitor(ksp,i+1,0.0);CHKERRQ(ierr);
      break;
    }
    omega = d1/d2; /* omega <- (y'q) / (y'y) */
    ierr = VecAXPBYPCZ(X,alpha,omega,1.0,P2,Q2);CHKERRQ(ierr); /* x <- alpha * p2 + omega * q2 + x */
    ierr = VecWAXPY(R,-omega,Y,Q);CHKERRQ(ierr);    /* r <- q - omega y */
    ierr = VecWAXPY(R2,-alpha,Z2,W2);CHKERRQ(ierr); /* r2 <- w2 - alpha z2 */
    ierr = VecAYPX(R2,-omega,Q2);CHKERRQ(ierr);     /* r2 <- q2 - omega r2 */
    ierr = VecWAXPY(W,-alpha,V,T);CHKERRQ(ierr);    /* w <- t - alpha v */
    ierr = VecAYPX(W,-omega,Y);CHKERRQ(ierr);       /* w <- y - omega w */
    rhoold = rho;

    if (ksp->normtype != KSP_NORM_NONE && ksp->chknorm < i+2) {
      ierr = VecNormBegin(R,NORM_2,&dp);CHKERRQ(ierr); /* dp <- norm(r) */
    }
    ierr = VecDotBegin(R,RP,&rho);CHKERRQ(ierr); /* rho <- (r,rp) */
    ierr = VecDotBegin(S,RP,&d1);CHKERRQ(ierr);  /* d1 <- (s,rp) */
    ierr = VecDotBegin(W,RP,&d2);CHKERRQ(ierr);  /* d2 <- (w,rp) */
    ierr = VecDotBegin(Z,RP,&d3);CHKERRQ(ierr);  /* d3 <- (z,rp) */

    ierr = PetscCommSplitReductionBegin(PetscObjectComm((PetscObject)R));CHKERRQ(ierr);
    ierr = KSP_PCApply(ksp,W,W2);CHKERRQ(ierr); /* w2 <- K w */
    ierr = KSP_MatMult(ksp,pc->mat,W2,T);CHKERRQ(ierr); /* t <- A w2 */

    if (ksp->normtype != KSP_NORM_NONE && ksp->chknorm < i+2) {
      ierr = VecNormEnd(R,NORM_2,&dp);CHKERRQ(ierr);
    }
    ierr = VecDotEnd(R,RP,&rho);CHKERRQ(ierr);
    ierr = VecDotEnd(S,RP,&d1);CHKERRQ(ierr);
    ierr = VecDotEnd(W,RP,&d2);CHKERRQ(ierr);
    ierr = VecDotEnd(Z,RP,&d3);CHKERRQ(ierr);

    if (d2 + beta * d1 - beta * omega * d3 == 0.0) SETERRQ(PetscObjectComm((PetscObject)ksp),PETSC_ERR_PLIB,"Divide by zero");

    beta = (rho/rhoold) * (alpha/omega);
    alpha = rho/(d2 + beta * d1 - beta * omega * d3); /* alpha <- rho / (d2 + beta * d1 - beta * omega * d3) */

    ierr = PetscObjectSAWsTakeAccess((PetscObject)ksp);CHKERRQ(ierr);
    ksp->its++;

    /* Residual replacement step  */
    if (i > 0 && i%100 == 0 && i < 1001) {
      ierr = KSP_MatMult(ksp,pc->mat,X,R);CHKERRQ(ierr);
      ierr = VecAYPX(R,-1.0,B);CHKERRQ(ierr);              /* r  <- b - Ax */
      ierr = KSP_PCApply(ksp,R,R2);CHKERRQ(ierr);          /* r2 <- K r */
      ierr = KSP_MatMult(ksp,pc->mat,R2,W);CHKERRQ(ierr);  /* w  <- A r2 */
      ierr = KSP_PCApply(ksp,W,W2);CHKERRQ(ierr);          /* w2 <- K w */
      ierr = KSP_MatMult(ksp,pc->mat,W2,T);CHKERRQ(ierr);  /* t  <- A w2 */
      ierr = KSP_MatMult(ksp,pc->mat,P2,S);CHKERRQ(ierr);  /* s  <- A p2 */
      ierr = KSP_PCApply(ksp,S,S2);CHKERRQ(ierr);          /* s2 <- K s */
      ierr = KSP_MatMult(ksp,pc->mat,S2,Z);CHKERRQ(ierr);  /* z  <- A s2 */
      ierr = KSP_PCApply(ksp,Z,Z2);CHKERRQ(ierr);          /* z2 <- K z */
      ierr = KSP_MatMult(ksp,pc->mat,Z2,V);CHKERRQ(ierr);  /* v  <- A z2 */
    }

    ksp->rnorm = dp;
    ierr = PetscObjectSAWsGrantAccess((PetscObject)ksp);CHKERRQ(ierr);
    ierr = KSPLogResidualHistory(ksp,dp);CHKERRQ(ierr);
    ierr = KSPMonitor(ksp,i+1,dp);CHKERRQ(ierr);
    ierr = (*ksp->converged)(ksp,i+1,dp,&ksp->reason,ksp->cnvP);CHKERRQ(ierr);
    if (ksp->reason) break;
    if (rho == 0.0) {
      ksp->reason = KSP_DIVERGED_BREAKDOWN;
      break;
    }
    i++;
  } while (i<ksp->max_it);

  if (i >= ksp->max_it) ksp->reason = KSP_DIVERGED_ITS;
  PetscFunctionReturn(0);
}

/*MC
    KSPPIPEBCGS - Implements the pipelined BiCGStab method.

    This method has only two non-blocking reductions per iteration, compared to 3 blocking for standard FBCGS.  The
    non-blocking reductions are overlapped by matrix-vector products and preconditioner applications.

    Periodic residual replacement may be used to increase robustness and maximal attainable accuracy.

    Options Database Keys:
.see KSPSolve()

    Level: intermediate

    Notes:
    Like KSPFBCGS, the KSPPIPEBCGS implementation only allows for right preconditioning.
    MPI configuration may be necessary for reductions to make asynchronous progress, which is important for
    performance of pipelined methods. See the FAQ on the PETSc website for details.

    Contributed by:
    Siegfried Cools, Universiteit Antwerpen,
    EXA2CT European Project on EXascale Algorithms and Advanced Computational Techniques, 2016.

    Reference:
    S. Cools and W. Vanroose,
    "The communication-hiding pipelined BiCGStab method for the parallel solution of large unsymmetric linear systems",
    Parallel Computing, 65:1-20, 2017.

.seealso:  KSPCreate(), KSPSetType(), KSPType (for list of available types), KSP, KSPBICG, KSPFBCGS, KSPFBCGSL, KSPSetPCSide()
M*/
PETSC_EXTERN PetscErrorCode KSPCreate_PIPEBCGS(KSP ksp)
{
  PetscErrorCode ierr;
  KSP_BCGS       *bcgs;

  PetscFunctionBegin;
  ierr = PetscNewLog(ksp,&bcgs);CHKERRQ(ierr);

  ksp->data                = bcgs;
  ksp->ops->setup          = KSPSetUp_PIPEBCGS;
  ksp->ops->solve          = KSPSolve_PIPEBCGS;
  ksp->ops->destroy        = KSPDestroy_BCGS;
  ksp->ops->reset          = KSPReset_BCGS;
  ksp->ops->buildresidual  = KSPBuildResidualDefault;
  ksp->ops->setfromoptions = KSPSetFromOptions_BCGS;

  ierr = KSPSetSupportedNorm(ksp,KSP_NORM_UNPRECONDITIONED,PC_RIGHT,2);CHKERRQ(ierr);
  ierr = KSPSetSupportedNorm(ksp,KSP_NORM_NONE,PC_RIGHT,1);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
