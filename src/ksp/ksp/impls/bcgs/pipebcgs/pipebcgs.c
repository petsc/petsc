
/*
    This file implements pipelined BiCGStab (pipe-BiCGStab).
    Only allow right preconditioning.
*/
#include <../src/ksp/ksp/impls/bcgs/bcgsimpl.h>       /*I  "petscksp.h"  I*/

static PetscErrorCode KSPSetUp_PIPEBCGS(KSP ksp)
{
  PetscFunctionBegin;
  PetscCall(KSPSetWorkVecs(ksp,15));
  PetscFunctionReturn(0);
}

/* Only need a few hacks from KSPSolve_BCGS */
#include <petsc/private/pcimpl.h>            /*I "petscksp.h" I*/
static PetscErrorCode  KSPSolve_PIPEBCGS(KSP ksp)
{
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
  PetscCheck(ksp->pc_side == PC_RIGHT,PetscObjectComm((PetscObject)ksp),PETSC_ERR_SUP,"KSP pipebcgs does not support %s",PCSides[ksp->pc_side]);
  if (!ksp->guess_zero) {
    if (!bcgs->guess) {
      PetscCall(VecDuplicate(X,&bcgs->guess));
    }
    PetscCall(VecCopy(X,bcgs->guess));
  } else {
    PetscCall(VecSet(X,0.0));
  }

  /* Compute initial residual */
  PetscCall(KSPGetPC(ksp,&pc));
  PetscCall(PCSetUp(pc));
  if (!ksp->guess_zero) {
    PetscCall(KSP_MatMult(ksp,pc->mat,X,Q2));
    PetscCall(VecCopy(B,R));
    PetscCall(VecAXPY(R,-1.0,Q2));
  } else {
    PetscCall(VecCopy(B,R));
  }

  /* Test for nothing to do */
  if (ksp->normtype != KSP_NORM_NONE) {
    PetscCall(VecNorm(R,NORM_2,&dp));
  } else dp = 0.0;
  PetscCall(PetscObjectSAWsTakeAccess((PetscObject)ksp));
  ksp->its   = 0;
  ksp->rnorm = dp;
  PetscCall(PetscObjectSAWsGrantAccess((PetscObject)ksp));
  PetscCall(KSPLogResidualHistory(ksp,dp));
  PetscCall(KSPMonitor(ksp,0,dp));
  PetscCall((*ksp->converged)(ksp,0,dp,&ksp->reason,ksp->cnvP));
  if (ksp->reason) PetscFunctionReturn(0);

  /* Initialize */
  PetscCall(VecCopy(R,RP)); /* rp <- r */

  PetscCall(VecDotBegin(R,RP,&rho)); /* rho <- (r,rp) */
  PetscCall(PetscCommSplitReductionBegin(PetscObjectComm((PetscObject)R)));
  PetscCall(KSP_PCApply(ksp,R,R2)); /* r2 <- K r */
  PetscCall(KSP_MatMult(ksp,pc->mat,R2,W)); /* w <- A r2 */
  PetscCall(VecDotEnd(R,RP,&rho));

  PetscCall(VecDotBegin(W,RP,&d2)); /* d2 <- (w,rp) */
  PetscCall(PetscCommSplitReductionBegin(PetscObjectComm((PetscObject)W)));
  PetscCall(KSP_PCApply(ksp,W,W2)); /* w2 <- K w */
  PetscCall(KSP_MatMult(ksp,pc->mat,W2,T)); /* t <- A w2 */
  PetscCall(VecDotEnd(W,RP,&d2));

  alpha = rho/d2;
  beta = 0.0;

  /* Main loop */
  i=0;
  do {
    if (i == 0) {
      PetscCall(VecCopy(R2,P2)); /* p2 <- r2 */
      PetscCall(VecCopy(W,S));   /* s  <- w  */
      PetscCall(VecCopy(W2,S2)); /* s2 <- w2 */
      PetscCall(VecCopy(T,Z));   /* z  <- t  */
    } else {
      PetscCall(VecAXPBYPCZ(P2,1.0,-beta*omega,beta,R2,S2)); /* p2 <- beta * p2 + r2 - beta * omega * s2 */
      PetscCall(VecAXPBYPCZ(S,1.0,-beta*omega,beta,W,Z));    /* s  <- beta * s  + w  - beta * omega * z  */
      PetscCall(VecAXPBYPCZ(S2,1.0,-beta*omega,beta,W2,Z2)); /* s2 <- beta * s2 + w2 - beta * omega * z2 */
      PetscCall(VecAXPBYPCZ(Z,1.0,-beta*omega,beta,T,V));    /* z  <- beta * z  + t  - beta * omega * v  */
    }
    PetscCall(VecWAXPY(Q,-alpha,S,R));    /* q  <- r  - alpha s  */
    PetscCall(VecWAXPY(Q2,-alpha,S2,R2)); /* q2 <- r2 - alpha s2 */
    PetscCall(VecWAXPY(Y,-alpha,Z,W));    /* y  <- w  - alpha z  */

    PetscCall(VecDotBegin(Q,Y,&d1)); /* d1 <- (q,y) */
    PetscCall(VecDotBegin(Y,Y,&d2)); /* d2 <- (y,y) */

    PetscCall(PetscCommSplitReductionBegin(PetscObjectComm((PetscObject)Q)));
    PetscCall(KSP_PCApply(ksp,Z,Z2)); /* z2 <- K z */
    PetscCall(KSP_MatMult(ksp,pc->mat,Z2,V)); /* v <- A z2 */

    PetscCall(VecDotEnd(Q,Y,&d1));
    PetscCall(VecDotEnd(Y,Y,&d2));

    if (d2 == 0.0) {
      /* y is 0. if q is 0, then alpha s == r, and hence alpha p may be our solution. Give it a try? */
      PetscCall(VecDot(Q,Q,&d1));
      if (d1 != 0.0) {
        ksp->reason = KSP_DIVERGED_BREAKDOWN;
        break;
      }
      PetscCall(VecAXPY(X,alpha,P2));   /* x <- x + alpha p2 */
      PetscCall(PetscObjectSAWsTakeAccess((PetscObject)ksp));
      ksp->its++;
      ksp->rnorm  = 0.0;
      ksp->reason = KSP_CONVERGED_RTOL;
      PetscCall(PetscObjectSAWsGrantAccess((PetscObject)ksp));
      PetscCall(KSPLogResidualHistory(ksp,dp));
      PetscCall(KSPMonitor(ksp,i+1,0.0));
      break;
    }
    omega = d1/d2; /* omega <- (y'q) / (y'y) */
    PetscCall(VecAXPBYPCZ(X,alpha,omega,1.0,P2,Q2)); /* x <- alpha * p2 + omega * q2 + x */
    PetscCall(VecWAXPY(R,-omega,Y,Q));    /* r <- q - omega y */
    PetscCall(VecWAXPY(R2,-alpha,Z2,W2)); /* r2 <- w2 - alpha z2 */
    PetscCall(VecAYPX(R2,-omega,Q2));     /* r2 <- q2 - omega r2 */
    PetscCall(VecWAXPY(W,-alpha,V,T));    /* w <- t - alpha v */
    PetscCall(VecAYPX(W,-omega,Y));       /* w <- y - omega w */
    rhoold = rho;

    if (ksp->normtype != KSP_NORM_NONE && ksp->chknorm < i+2) {
      PetscCall(VecNormBegin(R,NORM_2,&dp)); /* dp <- norm(r) */
    }
    PetscCall(VecDotBegin(R,RP,&rho)); /* rho <- (r,rp) */
    PetscCall(VecDotBegin(S,RP,&d1));  /* d1 <- (s,rp) */
    PetscCall(VecDotBegin(W,RP,&d2));  /* d2 <- (w,rp) */
    PetscCall(VecDotBegin(Z,RP,&d3));  /* d3 <- (z,rp) */

    PetscCall(PetscCommSplitReductionBegin(PetscObjectComm((PetscObject)R)));
    PetscCall(KSP_PCApply(ksp,W,W2)); /* w2 <- K w */
    PetscCall(KSP_MatMult(ksp,pc->mat,W2,T)); /* t <- A w2 */

    if (ksp->normtype != KSP_NORM_NONE && ksp->chknorm < i+2) {
      PetscCall(VecNormEnd(R,NORM_2,&dp));
    }
    PetscCall(VecDotEnd(R,RP,&rho));
    PetscCall(VecDotEnd(S,RP,&d1));
    PetscCall(VecDotEnd(W,RP,&d2));
    PetscCall(VecDotEnd(Z,RP,&d3));

    PetscCheck(d2 + beta * d1 - beta * omega * d3 != 0.0,PetscObjectComm((PetscObject)ksp),PETSC_ERR_PLIB,"Divide by zero");

    beta = (rho/rhoold) * (alpha/omega);
    alpha = rho/(d2 + beta * d1 - beta * omega * d3); /* alpha <- rho / (d2 + beta * d1 - beta * omega * d3) */

    PetscCall(PetscObjectSAWsTakeAccess((PetscObject)ksp));
    ksp->its++;

    /* Residual replacement step  */
    if (i > 0 && i%100 == 0 && i < 1001) {
      PetscCall(KSP_MatMult(ksp,pc->mat,X,R));
      PetscCall(VecAYPX(R,-1.0,B));              /* r  <- b - Ax */
      PetscCall(KSP_PCApply(ksp,R,R2));          /* r2 <- K r */
      PetscCall(KSP_MatMult(ksp,pc->mat,R2,W));  /* w  <- A r2 */
      PetscCall(KSP_PCApply(ksp,W,W2));          /* w2 <- K w */
      PetscCall(KSP_MatMult(ksp,pc->mat,W2,T));  /* t  <- A w2 */
      PetscCall(KSP_MatMult(ksp,pc->mat,P2,S));  /* s  <- A p2 */
      PetscCall(KSP_PCApply(ksp,S,S2));          /* s2 <- K s */
      PetscCall(KSP_MatMult(ksp,pc->mat,S2,Z));  /* z  <- A s2 */
      PetscCall(KSP_PCApply(ksp,Z,Z2));          /* z2 <- K z */
      PetscCall(KSP_MatMult(ksp,pc->mat,Z2,V));  /* v  <- A z2 */
    }

    ksp->rnorm = dp;
    PetscCall(PetscObjectSAWsGrantAccess((PetscObject)ksp));
    PetscCall(KSPLogResidualHistory(ksp,dp));
    PetscCall(KSPMonitor(ksp,i+1,dp));
    PetscCall((*ksp->converged)(ksp,i+1,dp,&ksp->reason,ksp->cnvP));
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
    see KSPSolve()

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

.seealso: `KSPCreate()`, `KSPSetType()`, `KSPType`, `KSP`, `KSPBICG`, `KSPFBCGS`, `KSPFBCGSL`, `KSPSetPCSide()`
M*/
PETSC_EXTERN PetscErrorCode KSPCreate_PIPEBCGS(KSP ksp)
{
  KSP_BCGS       *bcgs;

  PetscFunctionBegin;
  PetscCall(PetscNewLog(ksp,&bcgs));

  ksp->data                = bcgs;
  ksp->ops->setup          = KSPSetUp_PIPEBCGS;
  ksp->ops->solve          = KSPSolve_PIPEBCGS;
  ksp->ops->destroy        = KSPDestroy_BCGS;
  ksp->ops->reset          = KSPReset_BCGS;
  ksp->ops->buildresidual  = KSPBuildResidualDefault;
  ksp->ops->setfromoptions = KSPSetFromOptions_BCGS;

  PetscCall(KSPSetSupportedNorm(ksp,KSP_NORM_UNPRECONDITIONED,PC_RIGHT,2));
  PetscCall(KSPSetSupportedNorm(ksp,KSP_NORM_NONE,PC_RIGHT,1));
  PetscFunctionReturn(0);
}
