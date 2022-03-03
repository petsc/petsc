
#include <petsc/private/kspimpl.h>

/*
     KSPSetUp_PIPECG - Sets up the workspace needed by the PIPECG method.

      This is called once, usually automatically by KSPSolve() or KSPSetUp()
     but can be called directly by KSPSetUp()
*/
static PetscErrorCode KSPSetUp_PIPECG(KSP ksp)
{
  PetscFunctionBegin;
  /* get work vectors needed by PIPECG */
  CHKERRQ(KSPSetWorkVecs(ksp,9));
  PetscFunctionReturn(0);
}

/*
 KSPSolve_PIPECG - This routine actually applies the pipelined conjugate gradient method
*/
static PetscErrorCode  KSPSolve_PIPECG(KSP ksp)
{
  PetscInt       i;
  PetscScalar    alpha = 0.0,beta = 0.0,gamma = 0.0,gammaold = 0.0,delta = 0.0;
  PetscReal      dp    = 0.0;
  Vec            X,B,Z,P,W,Q,U,M,N,R,S;
  Mat            Amat,Pmat;
  PetscBool      diagonalscale;

  PetscFunctionBegin;
  CHKERRQ(PCGetDiagonalScale(ksp->pc,&diagonalscale));
  PetscCheck(!diagonalscale,PetscObjectComm((PetscObject)ksp),PETSC_ERR_SUP,"Krylov method %s does not support diagonal scaling",((PetscObject)ksp)->type_name);

  X = ksp->vec_sol;
  B = ksp->vec_rhs;
  R = ksp->work[0];
  Z = ksp->work[1];
  P = ksp->work[2];
  N = ksp->work[3];
  W = ksp->work[4];
  Q = ksp->work[5];
  U = ksp->work[6];
  M = ksp->work[7];
  S = ksp->work[8];

  CHKERRQ(PCGetOperators(ksp->pc,&Amat,&Pmat));

  ksp->its = 0;
  if (!ksp->guess_zero) {
    CHKERRQ(KSP_MatMult(ksp,Amat,X,R));            /*     r <- b - Ax     */
    CHKERRQ(VecAYPX(R,-1.0,B));
  } else {
    CHKERRQ(VecCopy(B,R));                         /*     r <- b (x is 0) */
  }

  CHKERRQ(KSP_PCApply(ksp,R,U));                   /*     u <- Br   */

  switch (ksp->normtype) {
  case KSP_NORM_PRECONDITIONED:
    CHKERRQ(VecNormBegin(U,NORM_2,&dp));                /*     dp <- u'*u = e'*A'*B'*B*A'*e'     */
    CHKERRQ(PetscCommSplitReductionBegin(PetscObjectComm((PetscObject)U)));
    CHKERRQ(KSP_MatMult(ksp,Amat,U,W));              /*     w <- Au   */
    CHKERRQ(VecNormEnd(U,NORM_2,&dp));
    break;
  case KSP_NORM_UNPRECONDITIONED:
    CHKERRQ(VecNormBegin(R,NORM_2,&dp));                /*     dp <- r'*r = e'*A'*A*e            */
    CHKERRQ(PetscCommSplitReductionBegin(PetscObjectComm((PetscObject)R)));
    CHKERRQ(KSP_MatMult(ksp,Amat,U,W));              /*     w <- Au   */
    CHKERRQ(VecNormEnd(R,NORM_2,&dp));
    break;
  case KSP_NORM_NATURAL:
    CHKERRQ(VecDotBegin(R,U,&gamma));                  /*     gamma <- u'*r       */
    CHKERRQ(PetscCommSplitReductionBegin(PetscObjectComm((PetscObject)R)));
    CHKERRQ(KSP_MatMult(ksp,Amat,U,W));              /*     w <- Au   */
    CHKERRQ(VecDotEnd(R,U,&gamma));
    KSPCheckDot(ksp,gamma);
    dp = PetscSqrtReal(PetscAbsScalar(gamma));                  /*     dp <- r'*u = r'*B*r = e'*A'*B*A*e */
    break;
  case KSP_NORM_NONE:
    CHKERRQ(KSP_MatMult(ksp,Amat,U,W));
    dp   = 0.0;
    break;
  default: SETERRQ(PetscObjectComm((PetscObject)ksp),PETSC_ERR_SUP,"%s",KSPNormTypes[ksp->normtype]);
  }
  CHKERRQ(KSPLogResidualHistory(ksp,dp));
  CHKERRQ(KSPMonitor(ksp,0,dp));
  ksp->rnorm = dp;
  CHKERRQ((*ksp->converged)(ksp,0,dp,&ksp->reason,ksp->cnvP)); /* test for convergence */
  if (ksp->reason) PetscFunctionReturn(0);

  i = 0;
  do {
    if (i > 0 && ksp->normtype == KSP_NORM_UNPRECONDITIONED) {
      CHKERRQ(VecNormBegin(R,NORM_2,&dp));
    } else if (i > 0 && ksp->normtype == KSP_NORM_PRECONDITIONED) {
      CHKERRQ(VecNormBegin(U,NORM_2,&dp));
    }
    if (!(i == 0 && ksp->normtype == KSP_NORM_NATURAL)) {
      CHKERRQ(VecDotBegin(R,U,&gamma));
    }
    CHKERRQ(VecDotBegin(W,U,&delta));
    CHKERRQ(PetscCommSplitReductionBegin(PetscObjectComm((PetscObject)R)));

    CHKERRQ(KSP_PCApply(ksp,W,M));           /*   m <- Bw       */
    CHKERRQ(KSP_MatMult(ksp,Amat,M,N));      /*   n <- Am       */

    if (i > 0 && ksp->normtype == KSP_NORM_UNPRECONDITIONED) {
      CHKERRQ(VecNormEnd(R,NORM_2,&dp));
    } else if (i > 0 && ksp->normtype == KSP_NORM_PRECONDITIONED) {
      CHKERRQ(VecNormEnd(U,NORM_2,&dp));
    }
    if (!(i == 0 && ksp->normtype == KSP_NORM_NATURAL)) {
      CHKERRQ(VecDotEnd(R,U,&gamma));
    }
    CHKERRQ(VecDotEnd(W,U,&delta));

    if (i > 0) {
      if (ksp->normtype == KSP_NORM_NATURAL) dp = PetscSqrtReal(PetscAbsScalar(gamma));
      else if (ksp->normtype == KSP_NORM_NONE) dp = 0.0;

      ksp->rnorm = dp;
      CHKERRQ(KSPLogResidualHistory(ksp,dp));
      CHKERRQ(KSPMonitor(ksp,i,dp));
      CHKERRQ((*ksp->converged)(ksp,i,dp,&ksp->reason,ksp->cnvP));
      if (ksp->reason) PetscFunctionReturn(0);
    }

    if (i == 0) {
      alpha = gamma / delta;
      CHKERRQ(VecCopy(N,Z));        /*     z <- n          */
      CHKERRQ(VecCopy(M,Q));        /*     q <- m          */
      CHKERRQ(VecCopy(U,P));        /*     p <- u          */
      CHKERRQ(VecCopy(W,S));        /*     s <- w          */
    } else {
      beta  = gamma / gammaold;
      alpha = gamma / (delta - beta / alpha * gamma);
      CHKERRQ(VecAYPX(Z,beta,N));   /*     z <- n + beta * z   */
      CHKERRQ(VecAYPX(Q,beta,M));   /*     q <- m + beta * q   */
      CHKERRQ(VecAYPX(P,beta,U));   /*     p <- u + beta * p   */
      CHKERRQ(VecAYPX(S,beta,W));   /*     s <- w + beta * s   */
    }
    CHKERRQ(VecAXPY(X, alpha,P)); /*     x <- x + alpha * p   */
    CHKERRQ(VecAXPY(U,-alpha,Q)); /*     u <- u - alpha * q   */
    CHKERRQ(VecAXPY(W,-alpha,Z)); /*     w <- w - alpha * z   */
    CHKERRQ(VecAXPY(R,-alpha,S)); /*     r <- r - alpha * s   */
    gammaold = gamma;
    i++;
    ksp->its = i;

    /* if (i%50 == 0) { */
    /*   CHKERRQ(KSP_MatMult(ksp,Amat,X,R));            /\*     w <- b - Ax     *\/ */
    /*   CHKERRQ(VecAYPX(R,-1.0,B)); */
    /*   CHKERRQ(KSP_PCApply(ksp,R,U)); */
    /*   CHKERRQ(KSP_MatMult(ksp,Amat,U,W)); */
    /* } */

  } while (i<=ksp->max_it);
  if (!ksp->reason) ksp->reason = KSP_DIVERGED_ITS;
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode KSPBuildResidual_CG(KSP,Vec,Vec,Vec*);

/*MC
   KSPPIPECG - Pipelined conjugate gradient method.

   This method has only a single non-blocking reduction per iteration, compared to 2 blocking for standard CG.  The
   non-blocking reduction is overlapped by the matrix-vector product and preconditioner application.

   See also KSPPIPECR, where the reduction is only overlapped with the matrix-vector product.

   Level: intermediate

   Notes:
   MPI configuration may be necessary for reductions to make asynchronous progress, which is important for performance of pipelined methods.
   See the FAQ on the PETSc website for details.

   Contributed by:
   Pieter Ghysels, Universiteit Antwerpen, Intel Exascience lab Flanders

   Reference:
   P. Ghysels and W. Vanroose, "Hiding global synchronization latency in the preconditioned Conjugate Gradient algorithm",
   Submitted to Parallel Computing, 2012.

.seealso: KSPCreate(), KSPSetType(), KSPPIPECR, KSPGROPPCG, KSPPGMRES, KSPCG, KSPCGUseSingleReduction()
M*/
PETSC_EXTERN PetscErrorCode KSPCreate_PIPECG(KSP ksp)
{
  PetscFunctionBegin;
  CHKERRQ(KSPSetSupportedNorm(ksp,KSP_NORM_UNPRECONDITIONED,PC_LEFT,2));
  CHKERRQ(KSPSetSupportedNorm(ksp,KSP_NORM_PRECONDITIONED,PC_LEFT,2));
  CHKERRQ(KSPSetSupportedNorm(ksp,KSP_NORM_NATURAL,PC_LEFT,2));
  CHKERRQ(KSPSetSupportedNorm(ksp,KSP_NORM_NONE,PC_LEFT,1));

  ksp->ops->setup          = KSPSetUp_PIPECG;
  ksp->ops->solve          = KSPSolve_PIPECG;
  ksp->ops->destroy        = KSPDestroyDefault;
  ksp->ops->view           = NULL;
  ksp->ops->setfromoptions = NULL;
  ksp->ops->buildsolution  = KSPBuildSolutionDefault;
  ksp->ops->buildresidual  = KSPBuildResidual_CG;
  PetscFunctionReturn(0);
}
