
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
  PetscCall(KSPSetWorkVecs(ksp,9));
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
  PetscCall(PCGetDiagonalScale(ksp->pc,&diagonalscale));
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

  PetscCall(PCGetOperators(ksp->pc,&Amat,&Pmat));

  ksp->its = 0;
  if (!ksp->guess_zero) {
    PetscCall(KSP_MatMult(ksp,Amat,X,R));            /*     r <- b - Ax     */
    PetscCall(VecAYPX(R,-1.0,B));
  } else {
    PetscCall(VecCopy(B,R));                         /*     r <- b (x is 0) */
  }

  PetscCall(KSP_PCApply(ksp,R,U));                   /*     u <- Br   */

  switch (ksp->normtype) {
  case KSP_NORM_PRECONDITIONED:
    PetscCall(VecNormBegin(U,NORM_2,&dp));                /*     dp <- u'*u = e'*A'*B'*B*A'*e'     */
    PetscCall(PetscCommSplitReductionBegin(PetscObjectComm((PetscObject)U)));
    PetscCall(KSP_MatMult(ksp,Amat,U,W));              /*     w <- Au   */
    PetscCall(VecNormEnd(U,NORM_2,&dp));
    break;
  case KSP_NORM_UNPRECONDITIONED:
    PetscCall(VecNormBegin(R,NORM_2,&dp));                /*     dp <- r'*r = e'*A'*A*e            */
    PetscCall(PetscCommSplitReductionBegin(PetscObjectComm((PetscObject)R)));
    PetscCall(KSP_MatMult(ksp,Amat,U,W));              /*     w <- Au   */
    PetscCall(VecNormEnd(R,NORM_2,&dp));
    break;
  case KSP_NORM_NATURAL:
    PetscCall(VecDotBegin(R,U,&gamma));                  /*     gamma <- u'*r       */
    PetscCall(PetscCommSplitReductionBegin(PetscObjectComm((PetscObject)R)));
    PetscCall(KSP_MatMult(ksp,Amat,U,W));              /*     w <- Au   */
    PetscCall(VecDotEnd(R,U,&gamma));
    KSPCheckDot(ksp,gamma);
    dp = PetscSqrtReal(PetscAbsScalar(gamma));                  /*     dp <- r'*u = r'*B*r = e'*A'*B*A*e */
    break;
  case KSP_NORM_NONE:
    PetscCall(KSP_MatMult(ksp,Amat,U,W));
    dp   = 0.0;
    break;
  default: SETERRQ(PetscObjectComm((PetscObject)ksp),PETSC_ERR_SUP,"%s",KSPNormTypes[ksp->normtype]);
  }
  PetscCall(KSPLogResidualHistory(ksp,dp));
  PetscCall(KSPMonitor(ksp,0,dp));
  ksp->rnorm = dp;
  PetscCall((*ksp->converged)(ksp,0,dp,&ksp->reason,ksp->cnvP)); /* test for convergence */
  if (ksp->reason) PetscFunctionReturn(0);

  i = 0;
  do {
    if (i > 0 && ksp->normtype == KSP_NORM_UNPRECONDITIONED) {
      PetscCall(VecNormBegin(R,NORM_2,&dp));
    } else if (i > 0 && ksp->normtype == KSP_NORM_PRECONDITIONED) {
      PetscCall(VecNormBegin(U,NORM_2,&dp));
    }
    if (!(i == 0 && ksp->normtype == KSP_NORM_NATURAL)) {
      PetscCall(VecDotBegin(R,U,&gamma));
    }
    PetscCall(VecDotBegin(W,U,&delta));
    PetscCall(PetscCommSplitReductionBegin(PetscObjectComm((PetscObject)R)));

    PetscCall(KSP_PCApply(ksp,W,M));           /*   m <- Bw       */
    PetscCall(KSP_MatMult(ksp,Amat,M,N));      /*   n <- Am       */

    if (i > 0 && ksp->normtype == KSP_NORM_UNPRECONDITIONED) {
      PetscCall(VecNormEnd(R,NORM_2,&dp));
    } else if (i > 0 && ksp->normtype == KSP_NORM_PRECONDITIONED) {
      PetscCall(VecNormEnd(U,NORM_2,&dp));
    }
    if (!(i == 0 && ksp->normtype == KSP_NORM_NATURAL)) {
      PetscCall(VecDotEnd(R,U,&gamma));
    }
    PetscCall(VecDotEnd(W,U,&delta));

    if (i > 0) {
      if (ksp->normtype == KSP_NORM_NATURAL) dp = PetscSqrtReal(PetscAbsScalar(gamma));
      else if (ksp->normtype == KSP_NORM_NONE) dp = 0.0;

      ksp->rnorm = dp;
      PetscCall(KSPLogResidualHistory(ksp,dp));
      PetscCall(KSPMonitor(ksp,i,dp));
      PetscCall((*ksp->converged)(ksp,i,dp,&ksp->reason,ksp->cnvP));
      if (ksp->reason) PetscFunctionReturn(0);
    }

    if (i == 0) {
      alpha = gamma / delta;
      PetscCall(VecCopy(N,Z));        /*     z <- n          */
      PetscCall(VecCopy(M,Q));        /*     q <- m          */
      PetscCall(VecCopy(U,P));        /*     p <- u          */
      PetscCall(VecCopy(W,S));        /*     s <- w          */
    } else {
      beta  = gamma / gammaold;
      alpha = gamma / (delta - beta / alpha * gamma);
      PetscCall(VecAYPX(Z,beta,N));   /*     z <- n + beta * z   */
      PetscCall(VecAYPX(Q,beta,M));   /*     q <- m + beta * q   */
      PetscCall(VecAYPX(P,beta,U));   /*     p <- u + beta * p   */
      PetscCall(VecAYPX(S,beta,W));   /*     s <- w + beta * s   */
    }
    PetscCall(VecAXPY(X, alpha,P)); /*     x <- x + alpha * p   */
    PetscCall(VecAXPY(U,-alpha,Q)); /*     u <- u - alpha * q   */
    PetscCall(VecAXPY(W,-alpha,Z)); /*     w <- w - alpha * z   */
    PetscCall(VecAXPY(R,-alpha,S)); /*     r <- r - alpha * s   */
    gammaold = gamma;
    i++;
    ksp->its = i;

    /* if (i%50 == 0) { */
    /*   PetscCall(KSP_MatMult(ksp,Amat,X,R));            /\*     w <- b - Ax     *\/ */
    /*   PetscCall(VecAYPX(R,-1.0,B)); */
    /*   PetscCall(KSP_PCApply(ksp,R,U)); */
    /*   PetscCall(KSP_MatMult(ksp,Amat,U,W)); */
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
  PetscCall(KSPSetSupportedNorm(ksp,KSP_NORM_UNPRECONDITIONED,PC_LEFT,2));
  PetscCall(KSPSetSupportedNorm(ksp,KSP_NORM_PRECONDITIONED,PC_LEFT,2));
  PetscCall(KSPSetSupportedNorm(ksp,KSP_NORM_NATURAL,PC_LEFT,2));
  PetscCall(KSPSetSupportedNorm(ksp,KSP_NORM_NONE,PC_LEFT,1));

  ksp->ops->setup          = KSPSetUp_PIPECG;
  ksp->ops->solve          = KSPSolve_PIPECG;
  ksp->ops->destroy        = KSPDestroyDefault;
  ksp->ops->view           = NULL;
  ksp->ops->setfromoptions = NULL;
  ksp->ops->buildsolution  = KSPBuildSolutionDefault;
  ksp->ops->buildresidual  = KSPBuildResidual_CG;
  PetscFunctionReturn(0);
}
