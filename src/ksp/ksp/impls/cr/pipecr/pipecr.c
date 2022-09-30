#include <petsc/private/kspimpl.h>

/*
     KSPSetUp_PIPECR - Sets up the workspace needed by the PIPECR method.

      This is called once, usually automatically by KSPSolve() or KSPSetUp()
     but can be called directly by KSPSetUp()
*/
static PetscErrorCode KSPSetUp_PIPECR(KSP ksp)
{
  PetscFunctionBegin;
  /* get work vectors needed by PIPECR */
  PetscCall(KSPSetWorkVecs(ksp, 7));
  PetscFunctionReturn(0);
}

/*
 KSPSolve_PIPECR - This routine actually applies the pipelined conjugate residual method
*/
static PetscErrorCode KSPSolve_PIPECR(KSP ksp)
{
  PetscInt    i;
  PetscScalar alpha = 0.0, beta = 0.0, gamma, gammaold = 0.0, delta;
  PetscReal   dp = 0.0;
  Vec         X, B, Z, P, W, Q, U, M, N;
  Mat         Amat, Pmat;
  PetscBool   diagonalscale;

  PetscFunctionBegin;
  PetscCall(PCGetDiagonalScale(ksp->pc, &diagonalscale));
  PetscCheck(!diagonalscale, PetscObjectComm((PetscObject)ksp), PETSC_ERR_SUP, "Krylov method %s does not support diagonal scaling", ((PetscObject)ksp)->type_name);

  X = ksp->vec_sol;
  B = ksp->vec_rhs;
  M = ksp->work[0];
  Z = ksp->work[1];
  P = ksp->work[2];
  N = ksp->work[3];
  W = ksp->work[4];
  Q = ksp->work[5];
  U = ksp->work[6];

  PetscCall(PCGetOperators(ksp->pc, &Amat, &Pmat));

  ksp->its = 0;
  /* we don't have an R vector, so put the (unpreconditioned) residual in w for now */
  if (!ksp->guess_zero) {
    PetscCall(KSP_MatMult(ksp, Amat, X, W)); /*     w <- b - Ax     */
    PetscCall(VecAYPX(W, -1.0, B));
  } else {
    PetscCall(VecCopy(B, W)); /*     w <- b (x is 0) */
  }
  PetscCall(KSP_PCApply(ksp, W, U)); /*     u <- Bw   */

  switch (ksp->normtype) {
  case KSP_NORM_PRECONDITIONED:
    PetscCall(VecNormBegin(U, NORM_2, &dp)); /*     dp <- u'*u = e'*A'*B'*B*A'*e'     */
    PetscCall(PetscCommSplitReductionBegin(PetscObjectComm((PetscObject)U)));
    PetscCall(KSP_MatMult(ksp, Amat, U, W)); /*     w <- Au   */
    PetscCall(VecNormEnd(U, NORM_2, &dp));
    break;
  case KSP_NORM_NONE:
    PetscCall(KSP_MatMult(ksp, Amat, U, W));
    dp = 0.0;
    break;
  default:
    SETERRQ(PetscObjectComm((PetscObject)ksp), PETSC_ERR_SUP, "%s", KSPNormTypes[ksp->normtype]);
  }
  PetscCall(KSPLogResidualHistory(ksp, dp));
  PetscCall(KSPMonitor(ksp, 0, dp));
  ksp->rnorm = dp;
  PetscCall((*ksp->converged)(ksp, 0, dp, &ksp->reason, ksp->cnvP)); /* test for convergence */
  if (ksp->reason) PetscFunctionReturn(0);

  i = 0;
  do {
    PetscCall(KSP_PCApply(ksp, W, M)); /*   m <- Bw       */

    if (i > 0 && ksp->normtype == KSP_NORM_PRECONDITIONED) PetscCall(VecNormBegin(U, NORM_2, &dp));
    PetscCall(VecDotBegin(W, U, &gamma));
    PetscCall(VecDotBegin(M, W, &delta));
    PetscCall(PetscCommSplitReductionBegin(PetscObjectComm((PetscObject)U)));

    PetscCall(KSP_MatMult(ksp, Amat, M, N)); /*   n <- Am       */

    if (i > 0 && ksp->normtype == KSP_NORM_PRECONDITIONED) PetscCall(VecNormEnd(U, NORM_2, &dp));
    PetscCall(VecDotEnd(W, U, &gamma));
    PetscCall(VecDotEnd(M, W, &delta));

    if (i > 0) {
      if (ksp->normtype == KSP_NORM_NONE) dp = 0.0;
      ksp->rnorm = dp;
      PetscCall(KSPLogResidualHistory(ksp, dp));
      PetscCall(KSPMonitor(ksp, i, dp));
      PetscCall((*ksp->converged)(ksp, i, dp, &ksp->reason, ksp->cnvP));
      if (ksp->reason) PetscFunctionReturn(0);
    }

    if (i == 0) {
      alpha = gamma / delta;
      PetscCall(VecCopy(N, Z)); /*     z <- n          */
      PetscCall(VecCopy(M, Q)); /*     q <- m          */
      PetscCall(VecCopy(U, P)); /*     p <- u          */
    } else {
      beta  = gamma / gammaold;
      alpha = gamma / (delta - beta / alpha * gamma);
      PetscCall(VecAYPX(Z, beta, N)); /*     z <- n + beta * z   */
      PetscCall(VecAYPX(Q, beta, M)); /*     q <- m + beta * q   */
      PetscCall(VecAYPX(P, beta, U)); /*     p <- u + beta * p   */
    }
    PetscCall(VecAXPY(X, alpha, P));  /*     x <- x + alpha * p   */
    PetscCall(VecAXPY(U, -alpha, Q)); /*     u <- u - alpha * q   */
    PetscCall(VecAXPY(W, -alpha, Z)); /*     w <- w - alpha * z   */
    gammaold = gamma;
    i++;
    ksp->its = i;

    /* if (i%50 == 0) { */
    /*   PetscCall(KSP_MatMult(ksp,Amat,X,W));            /\*     w <- b - Ax     *\/ */
    /*   PetscCall(VecAYPX(W,-1.0,B)); */
    /*   PetscCall(KSP_PCApply(ksp,W,U)); */
    /*   PetscCall(KSP_MatMult(ksp,Amat,U,W)); */
    /* } */

  } while (i <= ksp->max_it);
  if (i >= ksp->max_it) ksp->reason = KSP_DIVERGED_ITS;
  PetscFunctionReturn(0);
}

/*MC
   KSPPIPECR - Pipelined conjugate residual method. [](sec_pipelineksp)

   Level: intermediate

   Notes:
   This method has only a single non-blocking reduction per iteration, compared to 2 blocking for standard `KSPCR`.  The
   non-blocking reduction is overlapped by the matrix-vector product, but not the preconditioner application.

   See also `KSPPIPECG`, where the reduction is only overlapped with the matrix-vector product.

   MPI configuration may be necessary for reductions to make asynchronous progress, which is important for performance of pipelined methods.
   See [](doc_faq_pipelined)

   Contributed by:
   Pieter Ghysels, Universiteit Antwerpen, Intel Exascience lab Flanders

   Reference:
   P. Ghysels and W. Vanroose, "Hiding global synchronization latency in the preconditioned Conjugate Gradient algorithm",
   Submitted to Parallel Computing, 2012.

.seealso: [](chapter_ksp), [](sec_pipelineksp), [](doc_faq_pipelined), `KSPCreate()`, `KSPSetType()`, `KSPPIPECG`, `KSPGROPPCG`, `KSPPGMRES`, `KSPCG`, `KSPCGUseSingleReduction()`
M*/

PETSC_EXTERN PetscErrorCode KSPCreate_PIPECR(KSP ksp)
{
  PetscFunctionBegin;
  PetscCall(KSPSetSupportedNorm(ksp, KSP_NORM_PRECONDITIONED, PC_LEFT, 2));
  PetscCall(KSPSetSupportedNorm(ksp, KSP_NORM_NONE, PC_LEFT, 1));

  ksp->ops->setup          = KSPSetUp_PIPECR;
  ksp->ops->solve          = KSPSolve_PIPECR;
  ksp->ops->destroy        = KSPDestroyDefault;
  ksp->ops->view           = NULL;
  ksp->ops->setfromoptions = NULL;
  ksp->ops->buildsolution  = KSPBuildSolutionDefault;
  ksp->ops->buildresidual  = KSPBuildResidualDefault;
  PetscFunctionReturn(0);
}
