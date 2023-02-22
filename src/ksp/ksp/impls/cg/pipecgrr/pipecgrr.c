
#include <petsc/private/kspimpl.h>

/*
     KSPSetUp_PIPECGRR - Sets up the workspace needed by the PIPECGRR method.

      This is called once, usually automatically by KSPSolve() or KSPSetUp()
     but can be called directly by KSPSetUp()
*/
static PetscErrorCode KSPSetUp_PIPECGRR(KSP ksp)
{
  PetscFunctionBegin;
  /* get work vectors needed by PIPECGRR */
  PetscCall(KSPSetWorkVecs(ksp, 9));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
 KSPSolve_PIPECGRR - This routine actually applies the pipelined conjugate gradient method with automated residual replacement
*/
static PetscErrorCode KSPSolve_PIPECGRR(KSP ksp)
{
  PetscInt    i = 0, replace = 0, nsize;
  PetscScalar alpha = 0.0, beta = 0.0, gamma = 0.0, gammaold = 0.0, delta = 0.0, alphap = 0.0, betap = 0.0;
  PetscReal   dp = 0.0, nsi = 0.0, sqn = 0.0, Anorm = 0.0, rnp = 0.0, pnp = 0.0, snp = 0.0, unp = 0.0, wnp = 0.0, xnp = 0.0, qnp = 0.0, znp = 0.0, mnz = 5.0, tol = PETSC_SQRT_MACHINE_EPSILON, eps = PETSC_MACHINE_EPSILON;
  PetscReal   ds = 0.0, dz = 0.0, dx = 0.0, dpp = 0.0, dq = 0.0, dm = 0.0, du = 0.0, dw = 0.0, db = 0.0, errr = 0.0, errrprev = 0.0, errs = 0.0, errw = 0.0, errz = 0.0, errncr = 0.0, errncs = 0.0, errncw = 0.0, errncz = 0.0;
  Vec         X, B, Z, P, W, Q, U, M, N, R, S;
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
  R = ksp->work[7];
  S = ksp->work[8];

  PetscCall(PCGetOperators(ksp->pc, &Amat, &Pmat));

  ksp->its = 0;
  if (!ksp->guess_zero) {
    PetscCall(KSP_MatMult(ksp, Amat, X, R)); /*  r <- b - Ax  */
    PetscCall(VecAYPX(R, -1.0, B));
  } else {
    PetscCall(VecCopy(B, R)); /*  r <- b (x is 0)  */
  }

  PetscCall(KSP_PCApply(ksp, R, U)); /*  u <- Br  */

  switch (ksp->normtype) {
  case KSP_NORM_PRECONDITIONED:
    PetscCall(VecNormBegin(U, NORM_2, &dp)); /*  dp <- u'*u = e'*A'*B'*B*A'*e'  */
    PetscCall(VecNormBegin(B, NORM_2, &db));
    PetscCall(PetscCommSplitReductionBegin(PetscObjectComm((PetscObject)U)));
    PetscCall(KSP_MatMult(ksp, Amat, U, W)); /*  w <- Au  */
    PetscCall(VecNormEnd(U, NORM_2, &dp));
    PetscCall(VecNormEnd(B, NORM_2, &db));
    break;
  case KSP_NORM_UNPRECONDITIONED:
    PetscCall(VecNormBegin(R, NORM_2, &dp)); /*  dp <- r'*r = e'*A'*A*e  */
    PetscCall(VecNormBegin(B, NORM_2, &db));
    PetscCall(PetscCommSplitReductionBegin(PetscObjectComm((PetscObject)R)));
    PetscCall(KSP_MatMult(ksp, Amat, U, W)); /*  w <- Au  */
    PetscCall(VecNormEnd(R, NORM_2, &dp));
    PetscCall(VecNormEnd(B, NORM_2, &db));
    break;
  case KSP_NORM_NATURAL:
    PetscCall(VecDotBegin(R, U, &gamma)); /*  gamma <- u'*r  */
    PetscCall(VecNormBegin(B, NORM_2, &db));
    PetscCall(PetscCommSplitReductionBegin(PetscObjectComm((PetscObject)R)));
    PetscCall(KSP_MatMult(ksp, Amat, U, W)); /*  w <- Au  */
    PetscCall(VecDotEnd(R, U, &gamma));
    PetscCall(VecNormEnd(B, NORM_2, &db));
    KSPCheckDot(ksp, gamma);
    dp = PetscSqrtReal(PetscAbsScalar(gamma)); /*  dp <- r'*u = r'*B*r = e'*A'*B*A*e  */
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
  PetscCall((*ksp->converged)(ksp, 0, dp, &ksp->reason, ksp->cnvP)); /*  test for convergence  */
  if (ksp->reason) PetscFunctionReturn(PETSC_SUCCESS);

  PetscCall(MatNorm(Amat, NORM_INFINITY, &Anorm));
  PetscCall(VecGetSize(B, &nsize));
  nsi = (PetscReal)nsize;
  sqn = PetscSqrtReal(nsi);

  do {
    if (i > 1) {
      pnp = dpp;
      snp = ds;
      qnp = dq;
      znp = dz;
    }
    if (i > 0) {
      rnp    = dp;
      unp    = du;
      wnp    = dw;
      xnp    = dx;
      alphap = alpha;
      betap  = beta;
    }

    if (i > 0 && ksp->normtype == KSP_NORM_UNPRECONDITIONED) {
      PetscCall(VecNormBegin(R, NORM_2, &dp));
    } else if (i > 0 && ksp->normtype == KSP_NORM_PRECONDITIONED) {
      PetscCall(VecNormBegin(U, NORM_2, &dp));
    }
    if (!(i == 0 && ksp->normtype == KSP_NORM_NATURAL)) PetscCall(VecDotBegin(R, U, &gamma));
    PetscCall(VecDotBegin(W, U, &delta));

    if (i > 0) {
      PetscCall(VecNormBegin(S, NORM_2, &ds));
      PetscCall(VecNormBegin(Z, NORM_2, &dz));
      PetscCall(VecNormBegin(P, NORM_2, &dpp));
      PetscCall(VecNormBegin(Q, NORM_2, &dq));
      PetscCall(VecNormBegin(M, NORM_2, &dm));
    }
    PetscCall(VecNormBegin(X, NORM_2, &dx));
    PetscCall(VecNormBegin(U, NORM_2, &du));
    PetscCall(VecNormBegin(W, NORM_2, &dw));

    PetscCall(PetscCommSplitReductionBegin(PetscObjectComm((PetscObject)R)));
    PetscCall(KSP_PCApply(ksp, W, M));       /*   m <- Bw       */
    PetscCall(KSP_MatMult(ksp, Amat, M, N)); /*   n <- Am       */

    if (i > 0 && ksp->normtype == KSP_NORM_UNPRECONDITIONED) {
      PetscCall(VecNormEnd(R, NORM_2, &dp));
    } else if (i > 0 && ksp->normtype == KSP_NORM_PRECONDITIONED) {
      PetscCall(VecNormEnd(U, NORM_2, &dp));
    }
    if (!(i == 0 && ksp->normtype == KSP_NORM_NATURAL)) PetscCall(VecDotEnd(R, U, &gamma));
    PetscCall(VecDotEnd(W, U, &delta));

    if (i > 0) {
      PetscCall(VecNormEnd(S, NORM_2, &ds));
      PetscCall(VecNormEnd(Z, NORM_2, &dz));
      PetscCall(VecNormEnd(P, NORM_2, &dpp));
      PetscCall(VecNormEnd(Q, NORM_2, &dq));
      PetscCall(VecNormEnd(M, NORM_2, &dm));
    }
    PetscCall(VecNormEnd(X, NORM_2, &dx));
    PetscCall(VecNormEnd(U, NORM_2, &du));
    PetscCall(VecNormEnd(W, NORM_2, &dw));

    if (i > 0) {
      if (ksp->normtype == KSP_NORM_NATURAL) dp = PetscSqrtReal(PetscAbsScalar(gamma));
      else if (ksp->normtype == KSP_NORM_NONE) dp = 0.0;

      ksp->rnorm = dp;
      PetscCall(KSPLogResidualHistory(ksp, dp));
      PetscCall(KSPMonitor(ksp, i, dp));
      PetscCall((*ksp->converged)(ksp, i, dp, &ksp->reason, ksp->cnvP));
      if (ksp->reason) PetscFunctionReturn(PETSC_SUCCESS);
    }

    if (i == 0) {
      alpha = gamma / delta;
      PetscCall(VecCopy(N, Z)); /*  z <- n  */
      PetscCall(VecCopy(M, Q)); /*  q <- m  */
      PetscCall(VecCopy(U, P)); /*  p <- u  */
      PetscCall(VecCopy(W, S)); /*  s <- w  */
    } else {
      beta  = gamma / gammaold;
      alpha = gamma / (delta - beta / alpha * gamma);
      PetscCall(VecAYPX(Z, beta, N)); /*  z <- n + beta * z  */
      PetscCall(VecAYPX(Q, beta, M)); /*  q <- m + beta * q  */
      PetscCall(VecAYPX(P, beta, U)); /*  p <- u + beta * p  */
      PetscCall(VecAYPX(S, beta, W)); /*  s <- w + beta * s  */
    }
    PetscCall(VecAXPY(X, alpha, P));  /*  x <- x + alpha * p  */
    PetscCall(VecAXPY(U, -alpha, Q)); /*  u <- u - alpha * q  */
    PetscCall(VecAXPY(W, -alpha, Z)); /*  w <- w - alpha * z  */
    PetscCall(VecAXPY(R, -alpha, S)); /*  r <- r - alpha * s  */
    gammaold = gamma;

    if (i > 0) {
      errncr = PetscSqrtReal(Anorm * xnp + 2.0 * Anorm * PetscAbsScalar(alphap) * dpp + rnp + 2.0 * PetscAbsScalar(alphap) * ds) * eps;
      errncw = PetscSqrtReal(Anorm * unp + 2.0 * Anorm * PetscAbsScalar(alphap) * dq + wnp + 2.0 * PetscAbsScalar(alphap) * dz) * eps;
    }
    if (i > 1) {
      errncs = PetscSqrtReal(Anorm * unp + 2.0 * Anorm * PetscAbsScalar(betap) * pnp + wnp + 2.0 * PetscAbsScalar(betap) * snp) * eps;
      errncz = PetscSqrtReal((mnz * sqn + 2) * Anorm * dm + 2.0 * Anorm * PetscAbsScalar(betap) * qnp + 2.0 * PetscAbsScalar(betap) * znp) * eps;
    }

    if (i > 0) {
      if (i == 1) {
        errr = PetscSqrtReal((mnz * sqn + 1) * Anorm * xnp + db) * eps + PetscSqrtReal(PetscAbsScalar(alphap) * mnz * sqn * Anorm * dpp) * eps + errncr;
        errs = PetscSqrtReal(mnz * sqn * Anorm * dpp) * eps;
        errw = PetscSqrtReal(mnz * sqn * Anorm * unp) * eps + PetscSqrtReal(PetscAbsScalar(alphap) * mnz * sqn * Anorm * dq) * eps + errncw;
        errz = PetscSqrtReal(mnz * sqn * Anorm * dq) * eps;
      } else if (replace == 1) {
        errrprev = errr;
        errr     = PetscSqrtReal((mnz * sqn + 1) * Anorm * dx + db) * eps;
        errs     = PetscSqrtReal(mnz * sqn * Anorm * dpp) * eps;
        errw     = PetscSqrtReal(mnz * sqn * Anorm * du) * eps;
        errz     = PetscSqrtReal(mnz * sqn * Anorm * dq) * eps;
        replace  = 0;
      } else {
        errrprev = errr;
        errr     = errr + PetscAbsScalar(alphap) * PetscAbsScalar(betap) * errs + PetscAbsScalar(alphap) * errw + errncr + PetscAbsScalar(alphap) * errncs;
        errs     = errw + PetscAbsScalar(betap) * errs + errncs;
        errw     = errw + PetscAbsScalar(alphap) * PetscAbsScalar(betap) * errz + errncw + PetscAbsScalar(alphap) * errncz;
        errz     = PetscAbsScalar(betap) * errz + errncz;
      }
      if (i > 1 && errrprev <= (tol * rnp) && errr > (tol * dp)) {
        PetscCall(KSP_MatMult(ksp, Amat, X, R)); /*  r <- Ax - b  */
        PetscCall(VecAYPX(R, -1.0, B));
        PetscCall(KSP_PCApply(ksp, R, U));       /*  u <- Br  */
        PetscCall(KSP_MatMult(ksp, Amat, U, W)); /*  w <- Au  */
        PetscCall(KSP_MatMult(ksp, Amat, P, S)); /*  s <- Ap  */
        PetscCall(KSP_PCApply(ksp, S, Q));       /*  q <- Bs  */
        PetscCall(KSP_MatMult(ksp, Amat, Q, Z)); /*  z <- Aq  */
        replace = 1;
      }
    }

    i++;
    ksp->its = i;

  } while (i <= ksp->max_it);
  if (!ksp->reason) ksp->reason = KSP_DIVERGED_ITS;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*MC
   KSPPIPECGRR - Pipelined conjugate gradient method with automated residual replacements. [](sec_pipelineksp)

   Level: intermediate

   Notes:
   This method has only a single non-blocking reduction per iteration, compared to 2 blocking for standard `KSPCG`.  The
   non-blocking reduction is overlapped by the matrix-vector product and preconditioner application.

   `KSPPIPECGRR` improves the robustness of `KSPPIPECG` by adding an automated residual replacement strategy.
   True residual and other auxiliary variables are computed explicitly in a number of dynamically determined
   iterations to counteract the accumulation of rounding errors and thus attain a higher maximal final accuracy.

   See also `KSPPIPECG`, which is identical to `KSPPIPECGRR` without residual replacements.
   See also `KSPPIPECR`, where the reduction is only overlapped with the matrix-vector product.

   MPI configuration may be necessary for reductions to make asynchronous progress, which is important for
   performance of pipelined methods. See [](doc_faq_pipelined)

   Contributed by:
   Siegfried Cools, Universiteit Antwerpen, Dept. Mathematics & Computer Science,
   European FP7 Project on EXascale Algorithms and Advanced Computational Techniques (EXA2CT) / Research Foundation Flanders (FWO)

   Reference:
   S. Cools, E.F. Yetkin, E. Agullo, L. Giraud, W. Vanroose, "Analyzing the effect of local rounding error
   propagation on the maximal attainable accuracy of the pipelined Conjugate Gradients method",
   SIAM Journal on Matrix Analysis and Applications (SIMAX), 39(1):426--450, 2018.

.seealso: [](chapter_ksp), [](doc_faq_pipelined), [](sec_pipelineksp), `KSPCreate()`, `KSPSetType()`, `KSPPIPECR`, `KSPGROPPCG`, `KSPPIPECG`, `KSPPGMRES`, `KSPCG`, `KSPPIPEBCGS`, `KSPCGUseSingleReduction()`
M*/
PETSC_EXTERN PetscErrorCode KSPCreate_PIPECGRR(KSP ksp)
{
  PetscFunctionBegin;
  PetscCall(KSPSetSupportedNorm(ksp, KSP_NORM_UNPRECONDITIONED, PC_LEFT, 2));
  PetscCall(KSPSetSupportedNorm(ksp, KSP_NORM_PRECONDITIONED, PC_LEFT, 2));
  PetscCall(KSPSetSupportedNorm(ksp, KSP_NORM_NATURAL, PC_LEFT, 2));
  PetscCall(KSPSetSupportedNorm(ksp, KSP_NORM_NONE, PC_LEFT, 1));

  ksp->ops->setup          = KSPSetUp_PIPECGRR;
  ksp->ops->solve          = KSPSolve_PIPECGRR;
  ksp->ops->destroy        = KSPDestroyDefault;
  ksp->ops->view           = NULL;
  ksp->ops->setfromoptions = NULL;
  ksp->ops->buildsolution  = KSPBuildSolutionDefault;
  ksp->ops->buildresidual  = KSPBuildResidualDefault;
  PetscFunctionReturn(PETSC_SUCCESS);
}
