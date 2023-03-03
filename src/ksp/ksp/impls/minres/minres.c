
#include <petsc/private/kspimpl.h>

typedef struct {
  PetscReal haptol;
} KSP_MINRES;

static PetscErrorCode KSPSetUp_MINRES(KSP ksp)
{
  PetscFunctionBegin;
  PetscCall(KSPSetWorkVecs(ksp, 9));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#define KSPMinresSwap3(V1, V2, V3) \
  do { \
    Vec T = V1; \
    V1    = V2; \
    V2    = V3; \
    V3    = T; \
  } while (0)

static PetscErrorCode KSPSolve_MINRES(KSP ksp)
{
  PetscInt          i;
  PetscScalar       alpha, beta, betaold, eta, c = 1.0, ceta, cold = 1.0, coold, s = 0.0, sold = 0.0, soold;
  PetscScalar       rho0, rho1, rho2, rho3, dp = 0.0;
  const PetscScalar none = -1.0;
  PetscReal         np;
  Vec               X, B, R, Z, U, V, W, UOLD, VOLD, WOLD, WOOLD;
  Mat               Amat;
  KSP_MINRES       *minres = (KSP_MINRES *)ksp->data;
  PetscBool         diagonalscale;

  PetscFunctionBegin;
  PetscCall(PCGetDiagonalScale(ksp->pc, &diagonalscale));
  PetscCheck(!diagonalscale, PetscObjectComm((PetscObject)ksp), PETSC_ERR_SUP, "Krylov method %s does not support diagonal scaling", ((PetscObject)ksp)->type_name);

  X     = ksp->vec_sol;
  B     = ksp->vec_rhs;
  R     = ksp->work[0];
  Z     = ksp->work[1];
  U     = ksp->work[2];
  V     = ksp->work[3];
  W     = ksp->work[4];
  UOLD  = ksp->work[5];
  VOLD  = ksp->work[6];
  WOLD  = ksp->work[7];
  WOOLD = ksp->work[8];

  PetscCall(PCGetOperators(ksp->pc, &Amat, NULL));

  ksp->its = 0;

  if (!ksp->guess_zero) {
    PetscCall(KSP_MatMult(ksp, Amat, X, R)); /*     r <- b - A*x    */
    PetscCall(VecAYPX(R, -1.0, B));
  } else {
    PetscCall(VecCopy(B, R)); /*     r <- b (x is 0) */
  }
  PetscCall(KSP_PCApply(ksp, R, Z));  /*     z  <- B*r       */
  PetscCall(VecNorm(Z, NORM_2, &np)); /*   np <- ||z||        */
  KSPCheckNorm(ksp, np);
  PetscCall(VecDot(R, Z, &dp));
  KSPCheckDot(ksp, dp);

  if (PetscRealPart(dp) < minres->haptol && np > minres->haptol) {
    PetscCheck(!ksp->errorifnotconverged, PetscObjectComm((PetscObject)ksp), PETSC_ERR_CONV_FAILED, "Detected indefinite operator %g tolerance %g", (double)PetscRealPart(dp), (double)minres->haptol);
    PetscCall(PetscInfo(ksp, "Detected indefinite operator %g tolerance %g\n", (double)PetscRealPart(dp), (double)minres->haptol));
    ksp->reason = KSP_DIVERGED_INDEFINITE_MAT;
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  ksp->rnorm = 0.0;
  if (ksp->normtype != KSP_NORM_NONE) ksp->rnorm = np;
  PetscCall(KSPLogResidualHistory(ksp, ksp->rnorm));
  PetscCall(KSPMonitor(ksp, 0, ksp->rnorm));
  PetscCall((*ksp->converged)(ksp, 0, ksp->rnorm, &ksp->reason, ksp->cnvP)); /* test for convergence */
  if (ksp->reason) PetscFunctionReturn(PETSC_SUCCESS);

  dp   = PetscAbsScalar(dp);
  dp   = PetscSqrtScalar(dp);
  beta = dp; /*  beta <- sqrt(r'*z  */
  eta  = beta;
  PetscCall(VecAXPBY(V, 1.0 / beta, 0, R)); /* v <- r / beta */
  PetscCall(VecAXPBY(U, 1.0 / beta, 0, Z)); /* u <- z / beta */

  i = 0;
  do {
    ksp->its = i + 1;

    /*   Lanczos  */

    PetscCall(KSP_MatMult(ksp, Amat, U, R)); /*      r <- A*u   */
    PetscCall(VecDot(U, R, &alpha));         /*  alpha <- r'*u  */
    PetscCall(KSP_PCApply(ksp, R, Z));       /*      z <- B*r   */

    if (ksp->its > 1) {
      Vec         T[2];
      PetscScalar alphas[] = {-alpha, -beta};
      /*  r <- r - alpha v - beta v_old    */
      T[0] = V;
      T[1] = VOLD;
      PetscCall(VecMAXPY(R, 2, alphas, T));
      /*  z <- z - alpha u - beta u_old    */
      T[0] = U;
      T[1] = UOLD;
      PetscCall(VecMAXPY(Z, 2, alphas, T));
    } else {
      PetscCall(VecAXPY(R, -alpha, V)); /*  r <- r - alpha v     */
      PetscCall(VecAXPY(Z, -alpha, U)); /*  z <- z - alpha u     */
    }

    betaold = beta;

    PetscCall(VecDot(R, Z, &dp));
    KSPCheckDot(ksp, dp);
    dp   = PetscAbsScalar(dp);
    beta = PetscSqrtScalar(dp); /*  beta <- sqrt(r'*z)   */

    /*    QR factorisation    */

    coold = cold;
    cold  = c;
    soold = sold;
    sold  = s;

    rho0 = cold * alpha - coold * sold * betaold;
    rho1 = PetscSqrtScalar(rho0 * rho0 + beta * beta);
    rho2 = sold * alpha + coold * cold * betaold;
    rho3 = soold * betaold;

    /*     Givens rotation    */

    c = rho0 / rho1;
    s = beta / rho1;

    /* Update */
    /*  w_oold <- w_old */
    /*  w_old  <- w     */
    KSPMinresSwap3(WOOLD, WOLD, W);

    /* w <- (u - rho2 w_old - rho3 w_oold)/rho1 */
    PetscCall(VecAXPBY(W, 1.0 / rho1, 0.0, U));
    if (ksp->its > 1) {
      Vec         T[]      = {WOLD, WOOLD};
      PetscScalar alphas[] = {-rho2 / rho1, -rho3 / rho1};
      PetscInt    nv       = (ksp->its == 2 ? 1 : 2);

      PetscCall(VecMAXPY(W, nv, alphas, T));
    }

    ceta = c * eta;
    PetscCall(VecAXPY(X, ceta, W)); /*  x <- x + c eta w     */

    /*
        when dp is really small we have either convergence or an indefinite operator so compute true
        residual norm to check for convergence
    */
    if (PetscRealPart(dp) < minres->haptol) {
      PetscCall(PetscInfo(ksp, "Possible indefinite operator %g tolerance %g\n", (double)PetscRealPart(dp), (double)minres->haptol));
      PetscCall(KSP_MatMult(ksp, Amat, X, VOLD));
      PetscCall(VecAXPY(VOLD, none, B));
      PetscCall(VecNorm(VOLD, NORM_2, &np));
      KSPCheckNorm(ksp, np);
    } else {
      /* otherwise compute new residual norm via recurrence relation */
      np *= PetscAbsScalar(s);
    }

    if (ksp->normtype != KSP_NORM_NONE) ksp->rnorm = np;
    PetscCall(KSPLogResidualHistory(ksp, ksp->rnorm));
    PetscCall(KSPMonitor(ksp, i + 1, ksp->rnorm));
    PetscCall((*ksp->converged)(ksp, i + 1, ksp->rnorm, &ksp->reason, ksp->cnvP)); /* test for convergence */
    if (ksp->reason) break;

    if (PetscRealPart(dp) < minres->haptol) {
      PetscCheck(!ksp->errorifnotconverged, PetscObjectComm((PetscObject)ksp), PETSC_ERR_CONV_FAILED, "Detected indefinite operator %g tolerance %g", (double)PetscRealPart(dp), (double)minres->haptol);
      PetscCall(PetscInfo(ksp, "Detected indefinite operator %g tolerance %g\n", (double)PetscRealPart(dp), (double)minres->haptol));
      ksp->reason = KSP_DIVERGED_INDEFINITE_MAT;
      break;
    }

    eta = -s * eta;
    KSPMinresSwap3(VOLD, V, R);
    KSPMinresSwap3(UOLD, U, Z);
    PetscCall(VecScale(V, 1.0 / beta)); /* v <- r / beta */
    PetscCall(VecScale(U, 1.0 / beta)); /* u <- z / beta */

    i++;
  } while (i < ksp->max_it);
  if (i >= ksp->max_it) ksp->reason = KSP_DIVERGED_ITS;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*MC
     KSPMINRES - This code implements the MINRES (Minimum Residual) method.

   Level: beginner

   Notes:
   The operator and the preconditioner must be symmetric and the preconditioner must be positive definite for this method.

   Supports only left preconditioning.

   Reference:
. * - Paige & Saunders, Solution of sparse indefinite systems of linear equations, SIAM J. Numer. Anal. 12, 1975.

   Contributed by:
   Robert Scheichl: maprs@maths.bath.ac.uk

.seealso: [](chapter_ksp), `KSPCreate()`, `KSPSetType()`, `KSPType`, `KSP`, `KSPCG`, `KSPCR`
M*/
PETSC_EXTERN PetscErrorCode KSPCreate_MINRES(KSP ksp)
{
  KSP_MINRES *minres;

  PetscFunctionBegin;
  PetscCall(KSPSetSupportedNorm(ksp, KSP_NORM_PRECONDITIONED, PC_LEFT, 3));
  PetscCall(KSPSetSupportedNorm(ksp, KSP_NORM_NONE, PC_LEFT, 1));
  PetscCall(PetscNew(&minres));

  /* this parameter is arbitrary; but e-50 didn't work for __float128 in one example */
#if defined(PETSC_USE_REAL___FLOAT128)
  minres->haptol = 1.e-100;
#elif defined(PETSC_USE_REAL_SINGLE)
  minres->haptol = 1.e-25;
#else
  minres->haptol = 1.e-50;
#endif
  ksp->data = (void *)minres;

  /*
       Sets the functions that are associated with this data structure
       (in C++ this is the same as defining virtual functions)
  */
  ksp->ops->setup          = KSPSetUp_MINRES;
  ksp->ops->solve          = KSPSolve_MINRES;
  ksp->ops->destroy        = KSPDestroyDefault;
  ksp->ops->setfromoptions = NULL;
  ksp->ops->buildsolution  = KSPBuildSolutionDefault;
  ksp->ops->buildresidual  = KSPBuildResidualDefault;
  PetscFunctionReturn(PETSC_SUCCESS);
}
