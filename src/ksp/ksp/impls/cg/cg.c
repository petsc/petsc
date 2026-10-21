/*
    This file implements the conjugate gradient method in PETSc as part of
    KSP. You can use this as a starting point for implementing your own
    Krylov method that is not provided with PETSc.

    The following basic routines are required for each Krylov method.
        KSPCreate_XXX()          - Creates the Krylov context
        KSPSetFromOptions_XXX()  - Sets runtime options
        KSPSolve_XXX()           - Runs the Krylov method
        KSPDestroy_XXX()         - Destroys the Krylov context, freeing all
                                   memory it needed
    Here the "_XXX" denotes a particular implementation, in this case
    we use _CG (e.g. KSPCreate_CG, KSPDestroy_CG). These routines
    are actually called via the common user interface routines
    KSPSetType(), KSPSetFromOptions(), KSPSolve(), and KSPDestroy() so the
    application code interface remains identical for all preconditioners.

    Other basic routines for the KSP objects include
        KSPSetUp_XXX()
        KSPView_XXX()            - Prints details of solver being used.

    Detailed Notes:
    By default, this code implements the CG (Conjugate Gradient) method,
    which is valid for real symmetric (and complex Hermitian) positive
    definite matrices. Note that for the complex Hermitian case, the
    VecDot() arguments within the code MUST remain in the order given
    for correct computation of inner products.

    Reference: Hestenes and Steifel, 1952.

    By switching to the indefinite vector inner product, VecTDot(), the
    same code is used for the complex symmetric case as well.  The user
    must call KSPCGSetType(ksp,KSP_CG_SYMMETRIC) or use the option
    -ksp_cg_type symmetric to invoke this variant for the complex case.
    Note, however, that the complex symmetric code is NOT valid for
    all such matrices ... and thus we don't recommend using this method.
*/
/*
    cgimpl.h defines the simple data structured used to store information
    related to the type of matrix (e.g. complex symmetric) being solved and
    data used during the optional Lanczos process used to compute eigenvalues
*/
#include <../src/ksp/ksp/impls/cg/cgimpl.h> /*I "petscksp.h" I*/
extern PetscErrorCode KSPComputeExtremeSingularValues_CG(KSP, PetscReal *, PetscReal *);
extern PetscErrorCode KSPComputeEigenvalues_CG(KSP, PetscInt, PetscReal *, PetscReal *, PetscInt *);

static PetscErrorCode KSPCGSetObjectiveTarget_CG(KSP ksp, PetscReal obj_min)
{
  KSP_CG *cg = (KSP_CG *)ksp->data;

  PetscFunctionBegin;
  cg->obj_min = obj_min;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode KSPCGSetRadius_CG(KSP ksp, PetscReal radius)
{
  KSP_CG *cg = (KSP_CG *)ksp->data;

  PetscFunctionBegin;
  cg->radius = radius;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode KSPCGGetObjFcn_CG(KSP ksp, PetscReal *obj)
{
  KSP_CG *cg = (KSP_CG *)ksp->data;

  PetscFunctionBegin;
  *obj = cg->obj;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode KSPCGSetTRNormType_CG(KSP ksp, KSPCGTRNormType dtype)
{
  KSP_CG *cg = (KSP_CG *)ksp->data;

  PetscFunctionBegin;
  cg->dtype = dtype;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode KSPCGGetTRNormType_CG(KSP ksp, KSPCGTRNormType *dtype)
{
  KSP_CG *cg = (KSP_CG *)ksp->data;

  PetscFunctionBegin;
  *dtype = cg->dtype;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode KSPCGGetTRSolutionNorm_CG(KSP ksp, PetscReal *norm_d)
{
  KSP_CG *cg = (KSP_CG *)ksp->data;

  PetscFunctionBegin;
  *norm_d = cg->norm_d;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Intersections of x + t p with the trust-region boundary, avoiding cancellation near the boundary. */
static PetscErrorCode KSPCGBoundarySteps_Private(PetscReal norm_p, PetscReal dMp, PetscReal gap, PetscReal *positive, PetscReal *negative)
{
  PetscReal root;

  PetscFunctionBegin;
  *positive = *negative = 0.0;
  if (norm_p > 0.0) {
    gap  = PetscMax(0.0, gap);
    root = PetscSqrtReal(dMp * dMp + norm_p * gap);
    if (dMp >= 0.0) {
      *negative = -(dMp + root) / norm_p;
      if (*negative) *positive = -gap / (norm_p * *negative);
    } else {
      *positive = (root - dMp) / norm_p;
      *negative = -gap / (norm_p * *positive);
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
     KSPSetUp_CG - Sets up the workspace needed by the CG method.

      This is called once, usually automatically by KSPSolve() or KSPSetUp()
     but can be called directly by KSPSetUp()
*/
static PetscErrorCode KSPSetUp_CG(KSP ksp)
{
  KSP_CG  *cgP   = (KSP_CG *)ksp->data;
  PetscInt maxit = ksp->max_it, nwork = 3;

  PetscFunctionBegin;
  /* get work vectors needed by CG */
  if (cgP->singlereduction) nwork += 2;
  PetscCall(KSPSetWorkVecs(ksp, nwork));

  /*
     If user requested computations of eigenvalues then allocate
     work space needed
  */
  if (ksp->calc_sings) {
    PetscCall(PetscFree4(cgP->e, cgP->d, cgP->ee, cgP->dd));
    PetscCall(PetscMalloc4(maxit + 1, &cgP->e, maxit, &cgP->d, maxit, &cgP->ee, maxit, &cgP->dd));

    ksp->ops->computeextremesingularvalues = KSPComputeExtremeSingularValues_CG;
    ksp->ops->computeeigenvalues           = KSPComputeEigenvalues_CG;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
     A macro used in the following KSPSolve_CG and KSPSolve_CG_SingleReduction routines
*/
#define VecXDot(x, y, a) (cg->type == KSP_CG_HERMITIAN ? VecDot(x, y, a) : VecTDot(x, y, a))

/* Return a bounded gradient step when the first preconditioner application is indefinite. */
static PetscErrorCode KSPCGGradientStep_Private(KSP ksp, Mat Amat, PetscReal norm_d)
{
  KSP_CG     *cg = (KSP_CG *)ksp->data;
  Vec         X = ksp->vec_sol, R = ksp->work[0], Z = ksp->work[1];
  PetscScalar rr, rAr, beta;
  PetscReal   xTr = 0.0, positive, negative, step, dp = 0.0;

  PetscFunctionBegin;
  PetscCall(VecXDot(R, R, &rr));
  if (!ksp->guess_zero) PetscCall(VecDotRealPart(X, R, &xTr));
  PetscCall(KSPCGBoundarySteps_Private(PetscRealPart(rr), xTr, PetscSqr(cg->radius) - norm_d, &positive, &negative));
  step = PetscMin(1.0, positive);
  PetscCall(KSP_MatMult(ksp, Amat, R, Z));
  PetscCall(VecXDot(R, Z, &rAr));
  KSPCheckDot(ksp, rAr);
  if (ksp->calc_sings) cg->d[0] = rAr / rr;
  PetscCall(VecAXPY(X, step, R));
  PetscCall(VecAXPY(R, -step, Z));
  PetscCall(VecDotRealPart(X, X, &norm_d));
  cg->norm_d = positive <= 1.0 ? cg->radius : PetscSqrtReal(norm_d);
  cg->obj += PetscRealPart(step * (0.5 * step * rAr - rr));

  switch (ksp->normtype) {
  case KSP_NORM_PRECONDITIONED:
    PetscCall(KSP_PCApply(ksp, R, Z));
    PetscCall(VecNorm(Z, NORM_2, &dp));
    KSPCheckNorm(ksp, dp);
    break;
  case KSP_NORM_UNPRECONDITIONED:
    PetscCall(VecNorm(R, NORM_2, &dp));
    KSPCheckNorm(ksp, dp);
    break;
  case KSP_NORM_NATURAL:
    PetscCall(KSP_PCApply(ksp, R, Z));
    PetscCall(VecXDot(Z, R, &beta));
    KSPCheckDot(ksp, beta);
    dp = PetscSqrtReal(PetscAbsScalar(beta));
    break;
  case KSP_NORM_NONE:
    break;
  default:
    SETERRQ(PetscObjectComm((PetscObject)ksp), PETSC_ERR_SUP, "%s", KSPNormTypes[ksp->normtype]);
  }
  if (cg->obj_min < 0.0) PetscCall(PetscInfo(ksp, "it %" PetscInt_FMT " obj %g\n", ksp->its, (double)cg->obj));
  ksp->rnorm = dp;
  PetscCall(KSPLogResidualHistory(ksp, dp));
  PetscCall(KSPMonitor(ksp, ksp->its, dp));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
     KSPSolve_CG - This routine actually applies the conjugate gradient method

     Note : this routine can be replaced with another one (see below) which implements
            another variant of CG.

   Input Parameter:
.     ksp - the Krylov space object that was set to use conjugate gradient, by, for
            example, KSPCreate(MPI_Comm,KSP *ksp); KSPSetType(ksp,KSPCG);
*/
static PetscErrorCode KSPSolve_CG(KSP ksp)
{
  PetscInt    i, stored_max_it, eigs;
  PetscScalar dpi = 0.0, a = 1.0, beta, betaold = 1.0, b = 0, *e = NULL, *d = NULL, dpiold;
  PetscReal   dp = 0.0;
  PetscReal   r2, norm_p = 0.0, norm_d = 0.0, dMp = 0.0;
  Vec         X, B, Z, R, P, W;
  KSP_CG     *cg;
  Mat         Amat, Pmat;
  PetscBool   testobj, metric, computenorm, boundary = PETSC_FALSE;

  PetscFunctionBegin;
  cg            = (KSP_CG *)ksp->data;
  eigs          = ksp->calc_sings;
  stored_max_it = ksp->max_it;
  X             = ksp->vec_sol;
  B             = ksp->vec_rhs;
  R             = ksp->work[0];
  Z             = ksp->work[1];
  P             = ksp->work[2];
  W             = Z;
  r2            = PetscSqr(cg->radius);
  metric        = (PetscBool)(cg->dtype == KSP_CG_TR_NORM_PRECONDITIONED);
  testobj       = (PetscBool)(cg->obj_min < 0.0);

  PetscCheck(cg->radius >= 0.0, PetscObjectComm((PetscObject)ksp), PETSC_ERR_ARG_OUTOFRANGE, "Trust-region radius must be nonnegative");
  PetscCheck(!metric || ksp->guess_zero, PetscObjectComm((PetscObject)ksp), PETSC_ERR_SUP, "A nonzero initial guess is not supported with the preconditioned trust-region norm");
  PetscCheck(!PetscDefined(USE_COMPLEX) || cg->type == KSP_CG_HERMITIAN || (!cg->radius && !testobj && !metric), PetscObjectComm((PetscObject)ksp), PETSC_ERR_SUP, "Trust-region constraints and objective targets require Hermitian CG for complex scalars");

  if (eigs) {
    e    = cg->e;
    d    = cg->d;
    e[0] = 0.0;
  }
  PetscCall(PCGetOperators(ksp->pc, &Amat, &Pmat));

  ksp->its = 0;
  if (!ksp->guess_zero) {
    PetscCall(KSP_MatMult(ksp, Amat, X, R)); /*    r <- b - Ax                       */

    PetscCall(VecAYPX(R, -1.0, B));
    PetscCall(VecNorm(X, NORM_2, &norm_d));
    PetscCheck(!cg->radius || norm_d <= cg->radius * (1.0 + 10.0 * PETSC_MACHINE_EPSILON), PetscObjectComm((PetscObject)ksp), PETSC_ERR_ARG_OUTOFRANGE, "Initial guess norm %g exceeds trust-region radius %g", (double)norm_d, (double)cg->radius);
    norm_d *= norm_d;
  } else {
    PetscCall(VecCopy(B, R)); /*    r <- b (x is 0)                   */
    norm_d = 0.0;
  }
  cg->norm_d = PetscSqrtReal(norm_d);
  /* This may be true only on a subset of MPI ranks; setting it here so it will be detected by the first norm computation below */
  PetscCall(VecFlag(R, ksp->reason == KSP_DIVERGED_PC_FAILED));

  switch (ksp->normtype) {
  case KSP_NORM_PRECONDITIONED:
    PetscCall(KSP_PCApply(ksp, R, Z));  /*    z <- Br                           */
    PetscCall(VecNorm(Z, NORM_2, &dp)); /*    dp <- z'*z = e'*A'*B'*B*A*e       */
    KSPCheckNorm(ksp, dp);
    break;
  case KSP_NORM_UNPRECONDITIONED:
    PetscCall(VecNorm(R, NORM_2, &dp)); /*    dp <- r'*r = e'*A'*A*e            */
    KSPCheckNorm(ksp, dp);
    break;
  case KSP_NORM_NATURAL:
    PetscCall(KSP_PCApply(ksp, R, Z)); /*    z <- Br                           */
    PetscCall(VecXDot(Z, R, &beta));   /*    beta <- z'*r                      */
    KSPCheckDot(ksp, beta);
    dp = PetscSqrtReal(PetscAbsScalar(beta)); /*    dp <- r'*z = r'*B*r = e'*A'*B*A*e */
    break;
  case KSP_NORM_NONE:
    dp = 0.0;
    break;
  default:
    SETERRQ(PetscObjectComm((PetscObject)ksp), PETSC_ERR_SUP, "%s", KSPNormTypes[ksp->normtype]);
  }

  /* Initialize objective function
     obj = 1/2 x^T A x - x^T b */
  cg->obj = 0.0;
  if (!ksp->guess_zero) {
    PetscCall(VecXDot(R, X, &a));
    cg->obj = -0.5 * PetscRealPart(a);
    PetscCall(VecXDot(B, X, &a));
    cg->obj -= 0.5 * PetscRealPart(a);
  }

  if (testobj) PetscCall(PetscInfo(ksp, "it %" PetscInt_FMT " obj %g\n", ksp->its, (double)cg->obj));
  ksp->rnorm = dp;
  PetscCall(KSPLogResidualHistory(ksp, dp));
  PetscCall(KSPMonitor(ksp, ksp->its, dp));

  PetscCall((*ksp->converged)(ksp, ksp->its, dp, &ksp->reason, ksp->cnvP)); /* test for convergence */

  if (!ksp->reason && testobj && cg->obj <= cg->obj_min) {
    PetscCall(PetscInfo(ksp, "converged to objective target minimum\n"));
    ksp->reason = KSP_CONVERGED_ATOL;
  }

  if (ksp->reason) PetscFunctionReturn(PETSC_SUCCESS);

  if (ksp->normtype != KSP_NORM_PRECONDITIONED && (ksp->normtype != KSP_NORM_NATURAL)) PetscCall(KSP_PCApply(ksp, R, Z)); /*     z <- Br                           */
  if (ksp->normtype != KSP_NORM_NATURAL) {
    PetscCall(VecXDot(Z, R, &beta)); /*     beta <- z'*r                      */
    KSPCheckDot(ksp, beta);
  }

  i = 0;
  do {
    if (beta == 0.0) {
      ksp->reason = KSP_CONVERGED_ATOL;
      PetscCall(PetscInfo(ksp, "converged due to beta = 0\n"));
      break;
    } else if ((metric || cg->radius || testobj) && PetscRealPart(beta) < 0.0) {
      PetscCheck(!ksp->errorifnotconverged, PetscObjectComm((PetscObject)ksp), PETSC_ERR_NOT_CONVERGED, "Trust-region minimization requires a positive definite preconditioner, r^H z = %g", (double)PetscRealPart(beta));
      ksp->reason = KSP_DIVERGED_INDEFINITE_PC;
      if (!i && !metric && cg->radius) {
        ksp->its = i + 1;
        PetscCall(KSPCGGradientStep_Private(ksp, Amat, norm_d));
      }
      break;
#if !PetscDefined(USE_COMPLEX)
    } else if (i > 0 && beta * betaold < 0.0) {
      PetscCheck(!ksp->errorifnotconverged, PetscObjectComm((PetscObject)ksp), PETSC_ERR_NOT_CONVERGED, "Diverged due to indefinite preconditioner, beta %g, betaold %g", (double)PetscRealPart(beta), (double)PetscRealPart(betaold));
      ksp->reason = KSP_DIVERGED_INDEFINITE_PC;
      PetscCall(PetscInfo(ksp, "diverging due to indefinite preconditioner\n"));
      break;
#endif
    }
    ksp->its = i + 1;
    if (!i) {
      PetscCall(VecCopy(Z, P)); /*     p <- z                           */
      if (metric) norm_p = PetscRealPart(beta);
      else if (cg->radius) {
        PetscCall(VecDotRealPart(P, P, &norm_p));
        dMp = 0.0;
        if (!ksp->guess_zero) PetscCall(VecDotRealPart(X, P, &dMp));
      }
      b = 0.0;
    } else {
      b = beta / betaold;
      if (eigs) {
        PetscCheck(ksp->max_it == stored_max_it, PetscObjectComm((PetscObject)ksp), PETSC_ERR_SUP, "Can not change maxit AND calculate eigenvalues");
        e[i] = PetscSqrtReal(PetscAbsScalar(b)) / a;
      }
      PetscCall(VecAYPX(P, b, Z)); /*     p <- z + b* p                    */
      if (metric) {
        dMp    = PetscRealPart(b) * (dMp + PetscRealPart(a) * norm_p);
        norm_p = PetscRealPart(beta) + PetscSqr(PetscRealPart(b)) * norm_p;
      } else if (cg->radius) {
        PetscCall(VecDotRealPart(X, P, &dMp));
        PetscCall(VecDotRealPart(P, P, &norm_p));
      }
    }
    dpiold = dpi;
    PetscCall(KSP_MatMult(ksp, Amat, P, W)); /*     w <- Ap                          */
    PetscCall(VecXDot(P, W, &dpi));          /*     dpi <- p'w                       */
    KSPCheckDot(ksp, dpi);
    betaold = beta;

    if (eigs) d[i] = PetscSqrtReal(PetscAbsScalar(b)) * e[i] + dpi / beta;
    /* Stop for nonpositive curvature if requested or a trust-region radius is active.
       Always stop at zero curvature (avoiding division by zero) or a curvature sign
       change (indicating an indefinite matrix). */
    if (((ksp->converged_neg_curve || cg->radius) && PetscRealPart(dpi) <= 0.0) || (dpi == 0.0) || ((i > 0) && ((PetscSign(PetscRealPart(dpi)) * PetscSign(PetscRealPart(dpiold))) < 0.0))) {
      a = 0.0;
      if (cg->radius) {
        PetscReal positive, negative;

        PetscCall(KSPCGBoundarySteps_Private(norm_p, dMp, r2 - norm_d, &positive, &negative));
        a = positive;
        if (PetscRealPart(negative * (0.5 * negative * dpi - beta)) < PetscRealPart(positive * (0.5 * positive * dpi - beta))) a = negative;
        boundary = (PetscBool)(norm_p > 0.0);
      }
      if (ksp->converged_neg_curve) {
        PetscCall(PetscInfo(ksp, "converged due to negative curvature: %g\n", (double)(PetscRealPart(dpi))));
        ksp->reason = KSP_CONVERGED_NEG_CURVE;
      } else if (cg->radius) {
        PetscCall(PetscInfo(ksp, "converged to the trust region radius %g along a nonpositive curvature direction\n", (double)cg->radius));
        ksp->reason = KSP_CONVERGED_STEP_LENGTH;
      } else {
        PetscCheck(!ksp->errorifnotconverged, PetscObjectComm((PetscObject)ksp), PETSC_ERR_NOT_CONVERGED, "Diverged due to indefinite matrix, dpi %g, dpiold %g", (double)PetscRealPart(dpi), (double)PetscRealPart(dpiold));
        ksp->reason = KSP_DIVERGED_INDEFINITE_MAT;
        PetscCall(PetscInfo(ksp, "diverging due to indefinite matrix\n"));
      }
    } else {
      PetscCheck(PetscRealPart(dpi) >= 0.0 || !testobj, PetscObjectComm((PetscObject)ksp), PETSC_ERR_SUP, "An objective target is not supported when continuing CG with negative curvature");
      a = beta / dpi; /*     a = beta/p'w                     */
      if (cg->radius) {
        PetscReal norm_dp1 = norm_d + PetscRealPart(a) * (2.0 * dMp + PetscRealPart(a) * norm_p);

        if (norm_dp1 >= r2) {
          PetscReal positive, negative;

          ksp->reason = KSP_CONVERGED_STEP_LENGTH;
          PetscCall(PetscInfo(ksp, "converged to the trust region radius %g\n", (double)cg->radius));
          PetscCall(KSPCGBoundarySteps_Private(norm_p, dMp, r2 - norm_d, &positive, &negative));
          a        = positive;
          boundary = (PetscBool)(norm_p > 0.0);
        }
      }
    }
    PetscCall(VecAXPY(X, a, P));  /*     x <- x + ap                      */
    PetscCall(VecAXPY(R, -a, W)); /*     r <- r - aw                      */
    if (metric) norm_d = PetscMax(0.0, norm_d + PetscRealPart(a) * (2.0 * dMp + PetscRealPart(a) * norm_p));
    else if (cg->radius) PetscCall(VecDotRealPart(X, X, &norm_d));
    if (metric || cg->radius) cg->norm_d = PetscSqrtReal(norm_d);
    /* TAO uses exact equality with the radius to recognize a boundary step. */
    if (boundary) cg->norm_d = cg->radius;
    cg->obj += PetscRealPart(a * (0.5 * a * dpi - betaold));
    computenorm = (PetscBool)(ksp->reason || ksp->chknorm < i + 2 || i + 1 == ksp->max_it || (testobj && cg->obj <= cg->obj_min));
    if (ksp->normtype == KSP_NORM_PRECONDITIONED && computenorm) {
      PetscCall(KSP_PCApply(ksp, R, Z));  /*     z <- Br                          */
      PetscCall(VecNorm(Z, NORM_2, &dp)); /*     dp <- z'*z                       */
      KSPCheckNorm(ksp, dp);
    } else if (ksp->normtype == KSP_NORM_UNPRECONDITIONED && computenorm) {
      PetscCall(VecNorm(R, NORM_2, &dp)); /*     dp <- r'*r                       */
      KSPCheckNorm(ksp, dp);
    } else if (ksp->normtype == KSP_NORM_NATURAL) {
      PetscCall(KSP_PCApply(ksp, R, Z)); /*     z <- Br                          */
      PetscCall(VecXDot(Z, R, &beta));   /*     beta <- r'*z                     */
      KSPCheckDot(ksp, beta);
      dp = PetscSqrtReal(PetscAbsScalar(beta));
    } else {
      dp = 0.0;
    }
    if (testobj) PetscCall(PetscInfo(ksp, "it %" PetscInt_FMT " obj %g\n", i + 1, (double)cg->obj));

    ksp->rnorm = dp;
    PetscCall(KSPLogResidualHistory(ksp, dp));
    PetscCall(KSPMonitor(ksp, i + 1, dp));
    if (!ksp->reason) PetscCall((*ksp->converged)(ksp, i + 1, dp, &ksp->reason, ksp->cnvP));

    if (!ksp->reason && testobj && cg->obj <= cg->obj_min) {
      PetscCall(PetscInfo(ksp, "converged to objective target minimum\n"));
      ksp->reason = KSP_CONVERGED_ATOL;
    }

    if (ksp->reason) break;

    if ((ksp->normtype != KSP_NORM_PRECONDITIONED && ksp->normtype != KSP_NORM_NATURAL) || ksp->chknorm >= i + 2) PetscCall(KSP_PCApply(ksp, R, Z)); /*     z <- Br                          */
    if (ksp->normtype != KSP_NORM_NATURAL || ksp->chknorm >= i + 2) {
      PetscCall(VecXDot(Z, R, &beta)); /*     beta <- z'*r                     */
      KSPCheckDot(ksp, beta);
    }

    i++;
  } while (i < ksp->max_it);
  if (i >= ksp->max_it) ksp->reason = KSP_DIVERGED_ITS;
  if (!metric && !cg->radius) PetscCall(VecNorm(X, NORM_2, &cg->norm_d));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
       KSPSolve_CG_SingleReduction

       This variant of CG is identical in exact arithmetic to the standard algorithm,
       but is rearranged to use only a single reduction stage per iteration, using additional
       intermediate vectors.

       See KSPCGUseSingleReduction_CG()

*/
static PetscErrorCode KSPSolve_CG_SingleReduction(KSP ksp)
{
  PetscInt    i, stored_max_it, eigs;
  PetscScalar dpi = 0.0, a = 1.0, beta, betaold = 1.0, b = 0, *e = NULL, *d = NULL, delta, dpiold, tmp[2];
  PetscReal   dp = 0.0;
  Vec         X, B, Z, R, P, S, W, tmpvecs[2];
  KSP_CG     *cg;
  Mat         Amat, Pmat;

  PetscFunctionBegin;
  PetscCheck(ksp->nwork == 5, PetscObjectComm((PetscObject)ksp), PETSC_ERR_COR, "Unexpected number of work vectors %" PetscInt_FMT " != 5", ksp->nwork);
  cg            = (KSP_CG *)ksp->data;
  eigs          = ksp->calc_sings;
  stored_max_it = ksp->max_it;
  X             = ksp->vec_sol;
  B             = ksp->vec_rhs;
  R             = ksp->work[0];
  Z             = ksp->work[1];
  P             = ksp->work[2];
  S             = ksp->work[3];
  W             = ksp->work[4];

  PetscCheck(!cg->radius, PetscObjectComm((PetscObject)ksp), PETSC_ERR_SUP, "Single-reduction CG does not support a trust-region radius");
  PetscCheck(cg->obj_min >= 0.0, PetscObjectComm((PetscObject)ksp), PETSC_ERR_SUP, "Single-reduction CG does not support an objective target");
  PetscCheck(!ksp->converged_neg_curve, PetscObjectComm((PetscObject)ksp), PETSC_ERR_SUP, "Single-reduction CG does not support convergence at negative curvature");
  PetscCheck(cg->dtype == KSP_CG_TR_NORM_UNPRECONDITIONED, PetscObjectComm((PetscObject)ksp), PETSC_ERR_SUP, "Single-reduction CG does not support the preconditioned trust-region norm");

  if (eigs) {
    e    = cg->e;
    d    = cg->d;
    e[0] = 0.0;
  }
  PetscCall(PCGetOperators(ksp->pc, &Amat, &Pmat));

  ksp->its = 0;
  if (!ksp->guess_zero) {
    PetscCall(KSP_MatMult(ksp, Amat, X, R)); /*    r <- b - Ax                       */
    PetscCall(VecAYPX(R, -1.0, B));
  } else {
    PetscCall(VecCopy(B, R)); /*    r <- b (x is 0)                   */
  }

  switch (ksp->normtype) {
  case KSP_NORM_PRECONDITIONED:
    PetscCall(KSP_PCApply(ksp, R, Z));  /*    z <- Br                           */
    PetscCall(VecNorm(Z, NORM_2, &dp)); /*    dp <- z'*z = e'*A'*B'*B*A'*e'     */
    KSPCheckNorm(ksp, dp);
    break;
  case KSP_NORM_UNPRECONDITIONED:
    PetscCall(VecNorm(R, NORM_2, &dp)); /*    dp <- r'*r = e'*A'*A*e            */
    KSPCheckNorm(ksp, dp);
    break;
  case KSP_NORM_NATURAL:
    PetscCall(KSP_PCApply(ksp, R, Z)); /*    z <- Br                           */
    PetscCall(KSP_MatMult(ksp, Amat, Z, S));
    PetscCall(VecXDot(Z, S, &delta)); /*    delta <- z'*A*z = r'*B*A*B*r      */
    PetscCall(VecXDot(Z, R, &beta));  /*    beta <- z'*r                      */
    KSPCheckDot(ksp, beta);
    dp = PetscSqrtReal(PetscAbsScalar(beta)); /*    dp <- r'*z = r'*B*r = e'*A'*B*A*e */
    break;
  case KSP_NORM_NONE:
    dp = 0.0;
    break;
  default:
    SETERRQ(PetscObjectComm((PetscObject)ksp), PETSC_ERR_SUP, "%s", KSPNormTypes[ksp->normtype]);
  }
  PetscCall(KSPLogResidualHistory(ksp, dp));
  PetscCall(KSPMonitor(ksp, 0, dp));
  ksp->rnorm = dp;

  PetscCall((*ksp->converged)(ksp, 0, dp, &ksp->reason, ksp->cnvP)); /* test for convergence */
  if (ksp->reason) PetscFunctionReturn(PETSC_SUCCESS);

  if (ksp->normtype != KSP_NORM_PRECONDITIONED && (ksp->normtype != KSP_NORM_NATURAL)) PetscCall(KSP_PCApply(ksp, R, Z)); /*    z <- Br                           */
  if (ksp->normtype != KSP_NORM_NATURAL) {
    PetscCall(KSP_MatMult(ksp, Amat, Z, S));
    PetscCall(VecXDot(Z, S, &delta)); /*    delta <- z'*A*z = r'*B*A*B*r      */
    PetscCall(VecXDot(Z, R, &beta));  /*    beta <- z'*r                      */
    KSPCheckDot(ksp, beta);
  }

  i = 0;
  do {
    ksp->its = i + 1;
    if (beta == 0.0) {
      ksp->reason = KSP_CONVERGED_ATOL;
      PetscCall(PetscInfo(ksp, "converged due to beta = 0\n"));
      break;
#if !PetscDefined(USE_COMPLEX)
    } else if (i > 0 && beta * betaold < 0.0) {
      PetscCheck(!ksp->errorifnotconverged, PetscObjectComm((PetscObject)ksp), PETSC_ERR_NOT_CONVERGED, "Diverged due to indefinite preconditioner");
      ksp->reason = KSP_DIVERGED_INDEFINITE_PC;
      PetscCall(PetscInfo(ksp, "diverging due to indefinite preconditioner\n"));
      break;
#endif
    }
    if (!i) {
      PetscCall(VecCopy(Z, P)); /*    p <- z                           */
      b = 0.0;
    } else {
      b = beta / betaold;
      if (eigs) {
        PetscCheck(ksp->max_it == stored_max_it, PetscObjectComm((PetscObject)ksp), PETSC_ERR_SUP, "Can not change maxit AND calculate eigenvalues");
        e[i] = PetscSqrtReal(PetscAbsScalar(b)) / a;
      }
      PetscCall(VecAYPX(P, b, Z)); /*    p <- z + b* p                     */
    }
    dpiold = dpi;
    if (!i) {
      PetscCall(KSP_MatMult(ksp, Amat, P, W)); /*    w <- Ap                           */
      PetscCall(VecXDot(P, W, &dpi));          /*    dpi <- p'w                        */
    } else {
      PetscCall(VecAYPX(W, beta / betaold, S));                 /*    w <- Ap                           */
      dpi = delta - beta * beta * dpiold / (betaold * betaold); /*    dpi <- p'w                        */
    }
    betaold = beta;
    KSPCheckDot(ksp, beta);

    if ((dpi == 0.0) || ((i > 0) && (PetscRealPart(dpi * dpiold) <= 0.0))) {
      PetscCheck(!ksp->errorifnotconverged, PetscObjectComm((PetscObject)ksp), PETSC_ERR_NOT_CONVERGED, "Diverged due to indefinite matrix");
      ksp->reason = KSP_DIVERGED_INDEFINITE_MAT;
      PetscCall(PetscInfo(ksp, "diverging due to indefinite or negative definite matrix\n"));
      break;
    }
    a = beta / dpi; /*    a = beta/p'w                      */
    if (eigs) d[i] = PetscSqrtReal(PetscAbsScalar(b)) * e[i] + 1.0 / a;
    PetscCall(VecAXPY(X, a, P));  /*    x <- x + ap                       */
    PetscCall(VecAXPY(R, -a, W)); /*    r <- r - aw                       */
    if (ksp->normtype == KSP_NORM_PRECONDITIONED && ksp->chknorm < i + 2) {
      PetscCall(KSP_PCApply(ksp, R, Z)); /*    z <- Br                           */
      PetscCall(KSP_MatMult(ksp, Amat, Z, S));
      PetscCall(VecNorm(Z, NORM_2, &dp)); /*    dp <- z'*z                        */
      KSPCheckNorm(ksp, dp);
    } else if (ksp->normtype == KSP_NORM_UNPRECONDITIONED && ksp->chknorm < i + 2) {
      PetscCall(VecNorm(R, NORM_2, &dp)); /*    dp <- r'*r                        */
      KSPCheckNorm(ksp, dp);
    } else if (ksp->normtype == KSP_NORM_NATURAL) {
      PetscCall(KSP_PCApply(ksp, R, Z)); /*    z <- Br                           */
      tmpvecs[0] = S;
      tmpvecs[1] = R;
      PetscCall(KSP_MatMult(ksp, Amat, Z, S));
      PetscCall(VecMDot(Z, 2, tmpvecs, tmp)); /*    delta <- z'*A*z = r'*B*A*B*r      */
      delta = tmp[0];
      beta  = tmp[1]; /*    beta <- z'*r                      */
      KSPCheckDot(ksp, beta);
      dp = PetscSqrtReal(PetscAbsScalar(beta)); /*    dp <- r'*z = r'*B*r = e'*A'*B*A*e */
    } else {
      dp = 0.0;
    }
    ksp->rnorm = dp;
    PetscCall(KSPLogResidualHistory(ksp, dp));
    PetscCall(KSPMonitor(ksp, i + 1, dp));
    PetscCall((*ksp->converged)(ksp, i + 1, dp, &ksp->reason, ksp->cnvP));
    if (ksp->reason) break;

    if ((ksp->normtype != KSP_NORM_PRECONDITIONED && ksp->normtype != KSP_NORM_NATURAL) || ksp->chknorm >= i + 2) {
      PetscCall(KSP_PCApply(ksp, R, Z)); /*    z <- Br                           */
      PetscCall(KSP_MatMult(ksp, Amat, Z, S));
    }
    if (ksp->normtype != KSP_NORM_NATURAL || ksp->chknorm >= i + 2) {
      tmpvecs[0] = S;
      tmpvecs[1] = R;
      PetscCall(VecMDot(Z, 2, tmpvecs, tmp));
      delta = tmp[0];
      beta  = tmp[1];         /*    delta <- z'*A*z = r'*B'*A*B*r     */
      KSPCheckDot(ksp, beta); /*    beta <- z'*r                      */
    }

    i++;
  } while (i < ksp->max_it);
  if (i >= ksp->max_it) ksp->reason = KSP_DIVERGED_ITS;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
     KSPDestroy_CG - Frees resources allocated in KSPSetup_CG and clears function
                     compositions from KSPCreate_CG. If adding your own KSP implementation,
                     you must be sure to free all allocated resources here to prevent
                     leaks.
*/
PetscErrorCode KSPDestroy_CG(KSP ksp)
{
  KSP_CG *cg = (KSP_CG *)ksp->data;

  PetscFunctionBegin;
  PetscCall(PetscFree4(cg->e, cg->d, cg->ee, cg->dd));
  PetscCall(KSPDestroyDefault(ksp));
  PetscCall(PetscObjectComposeFunction((PetscObject)ksp, "KSPCGSetObjectiveTarget_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)ksp, "KSPCGSetRadius_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)ksp, "KSPCGSetType_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)ksp, "KSPCGUseSingleReduction_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)ksp, "KSPCGGetObjFcn_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)ksp, "KSPCGSetTRNormType_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)ksp, "KSPCGGetTRNormType_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)ksp, "KSPCGGetTRSolutionNorm_C", NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
     KSPView_CG - Prints information about the current Krylov method being used.
                  If your Krylov method has special options or flags that information
                  should be printed here.
*/
PetscErrorCode KSPView_CG(KSP ksp, PetscViewer viewer)
{
  KSP_CG   *cg = (KSP_CG *)ksp->data;
  PetscBool isascii;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &isascii));
  if (isascii) {
    if (PetscDefined(USE_COMPLEX)) PetscCall(PetscViewerASCIIPrintf(viewer, "  variant %s\n", KSPCGTypes[cg->type]));
    if (cg->singlereduction) PetscCall(PetscViewerASCIIPrintf(viewer, "  using single-reduction variant\n"));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
    KSPSetFromOptions_CG - Checks the options database for options related to the
                           conjugate gradient method.
*/
PetscErrorCode KSPSetFromOptions_CG(KSP ksp, PetscOptionItems PetscOptionsObject)
{
  KSP_CG   *cg = (KSP_CG *)ksp->data;
  PetscBool flg, flg2, hasradius;

  PetscFunctionBegin;
  PetscOptionsHeadBegin(PetscOptionsObject, "KSP CG and CGNE options");
  if (PetscDefined(USE_COMPLEX)) PetscCall(PetscOptionsEnum("-ksp_cg_type", "Matrix is Hermitian or complex symmetric", "KSPCGSetType", KSPCGTypes, (PetscEnum)cg->type, (PetscEnum *)&cg->type, NULL));
  PetscCall(PetscOptionsBool("-ksp_cg_single_reduction", "Merge inner products into single MPI_Allreduce()", "KSPCGUseSingleReduction", cg->singlereduction, &flg2, &flg));
  if (flg) PetscCall(KSPCGUseSingleReduction(ksp, flg2));
  PetscCall(PetscObjectHasFunction((PetscObject)ksp, "KSPCGSetRadius_C", &hasradius));
  if (hasradius) {
    KSPCGTRNormType dtype  = cg->dtype;
    PetscReal       radius = cg->radius;

    PetscCall(PetscOptionsReal("-ksp_cg_radius", "Trust-region radius", "KSPCGSetRadius", radius, &radius, &flg));
    if (flg) PetscCall(KSPCGSetRadius(ksp, radius));
    PetscCall(PetscOptionsDeprecated("-ksp_cg_dtype", "-ksp_cg_tr_norm_type", "3.27", NULL));
    PetscCall(PetscOptionsEnum("-ksp_cg_tr_norm_type", "Norm used for the trust-region constraint", "KSPCGSetTRNormType", KSPCGTRNormTypes, (PetscEnum)dtype, (PetscEnum *)&dtype, &flg));
    if (flg) PetscCall(KSPCGSetTRNormType(ksp, dtype));
  }
  PetscOptionsHeadEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
    KSPCGSetType_CG - This is an option that is SPECIFIC to this particular Krylov method.
                      This routine is registered below in KSPCreate_CG() and called from the
                      routine KSPCGSetType() (see the file cgtype.c).
*/
PetscErrorCode KSPCGSetType_CG(KSP ksp, KSPCGType type)
{
  KSP_CG *cg = (KSP_CG *)ksp->data;

  PetscFunctionBegin;
  cg->type = type;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
    KSPCGUseSingleReduction_CG

    This routine sets a flag to use a variant of CG. Note that (in somewhat
    atypical fashion) it also swaps out the routine called when KSPSolve()
    is invoked.
*/
static PetscErrorCode KSPCGUseSingleReduction_CG(KSP ksp, PetscBool flg)
{
  KSP_CG *cg = (KSP_CG *)ksp->data;

  PetscFunctionBegin;
  if (cg->singlereduction != flg) ksp->setupstage = KSP_SETUP_NEW;
  cg->singlereduction = flg;
  if (cg->singlereduction) {
    ksp->ops->solve = KSPSolve_CG_SingleReduction;
    PetscCall(PetscObjectComposeFunction((PetscObject)ksp, "KSPCGGetObjFcn_C", NULL));
    PetscCall(PetscObjectComposeFunction((PetscObject)ksp, "KSPCGGetTRSolutionNorm_C", NULL));
  } else {
    ksp->ops->solve = KSPSolve_CG;
    PetscCall(PetscObjectComposeFunction((PetscObject)ksp, "KSPCGGetObjFcn_C", KSPCGGetObjFcn_CG));
    PetscCall(PetscObjectComposeFunction((PetscObject)ksp, "KSPCGGetTRSolutionNorm_C", KSPCGGetTRSolutionNorm_CG));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode KSPBuildResidual_CG(KSP ksp, Vec t, Vec v, Vec *V)
{
  PetscFunctionBegin;
  PetscCall(VecCopy(ksp->work[0], v));
  *V = v;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*MC
   KSPCG - The Preconditioned Conjugate Gradient (PCG) iterative method {cite}`hs:52` and {cite}`malek2014preconditioning` for solving linear systems using `KSP`.

   Options Database Keys:
+   -ksp_cg_type (hermitian|symmetric)                     - (for complex matrices only) indicates the matrix is Hermitian or symmetric, see `KSPCGSetType()`
.   -ksp_cg_single_reduction                               - performs both inner products needed in the algorithm with a single `MPI_Allreduce()` call, see `KSPCGUseSingleReduction()`
.   -ksp_cg_radius radius                                  - trust-region radius, see `KSPCGSetRadius()`
-   -ksp_cg_tr_norm_type (preconditioned|unpreconditioned) - norm used for the trust-region constraint, see `KSPCGSetTRNormType()`

   Level: beginner

   Notes:
   The `KSPCG` method requires both the matrix and preconditioner to be symmetric positive (or negative) (semi) definite.

   `KSPCG` is the best Krylov method, `KSPType`, when the matrix and preconditioner are symmetric positive definite (SPD).

   By default, consistently negative curvature is allowed when solving a negative definite system; a change of curvature sign indicates an indefinite matrix.
   An active trust-region radius makes CG stop at the first nonpositive curvature direction, take a boundary step, and return `KSP_CONVERGED_STEP_LENGTH`.
   With `KSPSetConvergedNegativeCurvature()`, CG instead stops at the first nonpositive curvature direction and returns `KSP_CONVERGED_NEG_CURVE`.
   When an objective target is enabled with `KSPCGSetObjectiveTarget()`, continuing ordinary CG through negative curvature is unsupported.
   Enable convergence on negative curvature with `KSPSetConvergedNegativeCurvature()` or provide a trust-region radius with `KSPCGSetRadius()`.
   The reported quadratic objective remains $\frac{1}{2} x^H A x - b^H x$; for a negative definite matrix the solution maximizes this quadratic.

   `KSPSTCG` uses this implementation with convergence at negative curvature enabled and the unpreconditioned residual norm by default.
   Trust-region minimization requires a fixed symmetric positive definite preconditioner. Nonzero initial guesses must be feasible and
   are supported only with the Euclidean norm. Complex trust-region problems require the Hermitian variant.

   Only left preconditioning is supported with `KSPCG`; there are several ways to motivate preconditioned CG, but they all produce the same algorithm.
   One can interpret preconditioning $A$ with $B$ to mean any of the following\:
.vb
   (1) Solve a left-preconditioned system $BAx = Bb$, using $B^{-1}$ to define an inner product in the algorithm.
   (2) Solve a right-preconditioned system $ABy = b, x = By,$ using $B$ to define an inner product in the algorithm.
   (3) Solve a symmetrically-preconditioned system, $E^TAEy = E^Tb, x = Ey,$ where $B = EE^T.$
   (4) Solve $Ax=b$ with CG, but use the inner product defined by $B$ to define the method.
   In all cases, the resulting algorithm only requires application of $B$ to vectors, the other inner-product does not appear explicitly in the code
.ve

   For complex numbers there are two different CG methods, one for Hermitian symmetric matrices and one for non-Hermitian symmetric matrices. Use
   `KSPCGSetType()` to indicate which type you are using.

   One can use `KSPSetComputeEigenvalues()` and `KSPComputeEigenvalues()` to compute the eigenvalues of the (preconditioned) operator

   There are two pipelined implementations of CG in PETSc `KSPPIPECG` and `KSPGROPPCG`. These may perform better for very large
   numbers of MPI processes since they overlap communication and computation so the reduction operations in CG, that is inner products and norms,
   do not dominate the compute time.

   Developer Note:
   KSPSolve_CG() should actually query the matrix to determine if it is Hermitian or symmetric and NOT require the user to
   indicate it to the `KSP` object.

.seealso: [](ch_ksp), `KSPCreate()`, `KSPSetType()`, `KSPType`, `KSP`, `KSPSetComputeEigenvalues()`, `KSPComputeEigenvalues()`,
          `KSPCGSetType()`, `KSPCGUseSingleReduction()`, `KSPCGSetRadius()`, `KSPCGSetTRNormType()`, `KSPSTCG`, `KSPPIPECG`, `KSPGROPPCG`
M*/

/*
    KSPCreate_CG - Creates the data structure for the Krylov method CG and sets the
       function pointers for all the routines it needs to call (KSPSolve_CG() etc)

    It must be labeled as PETSC_EXTERN to be dynamically linkable in C++
*/
PETSC_EXTERN PetscErrorCode KSPCreate_CG(KSP ksp)
{
  KSP_CG *cg;

  PetscFunctionBegin;
  PetscCall(PetscNew(&cg));
  cg->type    = !PetscDefined(USE_COMPLEX) ? KSP_CG_SYMMETRIC : KSP_CG_HERMITIAN;
  cg->dtype   = KSP_CG_TR_NORM_UNPRECONDITIONED;
  cg->obj_min = 0.0;
  ksp->data   = (void *)cg;

  PetscCall(KSPSetSupportedNorm(ksp, KSP_NORM_PRECONDITIONED, PC_LEFT, 3));
  PetscCall(KSPSetSupportedNorm(ksp, KSP_NORM_UNPRECONDITIONED, PC_LEFT, 2));
  PetscCall(KSPSetSupportedNorm(ksp, KSP_NORM_NATURAL, PC_LEFT, 2));
  PetscCall(KSPSetSupportedNorm(ksp, KSP_NORM_NONE, PC_LEFT, 1));

  /*
       Sets the functions that are associated with this data structure
       (in C++ this is the same as defining virtual functions)
  */
  ksp->ops->setup          = KSPSetUp_CG;
  ksp->ops->solve          = KSPSolve_CG;
  ksp->ops->destroy        = KSPDestroy_CG;
  ksp->ops->view           = KSPView_CG;
  ksp->ops->setfromoptions = KSPSetFromOptions_CG;
  ksp->ops->buildsolution  = KSPBuildSolutionDefault;
  ksp->ops->buildresidual  = KSPBuildResidual_CG;

  /*
      Attach the function KSPCGSetType_CG() to this object. The routine
      KSPCGSetType() checks for this attached function and calls it if it finds
      it. (Sort of like a dynamic member function that can be added at run time
  */
  PetscCall(PetscObjectComposeFunction((PetscObject)ksp, "KSPCGSetType_C", KSPCGSetType_CG));
  PetscCall(PetscObjectComposeFunction((PetscObject)ksp, "KSPCGUseSingleReduction_C", KSPCGUseSingleReduction_CG));
  PetscCall(PetscObjectComposeFunction((PetscObject)ksp, "KSPCGSetRadius_C", KSPCGSetRadius_CG));
  PetscCall(PetscObjectComposeFunction((PetscObject)ksp, "KSPCGSetObjectiveTarget_C", KSPCGSetObjectiveTarget_CG));
  PetscCall(PetscObjectComposeFunction((PetscObject)ksp, "KSPCGGetObjFcn_C", KSPCGGetObjFcn_CG));
  PetscCall(PetscObjectComposeFunction((PetscObject)ksp, "KSPCGSetTRNormType_C", KSPCGSetTRNormType_CG));
  PetscCall(PetscObjectComposeFunction((PetscObject)ksp, "KSPCGGetTRNormType_C", KSPCGGetTRNormType_CG));
  PetscCall(PetscObjectComposeFunction((PetscObject)ksp, "KSPCGGetTRSolutionNorm_C", KSPCGGetTRSolutionNorm_CG));
  PetscFunctionReturn(PETSC_SUCCESS);
}
