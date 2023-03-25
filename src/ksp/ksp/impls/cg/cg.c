
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
    data used during the optional Lanczo process used to compute eigenvalues
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
    PetscCall(PetscMalloc4(maxit, &cgP->e, maxit, &cgP->d, maxit, &cgP->ee, maxit, &cgP->dd));

    ksp->ops->computeextremesingularvalues = KSPComputeExtremeSingularValues_CG;
    ksp->ops->computeeigenvalues           = KSPComputeEigenvalues_CG;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
     A macro used in the following KSPSolve_CG and KSPSolve_CG_SingleReduction routines
*/
#define VecXDot(x, y, a) (((cg->type) == (KSP_CG_HERMITIAN)) ? VecDot(x, y, a) : VecTDot(x, y, a))

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
  PetscReal   r2, norm_p, norm_d, dMp;
  Vec         X, B, Z, R, P, W;
  KSP_CG     *cg;
  Mat         Amat, Pmat;
  PetscBool   diagonalscale, testobj;

  PetscFunctionBegin;
  PetscCall(PCGetDiagonalScale(ksp->pc, &diagonalscale));
  PetscCheck(!diagonalscale, PetscObjectComm((PetscObject)ksp), PETSC_ERR_SUP, "Krylov method %s does not support diagonal scaling", ((PetscObject)ksp)->type_name);

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
    if (cg->radius) { /* XXX direction? */
      PetscCall(VecNorm(X, NORM_2, &norm_d));
      norm_d *= norm_d;
    }
  } else {
    PetscCall(VecCopy(B, R)); /*    r <- b (x is 0)                   */
    norm_d = 0.0;
  }
  /* This may be true only on a subset of MPI ranks; setting it here so it will be detected by the first norm computation below */
  if (ksp->reason == KSP_DIVERGED_PC_FAILED) PetscCall(VecSetInf(R));

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
  testobj = (PetscBool)(cg->obj_min < 0.0);
  PetscCall(VecXDot(R, X, &a));
  cg->obj = 0.5 * PetscRealPart(a);
  PetscCall(VecXDot(B, X, &a));
  cg->obj -= 0.5 * PetscRealPart(a);

  PetscCall(PetscInfo(ksp, "it %" PetscInt_FMT " obj %g\n", ksp->its, (double)cg->obj));
  PetscCall(KSPLogResidualHistory(ksp, dp));
  PetscCall(KSPMonitor(ksp, ksp->its, dp));
  ksp->rnorm = dp;

  PetscCall((*ksp->converged)(ksp, ksp->its, dp, &ksp->reason, ksp->cnvP)); /* test for convergence */

  if (!ksp->reason && testobj && cg->obj <= cg->obj_min) {
    PetscCall(PetscInfo(ksp, "converged to objective target minimum\n"));
    ksp->reason = KSP_CONVERGED_ATOL;
  }

  if (ksp->reason) PetscFunctionReturn(PETSC_SUCCESS);

  if (ksp->normtype != KSP_NORM_PRECONDITIONED && (ksp->normtype != KSP_NORM_NATURAL)) { PetscCall(KSP_PCApply(ksp, R, Z)); /*     z <- Br                           */ }
  if (ksp->normtype != KSP_NORM_NATURAL) {
    PetscCall(VecXDot(Z, R, &beta)); /*     beta <- z'*r                      */
    KSPCheckDot(ksp, beta);
  }

  i = 0;
  do {
    ksp->its = i + 1;
    if (beta == 0.0) {
      ksp->reason = KSP_CONVERGED_ATOL;
      PetscCall(PetscInfo(ksp, "converged due to beta = 0\n"));
      break;
#if !defined(PETSC_USE_COMPLEX)
    } else if ((i > 0) && (beta * betaold < 0.0)) {
      PetscCheck(!ksp->errorifnotconverged, PetscObjectComm((PetscObject)ksp), PETSC_ERR_NOT_CONVERGED, "Diverged due to indefinite preconditioner, beta %g, betaold %g", (double)beta, (double)betaold);
      ksp->reason = KSP_DIVERGED_INDEFINITE_PC;
      PetscCall(PetscInfo(ksp, "diverging due to indefinite preconditioner\n"));
      break;
#endif
    }
    if (!i) {
      PetscCall(VecCopy(Z, P)); /*     p <- z                           */
      if (cg->radius) {
        PetscCall(VecNorm(P, NORM_2, &norm_p));
        norm_p *= norm_p;
        dMp = 0.0;
        if (!ksp->guess_zero) { PetscCall(VecDotRealPart(X, P, &dMp)); }
      }
      b = 0.0;
    } else {
      b = beta / betaold;
      if (eigs) {
        PetscCheck(ksp->max_it == stored_max_it, PetscObjectComm((PetscObject)ksp), PETSC_ERR_SUP, "Can not change maxit AND calculate eigenvalues");
        e[i] = PetscSqrtReal(PetscAbsScalar(b)) / a;
      }
      PetscCall(VecAYPX(P, b, Z)); /*     p <- z + b* p                    */
      if (cg->radius) {
        PetscCall(VecDotRealPart(X, P, &dMp));
        PetscCall(VecNorm(P, NORM_2, &norm_p));
        norm_p *= norm_p;
      }
    }
    dpiold = dpi;
    PetscCall(KSP_MatMult(ksp, Amat, P, W)); /*     w <- Ap                          */
    PetscCall(VecXDot(P, W, &dpi));          /*     dpi <- p'w                       */
    KSPCheckDot(ksp, dpi);
    betaold = beta;

    if ((dpi == 0.0) || ((i > 0) && ((PetscSign(PetscRealPart(dpi)) * PetscSign(PetscRealPart(dpiold))) < 0.0))) {
      if (cg->radius) {
        a = 0.0;
        if (i == 0) {
          if (norm_p > 0.0) {
            a = PetscSqrtReal(r2 / norm_p);
          } else {
            PetscCall(VecNorm(R, NORM_2, &dp));
            a = cg->radius > dp ? 1.0 : cg->radius / dp;
          }
        } else if (norm_p > 0.0) {
          a = (PetscSqrtReal(dMp * dMp + norm_p * (r2 - norm_d)) - dMp) / norm_p;
        }
        PetscCall(VecAXPY(X, a, P)); /*     x <- x + ap                      */
        cg->obj += PetscRealPart(a * (0.5 * a * dpi - betaold));
      }
      PetscCall(PetscInfo(ksp, "it %" PetscInt_FMT " N obj %g\n", i + 1, (double)cg->obj));
      if (ksp->converged_neg_curve) {
        PetscCall(PetscInfo(ksp, "converged due to negative curvature: %g\n", (double)(PetscRealPart(dpi))));
        ksp->reason = KSP_CONVERGED_NEG_CURVE;
      } else {
        PetscCheck(!ksp->errorifnotconverged, PetscObjectComm((PetscObject)ksp), PETSC_ERR_NOT_CONVERGED, "Diverged due to indefinite matrix, dpi %g, dpiold %g", (double)PetscRealPart(dpi), (double)PetscRealPart(dpiold));
        ksp->reason = KSP_DIVERGED_INDEFINITE_MAT;
        PetscCall(PetscInfo(ksp, "diverging due to indefinite matrix\n"));
      }
      break;
    }
    a = beta / dpi; /*     a = beta/p'w                     */
    if (eigs) d[i] = PetscSqrtReal(PetscAbsScalar(b)) * e[i] + 1.0 / a;
    if (cg->radius) { /* Steihaugh-Toint */
      PetscReal norm_dp1 = norm_d + PetscRealPart(a) * (2.0 * dMp + PetscRealPart(a) * norm_p);
      if (norm_dp1 > r2) {
        ksp->reason = KSP_CONVERGED_STEP_LENGTH;
        PetscCall(PetscInfo(ksp, "converged to the trust region radius %g\n", (double)cg->radius));
        if (norm_p > 0.0) {
          dp = (PetscSqrtReal(dMp * dMp + norm_p * (r2 - norm_d)) - dMp) / norm_p;
          PetscCall(VecAXPY(X, dp, P)); /*     x <- x + ap                      */
          cg->obj += PetscRealPart(dp * (0.5 * dp * dpi - beta));
        }
        PetscCall(PetscInfo(ksp, "it %" PetscInt_FMT " R obj %g\n", i + 1, (double)cg->obj));
        break;
      }
    }
    PetscCall(VecAXPY(X, a, P));  /*     x <- x + ap                      */
    PetscCall(VecAXPY(R, -a, W)); /*     r <- r - aw                      */
    if (ksp->normtype == KSP_NORM_PRECONDITIONED && ksp->chknorm < i + 2) {
      PetscCall(KSP_PCApply(ksp, R, Z));  /*     z <- Br                          */
      PetscCall(VecNorm(Z, NORM_2, &dp)); /*     dp <- z'*z                       */
      KSPCheckNorm(ksp, dp);
    } else if (ksp->normtype == KSP_NORM_UNPRECONDITIONED && ksp->chknorm < i + 2) {
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
    cg->obj -= PetscRealPart(0.5 * a * betaold);
    PetscCall(PetscInfo(ksp, "it %" PetscInt_FMT " obj %g\n", i + 1, (double)cg->obj));

    ksp->rnorm = dp;
    PetscCall(KSPLogResidualHistory(ksp, dp));
    PetscCall(KSPMonitor(ksp, i + 1, dp));
    PetscCall((*ksp->converged)(ksp, i + 1, dp, &ksp->reason, ksp->cnvP));

    if (!ksp->reason && testobj && cg->obj <= cg->obj_min) {
      PetscCall(PetscInfo(ksp, "converged to objective target minimum\n"));
      ksp->reason = KSP_CONVERGED_ATOL;
    }

    if (ksp->reason) break;

    if (cg->radius) {
      PetscCall(VecNorm(X, NORM_2, &norm_d));
      norm_d *= norm_d;
    }

    if ((ksp->normtype != KSP_NORM_PRECONDITIONED && (ksp->normtype != KSP_NORM_NATURAL)) || (ksp->chknorm >= i + 2)) { PetscCall(KSP_PCApply(ksp, R, Z)); /*     z <- Br                          */ }
    if ((ksp->normtype != KSP_NORM_NATURAL) || (ksp->chknorm >= i + 2)) {
      PetscCall(VecXDot(Z, R, &beta)); /*     beta <- z'*r                     */
      KSPCheckDot(ksp, beta);
    }

    i++;
  } while (i < ksp->max_it);
  if (i >= ksp->max_it) ksp->reason = KSP_DIVERGED_ITS;
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
  PetscBool   diagonalscale;

  PetscFunctionBegin;
  PetscCall(PCGetDiagonalScale(ksp->pc, &diagonalscale));
  PetscCheck(!diagonalscale, PetscObjectComm((PetscObject)ksp), PETSC_ERR_SUP, "Krylov method %s does not support diagonal scaling", ((PetscObject)ksp)->type_name);

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

  if (ksp->normtype != KSP_NORM_PRECONDITIONED && (ksp->normtype != KSP_NORM_NATURAL)) { PetscCall(KSP_PCApply(ksp, R, Z)); /*    z <- Br                           */ }
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
#if !defined(PETSC_USE_COMPLEX)
    } else if ((i > 0) && (beta * betaold < 0.0)) {
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

    if ((ksp->normtype != KSP_NORM_PRECONDITIONED && (ksp->normtype != KSP_NORM_NATURAL)) || (ksp->chknorm >= i + 2)) {
      PetscCall(KSP_PCApply(ksp, R, Z)); /*    z <- Br                           */
      PetscCall(KSP_MatMult(ksp, Amat, Z, S));
    }
    if ((ksp->normtype != KSP_NORM_NATURAL) || (ksp->chknorm >= i + 2)) {
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
  PetscBool iascii;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &iascii));
  if (iascii) {
#if defined(PETSC_USE_COMPLEX)
    PetscCall(PetscViewerASCIIPrintf(viewer, "  variant %s\n", KSPCGTypes[cg->type]));
#endif
    if (cg->singlereduction) PetscCall(PetscViewerASCIIPrintf(viewer, "  using single-reduction variant\n"));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
    KSPSetFromOptions_CG - Checks the options database for options related to the
                           conjugate gradient method.
*/
PetscErrorCode KSPSetFromOptions_CG(KSP ksp, PetscOptionItems *PetscOptionsObject)
{
  KSP_CG   *cg = (KSP_CG *)ksp->data;
  PetscBool flg;

  PetscFunctionBegin;
  PetscOptionsHeadBegin(PetscOptionsObject, "KSP CG and CGNE options");
#if defined(PETSC_USE_COMPLEX)
  PetscCall(PetscOptionsEnum("-ksp_cg_type", "Matrix is Hermitian or complex symmetric", "KSPCGSetType", KSPCGTypes, (PetscEnum)cg->type, (PetscEnum *)&cg->type, NULL));
#endif
  PetscCall(PetscOptionsBool("-ksp_cg_single_reduction", "Merge inner products into single MPI_Allreduce()", "KSPCGUseSingleReduction", cg->singlereduction, &cg->singlereduction, &flg));
  if (flg) PetscCall(KSPCGUseSingleReduction(ksp, cg->singlereduction));
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
  cg->singlereduction = flg;
  if (cg->singlereduction) {
    ksp->ops->solve = KSPSolve_CG_SingleReduction;
  } else {
    ksp->ops->solve = KSPSolve_CG;
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
     KSPCG - The Preconditioned Conjugate Gradient (PCG) iterative method

   Options Database Keys:
+   -ksp_cg_type Hermitian - (for complex matrices only) indicates the matrix is Hermitian, see `KSPCGSetType()`
.   -ksp_cg_type symmetric - (for complex matrices only) indicates the matrix is symmetric
-   -ksp_cg_single_reduction - performs both inner products needed in the algorithm with a single `MPI_Allreduce()` call, see `KSPCGUseSingleReduction()`

   Level: beginner

   Notes:
    The PCG method requires both the matrix and preconditioner to be symmetric positive (or negative) (semi) definite.

   Only left preconditioning is supported; there are several ways to motivate preconditioned CG, but they all produce the same algorithm.
   One can interpret preconditioning A with B to mean any of the following\:
.n  (1) Solve a left-preconditioned system BAx = Bb, using inv(B) to define an inner product in the algorithm.
.n  (2) Solve a right-preconditioned system ABy = b, x = By, using B to define an inner product in the algorithm.
.n  (3) Solve a symmetrically-preconditioned system, E^TAEy = E^Tb, x = Ey, where B = EE^T.
.n  (4) Solve Ax=b with CG, but use the inner product defined by B to define the method [2].
.n  In all cases, the resulting algorithm only requires application of B to vectors.

   For complex numbers there are two different CG methods, one for Hermitian symmetric matrices and one for non-Hermitian symmetric matrices. Use
   `KSPCGSetType()` to indicate which type you are using.

   One can use `KSPSetComputeEigenvalues()` and `KSPComputeEigenvalues()` to compute the eigenvalues of the (preconditioned) operator

   Developer Notes:
    KSPSolve_CG() should actually query the matrix to determine if it is Hermitian symmetric or not and NOT require the user to
   indicate it to the `KSP` object.

   References:
+  * - Magnus R. Hestenes and Eduard Stiefel, Methods of Conjugate Gradients for Solving Linear Systems,
   Journal of Research of the National Bureau of Standards Vol. 49, No. 6, December 1952 Research Paper 2379
-  * - Josef Malek and Zdenek Strakos, Preconditioning and the Conjugate Gradient Method in the Context of Solving PDEs,
    SIAM, 2014.

.seealso: [](chapter_ksp), `KSPCreate()`, `KSPSetType()`, `KSPType`, `KSP`, `KSPSetComputeEigenvalues()`, `KSPComputeEigenvalues()`
          `KSPCGSetType()`, `KSPCGUseSingleReduction()`, `KSPPIPECG`, `KSPGROPPCG`
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
#if !defined(PETSC_USE_COMPLEX)
  cg->type = KSP_CG_SYMMETRIC;
#else
  cg->type = KSP_CG_HERMITIAN;
#endif
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
  PetscFunctionReturn(PETSC_SUCCESS);
}
