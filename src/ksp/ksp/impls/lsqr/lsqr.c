
/* lourens.vanzanen@shell.com contributed the standard error estimates of the solution, Jul 25, 2006 */
/* Bas van't Hof contributed the preconditioned aspects Feb 10, 2010 */

#define SWAP(a, b, c) \
  { \
    c = a; \
    a = b; \
    b = c; \
  }

#include <petsc/private/kspimpl.h> /*I "petscksp.h" I*/
#include <petscdraw.h>

typedef struct {
  PetscInt  nwork_n, nwork_m;
  Vec      *vwork_m;    /* work vectors of length m, where the system is size m x n */
  Vec      *vwork_n;    /* work vectors of length n */
  Vec       se;         /* Optional standard error vector */
  PetscBool se_flg;     /* flag for -ksp_lsqr_set_standard_error */
  PetscBool exact_norm; /* flag for -ksp_lsqr_exact_mat_norm */
  PetscReal arnorm;     /* Good estimate of norm((A*inv(Pmat))'*r), where r = A*x - b, used in specific stopping criterion */
  PetscReal anorm;      /* Poor estimate of norm(A*inv(Pmat),'fro') used in specific stopping criterion */
  /* Backup previous convergence test */
  PetscErrorCode (*converged)(KSP, PetscInt, PetscReal, KSPConvergedReason *, void *);
  PetscErrorCode (*convergeddestroy)(void *);
  void *cnvP;
} KSP_LSQR;

static PetscErrorCode VecSquare(Vec v)
{
  PetscScalar *x;
  PetscInt     i, n;

  PetscFunctionBegin;
  PetscCall(VecGetLocalSize(v, &n));
  PetscCall(VecGetArray(v, &x));
  for (i = 0; i < n; i++) x[i] *= PetscConj(x[i]);
  PetscCall(VecRestoreArray(v, &x));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode KSPSetUp_LSQR(KSP ksp)
{
  KSP_LSQR *lsqr = (KSP_LSQR *)ksp->data;
  PetscBool nopreconditioner;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)ksp->pc, PCNONE, &nopreconditioner));

  if (lsqr->vwork_m) PetscCall(VecDestroyVecs(lsqr->nwork_m, &lsqr->vwork_m));

  if (lsqr->vwork_n) PetscCall(VecDestroyVecs(lsqr->nwork_n, &lsqr->vwork_n));

  lsqr->nwork_m = 2;
  if (nopreconditioner) lsqr->nwork_n = 4;
  else lsqr->nwork_n = 5;
  PetscCall(KSPCreateVecs(ksp, lsqr->nwork_n, &lsqr->vwork_n, lsqr->nwork_m, &lsqr->vwork_m));

  if (lsqr->se_flg && !lsqr->se) {
    PetscCall(VecDuplicate(lsqr->vwork_n[0], &lsqr->se));
    PetscCall(VecSet(lsqr->se, PETSC_INFINITY));
  } else if (!lsqr->se_flg) {
    PetscCall(VecDestroy(&lsqr->se));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode KSPSolve_LSQR(KSP ksp)
{
  PetscInt    i, size1, size2;
  PetscScalar rho, rhobar, phi, phibar, theta, c, s, tmp, tau;
  PetscReal   beta, alpha, rnorm;
  Vec         X, B, V, V1, U, U1, TMP, W, W2, Z = NULL;
  Mat         Amat, Pmat;
  KSP_LSQR   *lsqr = (KSP_LSQR *)ksp->data;
  PetscBool   diagonalscale, nopreconditioner;

  PetscFunctionBegin;
  PetscCall(PCGetDiagonalScale(ksp->pc, &diagonalscale));
  PetscCheck(!diagonalscale, PetscObjectComm((PetscObject)ksp), PETSC_ERR_SUP, "Krylov method %s does not support diagonal scaling", ((PetscObject)ksp)->type_name);

  PetscCall(PCGetOperators(ksp->pc, &Amat, &Pmat));
  PetscCall(PetscObjectTypeCompare((PetscObject)ksp->pc, PCNONE, &nopreconditioner));

  /* vectors of length m, where system size is mxn */
  B  = ksp->vec_rhs;
  U  = lsqr->vwork_m[0];
  U1 = lsqr->vwork_m[1];

  /* vectors of length n */
  X  = ksp->vec_sol;
  W  = lsqr->vwork_n[0];
  V  = lsqr->vwork_n[1];
  V1 = lsqr->vwork_n[2];
  W2 = lsqr->vwork_n[3];
  if (!nopreconditioner) Z = lsqr->vwork_n[4];

  /* standard error vector */
  if (lsqr->se) PetscCall(VecSet(lsqr->se, 0.0));

  /* Compute initial residual, temporarily use work vector u */
  if (!ksp->guess_zero) {
    PetscCall(KSP_MatMult(ksp, Amat, X, U)); /*   u <- b - Ax     */
    PetscCall(VecAYPX(U, -1.0, B));
  } else {
    PetscCall(VecCopy(B, U)); /*   u <- b (x is 0) */
  }

  /* Test for nothing to do */
  PetscCall(VecNorm(U, NORM_2, &rnorm));
  KSPCheckNorm(ksp, rnorm);
  PetscCall(PetscObjectSAWsTakeAccess((PetscObject)ksp));
  ksp->its   = 0;
  ksp->rnorm = rnorm;
  PetscCall(PetscObjectSAWsGrantAccess((PetscObject)ksp));
  PetscCall(KSPLogResidualHistory(ksp, rnorm));
  PetscCall(KSPMonitor(ksp, 0, rnorm));
  PetscCall((*ksp->converged)(ksp, 0, rnorm, &ksp->reason, ksp->cnvP));
  if (ksp->reason) PetscFunctionReturn(PETSC_SUCCESS);

  beta = rnorm;
  PetscCall(VecScale(U, 1.0 / beta));
  PetscCall(KSP_MatMultHermitianTranspose(ksp, Amat, U, V));
  if (nopreconditioner) {
    PetscCall(VecNorm(V, NORM_2, &alpha));
    KSPCheckNorm(ksp, rnorm);
  } else {
    /* this is an application of the preconditioner for the normal equations; not the operator, see the manual page */
    PetscCall(PCApply(ksp->pc, V, Z));
    PetscCall(VecDotRealPart(V, Z, &alpha));
    if (alpha <= 0.0) {
      ksp->reason = KSP_DIVERGED_BREAKDOWN;
      PetscFunctionReturn(PETSC_SUCCESS);
    }
    alpha = PetscSqrtReal(alpha);
    PetscCall(VecScale(Z, 1.0 / alpha));
  }
  PetscCall(VecScale(V, 1.0 / alpha));

  if (nopreconditioner) {
    PetscCall(VecCopy(V, W));
  } else {
    PetscCall(VecCopy(Z, W));
  }

  if (lsqr->exact_norm) {
    PetscCall(MatNorm(Amat, NORM_FROBENIUS, &lsqr->anorm));
  } else lsqr->anorm = 0.0;

  lsqr->arnorm = alpha * beta;
  phibar       = beta;
  rhobar       = alpha;
  i            = 0;
  do {
    if (nopreconditioner) {
      PetscCall(KSP_MatMult(ksp, Amat, V, U1));
    } else {
      PetscCall(KSP_MatMult(ksp, Amat, Z, U1));
    }
    PetscCall(VecAXPY(U1, -alpha, U));
    PetscCall(VecNorm(U1, NORM_2, &beta));
    KSPCheckNorm(ksp, beta);
    if (beta > 0.0) {
      PetscCall(VecScale(U1, 1.0 / beta)); /* beta*U1 = Amat*V - alpha*U */
      if (!lsqr->exact_norm) lsqr->anorm = PetscSqrtReal(PetscSqr(lsqr->anorm) + PetscSqr(alpha) + PetscSqr(beta));
    }

    PetscCall(KSP_MatMultHermitianTranspose(ksp, Amat, U1, V1));
    PetscCall(VecAXPY(V1, -beta, V));
    if (nopreconditioner) {
      PetscCall(VecNorm(V1, NORM_2, &alpha));
      KSPCheckNorm(ksp, alpha);
    } else {
      PetscCall(PCApply(ksp->pc, V1, Z));
      PetscCall(VecDotRealPart(V1, Z, &alpha));
      if (alpha <= 0.0) {
        ksp->reason = KSP_DIVERGED_BREAKDOWN;
        break;
      }
      alpha = PetscSqrtReal(alpha);
      PetscCall(VecScale(Z, 1.0 / alpha));
    }
    PetscCall(VecScale(V1, 1.0 / alpha)); /* alpha*V1 = Amat^T*U1 - beta*V */
    rho    = PetscSqrtScalar(rhobar * rhobar + beta * beta);
    c      = rhobar / rho;
    s      = beta / rho;
    theta  = s * alpha;
    rhobar = -c * alpha;
    phi    = c * phibar;
    phibar = s * phibar;
    tau    = s * phi;

    PetscCall(VecAXPY(X, phi / rho, W)); /*    x <- x + (phi/rho) w   */

    if (lsqr->se) {
      PetscCall(VecCopy(W, W2));
      PetscCall(VecSquare(W2));
      PetscCall(VecScale(W2, 1.0 / (rho * rho)));
      PetscCall(VecAXPY(lsqr->se, 1.0, W2)); /* lsqr->se <- lsqr->se + (w^2/rho^2) */
    }
    if (nopreconditioner) {
      PetscCall(VecAYPX(W, -theta / rho, V1)); /* w <- v - (theta/rho) w */
    } else {
      PetscCall(VecAYPX(W, -theta / rho, Z)); /* w <- z - (theta/rho) w */
    }

    lsqr->arnorm = alpha * PetscAbsScalar(tau);
    rnorm        = PetscRealPart(phibar);

    PetscCall(PetscObjectSAWsTakeAccess((PetscObject)ksp));
    ksp->its++;
    ksp->rnorm = rnorm;
    PetscCall(PetscObjectSAWsGrantAccess((PetscObject)ksp));
    PetscCall(KSPLogResidualHistory(ksp, rnorm));
    PetscCall(KSPMonitor(ksp, i + 1, rnorm));
    PetscCall((*ksp->converged)(ksp, i + 1, rnorm, &ksp->reason, ksp->cnvP));
    if (ksp->reason) break;
    SWAP(U1, U, TMP);
    SWAP(V1, V, TMP);

    i++;
  } while (i < ksp->max_it);
  if (i >= ksp->max_it && !ksp->reason) ksp->reason = KSP_DIVERGED_ITS;

  /* Finish off the standard error estimates */
  if (lsqr->se) {
    tmp = 1.0;
    PetscCall(MatGetSize(Amat, &size1, &size2));
    if (size1 > size2) tmp = size1 - size2;
    tmp = rnorm / PetscSqrtScalar(tmp);
    PetscCall(VecSqrtAbs(lsqr->se));
    PetscCall(VecScale(lsqr->se, tmp));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode KSPDestroy_LSQR(KSP ksp)
{
  KSP_LSQR *lsqr = (KSP_LSQR *)ksp->data;

  PetscFunctionBegin;
  /* Free work vectors */
  if (lsqr->vwork_n) PetscCall(VecDestroyVecs(lsqr->nwork_n, &lsqr->vwork_n));
  if (lsqr->vwork_m) PetscCall(VecDestroyVecs(lsqr->nwork_m, &lsqr->vwork_m));
  PetscCall(VecDestroy(&lsqr->se));
  /* Revert convergence test */
  PetscCall(KSPSetConvergenceTest(ksp, lsqr->converged, lsqr->cnvP, lsqr->convergeddestroy));
  /* Free the KSP_LSQR context */
  PetscCall(PetscFree(ksp->data));
  PetscCall(PetscObjectComposeFunction((PetscObject)ksp, "KSPLSQRMonitorResidual_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)ksp, "KSPLSQRMonitorResidualDrawLG_C", NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
   KSPLSQRSetComputeStandardErrorVec - Compute a vector of standard error estimates during `KSPSolve()` for  `KSPLSQR`.

   Logically Collective

   Input Parameters:
+  ksp   - iterative context
-  flg   - compute the vector of standard estimates or not

   Level: intermediate

   Developer Note:
   Vaclav: I'm not sure whether this vector is useful for anything.

.seealso: [](chapter_ksp), `KSPSolve()`, `KSPLSQR`, `KSPLSQRGetStandardErrorVec()`
@*/
PetscErrorCode KSPLSQRSetComputeStandardErrorVec(KSP ksp, PetscBool flg)
{
  KSP_LSQR *lsqr = (KSP_LSQR *)ksp->data;

  PetscFunctionBegin;
  lsqr->se_flg = flg;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
   KSPLSQRSetExactMatNorm - Compute exact matrix norm instead of iteratively refined estimate.

   Not Collective

   Input Parameters:
+  ksp   - iterative context
-  flg   - compute exact matrix norm or not

   Level: intermediate

   Notes:
   By default, flg = `PETSC_FALSE`. This is usually preferred to avoid possibly expensive computation of the norm.
   For flg = `PETSC_TRUE`, we call `MatNorm`(Amat,`NORM_FROBENIUS`,&lsqr->anorm) which will work only for some types of explicitly assembled matrices.
   This can affect convergence rate as `KSPLSQRConvergedDefault()` assumes different value of ||A|| used in normal equation stopping criterion.

.seealso: [](chapter_ksp), `KSPSolve()`, `KSPLSQR`, `KSPLSQRGetNorms()`, `KSPLSQRConvergedDefault()`
@*/
PetscErrorCode KSPLSQRSetExactMatNorm(KSP ksp, PetscBool flg)
{
  KSP_LSQR *lsqr = (KSP_LSQR *)ksp->data;

  PetscFunctionBegin;
  lsqr->exact_norm = flg;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
   KSPLSQRGetStandardErrorVec - Get vector of standard error estimates.
   Only available if -ksp_lsqr_set_standard_error was set to true
   or `KSPLSQRSetComputeStandardErrorVec`(ksp, `PETSC_TRUE`) was called.
   Otherwise returns NULL.

   Not Collective

   Input Parameter:
.  ksp   - iterative context

   Output Parameter:
.  se - vector of standard estimates

   Level: intermediate

   Developer Note:
   Vaclav: I'm not sure whether this vector is useful for anything.

.seealso: [](chapter_ksp), `KSPSolve()`, `KSPLSQR`, `KSPLSQRSetComputeStandardErrorVec()`
@*/
PetscErrorCode KSPLSQRGetStandardErrorVec(KSP ksp, Vec *se)
{
  KSP_LSQR *lsqr = (KSP_LSQR *)ksp->data;

  PetscFunctionBegin;
  *se = lsqr->se;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
   KSPLSQRGetNorms - Get the norm estimates that `KSPLSQR` computes internally during `KSPSolve()`.

   Not Collective

   Input Parameter:
.  ksp   - iterative context

   Output Parameters:
+  arnorm - good estimate of norm((A*inv(Pmat))'*r), where r = A*x - b, used in specific stopping criterion
-  anorm - poor estimate of norm(A*inv(Pmat),'fro') used in specific stopping criterion

   Notes:
   Output parameters are meaningful only after `KSPSolve()`.

   These are the same quantities as normar and norma in MATLAB's `lsqr()`, whose output lsvec is a vector of normar / norma for all iterations.

   If -ksp_lsqr_exact_mat_norm is set or `KSPLSQRSetExactMatNorm`(ksp, `PETSC_TRUE`) called, then anorm is the exact Frobenius norm.

   Level: intermediate

.seealso: [](chapter_ksp), `KSPSolve()`, `KSPLSQR`, `KSPLSQRSetExactMatNorm()`
@*/
PetscErrorCode KSPLSQRGetNorms(KSP ksp, PetscReal *arnorm, PetscReal *anorm)
{
  KSP_LSQR *lsqr = (KSP_LSQR *)ksp->data;

  PetscFunctionBegin;
  if (arnorm) *arnorm = lsqr->arnorm;
  if (anorm) *anorm = lsqr->anorm;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode KSPLSQRMonitorResidual_LSQR(KSP ksp, PetscInt n, PetscReal rnorm, PetscViewerAndFormat *vf)
{
  KSP_LSQR         *lsqr   = (KSP_LSQR *)ksp->data;
  PetscViewer       viewer = vf->viewer;
  PetscViewerFormat format = vf->format;
  char              normtype[256];
  PetscInt          tablevel;
  const char       *prefix;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetTabLevel((PetscObject)ksp, &tablevel));
  PetscCall(PetscObjectGetOptionsPrefix((PetscObject)ksp, &prefix));
  PetscCall(PetscStrncpy(normtype, KSPNormTypes[ksp->normtype], sizeof(normtype)));
  PetscCall(PetscStrtolower(normtype));
  PetscCall(PetscViewerPushFormat(viewer, format));
  PetscCall(PetscViewerASCIIAddTab(viewer, tablevel));
  if (n == 0 && prefix) PetscCall(PetscViewerASCIIPrintf(viewer, "  Residual norm, norm of normal equations, and matrix norm for %s solve.\n", prefix));
  if (!n) {
    PetscCall(PetscViewerASCIIPrintf(viewer, "%3" PetscInt_FMT " KSP resid norm %14.12e\n", n, (double)rnorm));
  } else {
    PetscCall(PetscViewerASCIIPrintf(viewer, "%3" PetscInt_FMT " KSP resid norm %14.12e normal eq resid norm %14.12e matrix norm %14.12e\n", n, (double)rnorm, (double)lsqr->arnorm, (double)lsqr->anorm));
  }
  PetscCall(PetscViewerASCIISubtractTab(viewer, tablevel));
  PetscCall(PetscViewerPopFormat(viewer));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  KSPLSQRMonitorResidual - Prints the residual norm, as well as the normal equation residual norm, at each iteration of an iterative solver for the `KSPLSQR` solver

  Collective

  Input Parameters:
+ ksp   - iterative context
. n     - iteration number
. rnorm - 2-norm (preconditioned) residual value (may be estimated).
- vf    - The viewer context

  Options Database Key:
. -ksp_lsqr_monitor - Activates `KSPLSQRMonitorResidual()`

  Level: intermediate

.seealso: [](chapter_ksp), `KSPLSQR`, `KSPMonitorSet()`, `KSPMonitorResidual()`, `KSPMonitorTrueResidualMaxNorm()`, `KSPLSQRMonitorResidualDrawLG()`
@*/
PetscErrorCode KSPLSQRMonitorResidual(KSP ksp, PetscInt n, PetscReal rnorm, PetscViewerAndFormat *vf)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp, KSP_CLASSID, 1);
  PetscValidPointer(vf, 4);
  PetscValidHeaderSpecific(vf->viewer, PETSC_VIEWER_CLASSID, 4);
  PetscTryMethod(ksp, "KSPLSQRMonitorResidual_C", (KSP, PetscInt, PetscReal, PetscViewerAndFormat *), (ksp, n, rnorm, vf));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode KSPLSQRMonitorResidualDrawLG_LSQR(KSP ksp, PetscInt n, PetscReal rnorm, PetscViewerAndFormat *vf)
{
  KSP_LSQR          *lsqr   = (KSP_LSQR *)ksp->data;
  PetscViewer        viewer = vf->viewer;
  PetscViewerFormat  format = vf->format;
  PetscDrawLG        lg     = vf->lg;
  KSPConvergedReason reason;
  PetscReal          x[2], y[2];

  PetscFunctionBegin;
  PetscCall(PetscViewerPushFormat(viewer, format));
  if (!n) PetscCall(PetscDrawLGReset(lg));
  x[0] = (PetscReal)n;
  if (rnorm > 0.0) y[0] = PetscLog10Real(rnorm);
  else y[0] = -15.0;
  x[1] = (PetscReal)n;
  if (lsqr->arnorm > 0.0) y[1] = PetscLog10Real(lsqr->arnorm);
  else y[1] = -15.0;
  PetscCall(PetscDrawLGAddPoint(lg, x, y));
  PetscCall(KSPGetConvergedReason(ksp, &reason));
  if (n <= 20 || !(n % 5) || reason) {
    PetscCall(PetscDrawLGDraw(lg));
    PetscCall(PetscDrawLGSave(lg));
  }
  PetscCall(PetscViewerPopFormat(viewer));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  KSPLSQRMonitorResidualDrawLG - Plots the true residual norm at each iteration of an iterative solver for the `KSPLSQR` solver

  Collective

  Input Parameters:
+ ksp   - iterative context
. n     - iteration number
. rnorm - 2-norm (preconditioned) residual value (may be estimated).
- vf    - The viewer context

  Options Database Key:
. -ksp_lsqr_monitor draw::draw_lg - Activates `KSPMonitorTrueResidualDrawLG()`

  Level: intermediate

.seealso: [](chapter_ksp), `KSPLSQR`, `KSPMonitorSet()`, `KSPMonitorTrueResidual()`, `KSPLSQRMonitorResidual()`, `KSPLSQRMonitorResidualDrawLGCreate()`
@*/
PetscErrorCode KSPLSQRMonitorResidualDrawLG(KSP ksp, PetscInt n, PetscReal rnorm, PetscViewerAndFormat *vf)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp, KSP_CLASSID, 1);
  PetscValidPointer(vf, 4);
  PetscValidHeaderSpecific(vf->viewer, PETSC_VIEWER_CLASSID, 4);
  PetscValidHeaderSpecific(vf->lg, PETSC_DRAWLG_CLASSID, 4);
  PetscTryMethod(ksp, "KSPLSQRMonitorResidualDrawLG_C", (KSP, PetscInt, PetscReal, PetscViewerAndFormat *), (ksp, n, rnorm, vf));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  KSPLSQRMonitorResidualDrawLGCreate - Creates the plotter for the `KSPLSQR` residual and normal equation residual norm

  Collective

  Input Parameters:
+ viewer - The PetscViewer
. format - The viewer format
- ctx    - An optional user context

  Output Parameter:
. vf    - The viewer context

  Level: intermediate

.seealso: [](chapter_ksp), `KSPLSQR`, `KSPMonitorSet()`, `KSPLSQRMonitorResidual()`, `KSPLSQRMonitorResidualDrawLG()`
@*/
PetscErrorCode KSPLSQRMonitorResidualDrawLGCreate(PetscViewer viewer, PetscViewerFormat format, void *ctx, PetscViewerAndFormat **vf)
{
  const char *names[] = {"residual", "normal eqn residual"};

  PetscFunctionBegin;
  PetscCall(PetscViewerAndFormatCreate(viewer, format, vf));
  (*vf)->data = ctx;
  PetscCall(KSPMonitorLGCreate(PetscObjectComm((PetscObject)viewer), NULL, NULL, "Log Residual Norm", 2, names, PETSC_DECIDE, PETSC_DECIDE, 400, 300, &(*vf)->lg));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode KSPSetFromOptions_LSQR(KSP ksp, PetscOptionItems *PetscOptionsObject)
{
  KSP_LSQR *lsqr = (KSP_LSQR *)ksp->data;

  PetscFunctionBegin;
  PetscOptionsHeadBegin(PetscOptionsObject, "KSP LSQR Options");
  PetscCall(PetscOptionsBool("-ksp_lsqr_compute_standard_error", "Set Standard Error Estimates of Solution", "KSPLSQRSetComputeStandardErrorVec", lsqr->se_flg, &lsqr->se_flg, NULL));
  PetscCall(PetscOptionsBool("-ksp_lsqr_exact_mat_norm", "Compute exact matrix norm instead of iteratively refined estimate", "KSPLSQRSetExactMatNorm", lsqr->exact_norm, &lsqr->exact_norm, NULL));
  PetscCall(KSPMonitorSetFromOptions(ksp, "-ksp_lsqr_monitor", "lsqr_residual", NULL));
  PetscOptionsHeadEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode KSPView_LSQR(KSP ksp, PetscViewer viewer)
{
  KSP_LSQR *lsqr = (KSP_LSQR *)ksp->data;
  PetscBool iascii;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &iascii));
  if (iascii) {
    if (lsqr->se) {
      PetscReal rnorm;
      PetscCall(VecNorm(lsqr->se, NORM_2, &rnorm));
      PetscCall(PetscViewerASCIIPrintf(viewer, "  norm of standard error %g, iterations %" PetscInt_FMT "\n", (double)rnorm, ksp->its));
    } else {
      PetscCall(PetscViewerASCIIPrintf(viewer, "  standard error not computed\n"));
    }
    if (lsqr->exact_norm) {
      PetscCall(PetscViewerASCIIPrintf(viewer, "  using exact matrix norm\n"));
    } else {
      PetscCall(PetscViewerASCIIPrintf(viewer, "  using inexact matrix norm\n"));
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
   KSPLSQRConvergedDefault - Determines convergence of the `KSPLSQR` Krylov method.

   Collective

   Input Parameters:
+  ksp   - iterative context
.  n     - iteration number
.  rnorm - 2-norm residual value (may be estimated)
-  ctx - convergence context which must be created by `KSPConvergedDefaultCreate()`

   reason is set to:
+   positive - if the iteration has converged;
.   negative - if residual norm exceeds divergence threshold;
-   0 - otherwise.

   Notes:
   `KSPConvergedDefault()` is called first to check for convergence in A*x=b.
   If that does not determine convergence then checks convergence for the least squares problem, i.e. in min{|b-A*x|}.
   Possible convergence for the least squares problem (which is based on the residual of the normal equations) are `KSP_CONVERGED_RTOL_NORMAL` norm
   and `KSP_CONVERGED_ATOL_NORMAL`.

   `KSP_CONVERGED_RTOL_NORMAL` is returned if ||A'*r|| < rtol * ||A|| * ||r||.
   Matrix norm ||A|| is iteratively refined estimate, see `KSPLSQRGetNorms()`.
   This criterion is now largely compatible with that in MATLAB `lsqr()`.

   Level: intermediate

.seealso: [](chapter_ksp), `KSPLSQR`, `KSPSetConvergenceTest()`, `KSPSetTolerances()`, `KSPConvergedSkip()`, `KSPConvergedReason`, `KSPGetConvergedReason()`,
          `KSPConvergedDefaultSetUIRNorm()`, `KSPConvergedDefaultSetUMIRNorm()`, `KSPConvergedDefaultCreate()`, `KSPConvergedDefaultDestroy()`, `KSPConvergedDefault()`, `KSPLSQRGetNorms()`, `KSPLSQRSetExactMatNorm()`
@*/
PetscErrorCode KSPLSQRConvergedDefault(KSP ksp, PetscInt n, PetscReal rnorm, KSPConvergedReason *reason, void *ctx)
{
  KSP_LSQR *lsqr = (KSP_LSQR *)ksp->data;

  PetscFunctionBegin;
  /* check for convergence in A*x=b */
  PetscCall(KSPConvergedDefault(ksp, n, rnorm, reason, ctx));
  if (!n || *reason) PetscFunctionReturn(PETSC_SUCCESS);

  /* check for convergence in min{|b-A*x|} */
  if (lsqr->arnorm < ksp->abstol) {
    PetscCall(PetscInfo(ksp, "LSQR solver has converged. Normal equation residual %14.12e is less than absolute tolerance %14.12e at iteration %" PetscInt_FMT "\n", (double)lsqr->arnorm, (double)ksp->abstol, n));
    *reason = KSP_CONVERGED_ATOL_NORMAL;
  } else if (lsqr->arnorm < ksp->rtol * lsqr->anorm * rnorm) {
    PetscCall(PetscInfo(ksp, "LSQR solver has converged. Normal equation residual %14.12e is less than rel. tol. %14.12e times %s Frobenius norm of matrix %14.12e times residual %14.12e at iteration %" PetscInt_FMT "\n", (double)lsqr->arnorm,
                        (double)ksp->rtol, lsqr->exact_norm ? "exact" : "approx.", (double)lsqr->anorm, (double)rnorm, n));
    *reason = KSP_CONVERGED_RTOL_NORMAL;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*MC
     KSPLSQR - Implements LSQR

   Options Database Keys:
+   -ksp_lsqr_set_standard_error  - set standard error estimates of solution, see `KSPLSQRSetComputeStandardErrorVec()` and `KSPLSQRGetStandardErrorVec()`
.   -ksp_lsqr_exact_mat_norm - compute exact matrix norm instead of iteratively refined estimate, see `KSPLSQRSetExactMatNorm()`
-   -ksp_lsqr_monitor - monitor residual norm, norm of residual of normal equations A'*A x = A' b, and estimate of matrix norm ||A||

   Level: beginner

   Notes:
     Supports non-square (rectangular) matrices.

     This variant, when applied with no preconditioning is identical to the original algorithm in exact arithmetic; however, in practice, with no preconditioning
     due to inexact arithmetic, it can converge differently. Hence when no preconditioner is used (`PCType` `PCNONE`) it automatically reverts to the original algorithm.

     With the PETSc built-in preconditioners, such as `PCICC`, one should call `KSPSetOperators`(ksp,A,A'*A)) since the preconditioner needs to work
     for the normal equations A'*A.

     Supports only left preconditioning.

     For least squares problems with nonzero residual A*x - b, there are additional convergence tests for the residual of the normal equations, A'*(b - Ax), see `KSPLSQRConvergedDefault()`.

     In exact arithmetic the LSQR method (with no preconditioning) is identical to the `KSPCG` algorithm applied to the normal equations.
     The preconditioned variant was implemented by Bas van't Hof and is essentially a left preconditioning for the Normal Equations.
     It appears the implementation with preconditioning tracks the true norm of the residual and uses that in the convergence test.

   Developer Note:
    How is this related to the `KSPCGNE` implementation? One difference is that `KSPCGNE` applies
    the preconditioner transpose times the preconditioner,  so one does not need to pass A'*A as the third argument to `KSPSetOperators()`.

   Reference:
.  * - The original unpreconditioned algorithm can be found in Paige and Saunders, ACM Transactions on Mathematical Software, Vol 8, 1982.

.seealso: [](chapter_ksp), `KSPCreate()`, `KSPSetType()`, `KSPType`, `KSP`, `KSPSolve()`, `KSPLSQRConvergedDefault()`, `KSPLSQRSetComputeStandardErrorVec()`, `KSPLSQRGetStandardErrorVec()`, `KSPLSQRSetExactMatNorm()`, `KSPLSQRMonitorResidualDrawLGCreate()`, `KSPLSQRMonitorResidualDrawLG()`, `KSPLSQRMonitorResidual()`
M*/
PETSC_EXTERN PetscErrorCode KSPCreate_LSQR(KSP ksp)
{
  KSP_LSQR *lsqr;
  void     *ctx;

  PetscFunctionBegin;
  PetscCall(PetscNew(&lsqr));
  lsqr->se         = NULL;
  lsqr->se_flg     = PETSC_FALSE;
  lsqr->exact_norm = PETSC_FALSE;
  lsqr->anorm      = -1.0;
  lsqr->arnorm     = -1.0;
  ksp->data        = (void *)lsqr;
  PetscCall(KSPSetSupportedNorm(ksp, KSP_NORM_UNPRECONDITIONED, PC_LEFT, 3));

  ksp->ops->setup          = KSPSetUp_LSQR;
  ksp->ops->solve          = KSPSolve_LSQR;
  ksp->ops->destroy        = KSPDestroy_LSQR;
  ksp->ops->setfromoptions = KSPSetFromOptions_LSQR;
  ksp->ops->view           = KSPView_LSQR;

  /* Backup current convergence test; remove destroy routine from KSP to prevent destroying the convergence context in KSPSetConvergenceTest() */
  PetscCall(KSPGetAndClearConvergenceTest(ksp, &lsqr->converged, &lsqr->cnvP, &lsqr->convergeddestroy));
  /* Override current convergence test */
  PetscCall(KSPConvergedDefaultCreate(&ctx));
  PetscCall(KSPSetConvergenceTest(ksp, KSPLSQRConvergedDefault, ctx, KSPConvergedDefaultDestroy));
  PetscCall(PetscObjectComposeFunction((PetscObject)ksp, "KSPLSQRMonitorResidual_C", KSPLSQRMonitorResidual_LSQR));
  PetscCall(PetscObjectComposeFunction((PetscObject)ksp, "KSPLSQRMonitorResidualDrawLG_C", KSPLSQRMonitorResidualDrawLG_LSQR));
  PetscFunctionReturn(PETSC_SUCCESS);
}
