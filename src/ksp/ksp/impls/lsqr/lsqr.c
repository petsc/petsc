
/* lourens.vanzanen@shell.com contributed the standard error estimates of the solution, Jul 25, 2006 */
/* Bas van't Hof contributed the preconditioned aspects Feb 10, 2010 */

#define SWAP(a,b,c) { c = a; a = b; b = c; }

#include <petsc/private/kspimpl.h>  /*I "petscksp.h" I*/
#include <petscdraw.h>

typedef struct {
  PetscInt  nwork_n,nwork_m;
  Vec       *vwork_m;   /* work vectors of length m, where the system is size m x n */
  Vec       *vwork_n;   /* work vectors of length n */
  Vec       se;         /* Optional standard error vector */
  PetscBool se_flg;     /* flag for -ksp_lsqr_set_standard_error */
  PetscBool exact_norm; /* flag for -ksp_lsqr_exact_mat_norm */
  PetscReal arnorm;     /* Good estimate of norm((A*inv(Pmat))'*r), where r = A*x - b, used in specific stopping criterion */
  PetscReal anorm;      /* Poor estimate of norm(A*inv(Pmat),'fro') used in specific stopping criterion */
  /* Backup previous convergence test */
  PetscErrorCode        (*converged)(KSP,PetscInt,PetscReal,KSPConvergedReason*,void*);
  PetscErrorCode        (*convergeddestroy)(void*);
  void                  *cnvP;
} KSP_LSQR;

static PetscErrorCode  VecSquare(Vec v)
{
  PetscScalar    *x;
  PetscInt       i, n;

  PetscFunctionBegin;
  CHKERRQ(VecGetLocalSize(v, &n));
  CHKERRQ(VecGetArray(v, &x));
  for (i = 0; i < n; i++) x[i] *= PetscConj(x[i]);
  CHKERRQ(VecRestoreArray(v, &x));
  PetscFunctionReturn(0);
}

static PetscErrorCode KSPSetUp_LSQR(KSP ksp)
{
  KSP_LSQR       *lsqr = (KSP_LSQR*)ksp->data;
  PetscBool      nopreconditioner;

  PetscFunctionBegin;
  CHKERRQ(PetscObjectTypeCompare((PetscObject)ksp->pc,PCNONE,&nopreconditioner));

  if (lsqr->vwork_m) {
    CHKERRQ(VecDestroyVecs(lsqr->nwork_m,&lsqr->vwork_m));
  }

  if (lsqr->vwork_n) {
    CHKERRQ(VecDestroyVecs(lsqr->nwork_n,&lsqr->vwork_n));
  }

  lsqr->nwork_m = 2;
  if (nopreconditioner) lsqr->nwork_n = 4;
  else lsqr->nwork_n = 5;
  CHKERRQ(KSPCreateVecs(ksp,lsqr->nwork_n,&lsqr->vwork_n,lsqr->nwork_m,&lsqr->vwork_m));

  if (lsqr->se_flg && !lsqr->se) {
    CHKERRQ(VecDuplicate(lsqr->vwork_n[0],&lsqr->se));
    CHKERRQ(VecSet(lsqr->se,PETSC_INFINITY));
  } else if (!lsqr->se_flg) {
    CHKERRQ(VecDestroy(&lsqr->se));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode KSPSolve_LSQR(KSP ksp)
{
  PetscInt       i,size1,size2;
  PetscScalar    rho,rhobar,phi,phibar,theta,c,s,tmp,tau;
  PetscReal      beta,alpha,rnorm;
  Vec            X,B,V,V1,U,U1,TMP,W,W2,Z = NULL;
  Mat            Amat,Pmat;
  KSP_LSQR       *lsqr = (KSP_LSQR*)ksp->data;
  PetscBool      diagonalscale,nopreconditioner;

  PetscFunctionBegin;
  CHKERRQ(PCGetDiagonalScale(ksp->pc,&diagonalscale));
  PetscCheck(!diagonalscale,PetscObjectComm((PetscObject)ksp),PETSC_ERR_SUP,"Krylov method %s does not support diagonal scaling",((PetscObject)ksp)->type_name);

  CHKERRQ(PCGetOperators(ksp->pc,&Amat,&Pmat));
  CHKERRQ(PetscObjectTypeCompare((PetscObject)ksp->pc,PCNONE,&nopreconditioner));

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
  if (lsqr->se) {
    CHKERRQ(VecSet(lsqr->se,0.0));
  }

  /* Compute initial residual, temporarily use work vector u */
  if (!ksp->guess_zero) {
    CHKERRQ(KSP_MatMult(ksp,Amat,X,U));       /*   u <- b - Ax     */
    CHKERRQ(VecAYPX(U,-1.0,B));
  } else {
    CHKERRQ(VecCopy(B,U));            /*   u <- b (x is 0) */
  }

  /* Test for nothing to do */
  CHKERRQ(VecNorm(U,NORM_2,&rnorm));
  KSPCheckNorm(ksp,rnorm);
  CHKERRQ(PetscObjectSAWsTakeAccess((PetscObject)ksp));
  ksp->its   = 0;
  ksp->rnorm = rnorm;
  CHKERRQ(PetscObjectSAWsGrantAccess((PetscObject)ksp));
  CHKERRQ(KSPLogResidualHistory(ksp,rnorm));
  CHKERRQ(KSPMonitor(ksp,0,rnorm));
  CHKERRQ((*ksp->converged)(ksp,0,rnorm,&ksp->reason,ksp->cnvP));
  if (ksp->reason) PetscFunctionReturn(0);

  beta = rnorm;
  CHKERRQ(VecScale(U,1.0/beta));
  CHKERRQ(KSP_MatMultHermitianTranspose(ksp,Amat,U,V));
  if (nopreconditioner) {
    CHKERRQ(VecNorm(V,NORM_2,&alpha));
    KSPCheckNorm(ksp,rnorm);
  } else {
    /* this is an application of the preconditioner for the normal equations; not the operator, see the manual page */
    CHKERRQ(PCApply(ksp->pc,V,Z));
    CHKERRQ(VecDotRealPart(V,Z,&alpha));
    if (alpha <= 0.0) {
      ksp->reason = KSP_DIVERGED_BREAKDOWN;
      PetscFunctionReturn(0);
    }
    alpha = PetscSqrtReal(alpha);
    CHKERRQ(VecScale(Z,1.0/alpha));
  }
  CHKERRQ(VecScale(V,1.0/alpha));

  if (nopreconditioner) {
    CHKERRQ(VecCopy(V,W));
  } else {
    CHKERRQ(VecCopy(Z,W));
  }

  if (lsqr->exact_norm) {
    CHKERRQ(MatNorm(Amat,NORM_FROBENIUS,&lsqr->anorm));
  } else lsqr->anorm = 0.0;

  lsqr->arnorm = alpha * beta;
  phibar       = beta;
  rhobar       = alpha;
  i            = 0;
  do {
    if (nopreconditioner) {
      CHKERRQ(KSP_MatMult(ksp,Amat,V,U1));
    } else {
      CHKERRQ(KSP_MatMult(ksp,Amat,Z,U1));
    }
    CHKERRQ(VecAXPY(U1,-alpha,U));
    CHKERRQ(VecNorm(U1,NORM_2,&beta));
    KSPCheckNorm(ksp,beta);
    if (beta > 0.0) {
      CHKERRQ(VecScale(U1,1.0/beta)); /* beta*U1 = Amat*V - alpha*U */
      if (!lsqr->exact_norm) {
        lsqr->anorm = PetscSqrtReal(PetscSqr(lsqr->anorm) + PetscSqr(alpha) + PetscSqr(beta));
      }
    }

    CHKERRQ(KSP_MatMultHermitianTranspose(ksp,Amat,U1,V1));
    CHKERRQ(VecAXPY(V1,-beta,V));
    if (nopreconditioner) {
      CHKERRQ(VecNorm(V1,NORM_2,&alpha));
      KSPCheckNorm(ksp,alpha);
    } else {
      CHKERRQ(PCApply(ksp->pc,V1,Z));
      CHKERRQ(VecDotRealPart(V1,Z,&alpha));
      if (alpha <= 0.0) {
        ksp->reason = KSP_DIVERGED_BREAKDOWN;
        break;
      }
      alpha = PetscSqrtReal(alpha);
      CHKERRQ(VecScale(Z,1.0/alpha));
    }
    CHKERRQ(VecScale(V1,1.0/alpha)); /* alpha*V1 = Amat^T*U1 - beta*V */
    rho    = PetscSqrtScalar(rhobar*rhobar + beta*beta);
    c      = rhobar / rho;
    s      = beta / rho;
    theta  = s * alpha;
    rhobar = -c * alpha;
    phi    = c * phibar;
    phibar = s * phibar;
    tau    = s * phi;

    CHKERRQ(VecAXPY(X,phi/rho,W));  /*    x <- x + (phi/rho) w   */

    if (lsqr->se) {
      CHKERRQ(VecCopy(W,W2));
      CHKERRQ(VecSquare(W2));
      CHKERRQ(VecScale(W2,1.0/(rho*rho)));
      CHKERRQ(VecAXPY(lsqr->se, 1.0, W2)); /* lsqr->se <- lsqr->se + (w^2/rho^2) */
    }
    if (nopreconditioner) {
      CHKERRQ(VecAYPX(W,-theta/rho,V1));  /* w <- v - (theta/rho) w */
    } else {
      CHKERRQ(VecAYPX(W,-theta/rho,Z));   /* w <- z - (theta/rho) w */
    }

    lsqr->arnorm = alpha*PetscAbsScalar(tau);
    rnorm        = PetscRealPart(phibar);

    CHKERRQ(PetscObjectSAWsTakeAccess((PetscObject)ksp));
    ksp->its++;
    ksp->rnorm = rnorm;
    CHKERRQ(PetscObjectSAWsGrantAccess((PetscObject)ksp));
    CHKERRQ(KSPLogResidualHistory(ksp,rnorm));
    CHKERRQ(KSPMonitor(ksp,i+1,rnorm));
    CHKERRQ((*ksp->converged)(ksp,i+1,rnorm,&ksp->reason,ksp->cnvP));
    if (ksp->reason) break;
    SWAP(U1,U,TMP);
    SWAP(V1,V,TMP);

    i++;
  } while (i<ksp->max_it);
  if (i >= ksp->max_it && !ksp->reason) ksp->reason = KSP_DIVERGED_ITS;

  /* Finish off the standard error estimates */
  if (lsqr->se) {
    tmp  = 1.0;
    CHKERRQ(MatGetSize(Amat,&size1,&size2));
    if (size1 > size2) tmp = size1 - size2;
    tmp  = rnorm / PetscSqrtScalar(tmp);
    CHKERRQ(VecSqrtAbs(lsqr->se));
    CHKERRQ(VecScale(lsqr->se,tmp));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode KSPDestroy_LSQR(KSP ksp)
{
  KSP_LSQR       *lsqr = (KSP_LSQR*)ksp->data;

  PetscFunctionBegin;
  /* Free work vectors */
  if (lsqr->vwork_n) {
    CHKERRQ(VecDestroyVecs(lsqr->nwork_n,&lsqr->vwork_n));
  }
  if (lsqr->vwork_m) {
    CHKERRQ(VecDestroyVecs(lsqr->nwork_m,&lsqr->vwork_m));
  }
  CHKERRQ(VecDestroy(&lsqr->se));
  /* Revert convergence test */
  CHKERRQ(KSPSetConvergenceTest(ksp,lsqr->converged,lsqr->cnvP,lsqr->convergeddestroy));
  /* Free the KSP_LSQR context */
  CHKERRQ(PetscFree(ksp->data));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)ksp,"KSPLSQRMonitorResidual_C",NULL));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)ksp,"KSPLSQRMonitorResidualDrawLG_C",NULL));
  PetscFunctionReturn(0);
}

/*@
   KSPLSQRSetComputeStandardErrorVec - Compute vector of standard error estimates during KSPSolve_LSQR().

   Not Collective

   Input Parameters:
+  ksp   - iterative context
-  flg   - compute the vector of standard estimates or not

   Developer notes:
   Vaclav: I'm not sure whether this vector is useful for anything.

   Level: intermediate

.seealso: KSPSolve(), KSPLSQR, KSPLSQRGetStandardErrorVec()
@*/
PetscErrorCode  KSPLSQRSetComputeStandardErrorVec(KSP ksp, PetscBool flg)
{
  KSP_LSQR       *lsqr = (KSP_LSQR*)ksp->data;

  PetscFunctionBegin;
  lsqr->se_flg = flg;
  PetscFunctionReturn(0);
}

/*@
   KSPLSQRSetExactMatNorm - Compute exact matrix norm instead of iteratively refined estimate.

   Not Collective

   Input Parameters:
+  ksp   - iterative context
-  flg   - compute exact matrix norm or not

   Notes:
   By default, flg=PETSC_FALSE. This is usually preferred to avoid possibly expensive computation of the norm.
   For flg=PETSC_TRUE, we call MatNorm(Amat,NORM_FROBENIUS,&lsqr->anorm) which will work only for some types of explicitly assembled matrices.
   This can affect convergence rate as KSPLSQRConvergedDefault() assumes different value of ||A|| used in normal equation stopping criterion.

   Level: intermediate

.seealso: KSPSolve(), KSPLSQR, KSPLSQRGetNorms(), KSPLSQRConvergedDefault()
@*/
PetscErrorCode  KSPLSQRSetExactMatNorm(KSP ksp, PetscBool flg)
{
  KSP_LSQR       *lsqr = (KSP_LSQR*)ksp->data;

  PetscFunctionBegin;
  lsqr->exact_norm = flg;
  PetscFunctionReturn(0);
}

/*@
   KSPLSQRGetStandardErrorVec - Get vector of standard error estimates.
   Only available if -ksp_lsqr_set_standard_error was set to true
   or KSPLSQRSetComputeStandardErrorVec(ksp, PETSC_TRUE) was called.
   Otherwise returns NULL.

   Not Collective

   Input Parameters:
.  ksp   - iterative context

   Output Parameters:
.  se - vector of standard estimates

   Options Database Keys:
.   -ksp_lsqr_set_standard_error  - set standard error estimates of solution

   Developer notes:
   Vaclav: I'm not sure whether this vector is useful for anything.

   Level: intermediate

.seealso: KSPSolve(), KSPLSQR, KSPLSQRSetComputeStandardErrorVec()
@*/
PetscErrorCode  KSPLSQRGetStandardErrorVec(KSP ksp,Vec *se)
{
  KSP_LSQR *lsqr = (KSP_LSQR*)ksp->data;

  PetscFunctionBegin;
  *se = lsqr->se;
  PetscFunctionReturn(0);
}

/*@
   KSPLSQRGetNorms - Get norm estimates that LSQR computes internally during KSPSolve().

   Not Collective

   Input Parameter:
.  ksp   - iterative context

   Output Parameters:
+  arnorm - good estimate of norm((A*inv(Pmat))'*r), where r = A*x - b, used in specific stopping criterion
-  anorm - poor estimate of norm(A*inv(Pmat),'fro') used in specific stopping criterion

   Notes:
   Output parameters are meaningful only after KSPSolve().
   These are the same quantities as normar and norma in MATLAB's lsqr(), whose output lsvec is a vector of normar / norma for all iterations.
   If -ksp_lsqr_exact_mat_norm is set or KSPLSQRSetExactMatNorm(ksp, PETSC_TRUE) called, then anorm is exact Frobenius norm.

   Level: intermediate

.seealso: KSPSolve(), KSPLSQR, KSPLSQRSetExactMatNorm()
@*/
PetscErrorCode  KSPLSQRGetNorms(KSP ksp,PetscReal *arnorm, PetscReal *anorm)
{
  KSP_LSQR       *lsqr = (KSP_LSQR*)ksp->data;

  PetscFunctionBegin;
  if (arnorm)   *arnorm = lsqr->arnorm;
  if (anorm)    *anorm = lsqr->anorm;
  PetscFunctionReturn(0);
}

PetscErrorCode KSPLSQRMonitorResidual_LSQR(KSP ksp, PetscInt n, PetscReal rnorm, PetscViewerAndFormat *vf)
{
  KSP_LSQR          *lsqr  = (KSP_LSQR*)ksp->data;
  PetscViewer       viewer = vf->viewer;
  PetscViewerFormat format = vf->format;
  char              normtype[256];
  PetscInt          tablevel;
  const char        *prefix;

  PetscFunctionBegin;
  CHKERRQ(PetscObjectGetTabLevel((PetscObject) ksp, &tablevel));
  CHKERRQ(PetscObjectGetOptionsPrefix((PetscObject) ksp, &prefix));
  CHKERRQ(PetscStrncpy(normtype, KSPNormTypes[ksp->normtype], sizeof(normtype)));
  CHKERRQ(PetscStrtolower(normtype));
  CHKERRQ(PetscViewerPushFormat(viewer, format));
  CHKERRQ(PetscViewerASCIIAddTab(viewer, tablevel));
  if (n == 0 && prefix) CHKERRQ(PetscViewerASCIIPrintf(viewer, "  Residual norm, norm of normal equations, and matrix norm for %s solve.\n", prefix));
  if (!n) {
    CHKERRQ(PetscViewerASCIIPrintf(viewer,"%3D KSP resid norm %14.12e\n",n,(double)rnorm));
  } else {
    CHKERRQ(PetscViewerASCIIPrintf(viewer,"%3D KSP resid norm %14.12e normal eq resid norm %14.12e matrix norm %14.12e\n",n,(double)rnorm,(double)lsqr->arnorm,(double)lsqr->anorm));
  }
  CHKERRQ(PetscViewerASCIISubtractTab(viewer, tablevel));
  CHKERRQ(PetscViewerPopFormat(viewer));
  PetscFunctionReturn(0);
}

/*@C
  KSPLSQRMonitorResidual - Prints the residual norm, as well as the normal equation residual norm, at each iteration of an iterative solver.

  Collective on ksp

  Input Parameters:
+ ksp   - iterative context
. n     - iteration number
. rnorm - 2-norm (preconditioned) residual value (may be estimated).
- vf    - The viewer context

  Options Database Key:
. -ksp_lsqr_monitor - Activates KSPLSQRMonitorResidual()

  Level: intermediate

.seealso: KSPMonitorSet(), KSPMonitorResidual(),KSPMonitorTrueResidualMaxNorm()
@*/
PetscErrorCode KSPLSQRMonitorResidual(KSP ksp, PetscInt n, PetscReal rnorm, PetscViewerAndFormat *vf)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp, KSP_CLASSID, 1);
  PetscValidPointer(vf, 4);
  PetscValidHeaderSpecific(vf->viewer, PETSC_VIEWER_CLASSID, 4);
  CHKERRQ(PetscTryMethod(ksp, "KSPLSQRMonitorResidual_C", (KSP,PetscInt,PetscReal,PetscViewerAndFormat*), (ksp,n,rnorm,vf)));
  PetscFunctionReturn(0);
}

PetscErrorCode KSPLSQRMonitorResidualDrawLG_LSQR(KSP ksp, PetscInt n, PetscReal rnorm, PetscViewerAndFormat *vf)
{
  KSP_LSQR           *lsqr  = (KSP_LSQR*)ksp->data;
  PetscViewer        viewer = vf->viewer;
  PetscViewerFormat  format = vf->format;
  PetscDrawLG        lg     = vf->lg;
  KSPConvergedReason reason;
  PetscReal          x[2], y[2];

  PetscFunctionBegin;
  CHKERRQ(PetscViewerPushFormat(viewer, format));
  if (!n) CHKERRQ(PetscDrawLGReset(lg));
  x[0] = (PetscReal) n;
  if (rnorm > 0.0) y[0] = PetscLog10Real(rnorm);
  else y[0] = -15.0;
  x[1] = (PetscReal) n;
  if (lsqr->arnorm > 0.0) y[1] = PetscLog10Real(lsqr->arnorm);
  else y[1] = -15.0;
  CHKERRQ(PetscDrawLGAddPoint(lg, x, y));
  CHKERRQ(KSPGetConvergedReason(ksp, &reason));
  if (n <= 20 || !(n % 5) || reason) {
    CHKERRQ(PetscDrawLGDraw(lg));
    CHKERRQ(PetscDrawLGSave(lg));
  }
  CHKERRQ(PetscViewerPopFormat(viewer));
  PetscFunctionReturn(0);
}

/*@C
  KSPLSQRMonitorResidualDrawLG - Plots the true residual norm at each iteration of an iterative solver.

  Collective on ksp

  Input Parameters:
+ ksp   - iterative context
. n     - iteration number
. rnorm - 2-norm (preconditioned) residual value (may be estimated).
- vf    - The viewer context

  Options Database Key:
. -ksp_lsqr_monitor draw::draw_lg - Activates KSPMonitorTrueResidualDrawLG()

  Level: intermediate

.seealso: KSPMonitorSet(), KSPMonitorTrueResidual()
@*/
PetscErrorCode KSPLSQRMonitorResidualDrawLG(KSP ksp, PetscInt n, PetscReal rnorm, PetscViewerAndFormat *vf)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp, KSP_CLASSID, 1);
  PetscValidPointer(vf, 4);
  PetscValidHeaderSpecific(vf->viewer, PETSC_VIEWER_CLASSID, 4);
  PetscValidHeaderSpecific(vf->lg, PETSC_DRAWLG_CLASSID, 4);
  CHKERRQ(PetscTryMethod(ksp, "KSPLSQRMonitorResidualDrawLG_C", (KSP,PetscInt,PetscReal,PetscViewerAndFormat*), (ksp,n,rnorm,vf)));
  PetscFunctionReturn(0);
}

/*@C
  KSPLSQRMonitorResidualDrawLGCreate - Creates the plotter for the LSQR residual and normal eqn residual.

  Collective on ksp

  Input Parameters:
+ viewer - The PetscViewer
. format - The viewer format
- ctx    - An optional user context

  Output Parameter:
. vf    - The viewer context

  Level: intermediate

.seealso: KSPMonitorSet(), KSPLSQRMonitorResidual()
@*/
PetscErrorCode KSPLSQRMonitorResidualDrawLGCreate(PetscViewer viewer, PetscViewerFormat format, void *ctx, PetscViewerAndFormat **vf)
{
  const char    *names[] = {"residual", "normal eqn residual"};

  PetscFunctionBegin;
  CHKERRQ(PetscViewerAndFormatCreate(viewer, format, vf));
  (*vf)->data = ctx;
  CHKERRQ(KSPMonitorLGCreate(PetscObjectComm((PetscObject) viewer), NULL, NULL, "Log Residual Norm", 2, names, PETSC_DECIDE, PETSC_DECIDE, 400, 300, &(*vf)->lg));
  PetscFunctionReturn(0);
}

PetscErrorCode KSPSetFromOptions_LSQR(PetscOptionItems *PetscOptionsObject,KSP ksp)
{
  KSP_LSQR       *lsqr = (KSP_LSQR*)ksp->data;

  PetscFunctionBegin;
  CHKERRQ(PetscOptionsHead(PetscOptionsObject,"KSP LSQR Options"));
  CHKERRQ(PetscOptionsBool("-ksp_lsqr_compute_standard_error","Set Standard Error Estimates of Solution","KSPLSQRSetComputeStandardErrorVec",lsqr->se_flg,&lsqr->se_flg,NULL));
  CHKERRQ(PetscOptionsBool("-ksp_lsqr_exact_mat_norm","Compute exact matrix norm instead of iteratively refined estimate","KSPLSQRSetExactMatNorm",lsqr->exact_norm,&lsqr->exact_norm,NULL));
  CHKERRQ(KSPMonitorSetFromOptions(ksp, "-ksp_lsqr_monitor", "lsqr_residual", NULL));
  CHKERRQ(PetscOptionsTail());
  PetscFunctionReturn(0);
}

PetscErrorCode KSPView_LSQR(KSP ksp,PetscViewer viewer)
{
  KSP_LSQR       *lsqr = (KSP_LSQR*)ksp->data;
  PetscBool      iascii;

  PetscFunctionBegin;
  CHKERRQ(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii));
  if (iascii) {
    if (lsqr->se) {
      PetscReal rnorm;
      CHKERRQ(VecNorm(lsqr->se,NORM_2,&rnorm));
      CHKERRQ(PetscViewerASCIIPrintf(viewer,"  norm of standard error %g, iterations %d\n",(double)rnorm,ksp->its));
    } else {
      CHKERRQ(PetscViewerASCIIPrintf(viewer,"  standard error not computed\n"));
    }
    if (lsqr->exact_norm) {
      CHKERRQ(PetscViewerASCIIPrintf(viewer,"  using exact matrix norm\n"));
    } else {
      CHKERRQ(PetscViewerASCIIPrintf(viewer,"  using inexact matrix norm\n"));
    }
  }
  PetscFunctionReturn(0);
}

/*@C
   KSPLSQRConvergedDefault - Determines convergence of the LSQR Krylov method.

   Collective on ksp

   Input Parameters:
+  ksp   - iterative context
.  n     - iteration number
.  rnorm - 2-norm residual value (may be estimated)
-  ctx - convergence context which must be created by KSPConvergedDefaultCreate()

   reason is set to:
+   positive - if the iteration has converged;
.   negative - if residual norm exceeds divergence threshold;
-   0 - otherwise.

   Notes:
   KSPConvergedDefault() is called first to check for convergence in A*x=b.
   If that does not determine convergence then checks convergence for the least squares problem, i.e. in min{|b-A*x|}.
   Possible convergence for the least squares problem (which is based on the residual of the normal equations) are KSP_CONVERGED_RTOL_NORMAL norm and KSP_CONVERGED_ATOL_NORMAL.
   KSP_CONVERGED_RTOL_NORMAL is returned if ||A'*r|| < rtol * ||A|| * ||r||.
   Matrix norm ||A|| is iteratively refined estimate, see KSPLSQRGetNorms().
   This criterion is now largely compatible with that in MATLAB lsqr().

   Level: intermediate

.seealso: KSPLSQR, KSPSetConvergenceTest(), KSPSetTolerances(), KSPConvergedSkip(), KSPConvergedReason, KSPGetConvergedReason(),
          KSPConvergedDefaultSetUIRNorm(), KSPConvergedDefaultSetUMIRNorm(), KSPConvergedDefaultCreate(), KSPConvergedDefaultDestroy(), KSPConvergedDefault(), KSPLSQRGetNorms(), KSPLSQRSetExactMatNorm()
@*/
PetscErrorCode  KSPLSQRConvergedDefault(KSP ksp,PetscInt n,PetscReal rnorm,KSPConvergedReason *reason,void *ctx)
{
  KSP_LSQR       *lsqr = (KSP_LSQR*)ksp->data;

  PetscFunctionBegin;
  /* check for convergence in A*x=b */
  CHKERRQ(KSPConvergedDefault(ksp,n,rnorm,reason,ctx));
  if (!n || *reason) PetscFunctionReturn(0);

  /* check for convergence in min{|b-A*x|} */
  if (lsqr->arnorm < ksp->abstol) {
    CHKERRQ(PetscInfo(ksp,"LSQR solver has converged. Normal equation residual %14.12e is less than absolute tolerance %14.12e at iteration %D\n",(double)lsqr->arnorm,(double)ksp->abstol,n));
    *reason = KSP_CONVERGED_ATOL_NORMAL;
  } else if (lsqr->arnorm < ksp->rtol * lsqr->anorm * rnorm) {
    CHKERRQ(PetscInfo(ksp,"LSQR solver has converged. Normal equation residual %14.12e is less than rel. tol. %14.12e times %s Frobenius norm of matrix %14.12e times residual %14.12e at iteration %D\n",(double)lsqr->arnorm,(double)ksp->rtol,lsqr->exact_norm?"exact":"approx.",(double)lsqr->anorm,(double)rnorm,n));
    *reason = KSP_CONVERGED_RTOL_NORMAL;
  }
  PetscFunctionReturn(0);
}

/*MC
     KSPLSQR - This implements LSQR

   Options Database Keys:
+   -ksp_lsqr_set_standard_error  - set standard error estimates of solution, see KSPLSQRSetComputeStandardErrorVec() and KSPLSQRGetStandardErrorVec()
.   -ksp_lsqr_exact_mat_norm - compute exact matrix norm instead of iteratively refined estimate, see KSPLSQRSetExactMatNorm()
-   -ksp_lsqr_monitor - monitor residual norm, norm of residual of normal equations A'*A x = A' b, and estimate of matrix norm ||A||

   Level: beginner

   Notes:
     Supports non-square (rectangular) matrices.

     This variant, when applied with no preconditioning is identical to the original algorithm in exact arithematic; however, in practice, with no preconditioning
     due to inexact arithmetic, it can converge differently. Hence when no preconditioner is used (PCType PCNONE) it automatically reverts to the original algorithm.

     With the PETSc built-in preconditioners, such as ICC, one should call KSPSetOperators(ksp,A,A'*A)) since the preconditioner needs to work
     for the normal equations A'*A.

     Supports only left preconditioning.

     For least squares problems with nonzero residual A*x - b, there are additional convergence tests for the residual of the normal equations, A'*(b - Ax), see KSPLSQRConvergedDefault().

   References:
.  * - The original unpreconditioned algorithm can be found in Paige and Saunders, ACM Transactions on Mathematical Software, Vol 8, 1982.

     In exact arithmetic the LSQR method (with no preconditioning) is identical to the KSPCG algorithm applied to the normal equations.
     The preconditioned variant was implemented by Bas van't Hof and is essentially a left preconditioning for the Normal Equations. It appears the implementation with preconditioner
     track the true norm of the residual and uses that in the convergence test.

   Developer Notes:
    How is this related to the KSPCGNE implementation? One difference is that KSPCGNE applies
            the preconditioner transpose times the preconditioner,  so one does not need to pass A'*A as the third argument to KSPSetOperators().

.seealso:  KSPCreate(), KSPSetType(), KSPType (for list of available types), KSP, KSPSolve(), KSPLSQRConvergedDefault(), KSPLSQRSetComputeStandardErrorVec(), KSPLSQRGetStandardErrorVec(), KSPLSQRSetExactMatNorm()

M*/
PETSC_EXTERN PetscErrorCode KSPCreate_LSQR(KSP ksp)
{
  KSP_LSQR       *lsqr;
  void           *ctx;

  PetscFunctionBegin;
  CHKERRQ(PetscNewLog(ksp,&lsqr));
  lsqr->se     = NULL;
  lsqr->se_flg = PETSC_FALSE;
  lsqr->exact_norm = PETSC_FALSE;
  lsqr->anorm  = -1.0;
  lsqr->arnorm = -1.0;
  ksp->data    = (void*)lsqr;
  CHKERRQ(KSPSetSupportedNorm(ksp,KSP_NORM_UNPRECONDITIONED,PC_LEFT,3));

  ksp->ops->setup          = KSPSetUp_LSQR;
  ksp->ops->solve          = KSPSolve_LSQR;
  ksp->ops->destroy        = KSPDestroy_LSQR;
  ksp->ops->setfromoptions = KSPSetFromOptions_LSQR;
  ksp->ops->view           = KSPView_LSQR;

  /* Backup current convergence test; remove destroy routine from KSP to prevent destroying the convergence context in KSPSetConvergenceTest() */
  CHKERRQ(KSPGetAndClearConvergenceTest(ksp,&lsqr->converged,&lsqr->cnvP,&lsqr->convergeddestroy));
  /* Override current convergence test */
  CHKERRQ(KSPConvergedDefaultCreate(&ctx));
  CHKERRQ(KSPSetConvergenceTest(ksp,KSPLSQRConvergedDefault,ctx,KSPConvergedDefaultDestroy));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)ksp,"KSPLSQRMonitorResidual_C",KSPLSQRMonitorResidual_LSQR));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)ksp,"KSPLSQRMonitorResidualDrawLG_C",KSPLSQRMonitorResidualDrawLG_LSQR));
  PetscFunctionReturn(0);
}
