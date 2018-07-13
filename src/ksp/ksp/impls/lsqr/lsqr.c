
/* lourens.vanzanen@shell.com contributed the standard error estimates of the solution, Jul 25, 2006 */
/* Bas van't Hof contributed the preconditioned aspects Feb 10, 2010 */

#define SWAP(a,b,c) { c = a; a = b; b = c; }

#include <petsc/private/kspimpl.h>

typedef struct {
  PetscInt  nwork_n,nwork_m;
  Vec       *vwork_m;   /* work vectors of length m, where the system is size m x n */
  Vec       *vwork_n;   /* work vectors of length n */
  Vec       se;         /* Optional standard error vector */
  PetscBool se_flg;     /* flag for -ksp_lsqr_set_standard_error */
  PetscReal arnorm;     /* Good estimate of norm((A*inv(Pmat))'*r), where r = A*x - b, used in specific stopping criterion */
  PetscReal anorm;      /* Poor estimate of norm(A*inv(Pmat),'fro') used in specific stopping criterion */
  PetscReal rhs_norm;   /* Norm of the right hand side */
} KSP_LSQR;

static PetscErrorCode  VecSquare(Vec v)
{
  PetscErrorCode ierr;
  PetscScalar    *x;
  PetscInt       i, n;

  PetscFunctionBegin;
  ierr = VecGetLocalSize(v, &n);CHKERRQ(ierr);
  ierr = VecGetArray(v, &x);CHKERRQ(ierr);
  for (i = 0; i < n; i++) x[i] *= PetscConj(x[i]);
  ierr = VecRestoreArray(v, &x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode KSPSetUp_LSQR(KSP ksp)
{
  PetscErrorCode ierr;
  KSP_LSQR       *lsqr = (KSP_LSQR*)ksp->data;
  PetscBool      nopreconditioner;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)ksp->pc,PCNONE,&nopreconditioner);CHKERRQ(ierr);
  /*  nopreconditioner =PETSC_FALSE; */

  lsqr->nwork_m = 2;
  if (lsqr->vwork_m) {
    ierr = VecDestroyVecs(lsqr->nwork_m,&lsqr->vwork_m);CHKERRQ(ierr);
  }
  if (nopreconditioner) lsqr->nwork_n = 4;
  else lsqr->nwork_n = 5;

  if (lsqr->vwork_n) {
    ierr = VecDestroyVecs(lsqr->nwork_n,&lsqr->vwork_n);CHKERRQ(ierr);
  }
  ierr = KSPCreateVecs(ksp,lsqr->nwork_n,&lsqr->vwork_n,lsqr->nwork_m,&lsqr->vwork_m);CHKERRQ(ierr);
  if (lsqr->se_flg && !lsqr->se) {
    /* lsqr->se is not set by user, get it from pmat */
    Vec *se;
    ierr     = KSPCreateVecs(ksp,1,&se,0,NULL);CHKERRQ(ierr);
    lsqr->se = *se;
    ierr     = PetscFree(se);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode KSPSolve_LSQR(KSP ksp)
{
  PetscErrorCode ierr;
  PetscInt       i,size1,size2;
  PetscScalar    rho,rhobar,phi,phibar,theta,c,s,tmp,tau;
  PetscReal      beta,alpha,rnorm;
  Vec            X,B,V,V1,U,U1,TMP,W,W2,SE,Z = NULL;
  Mat            Amat,Pmat;
  KSP_LSQR       *lsqr = (KSP_LSQR*)ksp->data;
  PetscBool      diagonalscale,nopreconditioner;

  PetscFunctionBegin;
  ierr = PCGetDiagonalScale(ksp->pc,&diagonalscale);CHKERRQ(ierr);
  if (diagonalscale) SETERRQ1(PetscObjectComm((PetscObject)ksp),PETSC_ERR_SUP,"Krylov method %s does not support diagonal scaling",((PetscObject)ksp)->type_name);

  ierr = PCGetOperators(ksp->pc,&Amat,&Pmat);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)ksp->pc,PCNONE,&nopreconditioner);CHKERRQ(ierr);

  /*  nopreconditioner =PETSC_FALSE; */
  /* Calculate norm of right hand side */
  ierr = VecNorm(ksp->vec_rhs,NORM_2,&lsqr->rhs_norm);CHKERRQ(ierr);

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
  SE = lsqr->se;
  if (SE) {
    ierr = VecGetSize(SE,&size1);CHKERRQ(ierr);
    ierr = VecGetSize(X,&size2);CHKERRQ(ierr);
    if (size1 != size2) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Standard error vector (size %D) does not match solution vector (size %D)",size1,size2);
    ierr = VecSet(SE,0.0);CHKERRQ(ierr);
  }

  /* Compute initial residual, temporarily use work vector u */
  if (!ksp->guess_zero) {
    ierr = KSP_MatMult(ksp,Amat,X,U);CHKERRQ(ierr);       /*   u <- b - Ax     */
    ierr = VecAYPX(U,-1.0,B);CHKERRQ(ierr);
  } else {
    ierr = VecCopy(B,U);CHKERRQ(ierr);            /*   u <- b (x is 0) */
  }

  /* Test for nothing to do */
  ierr       = VecNorm(U,NORM_2,&rnorm);CHKERRQ(ierr);
  ierr       = PetscObjectSAWsTakeAccess((PetscObject)ksp);CHKERRQ(ierr);
  ksp->its   = 0;
  ksp->rnorm = rnorm;
  ierr       = PetscObjectSAWsGrantAccess((PetscObject)ksp);CHKERRQ(ierr);
  ierr = KSPLogResidualHistory(ksp,rnorm);CHKERRQ(ierr);
  ierr = KSPMonitor(ksp,0,rnorm);CHKERRQ(ierr);
  ierr = (*ksp->converged)(ksp,0,rnorm,&ksp->reason,ksp->cnvP);CHKERRQ(ierr);
  if (ksp->reason) PetscFunctionReturn(0);

  beta = rnorm;
  ierr = VecScale(U,1.0/beta);CHKERRQ(ierr);
  ierr = KSP_MatMultTranspose(ksp,Amat,U,V);CHKERRQ(ierr);
  if (nopreconditioner) {
    ierr = VecNorm(V,NORM_2,&alpha);CHKERRQ(ierr);
  } else {
    /* this is an application of the preconditioner for the normal equations; not the operator, see the manual page */
    ierr = PCApply(ksp->pc,V,Z);CHKERRQ(ierr);
    ierr = VecDotRealPart(V,Z,&alpha);CHKERRQ(ierr);
    if (alpha <= 0.0) {
      ksp->reason = KSP_DIVERGED_BREAKDOWN;
      PetscFunctionReturn(0);
    }
    alpha = PetscSqrtReal(alpha);
    ierr  = VecScale(Z,1.0/alpha);CHKERRQ(ierr);
  }
  ierr = VecScale(V,1.0/alpha);CHKERRQ(ierr);

  if (nopreconditioner) {
    ierr = VecCopy(V,W);CHKERRQ(ierr);
  } else {
    ierr = VecCopy(Z,W);CHKERRQ(ierr);
  }

  lsqr->anorm  = 0.0;
  lsqr->arnorm = alpha * beta;
  phibar       = beta;
  rhobar       = alpha;
  i            = 0;
  do {
    if (nopreconditioner) {
      ierr = KSP_MatMult(ksp,Amat,V,U1);CHKERRQ(ierr);
    } else {
      ierr = KSP_MatMult(ksp,Amat,Z,U1);CHKERRQ(ierr);
    }
    ierr = VecAXPY(U1,-alpha,U);CHKERRQ(ierr);
    ierr = VecNorm(U1,NORM_2,&beta);CHKERRQ(ierr);
    if (beta > 0.0) {
      ierr = VecScale(U1,1.0/beta);CHKERRQ(ierr); /* beta*U1 = Amat*V - alpha*U */
      lsqr->anorm = PetscSqrtScalar(PetscSqr(lsqr->anorm) + PetscSqr(alpha) + PetscSqr(beta));
    }

    ierr = KSP_MatMultTranspose(ksp,Amat,U1,V1);CHKERRQ(ierr);
    ierr = VecAXPY(V1,-beta,V);CHKERRQ(ierr);
    if (nopreconditioner) {
      ierr = VecNorm(V1,NORM_2,&alpha);CHKERRQ(ierr);
    } else {
      ierr = PCApply(ksp->pc,V1,Z);CHKERRQ(ierr);
      ierr = VecDotRealPart(V1,Z,&alpha);CHKERRQ(ierr);
      if (alpha <= 0.0) {
        ksp->reason = KSP_DIVERGED_BREAKDOWN;
        break;
      }
      alpha = PetscSqrtReal(alpha);
      ierr  = VecScale(Z,1.0/alpha);CHKERRQ(ierr);
    }
    ierr   = VecScale(V1,1.0/alpha);CHKERRQ(ierr); /* alpha*V1 = Amat^T*U1 - beta*V */
    rho    = PetscSqrtScalar(rhobar*rhobar + beta*beta);
    c      = rhobar / rho;
    s      = beta / rho;
    theta  = s * alpha;
    rhobar = -c * alpha;
    phi    = c * phibar;
    phibar = s * phibar;
    tau    = s * phi;

    ierr = VecAXPY(X,phi/rho,W);CHKERRQ(ierr);  /*    x <- x + (phi/rho) w   */

    if (SE) {
      ierr = VecCopy(W,W2);CHKERRQ(ierr);
      ierr = VecSquare(W2);CHKERRQ(ierr);
      ierr = VecScale(W2,1.0/(rho*rho));CHKERRQ(ierr);
      ierr = VecAXPY(SE, 1.0, W2);CHKERRQ(ierr); /* SE <- SE + (w^2/rho^2) */
    }
    if (nopreconditioner) {
      ierr = VecAYPX(W,-theta/rho,V1);CHKERRQ(ierr);  /* w <- v - (theta/rho) w */
    } else {
      ierr = VecAYPX(W,-theta/rho,Z);CHKERRQ(ierr);   /* w <- z - (theta/rho) w */
    }

    lsqr->arnorm = alpha*PetscAbsScalar(tau);
    rnorm        = PetscRealPart(phibar);

    ierr = PetscObjectSAWsTakeAccess((PetscObject)ksp);CHKERRQ(ierr);
    ksp->its++;
    ksp->rnorm = rnorm;
    ierr       = PetscObjectSAWsGrantAccess((PetscObject)ksp);CHKERRQ(ierr);
    ierr = KSPLogResidualHistory(ksp,rnorm);CHKERRQ(ierr);
    ierr = KSPMonitor(ksp,i+1,rnorm);CHKERRQ(ierr);
    ierr = (*ksp->converged)(ksp,i+1,rnorm,&ksp->reason,ksp->cnvP);CHKERRQ(ierr);
    if (ksp->reason) break;
    SWAP(U1,U,TMP);
    SWAP(V1,V,TMP);

    i++;
  } while (i<ksp->max_it);
  if (i >= ksp->max_it && !ksp->reason) ksp->reason = KSP_DIVERGED_ITS;

  /* Finish off the standard error estimates */
  if (SE) {
    tmp  = 1.0;
    ierr = MatGetSize(Amat,&size1,&size2);CHKERRQ(ierr);
    if (size1 > size2) tmp = size1 - size2;
    tmp  = rnorm / PetscSqrtScalar(tmp);
    ierr = VecSqrtAbs(SE);CHKERRQ(ierr);
    ierr = VecScale(SE,tmp);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}


PetscErrorCode KSPDestroy_LSQR(KSP ksp)
{
  KSP_LSQR       *lsqr = (KSP_LSQR*)ksp->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* Free work vectors */
  if (lsqr->vwork_n) {
    ierr = VecDestroyVecs(lsqr->nwork_n,&lsqr->vwork_n);CHKERRQ(ierr);
  }
  if (lsqr->vwork_m) {
    ierr = VecDestroyVecs(lsqr->nwork_m,&lsqr->vwork_m);CHKERRQ(ierr);
  }
  if (lsqr->se_flg) {
    ierr = VecDestroy(&lsqr->se);CHKERRQ(ierr);
  }
  ierr = PetscFree(ksp->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode  KSPLSQRSetStandardErrorVec(KSP ksp, Vec se)
{
  KSP_LSQR       *lsqr = (KSP_LSQR*)ksp->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr     = VecDestroy(&lsqr->se);CHKERRQ(ierr);
  lsqr->se = se;
  PetscFunctionReturn(0);
}

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

   Input Parameters:
.  ksp   - iterative context

   Output Parameters:
+  arnorm - good estimate of norm((A*inv(Pmat))'*r), where r = A*x - b, used in specific stopping criterion
-  anorm - poor estimate of norm(A*inv(Pmat),'fro') used in specific stopping criterion

   Notes:
   These are the same quantities as normar and norma in MATLAB's lsqr().
   This function's output lsvec is a vector of normar / norma for all iterations.

   Level: intermediate

.keywords: KSP, KSPLSQR

.seealso: KSPSolve(), KSPLSQR
@*/
PetscErrorCode  KSPLSQRGetNorms(KSP ksp,PetscReal *arnorm, PetscReal *anorm)
{
  KSP_LSQR       *lsqr = (KSP_LSQR*)ksp->data;

  PetscFunctionBegin;
  if (arnorm)   *arnorm = lsqr->arnorm;
  if (anorm)    *anorm = lsqr->anorm;
  PetscFunctionReturn(0);
}

/*@C
   KSPLSQRMonitorDefault - Print the residual norm at each iteration of the LSQR method,
   norm of the residual of the normal equations A'*A x = A' b, and estimate of matrix norm ||A||.

   Collective on KSP

   Input Parameters:
+  ksp   - iterative context
.  n     - iteration number
.  rnorm - 2-norm (preconditioned) residual value (may be estimated).
-  dummy - viewer and format context

   Level: intermediate

.keywords: KSP, KSPLSQR, default, monitor, residual

.seealso: KSPLSQR, KSPMonitorSet(), KSPMonitorTrueResidualNorm(), KSPMonitorLGResidualNormCreate(), KSPMonitorDefault()
@*/
PetscErrorCode  KSPLSQRMonitorDefault(KSP ksp,PetscInt n,PetscReal rnorm,PetscViewerAndFormat *dummy)
{
  PetscErrorCode ierr;
  PetscViewer    viewer = dummy->viewer;
  KSP_LSQR       *lsqr  = (KSP_LSQR*)ksp->data;

  PetscFunctionBegin;
  ierr = PetscViewerPushFormat(viewer,dummy->format);CHKERRQ(ierr);
  ierr = PetscViewerASCIIAddTab(viewer,((PetscObject)ksp)->tablevel);CHKERRQ(ierr);
  if (!n && ((PetscObject)ksp)->prefix) {
    ierr = PetscViewerASCIIPrintf(viewer,"  Residual norm, norm of normal equations, and matrix norm for %s solve.\n",((PetscObject)ksp)->prefix);CHKERRQ(ierr);
  }

  if (!n) {
    ierr = PetscViewerASCIIPrintf(viewer,"%3D KSP resid norm %14.12e\n",n,rnorm);CHKERRQ(ierr);
  } else {
    ierr = PetscViewerASCIIPrintf(viewer,"%3D KSP resid norm %14.12e normal eq resid norm %14.12e matrix norm %14.12e\n",n,rnorm,lsqr->arnorm,lsqr->anorm);CHKERRQ(ierr);
  }

  ierr = PetscViewerASCIISubtractTab(viewer,((PetscObject)ksp)->tablevel);CHKERRQ(ierr);
  ierr = PetscViewerPopFormat(viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode KSPSetFromOptions_LSQR(PetscOptionItems *PetscOptionsObject,KSP ksp)
{
  PetscErrorCode ierr;
  KSP_LSQR       *lsqr = (KSP_LSQR*)ksp->data;

  PetscFunctionBegin;
  ierr = PetscOptionsHead(PetscOptionsObject,"KSP LSQR Options");CHKERRQ(ierr);
  ierr = PetscOptionsName("-ksp_lsqr_set_standard_error","Set Standard Error Estimates of Solution","KSPLSQRSetStandardErrorVec",&lsqr->se_flg);CHKERRQ(ierr);
  ierr = KSPMonitorSetFromOptions(ksp,"-ksp_lsqr_monitor","Monitor residual norm and norm of residual of normal equations","KSPMonitorSet",KSPLSQRMonitorDefault);CHKERRQ(ierr);
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode KSPView_LSQR(KSP ksp,PetscViewer viewer)
{
  KSP_LSQR       *lsqr = (KSP_LSQR*)ksp->data;
  PetscErrorCode ierr;
  PetscBool      iascii;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
    if (lsqr->se) {
      PetscReal rnorm;
      ierr = KSPLSQRGetStandardErrorVec(ksp,&lsqr->se);CHKERRQ(ierr);
      ierr = VecNorm(lsqr->se,NORM_2,&rnorm);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"  Norm of Standard Error %g, Iterations %D\n",(double)rnorm,ksp->its);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

/*@C
   KSPLSQRConvergedDefault - Determines convergence of the LSQR Krylov method.

   Collective on KSP

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
   This criterion is now largely compatible with that in MATLAB lsqr().

   Level: intermediate

.keywords: KSP, KSPLSQR, default, convergence, residual

.seealso: KSPLSQR, KSPSetConvergenceTest(), KSPSetTolerances(), KSPConvergedSkip(), KSPConvergedReason, KSPGetConvergedReason(),
          KSPConvergedDefaultSetUIRNorm(), KSPConvergedDefaultSetUMIRNorm(), KSPConvergedDefaultCreate(), KSPConvergedDefaultDestroy(), KSPConvergedDefault()
@*/
PetscErrorCode  KSPLSQRConvergedDefault(KSP ksp,PetscInt n,PetscReal rnorm,KSPConvergedReason *reason,void *ctx)
{
  PetscErrorCode ierr;
  KSP_LSQR       *lsqr = (KSP_LSQR*)ksp->data;

  PetscFunctionBegin;
  /* check for convergence in A*x=b */
  ierr = KSPConvergedDefault(ksp,n,rnorm,reason,ctx);CHKERRQ(ierr);
  if (!n || *reason) PetscFunctionReturn(0);

  /* check for convergence in min{|b-A*x|} */
  if (lsqr->arnorm < ksp->abstol) {
    ierr = PetscInfo3(ksp,"LSQR solver has converged. Normal equation residual %14.12e is less than absolute tolerance %14.12e at iteration %D\n",(double)lsqr->arnorm,(double)ksp->abstol,n);CHKERRQ(ierr);
    *reason = KSP_CONVERGED_ATOL_NORMAL;
  } else if (lsqr->arnorm < ksp->rtol * lsqr->anorm * rnorm) {
    ierr = PetscInfo5(ksp,"LSQR solver has converged. Normal equation residual %14.12e is less than rel. tol. %14.12e times approx. Frobenius norm of matrix %14.12e times residual %14.12e at iteration %D\n",(double)lsqr->arnorm,(double)ksp->rtol,(double)lsqr->anorm,(double)rnorm,n);CHKERRQ(ierr);
    *reason = KSP_CONVERGED_RTOL_NORMAL;
  }
  PetscFunctionReturn(0);
}

/*MC
     KSPLSQR - This implements LSQR

   Options Database Keys:
+   -ksp_lsqr_set_standard_error  - Set Standard Error Estimates of Solution see KSPLSQRSetStandardErrorVec()
.   -ksp_lsqr_monitor - Monitor residual norm and norm of residual of normal equations
-   see KSPSolve()

   Level: beginner

   Notes:
     Supports non-square (rectangular) matrices.

     This varient, when applied with no preconditioning is identical to the original algorithm in exact arithematic; however, in practice, with no preconditioning
     due to inexact arithematic, it can converge differently. Hence when no preconditioner is used (PCType PCNONE) it automatically reverts to the original algorithm.

     With the PETSc built-in preconditioners, such as ICC, one should call KSPSetOperators(ksp,A,A'*A)) since the preconditioner needs to work
     for the normal equations A'*A.

     Supports only left preconditioning.

   References:
.  1. - The original unpreconditioned algorithm can be found in Paige and Saunders, ACM Transactions on Mathematical Software, Vol 8, 1982.

     In exact arithmetic the LSQR method (with no preconditioning) is identical to the KSPCG algorithm applied to the normal equations.
     The preconditioned variant was implemented by Bas van't Hof and is essentially a left preconditioning for the Normal Equations. It appears the implementation with preconditioner
     track the true norm of the residual and uses that in the convergence test.

   Developer Notes:
    How is this related to the KSPCGNE implementation? One difference is that KSPCGNE applies
            the preconditioner transpose times the preconditioner,  so one does not need to pass A'*A as the third argument to KSPSetOperators().


   For least squares problems without a zero to A*x = b, there are additional convergence tests for the residual of the normal equations, A'*(b - Ax), see KSPLSQRDefaultConverged()

.seealso:  KSPCreate(), KSPSetType(), KSPType (for list of available types), KSP, KSPLSQRDefaultConverged()

M*/
PETSC_EXTERN PetscErrorCode KSPCreate_LSQR(KSP ksp)
{
  KSP_LSQR       *lsqr;
  void           *ctx;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr         = PetscNewLog(ksp,&lsqr);CHKERRQ(ierr);
  lsqr->se     = NULL;
  lsqr->se_flg = PETSC_FALSE;
  lsqr->anorm  = -1.0;
  lsqr->arnorm = -1.0;
  ksp->data    = (void*)lsqr;
  ierr         = KSPSetSupportedNorm(ksp,KSP_NORM_UNPRECONDITIONED,PC_LEFT,3);CHKERRQ(ierr);

  ksp->ops->setup          = KSPSetUp_LSQR;
  ksp->ops->solve          = KSPSolve_LSQR;
  ksp->ops->destroy        = KSPDestroy_LSQR;
  ksp->ops->setfromoptions = KSPSetFromOptions_LSQR;
  ksp->ops->view           = KSPView_LSQR;

  ierr = KSPConvergedDefaultCreate(&ctx);CHKERRQ(ierr);
  ierr = KSPSetConvergenceTest(ksp,KSPLSQRConvergedDefault,ctx,KSPConvergedDefaultDestroy);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


