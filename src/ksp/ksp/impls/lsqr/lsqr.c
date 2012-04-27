
/* lourens.vanzanen@shell.com contributed the standard error estimates of the solution, Jul 25, 2006 */
/* Bas van't Hof contributed the preconditioned aspects Feb 10, 2010 */

#define SWAP(a,b,c) { c = a; a = b; b = c; }

#include <petsc-private/kspimpl.h>
#include <../src/ksp/ksp/impls/lsqr/lsqr.h>

typedef struct {
  PetscInt   nwork_n,nwork_m; 
  Vec        *vwork_m;  /* work vectors of length m, where the system is size m x n */
  Vec        *vwork_n;  /* work vectors of length n */
  Vec        se;        /* Optional standard error vector */
  PetscBool  se_flg;   /* flag for -ksp_lsqr_set_standard_error */
  PetscReal  arnorm;   /* Norm of the vector A.r */
  PetscReal  anorm;    /* Frobenius norm of the matrix A */
  PetscReal  rhs_norm; /* Norm of the right hand side */
} KSP_LSQR;

extern PetscErrorCode  VecSquare(Vec);

#undef __FUNCT__  
#define __FUNCT__ "KSPSetUp_LSQR"
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
  if (nopreconditioner) {
     lsqr->nwork_n = 4;
  } else {
     lsqr->nwork_n = 5;
  }
  if (lsqr->vwork_n) {
    ierr = VecDestroyVecs(lsqr->nwork_n,&lsqr->vwork_n);CHKERRQ(ierr);
  }
  ierr = KSPGetVecs(ksp,lsqr->nwork_n,&lsqr->vwork_n,lsqr->nwork_m,&lsqr->vwork_m);CHKERRQ(ierr);
  if (lsqr->se_flg && !lsqr->se){
    /* lsqr->se is not set by user, get it from pmat */
    Vec *se;
    ierr = KSPGetVecs(ksp,1,&se,0,PETSC_NULL);CHKERRQ(ierr);
    lsqr->se = *se;
    ierr = PetscFree(se);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "KSPSolve_LSQR"
static PetscErrorCode KSPSolve_LSQR(KSP ksp)
{
  PetscErrorCode ierr;
  PetscInt       i,size1,size2;
  PetscScalar    rho,rhobar,phi,phibar,theta,c,s,tmp,tau,alphac;
  PetscReal      beta,alpha,rnorm;
  Vec            X,B,V,V1,U,U1,TMP,W,W2,SE,Z = PETSC_NULL;
  Mat            Amat,Pmat;
  MatStructure   pflag;
  KSP_LSQR       *lsqr = (KSP_LSQR*)ksp->data;
  PetscBool      diagonalscale,nopreconditioner;
  
  PetscFunctionBegin;
  ierr    = PCGetDiagonalScale(ksp->pc,&diagonalscale);CHKERRQ(ierr);
  if (diagonalscale) SETERRQ1(((PetscObject)ksp)->comm,PETSC_ERR_SUP,"Krylov method %s does not support diagonal scaling",((PetscObject)ksp)->type_name);

  ierr     = PCGetOperators(ksp->pc,&Amat,&Pmat,&pflag);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)ksp->pc,PCNONE,&nopreconditioner);CHKERRQ(ierr);

  /*  nopreconditioner =PETSC_FALSE; */
  /* Calculate norm of right hand side */
  ierr = VecNorm(ksp->vec_rhs,NORM_2,&lsqr->rhs_norm);CHKERRQ(ierr);

  /* mark norm of matrix with negative number to indicate it has not yet been computed */
  lsqr->anorm = -1.0;

  /* vectors of length m, where system size is mxn */
  B        = ksp->vec_rhs;
  U        = lsqr->vwork_m[0];
  U1       = lsqr->vwork_m[1];

  /* vectors of length n */
  X        = ksp->vec_sol;
  W        = lsqr->vwork_n[0];
  V        = lsqr->vwork_n[1];
  V1       = lsqr->vwork_n[2];
  W2       = lsqr->vwork_n[3];
  if (!nopreconditioner) {
     Z     = lsqr->vwork_n[4];
  }

  /* standard error vector */
  SE = lsqr->se;
  if (SE){
    ierr = VecGetSize(SE,&size1);CHKERRQ(ierr);
    ierr = VecGetSize(X ,&size2);CHKERRQ(ierr);
    if (size1 != size2) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Standard error vector (size %d) does not match solution vector (size %d)",size1,size2);
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
  ierr = VecNorm(U,NORM_2,&rnorm);CHKERRQ(ierr);
  ierr = PetscObjectTakeAccess(ksp);CHKERRQ(ierr);
  ksp->its   = 0;
  ksp->rnorm = rnorm;
  ierr = PetscObjectGrantAccess(ksp);CHKERRQ(ierr);
  KSPLogResidualHistory(ksp,rnorm);
  ierr = KSPMonitor(ksp,0,rnorm);CHKERRQ(ierr);
  ierr = (*ksp->converged)(ksp,0,rnorm,&ksp->reason,ksp->cnvP);CHKERRQ(ierr);
  if (ksp->reason) PetscFunctionReturn(0);

  beta = rnorm;
  ierr = VecScale(U,1.0/beta);CHKERRQ(ierr);
  ierr = KSP_MatMultTranspose(ksp,Amat,U,V); CHKERRQ(ierr);
  if (nopreconditioner) {
     ierr = VecNorm(V,NORM_2,&alpha); CHKERRQ(ierr);
  } else {
    ierr = PCApply(ksp->pc,V,Z);CHKERRQ(ierr);
    ierr = VecDot(V,Z,&alphac);CHKERRQ(ierr);
    if (PetscRealPart(alphac) <= 0.0) {
      ksp->reason = KSP_DIVERGED_BREAKDOWN;
      PetscFunctionReturn(0);
    }
    alpha = PetscSqrtReal(PetscRealPart(alphac));
    ierr = VecScale(Z,1.0/alpha); CHKERRQ(ierr);
  }
  ierr = VecScale(V,1.0/alpha);CHKERRQ(ierr);

  if (nopreconditioner){
    ierr = VecCopy(V,W);CHKERRQ(ierr);
  } else {
    ierr = VecCopy(Z,W);CHKERRQ(ierr);
  }

  lsqr->arnorm = alpha * beta;
  phibar = beta;
  rhobar = alpha;
  tau = -beta;
  i = 0;
  do {
    if (nopreconditioner) {
       ierr = KSP_MatMult(ksp,Amat,V,U1);CHKERRQ(ierr);
    } else {
       ierr = KSP_MatMult(ksp,Amat,Z,U1);CHKERRQ(ierr);
    }
    ierr = VecAXPY(U1,-alpha,U);CHKERRQ(ierr);
    ierr = VecNorm(U1,NORM_2,&beta);CHKERRQ(ierr);
    if (beta == 0.0){ 
      ksp->reason = KSP_DIVERGED_BREAKDOWN;
      break;
    }
    ierr = VecScale(U1,1.0/beta);CHKERRQ(ierr);

    ierr = KSP_MatMultTranspose(ksp,Amat,U1,V1);CHKERRQ(ierr);
    ierr = VecAXPY(V1,-beta,V);CHKERRQ(ierr);
    if (nopreconditioner) {
      ierr = VecNorm(V1,NORM_2,&alpha); CHKERRQ(ierr);
    } else {
      ierr = PCApply(ksp->pc,V1,Z);CHKERRQ(ierr);
      ierr = VecDot(V1,Z,&alphac);CHKERRQ(ierr);
      if (PetscRealPart(alphac) <= 0.0) {
        ksp->reason = KSP_DIVERGED_BREAKDOWN;
        break;
      }
      alpha = PetscSqrtReal(PetscRealPart(alphac));
      ierr = VecScale(Z,1.0/alpha);CHKERRQ(ierr);
    }
    ierr   = VecScale(V1,1.0/alpha);CHKERRQ(ierr);
    rho    = PetscSqrtScalar(rhobar*rhobar + beta*beta);
    c      = rhobar / rho;
    s      = beta / rho;
    theta  = s * alpha;
    rhobar = - c * alpha;
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
       ierr = VecAYPX(W,-theta/rho,V1);CHKERRQ(ierr); /* w <- v - (theta/rho) w */  
    } else {
       ierr = VecAYPX(W,-theta/rho,Z);CHKERRQ(ierr);  /* w <- z - (theta/rho) w */  
    } 

    lsqr->arnorm = alpha*PetscAbsScalar(tau);
    rnorm = PetscRealPart(phibar);

    ierr = PetscObjectTakeAccess(ksp);CHKERRQ(ierr);
    ksp->its++;
    ksp->rnorm = rnorm;
    ierr = PetscObjectGrantAccess(ksp);CHKERRQ(ierr);
    KSPLogResidualHistory(ksp,rnorm);
    ierr = KSPMonitor(ksp,i+1,rnorm);CHKERRQ(ierr);
    ierr = (*ksp->converged)(ksp,i+1,rnorm,&ksp->reason,ksp->cnvP);CHKERRQ(ierr);
    if (ksp->reason) break;
    SWAP(U1,U,TMP);
    SWAP(V1,V,TMP);

    i++;
  } while (i<ksp->max_it);
  if (i >= ksp->max_it && !ksp->reason) {
    ksp->reason = KSP_DIVERGED_ITS;
  }

  /* Finish off the standard error estimates */
  if (SE) {
    tmp = 1.0;
    ierr = MatGetSize(Amat,&size1,&size2);CHKERRQ(ierr);
    if ( size1 > size2 ) tmp = size1 - size2;
    tmp = rnorm / PetscSqrtScalar(tmp);
    ierr = VecSqrtAbs(SE);CHKERRQ(ierr);
    ierr = VecScale(SE,tmp);CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "KSPDestroy_LSQR" 
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
  if (lsqr->se_flg){
    ierr = VecDestroy(&lsqr->se);CHKERRQ(ierr);
  }
  ierr = PetscFree(ksp->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "KSPLSQRSetStandardErrorVec"
PetscErrorCode  KSPLSQRSetStandardErrorVec( KSP ksp, Vec se )
{
  KSP_LSQR       *lsqr = (KSP_LSQR*)ksp->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecDestroy(&lsqr->se);CHKERRQ(ierr);
  lsqr->se = se;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "KSPLSQRGetStandardErrorVec"
PetscErrorCode  KSPLSQRGetStandardErrorVec( KSP ksp,Vec *se )
{
  KSP_LSQR *lsqr = (KSP_LSQR*)ksp->data;

  PetscFunctionBegin;
  *se = lsqr->se;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "KSPLSQRGetArnorm"
PetscErrorCode  KSPLSQRGetArnorm( KSP ksp,PetscReal *arnorm, PetscReal *rhs_norm , PetscReal *anorm)
{
  KSP_LSQR       *lsqr = (KSP_LSQR*)ksp->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  *arnorm   = lsqr->arnorm;
  if (anorm) {
    if (lsqr->anorm < 0.0) {
      PC  pc;
      Mat Amat;
      ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
      ierr = PCGetOperators(pc,&Amat,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
      ierr = MatNorm(Amat,NORM_FROBENIUS,&lsqr->anorm);CHKERRQ(ierr);
    }
    *anorm    = lsqr->anorm;
  }
  if (rhs_norm) {
    *rhs_norm = lsqr->rhs_norm;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "KSPLSQRMonitorDefault"
/*@C
   KSPLSQRMonitorDefault - Print the residual norm at each iteration of the LSQR method and the norm of the residual of the normal equations A'*A x = A' b

   Collective on KSP

   Input Parameters:
+  ksp   - iterative context
.  n     - iteration number
.  rnorm - 2-norm (preconditioned) residual value (may be estimated).  
-  dummy - unused monitor context 

   Level: intermediate

.keywords: KSP, default, monitor, residual

.seealso: KSPMonitorSet(), KSPMonitorTrueResidualNorm(), KSPMonitorLGCreate(), KSPMonitorDefault()
@*/
PetscErrorCode  KSPLSQRMonitorDefault(KSP ksp,PetscInt n,PetscReal rnorm,void *dummy)
{
  PetscErrorCode   ierr;
  PetscViewer      viewer = dummy ? (PetscViewer) dummy : PETSC_VIEWER_STDOUT_(((PetscObject)ksp)->comm);
  KSP_LSQR         *lsqr = (KSP_LSQR*)ksp->data;

  PetscFunctionBegin;
  ierr = PetscViewerASCIIAddTab(viewer,((PetscObject)ksp)->tablevel);CHKERRQ(ierr);
  if (((PetscObject)ksp)->prefix) {
    ierr = PetscViewerASCIIPrintf(viewer,"  Residual norm and norm of normal equations for %s solve.\n",((PetscObject)ksp)->prefix);CHKERRQ(ierr);
  }
  if (!n) {
    ierr = PetscViewerASCIIPrintf(viewer,"%3D KSP Residual norm %14.12e\n",n,rnorm);CHKERRQ(ierr);
  } else {
    ierr = PetscViewerASCIIPrintf(viewer,"%3D KSP Residual norm %14.12e Residual norm normal equations %14.12e\n",n,rnorm,lsqr->arnorm);CHKERRQ(ierr);
  }
  ierr = PetscViewerASCIISubtractTab(viewer,((PetscObject)ksp)->tablevel);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "KSPSetFromOptions_LSQR"
PetscErrorCode KSPSetFromOptions_LSQR(KSP ksp)
{
  PetscErrorCode ierr;
  KSP_LSQR       *lsqr = (KSP_LSQR*)ksp->data;
  char           monfilename[PETSC_MAX_PATH_LEN];
  PetscViewer    monviewer;
  PetscBool      flg;

  PetscFunctionBegin;
  ierr = PetscOptionsHead("KSP LSQR Options");CHKERRQ(ierr);
  ierr = PetscOptionsName("-ksp_lsqr_set_standard_error","Set Standard Error Estimates of Solution","KSPLSQRSetStandardErrorVec",&lsqr->se_flg);CHKERRQ(ierr);
  ierr = PetscOptionsString("-ksp_lsqr_monitor","Monitor residual norm and norm of residual of normal equations","KSPMonitorSet","stdout",monfilename,PETSC_MAX_PATH_LEN,&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = PetscViewerASCIIOpen(((PetscObject)ksp)->comm,monfilename,&monviewer);CHKERRQ(ierr);
    ierr = KSPMonitorSet(ksp,KSPLSQRMonitorDefault,monviewer,(PetscErrorCode (*)(void**))PetscViewerDestroy);CHKERRQ(ierr);
  }
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "KSPView_LSQR" 
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
      ierr = PetscViewerASCIIPrintf(viewer,"  Norm of Standard Error %G, Iterations %D\n",rnorm,ksp->its);CHKERRQ(ierr);
    }
  } else SETERRQ1(((PetscObject)ksp)->comm,PETSC_ERR_SUP,"Viewer type %s not supported for KSP LSQR",((PetscObject)viewer)->type_name);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "KSPLSQRDefaultConverged"
/*@C
   KSPLSQRDefaultConverged - Determines convergence of the LSQR Krylov method. This calls KSPDefaultConverged() and if that does not determine convergence then checks 
      convergence for the least squares problem.

   Collective on KSP

   Input Parameters:
+  ksp   - iterative context
.  n     - iteration number
.  rnorm - 2-norm residual value (may be estimated)
-  ctx - convergence context which must be created by KSPDefaultConvergedCreate()

   reason is set to:
+   positive - if the iteration has converged;
.   negative - if residual norm exceeds divergence threshold;
-   0 - otherwise.

   Notes:
      Possible convergence for the least squares problem (which is based on the residual of the normal equations) are KSP_CONVERGED_RTOL_NORMAL norm and KSP_CONVERGED_ATOL_NORMAL.

   Level: intermediate

.keywords: KSP, default, convergence, residual

.seealso: KSPSetConvergenceTest(), KSPSetTolerances(), KSPSkipConverged(), KSPConvergedReason, KSPGetConvergedReason(),
          KSPDefaultConvergedSetUIRNorm(), KSPDefaultConvergedSetUMIRNorm(), KSPDefaultConvergedCreate(), KSPDefaultConvergedDestroy(), KSPDefaultConverged()
@*/
PetscErrorCode  KSPLSQRDefaultConverged(KSP ksp,PetscInt n,PetscReal rnorm,KSPConvergedReason *reason,void *ctx)
{
  PetscErrorCode         ierr;
  KSP_LSQR               *lsqr = (KSP_LSQR*)ksp->data;

  PetscFunctionBegin;
  ierr = KSPDefaultConverged(ksp,n,rnorm,reason,ctx);CHKERRQ(ierr);
  if (!n || *reason) PetscFunctionReturn(0);
  if (lsqr->arnorm/lsqr->rhs_norm < ksp->rtol) {
    *reason = KSP_CONVERGED_RTOL_NORMAL;
  }
  if (lsqr->arnorm < ksp->abstol) {
    *reason = KSP_CONVERGED_ATOL_NORMAL;
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
     This varient, when applied with no preconditioning is identical to the original algorithm in exact arithematic; however, in practice, with no preconditioning
     due to inexact arithematic, it can converge differently. Hence when no preconditioner is used (PCType PCNONE) it automatically reverts to the original algorithm.

     With the PETSc built-in preconditioners, such as ICC, one should call KSPSetOperators(ksp,A,A'*A,...) since the preconditioner needs to work 
     for the normal equations A'*A.

     Supports only left preconditioning.

   References:The original unpreconditioned algorithm can be found in Paige and Saunders, ACM Transactions on Mathematical Software, Vol 8, pp 43-71, 1982. 
     In exact arithmetic the LSQR method (with no preconditioning) is identical to the KSPCG algorithm applied to the normal equations.
     The preconditioned varient was implemented by Bas van't Hof and is essentially a left preconditioning for the Normal Equations. It appears the implementation with preconditioner
     track the true norm of the residual and uses that in the convergence test.

   Developer Notes: How is this related to the KSPCGNE implementation? One difference is that KSPCGNE applies
            the preconditioner transpose times the preconditioner,  so one does not need to pass A'*A as the third argument to KSPSetOperators().
            

   For least squares problems without a zero to A*x = b, there are additional convergence tests for the residual of the normal equations, A'*(b - Ax), see KSPLSQRDefaultConverged()

.seealso:  KSPCreate(), KSPSetType(), KSPType (for list of available types), KSP, KSPLSQRDefaultConverged()

M*/
EXTERN_C_BEGIN
#undef __FUNCT__ 
#define __FUNCT__ "KSPCreate_LSQR"
PetscErrorCode  KSPCreate_LSQR(KSP ksp)
{
  KSP_LSQR       *lsqr;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscNewLog(ksp,KSP_LSQR,&lsqr);CHKERRQ(ierr);
  lsqr->se     = PETSC_NULL;
  lsqr->se_flg = PETSC_FALSE;
  lsqr->arnorm = 0.0;
  ksp->data                      = (void*)lsqr;
  ierr = KSPSetSupportedNorm(ksp,KSP_NORM_UNPRECONDITIONED,PC_LEFT,2);CHKERRQ(ierr);

  ksp->ops->setup                = KSPSetUp_LSQR;
  ksp->ops->solve                = KSPSolve_LSQR;
  ksp->ops->destroy              = KSPDestroy_LSQR;
  ksp->ops->buildsolution        = KSPDefaultBuildSolution;
  ksp->ops->buildresidual        = KSPDefaultBuildResidual;
  ksp->ops->setfromoptions       = KSPSetFromOptions_LSQR;
  ksp->ops->view                 = KSPView_LSQR;
  ksp->converged                 = KSPLSQRDefaultConverged;
  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNCT__
#define __FUNCT__ "VecSquare"
PetscErrorCode  VecSquare(Vec v)
{
  PetscErrorCode ierr;
  PetscScalar    *x;
  PetscInt       i, n;

  PetscFunctionBegin;
  ierr = VecGetLocalSize(v, &n);CHKERRQ(ierr);
  ierr = VecGetArray(v, &x);CHKERRQ(ierr);
  for(i = 0; i < n; i++) {
    x[i] *= x[i];
  }
  ierr = VecRestoreArray(v, &x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

