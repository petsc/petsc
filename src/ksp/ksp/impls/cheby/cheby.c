
#include <petsc-private/kspimpl.h>                    /*I "petscksp.h" I*/
#include <../src/ksp/ksp/impls/cheby/chebychevimpl.h>

#undef __FUNCT__  
#define __FUNCT__ "KSPSetUp_Chebychev"
PetscErrorCode KSPSetUp_Chebychev(KSP ksp)
{
  KSP_Chebychev *cheb = (KSP_Chebychev*)ksp->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = KSPDefaultGetWork(ksp,3);CHKERRQ(ierr);
  cheb->estimate_current = PETSC_FALSE;
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "KSPChebychevSetEigenvalues_Chebychev"
PetscErrorCode  KSPChebychevSetEigenvalues_Chebychev(KSP ksp,PetscReal emax,PetscReal emin)
{
  KSP_Chebychev *chebychevP = (KSP_Chebychev*)ksp->data;

  PetscFunctionBegin;
  if (emax <= emin) SETERRQ2(((PetscObject)ksp)->comm,PETSC_ERR_ARG_INCOMP,"Maximum eigenvalue must be larger than minimum: max %g min %G",emax,emin);
  if (emax*emin <= 0.0) SETERRQ2(((PetscObject)ksp)->comm,PETSC_ERR_ARG_INCOMP,"Both eigenvalues must be of the same sign: max %G min %G",emax,emin);
  chebychevP->emax = emax;
  chebychevP->emin = emin;
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "KSPChebychevSetEstimateEigenvalues_Chebychev"
PetscErrorCode  KSPChebychevSetEstimateEigenvalues_Chebychev(KSP ksp,PetscReal a,PetscReal b,PetscReal c,PetscReal d)
{
  KSP_Chebychev *cheb = (KSP_Chebychev*)ksp->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (a != 0.0 || b != 0.0 || c != 0.0 || d != 0.0) {
    if (!cheb->kspest) {
      PetscBool nonzero;

      ierr = KSPCreate(((PetscObject)ksp)->comm,&cheb->kspest);CHKERRQ(ierr);
      ierr = PetscObjectIncrementTabLevel((PetscObject)cheb->kspest,(PetscObject)ksp,1);CHKERRQ(ierr);

      ierr = KSPGetPC(cheb->kspest,&cheb->pcnone);CHKERRQ(ierr);
      ierr = PetscObjectReference((PetscObject)cheb->pcnone);CHKERRQ(ierr);
      ierr = PCSetType(cheb->pcnone,PCNONE);CHKERRQ(ierr);
      ierr = KSPSetPC(cheb->kspest,ksp->pc);CHKERRQ(ierr);

      ierr = KSPGetInitialGuessNonzero(ksp,&nonzero);CHKERRQ(ierr);
      ierr = KSPSetInitialGuessNonzero(cheb->kspest,nonzero);CHKERRQ(ierr);
      ierr = KSPSetComputeSingularValues(cheb->kspest,PETSC_TRUE);CHKERRQ(ierr);

      /* Estimate with a fixed number of iterations */
      ierr = KSPSetConvergenceTest(cheb->kspest,KSPSkipConverged,0,0);CHKERRQ(ierr);
      ierr = KSPSetNormType(cheb->kspest,KSP_NORM_NONE);CHKERRQ(ierr);
      ierr = KSPSetTolerances(cheb->kspest,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT,10);CHKERRQ(ierr);
    }
    if (a >= 0) cheb->tform[0] = a;
    if (b >= 0) cheb->tform[1] = b;
    if (c >= 0) cheb->tform[2] = c;
    if (d >= 0) cheb->tform[3] = d;
    cheb->estimate_current = PETSC_FALSE;
  } else {
    ierr = KSPDestroy(&cheb->kspest);CHKERRQ(ierr);
    ierr = PCDestroy(&cheb->pcnone);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNCT__
#define __FUNCT__ "KSPChebychevSetNewMatrix_Chebychev"
PETSC_EXTERN_C PetscErrorCode  KSPChebychevSetNewMatrix_Chebychev(KSP ksp)
{
  KSP_Chebychev *cheb = (KSP_Chebychev*)ksp->data;

  PetscFunctionBegin;
  cheb->estimate_current = PETSC_FALSE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "KSPChebychevSetEigenvalues"
/*@
   KSPChebychevSetEigenvalues - Sets estimates for the extreme eigenvalues
   of the preconditioned problem.

   Logically Collective on KSP

   Input Parameters:
+  ksp - the Krylov space context
-  emax, emin - the eigenvalue estimates

  Options Database:
.  -ksp_chebychev_eigenvalues emin,emax

   Note: If you run with the Krylov method of KSPCG with the option -ksp_monitor_singular_value it will 
    for that given matrix and preconditioner an estimate of the extreme eigenvalues.

   Level: intermediate

.keywords: KSP, Chebyshev, set, eigenvalues
@*/
PetscErrorCode  KSPChebychevSetEigenvalues(KSP ksp,PetscReal emax,PetscReal emin)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  PetscValidLogicalCollectiveReal(ksp,emax,2);
  PetscValidLogicalCollectiveReal(ksp,emin,3);
  ierr = PetscTryMethod(ksp,"KSPChebychevSetEigenvalues_C",(KSP,PetscReal,PetscReal),(ksp,emax,emin));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "KSPChebychevSetEstimateEigenvalues"
/*@
   KSPChebychevSetEstimateEigenvalues - Automatically estimate the eigenvalues to use for Chebychev

   Logically Collective on KSP

   Input Parameters:
+  ksp - the Krylov space context
.  a - multiple of min eigenvalue estimate to use for min Chebychev bound (or PETSC_DECIDE)
.  b - multiple of max eigenvalue estimate to use for min Chebychev bound (or PETSC_DECIDE)
.  c - multiple of min eigenvalue estimate to use for max Chebychev bound (or PETSC_DECIDE)
-  d - multiple of max eigenvalue estimate to use for max Chebychev bound (or PETSC_DECIDE)

  Options Database:
.  -ksp_chebychev_estimate_eigenvalues a,b,c,d

   Notes:
   The Chebychev bounds are estimated using
.vb
   minbound = a*minest + b*maxest
   maxbound = c*minest + d*maxest
.ve
   The default configuration targets the upper part of the spectrum for use as a multigrid smoother, so only the maximum eigenvalue estimate is used.
   The minimum eigenvalue estimate obtained by Krylov iteration is typically not accurate until the method has converged.

   If 0.0 is passed for all transform arguments (a,b,c,d), eigenvalue estimation is disabled.

   Level: intermediate

.keywords: KSP, Chebyshev, set, eigenvalues
@*/
PetscErrorCode KSPChebychevSetEstimateEigenvalues(KSP ksp,PetscReal a,PetscReal b,PetscReal c,PetscReal d)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  PetscValidLogicalCollectiveReal(ksp,a,2);
  PetscValidLogicalCollectiveReal(ksp,b,3);
  PetscValidLogicalCollectiveReal(ksp,c,4);
  PetscValidLogicalCollectiveReal(ksp,d,5);
  ierr = PetscTryMethod(ksp,"KSPChebychevSetEstimateEigenvalues_C",(KSP,PetscReal,PetscReal,PetscReal,PetscReal),(ksp,a,b,c,d));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "KSPChebychevSetNewMatrix"
/*@
   KSPChebychevSetNewMatrix - Indicates that the matrix has changed, causes eigenvalue estimates to be recomputed if appropriate.

   Logically Collective on KSP

   Input Parameter:
.  ksp - the Krylov space context

   Level: developer

.keywords: KSP, Chebyshev, set, eigenvalues

.seealso: KSPChebychevSetEstimateEigenvalues()
@*/
PetscErrorCode KSPChebychevSetNewMatrix(KSP ksp)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  ierr = PetscTryMethod(ksp,"KSPChebychevSetNewMatrix_C",(KSP),(ksp));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "KSPSetFromOptions_Chebychev"
PetscErrorCode KSPSetFromOptions_Chebychev(KSP ksp)
{
  KSP_Chebychev *cheb = (KSP_Chebychev*)ksp->data;
  PetscErrorCode ierr;
  PetscInt       two = 2,four = 4;
  PetscReal      tform[4] = {PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE};
  PetscBool      flg;

  PetscFunctionBegin;
  ierr = PetscOptionsHead("KSP Chebychev Options");CHKERRQ(ierr);
  ierr = PetscOptionsRealArray("-ksp_chebychev_eigenvalues","extreme eigenvalues","KSPChebychevSetEigenvalues",&cheb->emin,&two,0);CHKERRQ(ierr);
  ierr = PetscOptionsRealArray("-ksp_chebychev_estimate_eigenvalues","estimate eigenvalues using a Krylov method, then use this transform for Chebychev eigenvalue bounds","KSPChebychevSetEstimateEigenvalues",tform,&four,&flg);CHKERRQ(ierr);
  if (flg) {
    switch (four) {
    case 0:
      ierr = KSPChebychevSetEstimateEigenvalues(ksp,PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE);CHKERRQ(ierr);
      break;
    case 2:                     /* Base everything on the max eigenvalues */
      ierr = KSPChebychevSetEstimateEigenvalues(ksp,PETSC_DECIDE,tform[0],PETSC_DECIDE,tform[1]);CHKERRQ(ierr);
      break;
    case 4:                     /* Use the full 2x2 linear transformation */
      ierr = KSPChebychevSetEstimateEigenvalues(ksp,tform[0],tform[1],tform[2],tform[3]);CHKERRQ(ierr);
      break;
    default: SETERRQ(((PetscObject)ksp)->comm,PETSC_ERR_ARG_INCOMP,"Must specify either 0, 2, or 4 parameters for eigenvalue estimation");
    }
  }
  if (cheb->kspest) {
    /* Mask the PC so that PCSetFromOptions does not do anything */
    ierr = KSPSetPC(cheb->kspest,cheb->pcnone);CHKERRQ(ierr);
    ierr = KSPSetOptionsPrefix(cheb->kspest,((PetscObject)ksp)->prefix);CHKERRQ(ierr);
    ierr = KSPAppendOptionsPrefix(cheb->kspest,"est_");CHKERRQ(ierr);
    if (!((PetscObject)cheb->kspest)->type_name) {
      ierr = KSPSetType(cheb->kspest,KSPGMRES);CHKERRQ(ierr);
    }
    ierr = KSPSetFromOptions(cheb->kspest);CHKERRQ(ierr);
    ierr = KSPSetPC(cheb->kspest,ksp->pc);CHKERRQ(ierr);
  }
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "KSPSolve_Chebychev"
PetscErrorCode KSPSolve_Chebychev(KSP ksp)
{
  KSP_Chebychev  *cheb = (KSP_Chebychev*)ksp->data;
  PetscErrorCode ierr;
  PetscInt       k,kp1,km1,maxit,ktmp,i;
  PetscScalar    alpha,omegaprod,mu,omega,Gamma,c[3],scale;
  PetscReal      rnorm = 0.0;
  Vec            x,b,p[3],r;
  Mat            Amat,Pmat;
  MatStructure   pflag;
  PetscBool      diagonalscale;

  PetscFunctionBegin;
  ierr    = PCGetDiagonalScale(ksp->pc,&diagonalscale);CHKERRQ(ierr);
  if (diagonalscale) SETERRQ1(((PetscObject)ksp)->comm,PETSC_ERR_SUP,"Krylov method %s does not support diagonal scaling",((PetscObject)ksp)->type_name);

  if (cheb->kspest && !cheb->estimate_current) {
    PetscReal max,min;
    PetscBool nonzero;
    Vec X = ksp->vec_sol;
    ierr = KSPGetInitialGuessNonzero(ksp,&nonzero);CHKERRQ(ierr);
    if (nonzero) {ierr = VecDuplicate(ksp->vec_sol,&X);CHKERRQ(ierr);}
    ierr = KSPSolve(cheb->kspest,ksp->vec_rhs,X);CHKERRQ(ierr);
    if (nonzero) {ierr = VecDestroy(&X);CHKERRQ(ierr);}
    else {ierr = VecZeroEntries(X);CHKERRQ(ierr);}
    ierr = KSPComputeExtremeSingularValues(cheb->kspest,&max,&min);CHKERRQ(ierr);
    cheb->emin = cheb->tform[0]*min + cheb->tform[1]*max;
    cheb->emax = cheb->tform[2]*min + cheb->tform[3]*max;
    cheb->estimate_current = PETSC_TRUE;
  }

  ksp->its = 0;
  ierr     = PCGetOperators(ksp->pc,&Amat,&Pmat,&pflag);CHKERRQ(ierr);
  maxit    = ksp->max_it;

  /* These three point to the three active solutions, we
     rotate these three at each solution update */
  km1    = 0; k = 1; kp1 = 2;
  x      = ksp->vec_sol;
  b      = ksp->vec_rhs;
  p[km1] = x;
  p[k]   = ksp->work[0];
  p[kp1] = ksp->work[1];
  r      = ksp->work[2];

  /* use scale*B as our preconditioner */
  scale  = 2.0/(cheb->emax + cheb->emin);

  /*   -alpha <=  scale*lambda(B^{-1}A) <= alpha   */
  alpha  = 1.0 - scale*(cheb->emin); ;
  Gamma  = 1.0;
  mu     = 1.0/alpha; 
  omegaprod = 2.0/alpha;

  c[km1] = 1.0;
  c[k]   = mu;

  if (!ksp->guess_zero) {
    ierr = KSP_MatMult(ksp,Amat,x,r);CHKERRQ(ierr);     /*  r = b - Ax     */
    ierr = VecAYPX(r,-1.0,b);CHKERRQ(ierr);
  } else {
    ierr = VecCopy(b,r);CHKERRQ(ierr);
  }
                  
  ierr = KSP_PCApply(ksp,r,p[k]);CHKERRQ(ierr);  /* p[k] = scale B^{-1}r + x */
  ierr = VecAYPX(p[k],scale,x);CHKERRQ(ierr);

  for (i=0; i<maxit; i++) {
    ierr = PetscObjectTakeAccess(ksp);CHKERRQ(ierr);
    ksp->its++;
    ierr = PetscObjectGrantAccess(ksp);CHKERRQ(ierr);
    c[kp1] = 2.0*mu*c[k] - c[km1];
    omega = omegaprod*c[k]/c[kp1];

    ierr = KSP_MatMult(ksp,Amat,p[k],r);CHKERRQ(ierr);                 /*  r = b - Ap[k]    */
    ierr = VecAYPX(r,-1.0,b);CHKERRQ(ierr);                       
    ierr = KSP_PCApply(ksp,r,p[kp1]);CHKERRQ(ierr);             /*  p[kp1] = B^{-1}z  */

    /* calculate residual norm if requested */
    if (ksp->normtype != KSP_NORM_NONE) {
      if (ksp->normtype == KSP_NORM_UNPRECONDITIONED) {ierr = VecNorm(r,NORM_2,&rnorm);CHKERRQ(ierr);}
      else {ierr = VecNorm(p[kp1],NORM_2,&rnorm);CHKERRQ(ierr);}
      ierr = PetscObjectTakeAccess(ksp);CHKERRQ(ierr);
      ksp->rnorm                              = rnorm;
      ierr = PetscObjectGrantAccess(ksp);CHKERRQ(ierr);
      ksp->vec_sol = p[k]; 
      KSPLogResidualHistory(ksp,rnorm);
      ierr = KSPMonitor(ksp,i,rnorm);CHKERRQ(ierr);
      ierr = (*ksp->converged)(ksp,i,rnorm,&ksp->reason,ksp->cnvP);CHKERRQ(ierr);
      if (ksp->reason) break;
    }

    /* y^{k+1} = omega(y^{k} - y^{k-1} + Gamma*r^{k}) + y^{k-1} */
    ierr = VecScale(p[kp1],omega*Gamma*scale);CHKERRQ(ierr);
    ierr = VecAXPY(p[kp1],1.0-omega,p[km1]);CHKERRQ(ierr);
    ierr = VecAXPY(p[kp1],omega,p[k]);CHKERRQ(ierr);

    ktmp = km1;
    km1  = k;
    k    = kp1;
    kp1  = ktmp;
  }
  if (!ksp->reason) {
    if (ksp->normtype != KSP_NORM_NONE) {
      ierr = KSP_MatMult(ksp,Amat,p[k],r);CHKERRQ(ierr);       /*  r = b - Ap[k]    */
      ierr = VecAYPX(r,-1.0,b);CHKERRQ(ierr);
      if (ksp->normtype == KSP_NORM_UNPRECONDITIONED) {
	ierr = VecNorm(r,NORM_2,&rnorm);CHKERRQ(ierr);
      } else {
	ierr = KSP_PCApply(ksp,r,p[kp1]);CHKERRQ(ierr); /* p[kp1] = B^{-1}z */
	ierr = VecNorm(p[kp1],NORM_2,&rnorm);CHKERRQ(ierr);
      }
      ierr = PetscObjectTakeAccess(ksp);CHKERRQ(ierr);
      ksp->rnorm = rnorm;
      ierr = PetscObjectGrantAccess(ksp);CHKERRQ(ierr);
      ksp->vec_sol = p[k]; 
      KSPLogResidualHistory(ksp,rnorm);
      ierr = KSPMonitor(ksp,i,rnorm);CHKERRQ(ierr);
    }
    if (ksp->its >= ksp->max_it) {
      if (ksp->normtype != KSP_NORM_NONE) {
	ierr = (*ksp->converged)(ksp,i,rnorm,&ksp->reason,ksp->cnvP);CHKERRQ(ierr);
	if (!ksp->reason) ksp->reason = KSP_DIVERGED_ITS;
      } else { 
	ksp->reason = KSP_CONVERGED_ITS;
      }
    }
  }

  /* make sure solution is in vector x */
  ksp->vec_sol = x;
  if (k) {
    ierr = VecCopy(p[k],x);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "KSPView_Chebychev" 
PetscErrorCode KSPView_Chebychev(KSP ksp,PetscViewer viewer)
{
  KSP_Chebychev  *cheb = (KSP_Chebychev*)ksp->data;
  PetscErrorCode ierr;
  PetscBool      iascii;

  PetscFunctionBegin;
  ierr = PetscTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
    ierr = PetscViewerASCIIPrintf(viewer,"  Chebychev: eigenvalue estimates:  min = %G, max = %G\n",cheb->emin,cheb->emax);CHKERRQ(ierr);
    if (cheb->kspest) {
      ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
      ierr = KSPView(cheb->kspest,viewer);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
    }
  } else {
    SETERRQ1(((PetscObject)ksp)->comm,PETSC_ERR_SUP,"Viewer type %s not supported for KSP Chebychev",((PetscObject)viewer)->type_name);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "KSPDestroy_Chebychev"
PetscErrorCode KSPDestroy_Chebychev(KSP ksp)
{
  KSP_Chebychev  *cheb = (KSP_Chebychev*)ksp->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = KSPDestroy(&cheb->kspest);CHKERRQ(ierr);
  ierr = PCDestroy(&cheb->pcnone);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)ksp,"KSPChebychevSetEigenvalues_C","",PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)ksp,"KSPChebychevSetEstimateEigenvalues_C","",PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)ksp,"KSPChebychevSetNewMatrix_C","",PETSC_NULL);CHKERRQ(ierr);
  ierr = KSPDefaultDestroy(ksp);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*MC
     KSPCHEBYCHEV - The preconditioned Chebychev iterative method

   Options Database Keys:
+   -ksp_chebychev_eigenvalues <emin,emax> - set approximations to the smallest and largest eigenvalues
                  of the preconditioned operator. If these are accurate you will get much faster convergence.
-   -ksp_chebychev_estimate_eigenvalues <a,b,c,d> - estimate eigenvalues using a Krylov method, then use this
                  transform for Chebychev eigenvalue bounds (KSPChebychevSetEstimateEigenvalues)


   Level: beginner

   Notes: The Chebychev method requires both the matrix and preconditioner to 
          be symmetric positive (semi) definite.
          Only support for left preconditioning.

.seealso:  KSPCreate(), KSPSetType(), KSPType (for list of available types), KSP,
           KSPChebychevSetEigenvalues(), KSPRICHARDSON, KSPCG

M*/

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "KSPCreate_Chebychev"
PetscErrorCode  KSPCreate_Chebychev(KSP ksp)
{
  PetscErrorCode ierr;
  KSP_Chebychev  *chebychevP;

  PetscFunctionBegin;
  ierr = PetscNewLog(ksp,KSP_Chebychev,&chebychevP);CHKERRQ(ierr);

  ksp->data                      = (void*)chebychevP;
  ierr = KSPSetSupportedNorm(ksp,KSP_NORM_PRECONDITIONED,PC_LEFT,2);CHKERRQ(ierr);
  ierr = KSPSetSupportedNorm(ksp,KSP_NORM_UNPRECONDITIONED,PC_LEFT,1);CHKERRQ(ierr);

  chebychevP->emin               = 1.e-2;
  chebychevP->emax               = 1.e+2;

  chebychevP->tform[0]           = 0.0;
  chebychevP->tform[1]           = 0.02;
  chebychevP->tform[2]           = 0;
  chebychevP->tform[3]           = 1.1;

  ksp->ops->setup                = KSPSetUp_Chebychev;
  ksp->ops->solve                = KSPSolve_Chebychev;
  ksp->ops->destroy              = KSPDestroy_Chebychev;
  ksp->ops->buildsolution        = KSPDefaultBuildSolution;
  ksp->ops->buildresidual        = KSPDefaultBuildResidual;
  ksp->ops->setfromoptions       = KSPSetFromOptions_Chebychev;
  ksp->ops->view                 = KSPView_Chebychev;

  ierr = PetscObjectComposeFunctionDynamic((PetscObject)ksp,"KSPChebychevSetEigenvalues_C",
                                    "KSPChebychevSetEigenvalues_Chebychev",
                                    KSPChebychevSetEigenvalues_Chebychev);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)ksp,"KSPChebychevSetEstimateEigenvalues_C",
                                    "KSPChebychevSetEstimateEigenvalues_Chebychev",
                                    KSPChebychevSetEstimateEigenvalues_Chebychev);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)ksp,"KSPChebychevSetNewMatrix_C",
                                    "KSPChebychevSetNewMatrix_Chebychev",
                                    KSPChebychevSetNewMatrix_Chebychev);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END
