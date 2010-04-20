#define PETSCKSP_DLL

#include "private/kspimpl.h"                    /*I "petscksp.h" I*/
#include "../src/ksp/ksp/impls/cheby/chebychevimpl.h"

#undef __FUNCT__  
#define __FUNCT__ "KSPSetUp_Chebychev"
PetscErrorCode KSPSetUp_Chebychev(KSP ksp)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (ksp->pc_side == PC_SYMMETRIC) SETERRQ(PETSC_ERR_SUP,"no symmetric preconditioning for KSPCHEBYCHEV");
  if (ksp->pc_side == PC_RIGHT) SETERRQ(PETSC_ERR_SUP,"no right preconditioning for KSPCHEBYCHEV");
  ierr = KSPDefaultGetWork(ksp,3);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "KSPChebychevSetEigenvalues_Chebychev"
PetscErrorCode PETSCKSP_DLLEXPORT KSPChebychevSetEigenvalues_Chebychev(KSP ksp,PetscReal emax,PetscReal emin)
{
  KSP_Chebychev *chebychevP = (KSP_Chebychev*)ksp->data;

  PetscFunctionBegin;
  if (emax <= emin) SETERRQ2(PETSC_ERR_ARG_INCOMP,"Maximum eigenvalue must be larger than minimum: max %g min %G",emax,emin);
  if (emax*emin <= 0.0) SETERRQ2(PETSC_ERR_ARG_INCOMP,"Both eigenvalues must be of the same sign: max %G min %G",emax,emin);
  chebychevP->emax = emax;
  chebychevP->emin = emin;
  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNCT__  
#define __FUNCT__ "KSPChebychevSetEigenvalues"
/*@
   KSPChebychevSetEigenvalues - Sets estimates for the extreme eigenvalues
   of the preconditioned problem.

   Collective on KSP

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
PetscErrorCode PETSCKSP_DLLEXPORT KSPChebychevSetEigenvalues(KSP ksp,PetscReal emax,PetscReal emin)
{
  PetscErrorCode ierr,(*f)(KSP,PetscReal,PetscReal);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_COOKIE,1);
  ierr = PetscObjectQueryFunction((PetscObject)ksp,"KSPChebychevSetEigenvalues_C",(void (**)(void))&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(ksp,emax,emin);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "KSPSetFromOptions_Chebychev"
PetscErrorCode KSPSetFromOptions_Chebychev(KSP ksp)
{
  KSP_Chebychev *cheb = (KSP_Chebychev*)ksp->data;
  PetscErrorCode ierr;
  PetscInt       two = 2;

  PetscFunctionBegin;
  ierr = PetscOptionsHead("KSP Chebychev Options");CHKERRQ(ierr);
    ierr = PetscOptionsRealArray("-ksp_chebychev_eigenvalues","extreme eigenvalues","KSPChebychevSetEigenvalues",&cheb->emin,&two,0);CHKERRQ(ierr);
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "KSPSolve_Chebychev"
PetscErrorCode KSPSolve_Chebychev(KSP ksp)
{
  PetscErrorCode ierr;
  PetscInt       k,kp1,km1,maxit,ktmp,i;
  PetscScalar    alpha,omegaprod,mu,omega,Gamma,c[3],scale;
  PetscReal      rnorm = 0.0;
  Vec            x,b,p[3],r;
  KSP_Chebychev  *chebychevP = (KSP_Chebychev*)ksp->data;
  Mat            Amat,Pmat;
  MatStructure   pflag;
  PetscTruth     diagonalscale;

  PetscFunctionBegin;
  if (ksp->normtype == KSP_NORM_NATURAL) SETERRQ(PETSC_ERR_SUP,"Cannot use natural residual norm with KSPCHEBYCHEV");

  ierr    = PCDiagonalScale(ksp->pc,&diagonalscale);CHKERRQ(ierr);
  if (diagonalscale) SETERRQ1(PETSC_ERR_SUP,"Krylov method %s does not support diagonal scaling",((PetscObject)ksp)->type_name);

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
  scale  = 2.0/(chebychevP->emax + chebychevP->emin);

  /*   -alpha <=  scale*lambda(B^{-1}A) <= alpha   */
  alpha  = 1.0 - scale*(chebychevP->emin); ;
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
    if (ksp->normtype != KSP_NORM_NO) {
      if (ksp->normtype == KSP_NORM_UNPRECONDITIONED) {ierr = VecNorm(r,NORM_2,&rnorm);CHKERRQ(ierr);}
      else {ierr = VecNorm(p[kp1],NORM_2,&rnorm);CHKERRQ(ierr);}
      ierr = PetscObjectTakeAccess(ksp);CHKERRQ(ierr);
      ksp->rnorm                              = rnorm;
      ierr = PetscObjectGrantAccess(ksp);CHKERRQ(ierr);
      ksp->vec_sol = p[k]; 
      KSPLogResidualHistory(ksp,rnorm);
      KSPMonitor(ksp,i,rnorm);
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
    if (ksp->normtype != KSP_NORM_NO) {
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
      KSPMonitor(ksp,i,rnorm);
    }
    if (ksp->its >= ksp->max_it) {
      if (ksp->normtype != KSP_NORM_NO) {
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
  PetscTruth     iascii;

  PetscFunctionBegin;
  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_ASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
    ierr = PetscViewerASCIIPrintf(viewer,"  Chebychev: eigenvalue estimates:  min = %G, max = %G\n",cheb->emin,cheb->emax);CHKERRQ(ierr);
  } else {
    SETERRQ1(PETSC_ERR_SUP,"Viewer type %s not supported for KSP Chebychev",((PetscObject)viewer)->type_name);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "KSPDestroy_Chebychev"
PetscErrorCode KSPDestroy_Chebychev(KSP ksp)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = KSPDefaultDestroy(ksp);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)ksp,"KSPChebychevSetEigenvalues_C","",PETSC_NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*MC
     KSPCHEBYCHEV - The preconditioned Chebychev iterative method

   Options Database Keys:
.   -ksp_chebychev_eigenvalues <emin,emax> - set approximations to the smallest and largest eigenvalues
                  of the preconditioned operator. If these are accurate you will get much faster convergence.

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
PetscErrorCode PETSCKSP_DLLEXPORT KSPCreate_Chebychev(KSP ksp)
{
  PetscErrorCode ierr;
  KSP_Chebychev  *chebychevP;

  PetscFunctionBegin;
  ierr = PetscNewLog(ksp,KSP_Chebychev,&chebychevP);CHKERRQ(ierr);

  ksp->data                      = (void*)chebychevP;
  if (ksp->pc_side != PC_LEFT) {
     ierr = PetscInfo(ksp,"WARNING! Setting PC_SIDE for Chebychev to left!\n");CHKERRQ(ierr);
  }
  ksp->pc_side                   = PC_LEFT;

  chebychevP->emin               = 1.e-2;
  chebychevP->emax               = 1.e+2;

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
  PetscFunctionReturn(0);
}
EXTERN_C_END
