#define PETSCKSP_DLL

/*          
            This implements Richardson Iteration.       
*/
#include "private/kspimpl.h"              /*I "petscksp.h" I*/
#include "../src/ksp/ksp/impls/rich/richardsonimpl.h"

#undef __FUNCT__  
#define __FUNCT__ "KSPSetUp_Richardson"
PetscErrorCode KSPSetUp_Richardson(KSP ksp)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (ksp->pc_side == PC_RIGHT) {SETERRQ(PETSC_ERR_SUP,"no right preconditioning for KSPRICHARDSON");}
  else if (ksp->pc_side == PC_SYMMETRIC) {SETERRQ(PETSC_ERR_SUP,"no symmetric preconditioning for KSPRICHARDSON");}
  ierr = KSPDefaultGetWork(ksp,2);CHKERRQ(ierr);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "KSPSolve_Richardson"
PetscErrorCode  KSPSolve_Richardson(KSP ksp)
{
  PetscErrorCode ierr;
  PetscInt       i,maxit;
  MatStructure   pflag;
  PetscReal      rnorm = 0.0;
  PetscScalar    scale;
  Vec            x,b,r,z;
  PetscInt       xs, ws;
  Mat            Amat,Pmat;
  KSP_Richardson *richardsonP = (KSP_Richardson*)ksp->data;
  PetscTruth     exists,diagonalscale;

  PetscFunctionBegin;
  if (ksp->normtype == KSP_NORM_NATURAL) SETERRQ(PETSC_ERR_SUP,"Cannot use natural residual norm for KSPRICHARDSON");

  ierr    = PCDiagonalScale(ksp->pc,&diagonalscale);CHKERRQ(ierr);
  if (diagonalscale) SETERRQ1(PETSC_ERR_SUP,"Krylov method %s does not support diagonal scaling",((PetscObject)ksp)->type_name);

  ksp->its = 0;

  ierr    = PCGetOperators(ksp->pc,&Amat,&Pmat,&pflag);CHKERRQ(ierr);
  x       = ksp->vec_sol;
  b       = ksp->vec_rhs;
  ierr    = VecGetSize(x,&xs);CHKERRQ(ierr);
  ierr    = VecGetSize(ksp->work[0],&ws);CHKERRQ(ierr);
  if (xs != ws) {
    ierr  = KSPDefaultFreeWork(ksp);CHKERRQ(ierr);
    ierr  = KSPDefaultGetWork(ksp,2);CHKERRQ(ierr);
  }
  r       = ksp->work[0];
  z       = ksp->work[1];
  maxit   = ksp->max_it;

  /* if user has provided fast Richardson code use that */
  ierr = PCApplyRichardsonExists(ksp->pc,&exists);CHKERRQ(ierr);
  if (exists && !ksp->numbermonitors && !ksp->transpose_solve) {
    PCRichardsonConvergedReason reason;
    ierr = PCApplyRichardson(ksp->pc,b,x,r,ksp->rtol,ksp->abstol,ksp->divtol,maxit,ksp->guess_zero,&ksp->its,&reason);CHKERRQ(ierr);
    ksp->reason = (KSPConvergedReason)reason;
    PetscFunctionReturn(0);
  }

  scale   = richardsonP->scale;

  if (!ksp->guess_zero) {                          /*   r <- b - A x     */
    ierr = KSP_MatMult(ksp,Amat,x,r);CHKERRQ(ierr);
    ierr = VecAYPX(r,-1.0,b);CHKERRQ(ierr);
  } else {
    ierr = VecCopy(b,r);CHKERRQ(ierr);
  }

  ksp->its = 0;
  for (i=0; i<maxit; i++) {

    if (ksp->normtype == KSP_NORM_UNPRECONDITIONED) {
      ierr = VecNorm(r,NORM_2,&rnorm);CHKERRQ(ierr); /*   rnorm <- r'*r     */
      KSPMonitor(ksp,i,rnorm);
      ksp->rnorm = rnorm;
      KSPLogResidualHistory(ksp,rnorm);
      ierr = (*ksp->converged)(ksp,i,rnorm,&ksp->reason,ksp->cnvP);CHKERRQ(ierr);
      if (ksp->reason) break;
    }

    ierr = KSP_PCApply(ksp,r,z);CHKERRQ(ierr);    /*   z <- B r          */

    if (ksp->normtype == KSP_NORM_PRECONDITIONED) {
      ierr = VecNorm(z,NORM_2,&rnorm);CHKERRQ(ierr); /*   rnorm <- z'*z     */
      KSPMonitor(ksp,i,rnorm);
      ksp->rnorm = rnorm;
      KSPLogResidualHistory(ksp,rnorm);
      ierr = (*ksp->converged)(ksp,i,rnorm,&ksp->reason,ksp->cnvP);CHKERRQ(ierr);
      if (ksp->reason) break;
    }
   
    ierr = VecAXPY(x,scale,z);CHKERRQ(ierr);    /*   x  <- x + scale z */
    ksp->its++;

    ierr = KSP_MatMult(ksp,Amat,x,r);CHKERRQ(ierr);      /*   r  <- b - Ax      */
    ierr = VecAYPX(r,-1.0,b);CHKERRQ(ierr);
  }
  if (!ksp->reason) {
    if (ksp->normtype != KSP_NORM_NO) {
      if (ksp->normtype == KSP_NORM_UNPRECONDITIONED){
        ierr = VecNorm(r,NORM_2,&rnorm);CHKERRQ(ierr);     /*   rnorm <- r'*r     */
      } else {
        ierr = KSP_PCApply(ksp,r,z);CHKERRQ(ierr);   /*   z <- B r          */
        ierr = VecNorm(z,NORM_2,&rnorm);CHKERRQ(ierr);     /*   rnorm <- z'*z     */
      }
      ierr = PetscObjectTakeAccess(ksp);CHKERRQ(ierr);
      ksp->rnorm = rnorm;
      ierr = PetscObjectGrantAccess(ksp);CHKERRQ(ierr);
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
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "KSPView_Richardson" 
PetscErrorCode KSPView_Richardson(KSP ksp,PetscViewer viewer)
{
  KSP_Richardson *richardsonP = (KSP_Richardson*)ksp->data;
  PetscErrorCode ierr;
  PetscTruth     iascii;

  PetscFunctionBegin;
  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_ASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
    ierr = PetscViewerASCIIPrintf(viewer,"  Richardson: damping factor=%G\n",richardsonP->scale);CHKERRQ(ierr);
  } else {
    SETERRQ1(PETSC_ERR_SUP,"Viewer type %s not supported for KSP Richardson",((PetscObject)viewer)->type_name);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "KSPSetFromOptions_Richardson"
PetscErrorCode KSPSetFromOptions_Richardson(KSP ksp)
{
  KSP_Richardson *rich = (KSP_Richardson*)ksp->data;
  PetscErrorCode ierr;
  PetscReal      tmp;
  PetscTruth     flg;

  PetscFunctionBegin;
  ierr = PetscOptionsHead("KSP Richardson Options");CHKERRQ(ierr);
    ierr = PetscOptionsReal("-ksp_richardson_scale","damping factor","KSPRichardsonSetScale",rich->scale,&tmp,&flg);CHKERRQ(ierr);
    if (flg) { ierr = KSPRichardsonSetScale(ksp,tmp);CHKERRQ(ierr); }
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "KSPDestroy_Richardson"
PetscErrorCode KSPDestroy_Richardson(KSP ksp)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = KSPDefaultDestroy(ksp);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)ksp,"KSPRichardsonSetScale_C","",PETSC_NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "KSPRichardsonSetScale_Richardson"
PetscErrorCode PETSCKSP_DLLEXPORT KSPRichardsonSetScale_Richardson(KSP ksp,PetscReal scale)
{
  KSP_Richardson *richardsonP;

  PetscFunctionBegin;
  richardsonP = (KSP_Richardson*)ksp->data;
  richardsonP->scale = scale;
  PetscFunctionReturn(0);
}
EXTERN_C_END

/*MC
     KSPRICHARDSON - The preconditioned Richardson iterative method

   Options Database Keys:
.   -ksp_richardson_scale - damping factor on the correction (defaults to 1.0)

   Level: beginner

   Notes: x^{n+1} = x^{n} + scale*B(b - A x^{n})
 
          Here B is the application of the preconditioner

          This method often (usually) will not converge unless scale is very small. It
is described in


   Notes: For some preconditioners, currently SOR, the convergence test is skipped to improve speed,
    thus it always iterates the maximum number of iterations you've selected. When -ksp_monitor 
    (or any other monitor) is turned on, the norm is computed at each iteration and so the convergence test is run unless
    you specifically call KSPSetNormType(ksp,KSP_NORM_NO);

         For some preconditioners, currently PCMG and PCHYPRE with BoomerAMG if -ksp_monitor (and also
    any other monitor) is not turned on then the convergence test is done by the preconditioner itself and
    so the solver may run more or fewer iterations then if -ksp_monitor is selected.

    Supports only left preconditioning

  References:
  "The Approximate Arithmetical Solution by Finite Differences of Physical Problems Involving
   Differential Equations, with an Application to the Stresses in a Masonry Dam",
  L. F. Richardson, Philosophical Transactions of the Royal Society of London. Series A,
  Containing Papers of a Mathematical or Physical Character, Vol. 210, 1911 (1911), pp. 307-357.

.seealso:  KSPCreate(), KSPSetType(), KSPType (for list of available types), KSP,
           KSPRichardsonSetScale()

M*/

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "KSPCreate_Richardson"
PetscErrorCode PETSCKSP_DLLEXPORT KSPCreate_Richardson(KSP ksp)
{
  PetscErrorCode ierr;
  KSP_Richardson *richardsonP;

  PetscFunctionBegin;
  ierr = PetscNewLog(ksp,KSP_Richardson,&richardsonP);CHKERRQ(ierr);
  ksp->data                        = (void*)richardsonP;

  ksp->normtype                    = KSP_NORM_PRECONDITIONED;
  if (ksp->pc_side != PC_LEFT) {
     ierr = PetscInfo(ksp,"WARNING! Setting PC_SIDE for Richardson to left!\n");CHKERRQ(ierr);
  }
  ksp->pc_side                     = PC_LEFT;

  ksp->ops->setup                  = KSPSetUp_Richardson;
  ksp->ops->solve                  = KSPSolve_Richardson;
  ksp->ops->destroy                = KSPDestroy_Richardson;
  ksp->ops->buildsolution          = KSPDefaultBuildSolution;
  ksp->ops->buildresidual          = KSPDefaultBuildResidual;
  ksp->ops->view                   = KSPView_Richardson;
  ksp->ops->setfromoptions         = KSPSetFromOptions_Richardson;

  ierr = PetscObjectComposeFunctionDynamic((PetscObject)ksp,"KSPRichardsonSetScale_C",
                                    "KSPRichardsonSetScale_Richardson",
                                    KSPRichardsonSetScale_Richardson);CHKERRQ(ierr);
  richardsonP->scale               = 1.0;
  PetscFunctionReturn(0);
}
EXTERN_C_END


