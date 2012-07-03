
/*          
            This implements Richardson Iteration.       
*/
#include <petsc-private/kspimpl.h>              /*I "petscksp.h" I*/
#include <../src/ksp/ksp/impls/rich/richardsonimpl.h>

#undef __FUNCT__  
#define __FUNCT__ "KSPSetUp_Richardson"
PetscErrorCode KSPSetUp_Richardson(KSP ksp)
{
  PetscErrorCode ierr;
  KSP_Richardson *richardsonP = (KSP_Richardson*)ksp->data;

  PetscFunctionBegin;
  if (richardsonP->selfscale) {
    ierr  = KSPDefaultGetWork(ksp,4);CHKERRQ(ierr);
  } else {
    ierr  = KSPDefaultGetWork(ksp,2);CHKERRQ(ierr);
  }
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
  PetscScalar    scale,abr,rdot;
  Vec            x,b,r,z,w = PETSC_NULL,y = PETSC_NULL;
  PetscInt       xs, ws;
  Mat            Amat,Pmat;
  KSP_Richardson *richardsonP = (KSP_Richardson*)ksp->data;
  PetscBool      exists,diagonalscale;

  PetscFunctionBegin;
  ierr    = PCGetDiagonalScale(ksp->pc,&diagonalscale);CHKERRQ(ierr);
  if (diagonalscale) SETERRQ1(((PetscObject)ksp)->comm,PETSC_ERR_SUP,"Krylov method %s does not support diagonal scaling",((PetscObject)ksp)->type_name);

  ksp->its = 0;

  ierr    = PCGetOperators(ksp->pc,&Amat,&Pmat,&pflag);CHKERRQ(ierr);
  x       = ksp->vec_sol;
  b       = ksp->vec_rhs;
  ierr    = VecGetSize(x,&xs);CHKERRQ(ierr);
  ierr    = VecGetSize(ksp->work[0],&ws);CHKERRQ(ierr);
  if (xs != ws) {
    if (richardsonP->selfscale) {
      ierr  = KSPDefaultGetWork(ksp,4);CHKERRQ(ierr);
    } else {
      ierr  = KSPDefaultGetWork(ksp,2);CHKERRQ(ierr);
    }
  }
  r       = ksp->work[0];
  z       = ksp->work[1];
  if (richardsonP->selfscale) {
    w     = ksp->work[2];
    y     = ksp->work[3];
  }
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
  if (richardsonP->selfscale) {
    ierr = KSP_PCApply(ksp,r,z);CHKERRQ(ierr);         /*   z <- B r          */
    for (i=0; i<maxit; i++) {
      
      if (ksp->normtype == KSP_NORM_UNPRECONDITIONED) {
	ierr = VecNorm(r,NORM_2,&rnorm);CHKERRQ(ierr); /*   rnorm <- r'*r     */
	ierr = KSPMonitor(ksp,i,rnorm);CHKERRQ(ierr);
	ksp->rnorm = rnorm;
	KSPLogResidualHistory(ksp,rnorm);
	ierr = (*ksp->converged)(ksp,i,rnorm,&ksp->reason,ksp->cnvP);CHKERRQ(ierr);
	if (ksp->reason) break;
      } else if (ksp->normtype == KSP_NORM_PRECONDITIONED) {
	ierr = VecNorm(z,NORM_2,&rnorm);CHKERRQ(ierr); /*   rnorm <- z'*z     */
	ierr = KSPMonitor(ksp,i,rnorm);CHKERRQ(ierr);
	ksp->rnorm = rnorm;
	KSPLogResidualHistory(ksp,rnorm);
	ierr = (*ksp->converged)(ksp,i,rnorm,&ksp->reason,ksp->cnvP);CHKERRQ(ierr);
	if (ksp->reason) break;
      }
      ierr = KSP_PCApplyBAorAB(ksp,z,y,w);CHKERRQ(ierr);  /* y = BAz = BABr */
      ierr  = VecDotNorm2(z,y,&rdot,&abr);CHKERRQ(ierr);   /*   rdot = (Br)^T(BABR); abr = (BABr)^T (BABr) */
      scale = rdot/abr;
      ierr = PetscInfo1(ksp,"Self-scale factor %G\n",PetscRealPart(scale));CHKERRQ(ierr);
      ierr = VecAXPY(x,scale,z);CHKERRQ(ierr);    /*   x  <- x + scale z */
      ierr = VecAXPY(r,-scale,w);CHKERRQ(ierr);   /*  r <- r - scale*Az */
      ierr = VecAXPY(z,-scale,y);CHKERRQ(ierr);   /*  z <- z - scale*y */
      ksp->its++;
    }
  } else {
    for (i=0; i<maxit; i++) {

      if (ksp->normtype == KSP_NORM_UNPRECONDITIONED) {
	ierr = VecNorm(r,NORM_2,&rnorm);CHKERRQ(ierr); /*   rnorm <- r'*r     */
	ierr = KSPMonitor(ksp,i,rnorm);CHKERRQ(ierr);
	ksp->rnorm = rnorm;
	KSPLogResidualHistory(ksp,rnorm);
	ierr = (*ksp->converged)(ksp,i,rnorm,&ksp->reason,ksp->cnvP);CHKERRQ(ierr);
	if (ksp->reason) break;
      }

      ierr = KSP_PCApply(ksp,r,z);CHKERRQ(ierr);    /*   z <- B r          */

      if (ksp->normtype == KSP_NORM_PRECONDITIONED) {
	ierr = VecNorm(z,NORM_2,&rnorm);CHKERRQ(ierr); /*   rnorm <- z'*z     */
	ierr = KSPMonitor(ksp,i,rnorm);CHKERRQ(ierr);
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
  }
  if (!ksp->reason) {
    if (ksp->normtype != KSP_NORM_NONE) {
      if (ksp->normtype == KSP_NORM_UNPRECONDITIONED){
        ierr = VecNorm(r,NORM_2,&rnorm);CHKERRQ(ierr);     /*   rnorm <- r'*r     */
      } else {
        ierr = KSP_PCApply(ksp,r,z);CHKERRQ(ierr);   /*   z <- B r          */
        ierr = VecNorm(z,NORM_2,&rnorm);CHKERRQ(ierr);     /*   rnorm <- z'*z     */
      }
      ksp->rnorm = rnorm;
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
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "KSPView_Richardson" 
PetscErrorCode KSPView_Richardson(KSP ksp,PetscViewer viewer)
{
  KSP_Richardson *richardsonP = (KSP_Richardson*)ksp->data;
  PetscErrorCode ierr;
  PetscBool      iascii;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
    if (richardsonP->selfscale) {
      ierr = PetscViewerASCIIPrintf(viewer,"  Richardson: using self-scale best computed damping factor\n");CHKERRQ(ierr);
    } else {
      ierr = PetscViewerASCIIPrintf(viewer,"  Richardson: damping factor=%G\n",richardsonP->scale);CHKERRQ(ierr);
    }
  } else {
    SETERRQ1(((PetscObject)ksp)->comm,PETSC_ERR_SUP,"Viewer type %s not supported for KSP Richardson",((PetscObject)viewer)->type_name);
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
  PetscBool      flg,flg2;

  PetscFunctionBegin;
  ierr = PetscOptionsHead("KSP Richardson Options");CHKERRQ(ierr);
    ierr = PetscOptionsReal("-ksp_richardson_scale","damping factor","KSPRichardsonSetScale",rich->scale,&tmp,&flg);CHKERRQ(ierr);
    if (flg) { ierr = KSPRichardsonSetScale(ksp,tmp);CHKERRQ(ierr); }
    ierr = PetscOptionsBool("-ksp_richardson_self_scale","dynamically determine optimal damping factor","KSPRichardsonSetSelfScale",rich->selfscale,&flg2,&flg);CHKERRQ(ierr);
    if (flg) { ierr = KSPRichardsonSetSelfScale(ksp,flg2);CHKERRQ(ierr); }
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "KSPDestroy_Richardson"
PetscErrorCode KSPDestroy_Richardson(KSP ksp)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)ksp,"KSPRichardsonSetScale_C","",PETSC_NULL);CHKERRQ(ierr);
  ierr = KSPDefaultDestroy(ksp);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "KSPRichardsonSetScale_Richardson"
PetscErrorCode  KSPRichardsonSetScale_Richardson(KSP ksp,PetscReal scale)
{
  KSP_Richardson *richardsonP;

  PetscFunctionBegin;
  richardsonP = (KSP_Richardson*)ksp->data;
  richardsonP->scale = scale;
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "KSPRichardsonSetSelfScale_Richardson"
PetscErrorCode  KSPRichardsonSetSelfScale_Richardson(KSP ksp,PetscBool  selfscale)
{
  KSP_Richardson *richardsonP;

  PetscFunctionBegin;
  richardsonP = (KSP_Richardson*)ksp->data;
  richardsonP->selfscale = selfscale;
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
    you specifically call KSPSetNormType(ksp,KSP_NORM_NONE);

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
PetscErrorCode  KSPCreate_Richardson(KSP ksp)
{
  PetscErrorCode ierr;
  KSP_Richardson *richardsonP;

  PetscFunctionBegin;
  ierr = PetscNewLog(ksp,KSP_Richardson,&richardsonP);CHKERRQ(ierr);
  ksp->data                        = (void*)richardsonP;

  ierr = KSPSetSupportedNorm(ksp,KSP_NORM_PRECONDITIONED,PC_LEFT,2);CHKERRQ(ierr);
  ierr = KSPSetSupportedNorm(ksp,KSP_NORM_UNPRECONDITIONED,PC_LEFT,1);CHKERRQ(ierr);

  ksp->ops->setup                  = KSPSetUp_Richardson;
  ksp->ops->solve                  = KSPSolve_Richardson;
  ksp->ops->destroy                = KSPDestroy_Richardson;
  ksp->ops->buildsolution          = KSPDefaultBuildSolution;
  ksp->ops->buildresidual          = KSPDefaultBuildResidual;
  ksp->ops->view                   = KSPView_Richardson;
  ksp->ops->setfromoptions         = KSPSetFromOptions_Richardson;

  ierr = PetscObjectComposeFunctionDynamic((PetscObject)ksp,"KSPRichardsonSetScale_C","KSPRichardsonSetScale_Richardson",KSPRichardsonSetScale_Richardson);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)ksp,"KSPRichardsonSetSelfScale_C","KSPRichardsonSetSelfScale_Richardson",KSPRichardsonSetSelfScale_Richardson);CHKERRQ(ierr);
  richardsonP->scale               = 1.0;
  PetscFunctionReturn(0);
}
EXTERN_C_END


