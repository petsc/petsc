#define PETSCKSP_DLL

/*          
            This implements Richardson Iteration.       
*/
#include "src/ksp/ksp/kspimpl.h"              /*I "petscksp.h" I*/
#include "src/ksp/ksp/impls/rich/richctx.h"

#undef __FUNCT__  
#define __FUNCT__ "KSPSetUp_Richardson"
PetscErrorCode KSPSetUp_Richardson(KSP ksp)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (ksp->pc_side == PC_RIGHT) {SETERRQ(2,"no right preconditioning for KSPRICHARDSON");}
  else if (ksp->pc_side == PC_SYMMETRIC) {SETERRQ(2,"no symmetric preconditioning for KSPRICHARDSON");}
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
  PetscScalar    scale,mone = -1.0;
  Vec            x,b,r,z;
  Mat            Amat,Pmat;
  KSP_Richardson *richardsonP = (KSP_Richardson*)ksp->data;
  PetscTruth     exists,diagonalscale;

  PetscFunctionBegin;
  ierr    = PCDiagonalScale(ksp->pc,&diagonalscale);CHKERRQ(ierr);
  if (diagonalscale) SETERRQ1(PETSC_ERR_SUP,"Krylov method %s does not support diagonal scaling",ksp->type_name);

  ksp->its = 0;

  ierr    = PCGetOperators(ksp->pc,&Amat,&Pmat,&pflag);CHKERRQ(ierr);
  x       = ksp->vec_sol;
  b       = ksp->vec_rhs;
  r       = ksp->work[0];
  maxit   = ksp->max_it;

  /* if user has provided fast Richardson code use that */
  ierr = PCApplyRichardsonExists(ksp->pc,&exists);CHKERRQ(ierr);
  if (exists && !ksp->numbermonitors && !ksp->transpose_solve) {
    ierr = PCApplyRichardson(ksp->pc,b,x,r,ksp->rtol,ksp->abstol,ksp->divtol,maxit);CHKERRQ(ierr);
    ierr = VecNorm(r,NORM_2,&rnorm);CHKERRQ(ierr); /*   rnorm <- r'*r     */
    ierr = (*ksp->converged)(ksp,0,rnorm,&ksp->reason,ksp->cnvP);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }

  z       = ksp->work[1];
  scale   = richardsonP->scale;

  if (!ksp->guess_zero) {                          /*   r <- b - A x     */
    ierr = KSP_MatMult(ksp,Amat,x,r);CHKERRQ(ierr);
    ierr = VecAYPX(&mone,b,r);CHKERRQ(ierr);
  } else {
    ierr = VecCopy(b,r);CHKERRQ(ierr);
  }

  for (i=0; i<maxit; i++) {
    ierr = PetscObjectTakeAccess(ksp);CHKERRQ(ierr);
    ksp->its++;
    ierr = PetscObjectGrantAccess(ksp);CHKERRQ(ierr);

    if (ksp->normtype == KSP_UNPRECONDITIONED_NORM) {
      ierr = VecNorm(r,NORM_2,&rnorm);CHKERRQ(ierr); /*   rnorm <- r'*r     */
      KSPMonitor(ksp,i,rnorm);
    }

    ierr = KSP_PCApply(ksp,r,z);CHKERRQ(ierr);    /*   z <- B r          */

    if (ksp->normtype == KSP_PRECONDITIONED_NORM) {
      ierr = VecNorm(z,NORM_2,&rnorm);CHKERRQ(ierr); /*   rnorm <- z'*z     */
      KSPMonitor(ksp,i,rnorm);
    }

    ierr = VecAXPY(&scale,z,x);CHKERRQ(ierr);    /*   x  <- x + scale z */
    if (ksp->normtype != KSP_NO_NORM) {
      ierr       = PetscObjectTakeAccess(ksp);CHKERRQ(ierr);
      ksp->rnorm = rnorm;
      ierr       = PetscObjectGrantAccess(ksp);CHKERRQ(ierr);
      KSPLogResidualHistory(ksp,rnorm);

      ierr = (*ksp->converged)(ksp,i,rnorm,&ksp->reason,ksp->cnvP);CHKERRQ(ierr);
      if (ksp->reason) break;
    }
   
    ierr = KSP_MatMult(ksp,Amat,x,r);CHKERRQ(ierr);      /*   r  <- b - Ax      */
    ierr = VecAYPX(&mone,b,r);CHKERRQ(ierr);
  }
  if (!ksp->reason) {
    ksp->reason = KSP_DIVERGED_ITS;
    if (ksp->normtype != KSP_NO_NORM) {
      if (ksp->normtype == KSP_UNPRECONDITIONED_NORM){
        ierr = VecNorm(r,NORM_2,&rnorm);CHKERRQ(ierr);     /*   rnorm <- r'*r     */
      } else {
        ierr = KSP_PCApply(ksp,r,z);CHKERRQ(ierr);   /*   z <- B r          */
        ierr = VecNorm(z,NORM_2,&rnorm);CHKERRQ(ierr);     /*   rnorm <- z'*z     */
      }
    }
    ierr = PetscObjectTakeAccess(ksp);CHKERRQ(ierr);
    ksp->rnorm = rnorm;
    ierr = PetscObjectGrantAccess(ksp);CHKERRQ(ierr);
    KSPLogResidualHistory(ksp,rnorm);
    KSPMonitor(ksp,i,rnorm);
    i--;
  } else if (!ksp->reason) {
    ksp->reason = KSP_DIVERGED_ITS;
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
    ierr = PetscViewerASCIIPrintf(viewer,"  Richardson: damping factor=%g\n",richardsonP->scale);CHKERRQ(ierr);
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

          This method often (usually) will not converge unless scale is very small

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
  ierr = PetscNew(KSP_Richardson,&richardsonP);CHKERRQ(ierr);
  ierr = PetscLogObjectMemory(ksp,sizeof(KSP_Richardson));CHKERRQ(ierr);
  ksp->data                        = (void*)richardsonP;
  richardsonP->scale               = 1.0;
  ksp->ops->setup                  = KSPSetUp_Richardson;
  ksp->ops->solve                  = KSPSolve_Richardson;
  ksp->ops->destroy                = KSPDefaultDestroy;
  ksp->ops->buildsolution          = KSPDefaultBuildSolution;
  ksp->ops->buildresidual          = KSPDefaultBuildResidual;
  ksp->ops->view                   = KSPView_Richardson;
  ksp->ops->setfromoptions         = KSPSetFromOptions_Richardson;

  ierr = PetscObjectComposeFunctionDynamic((PetscObject)ksp,"KSPRichardsonSetScale_C",
                                    "KSPRichardsonSetScale_Richardson",
                                    KSPRichardsonSetScale_Richardson);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END


