/*$Id: rich.c,v 1.104 2001/08/21 21:03:36 bsmith Exp $*/
/*          
            This implements Richardson Iteration.       
*/
#include "src/sles/ksp/kspimpl.h"              /*I "petscksp.h" I*/
#include "src/sles/ksp/impls/rich/richctx.h"

#undef __FUNCT__  
#define __FUNCT__ "KSPSetUp_Richardson"
int KSPSetUp_Richardson(KSP ksp)
{
  int ierr;

  PetscFunctionBegin;
  if (ksp->pc_side == PC_RIGHT) {SETERRQ(2,"no right preconditioning for KSPRICHARDSON");}
  else if (ksp->pc_side == PC_SYMMETRIC) {SETERRQ(2,"no symmetric preconditioning for KSPRICHARDSON");}
  ierr = KSPDefaultGetWork(ksp,2);CHKERRQ(ierr);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "KSPSolve_Richardson"
int  KSPSolve_Richardson(KSP ksp,int *its)
{
  int             i,maxit,ierr;
  MatStructure    pflag;
  PetscReal       rnorm = 0.0;
  PetscScalar     scale,mone = -1.0;
  Vec             x,b,r,z;
  Mat             Amat,Pmat;
  KSP_Richardson  *richardsonP = (KSP_Richardson*)ksp->data;
  PetscTruth      exists,diagonalscale;

  PetscFunctionBegin;
  ierr    = PCDiagonalScale(ksp->B,&diagonalscale);CHKERRQ(ierr);
  if (diagonalscale) SETERRQ1(1,"Krylov method %s does not support diagonal scaling",ksp->type_name);

  ksp->its = 0;

  ierr    = PCGetOperators(ksp->B,&Amat,&Pmat,&pflag);CHKERRQ(ierr);
  x       = ksp->vec_sol;
  b       = ksp->vec_rhs;
  r       = ksp->work[0];
  maxit   = ksp->max_it;

  /* if user has provided fast Richardson code use that */
  ierr = PCApplyRichardsonExists(ksp->B,&exists);CHKERRQ(ierr);
  if (exists && !ksp->numbermonitors && !ksp->transpose_solve) {
    if (its) *its = maxit;
    ierr = PCApplyRichardson(ksp->B,b,x,r,ksp->rtol,ksp->atol,ksp->divtol,maxit);CHKERRQ(ierr);
    ksp->reason = KSP_DIVERGED_ITS; /* what should we really put here? */
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

    ierr = KSP_PCApply(ksp,ksp->B,r,z);CHKERRQ(ierr);    /*   z <- B r          */

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
        ierr = KSP_PCApply(ksp,ksp->B,r,z);CHKERRQ(ierr);   /*   z <- B r          */
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

  if (its) *its = ksp->its;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "KSPView_Richardson" 
int KSPView_Richardson(KSP ksp,PetscViewer viewer)
{
  KSP_Richardson *richardsonP = (KSP_Richardson*)ksp->data;
  int            ierr;
  PetscTruth     isascii;

  PetscFunctionBegin;
  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_ASCII,&isascii);CHKERRQ(ierr);
  if (isascii) {
    ierr = PetscViewerASCIIPrintf(viewer,"  Richardson: damping factor=%g\n",richardsonP->scale);CHKERRQ(ierr);
  } else {
    SETERRQ1(1,"Viewer type %s not supported for KSP Richardson",((PetscObject)viewer)->type_name);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "KSPSetFromOptions_Richardson"
int KSPSetFromOptions_Richardson(KSP ksp)
{
  KSP_Richardson *rich = (KSP_Richardson*)ksp->data;
  int            ierr;
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
int KSPRichardsonSetScale_Richardson(KSP ksp,PetscReal scale)
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
#define __FUNCT__ "KSPCreate_Richardson"
int KSPCreate_Richardson(KSP ksp)
{
  int            ierr;
  KSP_Richardson *richardsonP;

  PetscFunctionBegin;
  ierr = PetscNew(KSP_Richardson,&richardsonP);CHKERRQ(ierr);
  PetscLogObjectMemory(ksp,sizeof(KSP_Richardson));
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


