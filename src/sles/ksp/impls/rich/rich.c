/*$Id: rich.c,v 1.86 1999/11/24 21:54:54 bsmith Exp bsmith $*/
/*          
            This implements Richardson Iteration.       
*/
#include "src/sles/ksp/kspimpl.h"              /*I "ksp.h" I*/
#include "src/sles/ksp/impls/rich/richctx.h"

#undef __FUNC__  
#define __FUNC__ "KSPSetUp_Richardson"
int KSPSetUp_Richardson(KSP ksp)
{
  int ierr;

  PetscFunctionBegin;
  if (ksp->pc_side == PC_RIGHT) {SETERRQ(2,0,"no right preconditioning for KSPRICHARDSON");}
  else if (ksp->pc_side == PC_SYMMETRIC) {SETERRQ(2,0,"no symmetric preconditioning for KSPRICHARDSON");}
  ierr = KSPDefaultGetWork(ksp,2);CHKERRQ(ierr);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "KSPSolve_Richardson"
int  KSPSolve_Richardson(KSP ksp,int *its)
{
  int                i,maxit,ierr;
  MatStructure       pflag;
  PetscReal          rnorm = 0.0;
  Scalar             scale,mone = -1.0;
  Vec                x,b,r,z;
  Mat                Amat,Pmat;
  KSP_Richardson     *richardsonP = (KSP_Richardson*)ksp->data;
  PetscTruth         exists,pres;

  PetscFunctionBegin;

  ksp->its = 0;

  ierr    = PCGetOperators(ksp->B,&Amat,&Pmat,&pflag);CHKERRQ(ierr);
  x       = ksp->vec_sol;
  b       = ksp->vec_rhs;
  r       = ksp->work[0];
  maxit   = ksp->max_it;

  /* if user has provided fast Richardson code use that */
  ierr = PCApplyRichardsonExists(ksp->B,&exists);CHKERRQ(ierr);
  if (exists && !ksp->numbermonitors && !ksp->transpose_solve) {
    *its = maxit;
    ierr = PCApplyRichardson(ksp->B,b,x,r,maxit);CHKERRQ(ierr);
    ksp->reason = KSP_DIVERGED_ITS; /* what should we really put here? */
    PetscFunctionReturn(0);
  }

  z       = ksp->work[1];
  scale   = richardsonP->scale;
  pres    = ksp->use_pres;

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

     ierr = KSP_PCApply(ksp,ksp->B,r,z);CHKERRQ(ierr);    /*   z <- B r          */
     if (ksp->calc_res) {
       if (!ksp->avoidnorms) {
         if (!pres) {
           ierr = VecNorm(r,NORM_2,&rnorm);CHKERRQ(ierr); /*   rnorm <- r'*r     */
         } else {
           ierr = VecNorm(z,NORM_2,&rnorm);CHKERRQ(ierr); /*   rnorm <- z'*z     */
         }
       }
       ierr = PetscObjectTakeAccess(ksp);CHKERRQ(ierr);
       ksp->rnorm                              = rnorm;
       ierr = PetscObjectGrantAccess(ksp);CHKERRQ(ierr);
       KSPLogResidualHistory(ksp,rnorm);
       KSPMonitor(ksp,i,rnorm);
       ierr = (*ksp->converged)(ksp,i,rnorm,&ksp->reason,ksp->cnvP);CHKERRQ(ierr);
       if (ksp->reason) break;
     }
   
     ierr = VecAXPY(&scale,z,x);CHKERRQ(ierr);    /*   x  <- x + scale z */
     ierr = KSP_MatMult(ksp,Amat,x,r);CHKERRQ(ierr);      /*   r  <- b - Ax      */
     ierr = VecAYPX(&mone,b,r);CHKERRQ(ierr);
  }
  if (ksp->calc_res && !ksp->reason) {
    ksp->reason = KSP_DIVERGED_ITS;
    if (!ksp->avoidnorms) {
      if (!pres) {
        ierr = VecNorm(r,NORM_2,&rnorm);CHKERRQ(ierr);     /*   rnorm <- r'*r     */
      } else {
        ierr = KSP_PCApply(ksp,ksp->B,r,z);CHKERRQ(ierr);   /*   z <- B r          */
        ierr = VecNorm(z,NORM_2,&rnorm);CHKERRQ(ierr);     /*   rnorm <- z'*z     */
      }
    }
    ierr = PetscObjectTakeAccess(ksp);CHKERRQ(ierr);
    ksp->rnorm                              = rnorm;
    ierr = PetscObjectGrantAccess(ksp);CHKERRQ(ierr);
    KSPLogResidualHistory(ksp,rnorm);
    KSPMonitor(ksp,i,rnorm);
    i--;
  } else if (!ksp->reason) {
    ksp->reason = KSP_DIVERGED_ITS;
    i--;
  }

  *its = i+1;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "KSPView_Richardson" 
extern int KSPView_Richardson(KSP ksp,Viewer viewer)
{
  KSP_Richardson *richardsonP = (KSP_Richardson*)ksp->data;
  int            ierr;
  PetscTruth     isascii;

  PetscFunctionBegin;
  ierr = PetscTypeCompare((PetscObject)viewer,ASCII_VIEWER,&isascii);CHKERRQ(ierr);
  if (isascii) {
    ierr = ViewerASCIIPrintf(viewer,"  Richardson: damping factor=%g\n",richardsonP->scale);CHKERRQ(ierr);
  } else {
    SETERRQ1(1,1,"Viewer type %s not supported for KSP Richardson",((PetscObject)viewer)->type_name);
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "KSPPrintHelp_Richardson"
static int KSPPrintHelp_Richardson(KSP ksp,char *p)
{
  int ierr;

  PetscFunctionBegin;
  ierr = (*PetscHelpPrintf)(ksp->comm," Options for Richardson method:\n");CHKERRQ(ierr);
  ierr = (*PetscHelpPrintf)(ksp->comm,"   %sksp_richardson_scale <scale> : damping factor\n",p);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "KSPSetFromOptions_Richardson"
int KSPSetFromOptions_Richardson(KSP ksp)
{
  int        ierr;
  PetscReal  tmp;
  PetscTruth flg;

  PetscFunctionBegin;

  ierr = OptionsGetDouble(ksp->prefix,"-ksp_richardson_scale",&tmp,&flg);CHKERRQ(ierr);
  if (flg) { ierr = KSPRichardsonSetScale(ksp,tmp);CHKERRQ(ierr); }

  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNC__  
#define __FUNC__ "KSPRichardsonSetScale_Richardson"
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
#undef __FUNC__  
#define __FUNC__ "KSPCreate_Richardson"
int KSPCreate_Richardson(KSP ksp)
{
  int            ierr;
  KSP_Richardson *richardsonP = PetscNew(KSP_Richardson);CHKPTRQ(richardsonP);

  PetscFunctionBegin;
  PLogObjectMemory(ksp,sizeof(KSP_Richardson));
  ksp->data                        = (void*)richardsonP;
  richardsonP->scale               = 1.0;
  ksp->calc_res                    = PETSC_TRUE;
  ksp->ops->setup                  = KSPSetUp_Richardson;
  ksp->ops->solve                  = KSPSolve_Richardson;
  ksp->ops->destroy                = KSPDefaultDestroy;
  ksp->ops->buildsolution          = KSPDefaultBuildSolution;
  ksp->ops->buildresidual          = KSPDefaultBuildResidual;
  ksp->ops->view                   = KSPView_Richardson;
  ksp->ops->printhelp              = KSPPrintHelp_Richardson;
  ksp->ops->setfromoptions         = KSPSetFromOptions_Richardson;

  ierr = PetscObjectComposeFunctionDynamic((PetscObject)ksp,"KSPRichardsonSetScale_C",
                                    "KSPRichardsonSetScale_Richardson",
                                    (void*)KSPRichardsonSetScale_Richardson);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END


