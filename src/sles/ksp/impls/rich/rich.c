#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: rich.c,v 1.59 1998/03/06 00:11:32 bsmith Exp bsmith $";
#endif
/*          
            This implements Richardson Iteration.       
*/
#include <math.h>
#include "petsc.h"
#include "src/ksp/kspimpl.h"              /*I "ksp.h" I*/
#include "src/ksp/impls/rich/richctx.h"
#include "pinclude/pviewer.h"

#undef __FUNC__  
#define __FUNC__ "KSPSetUp_Richardson"
int KSPSetUp_Richardson(KSP ksp)
{
  int ierr;

  if (ksp->pc_side == PC_RIGHT) {SETERRQ(2,0,"no right preconditioning for KSPRICHARDSON");}
  else if (ksp->pc_side == PC_SYMMETRIC) {SETERRQ(2,0,"no symmetric preconditioning for KSPRICHARDSON");}
  ierr = KSPDefaultGetWork(ksp,2);CHKERRQ(ierr);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "KSPSolve_Richardson"
int  KSPSolve_Richardson(KSP ksp,int *its)
{
  int                i = 0,maxit,pres, brokeout = 0, hist_len, cerr = 0, ierr;
  MatStructure       pflag;
  double             rnorm,*history;
  Scalar             scale, mone = -1.0;
  Vec                x,b,r,z;
  Mat                Amat, Pmat;
  KSP_Richardson     *richardsonP = (KSP_Richardson *) ksp->data;
  PetscTruth         exists;

  PetscFunctionBegin;

  ksp->its = 0;

  ierr    = PCGetOperators(ksp->B,&Amat,&Pmat,&pflag); CHKERRQ(ierr);
  x       = ksp->vec_sol;
  b       = ksp->vec_rhs;
  r       = ksp->work[0];
  maxit   = ksp->max_it;

  /* if user has provided fast Richardson code use that */
  ierr = PCApplyRichardsonExists(ksp->B,&exists); CHKERRQ(ierr);
  if (exists && !ksp->numbermonitors) {
    *its = maxit;
    ierr = PCApplyRichardson(ksp->B,b,x,r,maxit);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }

  z       = ksp->work[1];
  history = ksp->residual_history;
  hist_len= ksp->res_hist_size;
  scale   = richardsonP->scale;
  pres    = ksp->use_pres;

  if (!ksp->guess_zero) {                          /*   r <- b - A x     */
    ierr = MatMult(Amat,x,r); CHKERRQ(ierr);
    ierr = VecAYPX(&mone,b,r); CHKERRQ(ierr);
  } else {
    ierr = VecCopy(b,r); CHKERRQ(ierr);
  }

  for ( i=0; i<maxit; i++ ) {
     ksp->its++;

     ierr = PCApply(ksp->B,r,z); CHKERRQ(ierr);    /*   z <- B r          */
     if (ksp->calc_res) {
	if (!pres) {
          ierr = VecNorm(r,NORM_2,&rnorm); CHKERRQ(ierr); /*   rnorm <- r'*r     */
        } else {
          ierr = VecNorm(z,NORM_2,&rnorm); CHKERRQ(ierr); /*   rnorm <- z'*z     */
        }
        ksp->rnorm                              = rnorm;
        if (history && hist_len > i) history[i] = rnorm;
        KSPMonitor(ksp,i,rnorm);
        cerr = (*ksp->converged)(ksp,i,rnorm,ksp->cnvP);
        if (cerr) {brokeout = 1; break;}
     }
   
     ierr = VecAXPY(&scale,z,x); CHKERRQ(ierr);    /*   x  <- x + scale z */
     ierr = MatMult(Amat,x,r); CHKERRQ(ierr);      /*   r  <- b - Ax      */
     ierr = VecAYPX(&mone,b,r); CHKERRQ(ierr);
  }
  if (ksp->calc_res && !brokeout) {
    if (!pres) {
      ierr = VecNorm(r,NORM_2,&rnorm); CHKERRQ(ierr);     /*   rnorm <- r'*r     */
    } else {
      ierr = PCApply(ksp->B,r,z); CHKERRQ(ierr);   /*   z <- B r          */
      ierr = VecNorm(z,NORM_2,&rnorm); CHKERRQ(ierr);     /*   rnorm <- z'*z     */
    }
    ksp->rnorm                              = rnorm;
    if (history && hist_len > i) history[i] = rnorm;
    KSPMonitor(ksp,i,rnorm);
  }
  if (history) ksp->res_act_size = (hist_len < i) ? hist_len : i;

  if (cerr <= 0) *its = -(i+1);
  else          *its = i+1;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "KSPView_Richardson" 
extern int KSPView_Richardson(PetscObject obj,Viewer viewer)
{
  KSP            ksp = (KSP)obj;
  KSP_Richardson *richardsonP = (KSP_Richardson *) ksp->data;
  FILE           *fd;
  int            ierr;
  ViewerType     vtype;

  PetscFunctionBegin;
  ierr = ViewerGetType(viewer,&vtype); CHKERRQ(ierr);
  if (vtype == ASCII_FILE_VIEWER || vtype == ASCII_FILES_VIEWER) {
    ierr = ViewerASCIIGetPointer(viewer,&fd); CHKERRQ(ierr);
    PetscFPrintf(ksp->comm,fd,"    Richardson: damping factor=%g\n",richardsonP->scale);
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "KSPPrintHelp_Richardson"
static int KSPPrintHelp_Richardson(KSP ksp,char *p)
{
  PetscFunctionBegin;

  (*PetscHelpPrintf)(ksp->comm," Options for Richardson method:\n");
  (*PetscHelpPrintf)(ksp->comm,"   %sksp_richardson_scale <scale> : damping factor\n",p);

  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "KSPSetFromOptions_Richardson"
int KSPSetFromOptions_Richardson(KSP ksp)
{
  int       ierr,flg;
  double    tmp;

  PetscFunctionBegin;

  ierr = OptionsGetDouble(ksp->prefix,"-ksp_richardson_scale",&tmp,&flg);CHKERRQ(ierr);
  if (flg) { ierr = KSPRichardsonSetScale(ksp,tmp); CHKERRQ(ierr); }

  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "KSPRichardsonSetScale_Richardson"
int KSPRichardsonSetScale_Richardson(KSP ksp,double scale)
{
  KSP_Richardson *richardsonP;

  PetscFunctionBegin;
  richardsonP = (KSP_Richardson *) ksp->data;
  richardsonP->scale = scale;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "KSPCreate_Richardson"
int KSPCreate_Richardson(KSP ksp)
{
  int            ierr;
  KSP_Richardson *richardsonP = PetscNew(KSP_Richardson); CHKPTRQ(richardsonP);

  PetscFunctionBegin;
  PLogObjectMemory(ksp,sizeof(KSP_Richardson));
  ksp->data                   = (void *) richardsonP;
  richardsonP->scale          = 1.0;
  ksp->setup                  = KSPSetUp_Richardson;
  ksp->solver                 = KSPSolve_Richardson;
  ksp->adjustwork             = KSPDefaultAdjustWork;
  ksp->destroy                = KSPDefaultDestroy;
  ksp->calc_res               = 1;
  ksp->converged              = KSPDefaultConverged;
  ksp->buildsolution          = KSPDefaultBuildSolution;
  ksp->buildresidual          = KSPDefaultBuildResidual;
  ksp->view                   = KSPView_Richardson;
  ksp->printhelp              = KSPPrintHelp_Richardson;
  ksp->setfromoptions         = KSPSetFromOptions_Richardson;

  ierr = DLRegister(&ksp->qlist,"KSPRichardsonSetScale","KSPRichardsonSetScale_Richardson",KSPRichardsonSetScale_Richardson);
  CHKERRQ(ierr);
  PetscFunctionReturn(0);
}



