#ifndef lint
static char vcid[] = "$Id: rich.c,v 1.37 1996/03/19 21:24:02 bsmith Exp bsmith $";
#endif
/*          
            This implements Richardson Iteration.       
*/
#include <stdio.h>
#include <math.h>
#include "petsc.h"
#include "kspimpl.h"         /*I "ksp.h" I*/
#include "richctx.h"
#include "pinclude/pviewer.h"

int KSPSetUp_Richardson(KSP ksp)
{
  int ierr;
  /* check user parameters and functions */
  if (ksp->pc_side == PC_RIGHT)
    {SETERRQ(2,"KSPSetUp_Richardson:no right preconditioning for KSPRICHARDSON");}
  else if (ksp->pc_side == PC_SYMMETRIC)
    {SETERRQ(2,"KSPSetUp_Richardson:no symmetric preconditioning for KSPRICHARDSON");}
  ierr = KSPCheckDef(ksp); CHKERRQ(ierr);
  /* get work vectors from user code */
  return KSPiDefaultGetWork(ksp,2);
}

/*@
    KSPRichardsonSetScale - Call after KSPCreate(KSPRICHARDSON) to set
    the damping factor; if this routine is not called, the factor 
    defaults to 1.0.

    Input Parameters:
.   ksp - the iterative context
.   scale - the relaxation factor

.keywords: KSP, Richardson, set, scale
@*/
int KSPRichardsonSetScale(KSP ksp,double scale)
{
  KSP_Richardson *richardsonP;
  PetscValidHeaderSpecific(ksp,KSP_COOKIE);
  if (ksp->type != KSPRICHARDSON) return 0;
  richardsonP = (KSP_Richardson *) ksp->data;
  richardsonP->scale = scale;
  return 0;
}

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

  ierr    = PCGetOperators(ksp->B,&Amat,&Pmat,&pflag); CHKERRQ(ierr);
  x       = ksp->vec_sol;
  b       = ksp->vec_rhs;
  r       = ksp->work[0];
  maxit   = ksp->max_it;

  /* if user has provided fast Richardson code use that */
  ierr = PCApplyRichardsonExists(ksp->B,&exists); CHKERRQ(ierr);
  if (exists) {
    *its = maxit;
    return PCApplyRichardson(ksp->B,b,x,r,maxit);
  }

  z       = ksp->work[1];
  history = ksp->residual_history;
  hist_len= ksp->res_hist_size;
  scale   = richardsonP->scale;
  pres    = ksp->use_pres;

  if (!ksp->guess_zero) {                          /*   r <- b - A x     */
    ierr = MatMult(Amat,x,r); CHKERRQ(ierr);
    ierr = VecAYPX(&mone,b,r); CHKERRQ(ierr);
  }
  else {
    ierr = VecCopy(b,r); CHKERRQ(ierr);
  }

  for ( i=0; i<maxit; i++ ) {
     ierr = PCApply(ksp->B,r,z); CHKERRQ(ierr);    /*   z <- B r          */
     if (ksp->calc_res) {
	if (!pres) {
          ierr = VecNorm(r,NORM_2,&rnorm); CHKERRQ(ierr); /*   rnorm <- r'*r     */
        }
	else {
          ierr = VecNorm(z,NORM_2,&rnorm); CHKERRQ(ierr); /*   rnorm <- z'*z     */
        }
        if (history && hist_len > i) history[i] = rnorm;
        KSPMonitor(ksp,rnorm,i);
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
    }
    else {
      ierr = PCApply(ksp->B,r,z); CHKERRQ(ierr);   /*   z <- B r          */
      ierr = VecNorm(z,NORM_2,&rnorm); CHKERRQ(ierr);     /*   rnorm <- z'*z     */
    }
    if (history && hist_len > i) history[i] = rnorm;
    KSPMonitor(ksp,rnorm,i);
  }
  if (history) ksp->res_act_size = (hist_len < i) ? hist_len : i;

  if (cerr <= 0) *its = -(i+1);
  else          *its = i+1;
  return 0;
}

static int KSPView_Richardson(PetscObject obj,Viewer viewer)
{
  KSP            ksp = (KSP)obj;
  KSP_Richardson *richardsonP = (KSP_Richardson *) ksp->data;
  FILE           *fd;
  int            ierr;
  ViewerType     vtype;

  ierr = ViewerGetType(viewer,&vtype); CHKERRQ(ierr);
  if (vtype == ASCII_FILE_VIEWER || vtype == ASCII_FILES_VIEWER) {
    ierr = ViewerASCIIGetPointer(viewer,&fd); CHKERRQ(ierr);
    PetscFPrintf(ksp->comm,fd,"    Richardson: damping factor=%g\n",richardsonP->scale);
  }
  return 0;
}

int KSPCreate_Richardson(KSP ksp)
{
  KSP_Richardson *richardsonP = PetscNew(KSP_Richardson); CHKPTRQ(richardsonP);
  ksp->data                   = (void *) richardsonP;
  ksp->type                   = KSPRICHARDSON;
  richardsonP->scale          = 1.0;
  ksp->setup                  = KSPSetUp_Richardson;
  ksp->solver                 = KSPSolve_Richardson;
  ksp->adjustwork             = KSPiDefaultAdjustWork;
  ksp->destroy                = KSPiDefaultDestroy;
  ksp->calc_res               = 1;
  ksp->converged              = KSPDefaultConverged;
  ksp->buildsolution          = KSPDefaultBuildSolution;
  ksp->buildresidual          = KSPDefaultBuildResidual;
  ksp->view                   = KSPView_Richardson;
  return 0;
}




