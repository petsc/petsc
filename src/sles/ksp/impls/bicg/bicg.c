#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: bicg.c,v 1.6 1999/01/31 16:09:06 bsmith Exp bsmith $";
#endif

/*                       
    This code implements the BiCG (BiConjugate Gradient) method

    Contibuted by: Victor Eijkhout

*/
#include "src/sles/ksp/kspimpl.h"

#undef __FUNC__  
#define __FUNC__ "KSPSetUp_BiCG"
int KSPSetUp_BiCG(KSP ksp)
{
  int ierr;

  PetscFunctionBegin;
  /* check user parameters and functions */
  if (ksp->pc_side == PC_RIGHT) {
    SETERRQ(2,0,"no right preconditioning for KSPBiCG");
  } else if (ksp->pc_side == PC_SYMMETRIC) {
    SETERRQ(2,0,"no symmetric preconditioning for KSPBiCG");
  }

  /* get work vectors from user code */
  ierr = KSPDefaultGetWork( ksp, 6 ); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "KSPSolve_BiCG"
int  KSPSolve_BiCG(KSP ksp,int *its)
{
  int          ierr, i = 0,maxit,pres, hist_len, cerr;
  Scalar       dpi, a = 1.0,beta,betaold = 1.0,b, mone = -1.0, ma; 
  double       *history, dp;
  Vec          X,B,Zl,Zr,Rl,Rr,Pl,Pr;
  Mat          Amat, Pmat;
  MatStructure pflag;

  PetscFunctionBegin;
  pres    = ksp->use_pres;
  maxit   = ksp->max_it;
  history = ksp->residual_history;
  hist_len= ksp->res_hist_size;
  X       = ksp->vec_sol;
  B       = ksp->vec_rhs;
  Rl       = ksp->work[0];
  Zl       = ksp->work[1];
  Pl       = ksp->work[2];
  Rr       = ksp->work[3];
  Zr       = ksp->work[4];
  Pr       = ksp->work[5];

  ierr = PCGetOperators(ksp->B,&Amat,&Pmat,&pflag); CHKERRQ(ierr);

  ksp->its = 0;
  if (!ksp->guess_zero) {
    ierr = MatMult(Amat,X,Rr); CHKERRQ(ierr);      /*   r <- b - Ax       */
    ierr = VecAYPX(&mone,B,Rr); CHKERRQ(ierr);
  } else { 
    ierr = VecCopy(B,Rr); CHKERRQ(ierr);           /*     r <- b (x is 0) */
  }
  ierr = VecCopy(Rr,Rl); CHKERRQ(ierr);
  ierr = PCApply(ksp->B,Rr,Zr); CHKERRQ(ierr);     /*     z <- Br         */
  ierr = PCApplyTrans(ksp->B,Rl,Zl); CHKERRQ(ierr);
  if (pres) {
      ierr = VecNorm(Zr,NORM_2,&dp); CHKERRQ(ierr);  /*    dp <- z'*z       */
  } else {
      ierr = VecNorm(Rr,NORM_2,&dp); CHKERRQ(ierr);  /*    dp <- r'*r       */
  }
  cerr = (*ksp->converged)(ksp,0,dp,ksp->cnvP);
  if (cerr) {*its =  0; PetscFunctionReturn(0);}
  KSPMonitor(ksp,0,dp);
  PetscAMSTakeAccess(ksp);
  ksp->rnorm              = dp;
  PetscAMSGrantAccess(ksp);
  if (history) history[0] = dp;

  for ( i=0; i<maxit; i++) {
     ksp->its = i+1;
     VecDot(Zr,Rl,&beta);                         /*     beta <- r'z     */
     if (i == 0) {
       if (beta == 0.0) break;
       ierr = VecCopy(Zr,Pr); CHKERRQ(ierr);       /*     p <- z          */
       ierr = VecCopy(Zl,Pl); CHKERRQ(ierr);
     } else {
         b = beta/betaold;
         ierr = VecAYPX(&b,Zr,Pr); CHKERRQ(ierr);  /*     p <- z + b* p   */
         ierr = VecAYPX(&b,Zl,Pl); CHKERRQ(ierr);
     }
     betaold = beta;
     ierr = MatMult(Amat,Pr,Zr); CHKERRQ(ierr);    /*     z <- Kp         */
     ierr = MatMultTrans(Amat,Pl,Zl); CHKERRQ(ierr);
     VecDot(Pl,Zr,&dpi);                          /*     dpi <- z'p      */
     a = beta/dpi;                                 /*     a = beta/p'z    */
     ierr = VecAXPY(&a,Pr,X); CHKERRQ(ierr);       /*     x <- x + ap     */
     ma = -a; VecAXPY(&ma,Zr,Rr);                  /*     r <- r - az     */
     VecAXPY(&ma,Zl,Rl);
     if (pres) {
       ierr = PCApply(ksp->B,Rr,Zr); CHKERRQ(ierr);  /*     z <- Br         */
       ierr = PCApplyTrans(ksp->B,Rl,Zl); CHKERRQ(ierr);
       ierr = VecNorm(Zr,NORM_2,&dp); CHKERRQ(ierr);  /*    dp <- z'*z       */
     } else {
       ierr = VecNorm(Rr,NORM_2,&dp); CHKERRQ(ierr);  /*    dp <- r'*r       */
     }
     PetscAMSTakeAccess(ksp);
     ksp->rnorm = dp;
     PetscAMSGrantAccess(ksp);
     if (history && hist_len > i + 1) history[i+1] = dp;
     KSPMonitor(ksp,i+1,dp);
     cerr = (*ksp->converged)(ksp,i+1,dp,ksp->cnvP);
     if (cerr) break;
     if (!pres) {
       ierr = PCApply(ksp->B,Rr,Zr); CHKERRQ(ierr);  /* z <- Br  */
       ierr = PCApplyTrans(ksp->B,Rl,Zl); CHKERRQ(ierr);
     }
  }
  if (i == maxit) {i--; ksp->its--;}
  if (history) ksp->res_act_size = (hist_len < i + 1) ? hist_len : i + 1;
  if (cerr <= 0) *its = -(i+1);
  else           *its = i+1;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "KSPDestroy_BiCG" 
int KSPDestroy_BiCG(KSP ksp)
{
  int ierr;

  PetscFunctionBegin;
  ierr = KSPDefaultFreeWork( ksp );CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNC__  
#define __FUNC__ "KSPCreate_BiCG"
int KSPCreate_BiCG(KSP ksp)
{
  PetscFunctionBegin;
  ksp->data                      = (void *) 0;
  ksp->pc_side                   = PC_LEFT;
  ksp->calc_res                  = 1;
  ksp->ops->setup                = KSPSetUp_BiCG;
  ksp->ops->solve                = KSPSolve_BiCG;
  ksp->ops->destroy              = KSPDestroy_BiCG;
  ksp->ops->view                 = 0;
  ksp->ops->printhelp            = 0;
  ksp->ops->setfromoptions       = 0;
  ksp->converged                 = KSPDefaultConverged;
  ksp->ops->buildsolution        = KSPDefaultBuildSolution;
  ksp->ops->buildresidual        = KSPDefaultBuildResidual;

  PetscFunctionReturn(0);
}
EXTERN_C_END





