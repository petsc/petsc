/*$Id: bicg.c,v 1.13 1999/10/13 20:38:19 bsmith Exp bsmith $*/

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
  ierr = KSPDefaultGetWork( ksp, 6 );CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "KSPSolve_BiCG"
int  KSPSolve_BiCG(KSP ksp,int *its)
{
  int          ierr,i,maxit,pres, cerr;
  Scalar       dpi,a=1.0,beta,betaold=1.0,b,mone=-1.0,ma; 
  double       dp;
  Vec          X,B,Zl,Zr,Rl,Rr,Pl,Pr;
  Mat          Amat,Pmat;
  MatStructure pflag;

  PetscFunctionBegin;
  pres    = ksp->use_pres;
  maxit   = ksp->max_it;
  X       = ksp->vec_sol;
  B       = ksp->vec_rhs;
  Rl       = ksp->work[0];
  Zl       = ksp->work[1];
  Pl       = ksp->work[2];
  Rr       = ksp->work[3];
  Zr       = ksp->work[4];
  Pr       = ksp->work[5];

  ierr = PCGetOperators(ksp->B,&Amat,&Pmat,&pflag);CHKERRQ(ierr);

  if (!ksp->guess_zero) {
    ierr = MatMult(Amat,X,Rr);CHKERRQ(ierr);      /*   r <- b - Ax       */
    ierr = VecAYPX(&mone,B,Rr);CHKERRQ(ierr);
  } else { 
    ierr = VecCopy(B,Rr);CHKERRQ(ierr);           /*     r <- b (x is 0) */
  }
  ierr = VecCopy(Rr,Rl);CHKERRQ(ierr);
  ierr = PCApply(ksp->B,Rr,Zr);CHKERRQ(ierr);     /*     z <- Br         */
  ierr = PCApplyTrans(ksp->B,Rl,Zl);CHKERRQ(ierr);
  if (pres) {
      ierr = VecNorm(Zr,NORM_2,&dp);CHKERRQ(ierr);  /*    dp <- z'*z       */
  } else {
      ierr = VecNorm(Rr,NORM_2,&dp);CHKERRQ(ierr);  /*    dp <- r'*r       */
  }
  cerr = (*ksp->converged)(ksp,0,dp,ksp->cnvP);
  if (cerr) {*its =  0; PetscFunctionReturn(0);}
  KSPMonitor(ksp,0,dp);
  ierr = PetscObjectTakeAccess(ksp);CHKERRQ(ierr);
  ksp->its   = 0;
  ksp->rnorm = dp;
  ierr = PetscObjectGrantAccess(ksp);CHKERRQ(ierr);
  KSPLogResidualHistory(ksp,dp);

  for ( i=0; i<maxit; i++) {
     VecDot(Zr,Rl,&beta);                         /*     beta <- r'z     */
     if (i == 0) {
       if (beta == 0.0) break;
       ierr = VecCopy(Zr,Pr);CHKERRQ(ierr);       /*     p <- z          */
       ierr = VecCopy(Zl,Pl);CHKERRQ(ierr);
     } else {
         b = beta/betaold;
         ierr = VecAYPX(&b,Zr,Pr);CHKERRQ(ierr);  /*     p <- z + b* p   */
         ierr = VecAYPX(&b,Zl,Pl);CHKERRQ(ierr);
     }
     betaold = beta;
     ierr = MatMult(Amat,Pr,Zr);CHKERRQ(ierr);    /*     z <- Kp         */
     ierr = MatMultTrans(Amat,Pl,Zl);CHKERRQ(ierr);
     VecDot(Pl,Zr,&dpi);                          /*     dpi <- z'p      */
     a = beta/dpi;                                 /*     a = beta/p'z    */
     ierr = VecAXPY(&a,Pr,X);CHKERRQ(ierr);       /*     x <- x + ap     */
     ma = -a; VecAXPY(&ma,Zr,Rr);                  /*     r <- r - az     */
     VecAXPY(&ma,Zl,Rl);
     if (pres) {
       ierr = PCApply(ksp->B,Rr,Zr);CHKERRQ(ierr);  /*     z <- Br         */
       ierr = PCApplyTrans(ksp->B,Rl,Zl);CHKERRQ(ierr);
       ierr = VecNorm(Zr,NORM_2,&dp);CHKERRQ(ierr);  /*    dp <- z'*z       */
     } else {
       ierr = VecNorm(Rr,NORM_2,&dp);CHKERRQ(ierr);  /*    dp <- r'*r       */
     }
     ierr = PetscObjectTakeAccess(ksp);CHKERRQ(ierr);
     ksp->its   = i+1;
     ksp->rnorm = dp;
     ierr = PetscObjectGrantAccess(ksp);CHKERRQ(ierr);
     KSPLogResidualHistory(ksp,dp);
     KSPMonitor(ksp,i+1,dp);
     cerr = (*ksp->converged)(ksp,i+1,dp,ksp->cnvP);
     if (cerr) break;
     if (!pres) {
       ierr = PCApply(ksp->B,Rr,Zr);CHKERRQ(ierr);  /* z <- Br  */
       ierr = PCApplyTrans(ksp->B,Rl,Zl);CHKERRQ(ierr);
     }
  }
  if (i == maxit) {i--; ksp->its--;}
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
  ksp->guess_zero                = 1; 
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





