/*$Id: bicg.c,v 1.28 2001/08/07 03:03:55 balay Exp $*/

/*                       
    This code implements the BiCG (BiConjugate Gradient) method

    Contributed by: Victor Eijkhout

*/
#include "src/ksp/ksp/kspimpl.h"

#undef __FUNCT__  
#define __FUNCT__ "KSPSetUp_BiCG"
int KSPSetUp_BiCG(KSP ksp)
{
  int ierr;

  PetscFunctionBegin;
  /* check user parameters and functions */
  if (ksp->pc_side == PC_RIGHT) {
    SETERRQ(2,"no right preconditioning for KSPBiCG");
  } else if (ksp->pc_side == PC_SYMMETRIC) {
    SETERRQ(2,"no symmetric preconditioning for KSPBiCG");
  }

  /* get work vectors from user code */
  ierr = KSPDefaultGetWork(ksp,6);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "KSPSolve_BiCG"
int  KSPSolve_BiCG(KSP ksp)
{
  int          ierr,i;
  PetscTruth   diagonalscale;
  PetscScalar  dpi,a=1.0,beta,betaold=1.0,b,mone=-1.0,ma; 
  PetscReal    dp;
  Vec          X,B,Zl,Zr,Rl,Rr,Pl,Pr;
  Mat          Amat,Pmat;
  MatStructure pflag;

  PetscFunctionBegin;
  ierr    = PCDiagonalScale(ksp->B,&diagonalscale);CHKERRQ(ierr);
  if (diagonalscale) SETERRQ1(1,"Krylov method %s does not support diagonal scaling",ksp->type_name);

  X       = ksp->vec_sol;
  B       = ksp->vec_rhs;
  Rl      = ksp->work[0];
  Zl      = ksp->work[1];
  Pl      = ksp->work[2];
  Rr      = ksp->work[3];
  Zr      = ksp->work[4];
  Pr      = ksp->work[5];

  ierr = PCGetOperators(ksp->B,&Amat,&Pmat,&pflag);CHKERRQ(ierr);

  if (!ksp->guess_zero) {
    ierr = KSP_MatMult(ksp,Amat,X,Rr);CHKERRQ(ierr);      /*   r <- b - Ax       */
    ierr = VecAYPX(&mone,B,Rr);CHKERRQ(ierr);
  } else { 
    ierr = VecCopy(B,Rr);CHKERRQ(ierr);           /*     r <- b (x is 0) */
  }
  ierr = VecCopy(Rr,Rl);CHKERRQ(ierr);
  ierr = KSP_PCApply(ksp,ksp->B,Rr,Zr);CHKERRQ(ierr);     /*     z <- Br         */
  ierr = VecConjugate(Rl);CHKERRQ(ierr);
  ierr = KSP_PCApplyTranspose(ksp,ksp->B,Rl,Zl);CHKERRQ(ierr);
  ierr = VecConjugate(Rl);CHKERRQ(ierr);
  ierr = VecConjugate(Zl);CHKERRQ(ierr);
  if (ksp->normtype == KSP_PRECONDITIONED_NORM) {
    ierr = VecNorm(Zr,NORM_2,&dp);CHKERRQ(ierr);  /*    dp <- z'*z       */
  } else {
    ierr = VecNorm(Rr,NORM_2,&dp);CHKERRQ(ierr);  /*    dp <- r'*r       */
  }
  KSPMonitor(ksp,0,dp);
  ierr = PetscObjectTakeAccess(ksp);CHKERRQ(ierr);
  ksp->its   = 0;
  ksp->rnorm = dp;
  ierr = PetscObjectGrantAccess(ksp);CHKERRQ(ierr);
  KSPLogResidualHistory(ksp,dp);
  ierr = (*ksp->converged)(ksp,0,dp,&ksp->reason,ksp->cnvP);CHKERRQ(ierr);
  if (ksp->reason) PetscFunctionReturn(0);

  i = 0;
  do {
     ierr = VecDot(Zr,Rl,&beta);CHKERRQ(ierr);       /*     beta <- r'z     */
     if (!i) {
       if (beta == 0.0) {
         ksp->reason = KSP_DIVERGED_BREAKDOWN_BICG;
         PetscFunctionReturn(0);
       }
       ierr = VecCopy(Zr,Pr);CHKERRQ(ierr);       /*     p <- z          */
       ierr = VecCopy(Zl,Pl);CHKERRQ(ierr);
     } else {
       b = beta/betaold;
       ierr = VecAYPX(&b,Zr,Pr);CHKERRQ(ierr);  /*     p <- z + b* p   */
       b = PetscConj(b);
       ierr = VecAYPX(&b,Zl,Pl);CHKERRQ(ierr);
     }
     betaold = beta;
     ierr = KSP_MatMult(ksp,Amat,Pr,Zr);CHKERRQ(ierr);    /*     z <- Kp         */
     ierr = VecConjugate(Pl);CHKERRQ(ierr);
     ierr = KSP_MatMultTranspose(ksp,Amat,Pl,Zl);CHKERRQ(ierr);
     ierr = VecConjugate(Pl);CHKERRQ(ierr);
     ierr = VecConjugate(Zl);CHKERRQ(ierr);
     ierr = VecDot(Zr,Pl,&dpi);CHKERRQ(ierr);               /*     dpi <- z'p      */
     a = beta/dpi;                                 /*     a = beta/p'z    */
     ierr = VecAXPY(&a,Pr,X);CHKERRQ(ierr);       /*     x <- x + ap     */
     ma = -a;
     ierr = VecAXPY(&ma,Zr,Rr);CHKERRQ(ierr)
     ma = PetscConj(ma);
     ierr = VecAXPY(&ma,Zl,Rl);CHKERRQ(ierr);
     if (ksp->normtype == KSP_PRECONDITIONED_NORM) {
       ierr = KSP_PCApply(ksp,ksp->B,Rr,Zr);CHKERRQ(ierr);  /*     z <- Br         */
       ierr = VecConjugate(Rl);CHKERRQ(ierr);
       ierr = KSP_PCApplyTranspose(ksp,ksp->B,Rl,Zl);CHKERRQ(ierr);
       ierr = VecConjugate(Rl);CHKERRQ(ierr);
       ierr = VecConjugate(Zl);CHKERRQ(ierr);
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
     ierr = (*ksp->converged)(ksp,i+1,dp,&ksp->reason,ksp->cnvP);CHKERRQ(ierr);
     if (ksp->reason) break;
     if (ksp->normtype == KSP_UNPRECONDITIONED_NORM) {
       ierr = KSP_PCApply(ksp,ksp->B,Rr,Zr);CHKERRQ(ierr);  /* z <- Br  */
       ierr = VecConjugate(Rl);CHKERRQ(ierr);
       ierr = KSP_PCApplyTranspose(ksp,ksp->B,Rl,Zl);CHKERRQ(ierr);
       ierr = VecConjugate(Rl);CHKERRQ(ierr);
       ierr = VecConjugate(Zl);CHKERRQ(ierr);
     }
     i++;
  } while (i<ksp->max_it);
  if (i == ksp->max_it) {
    ksp->reason = KSP_DIVERGED_ITS;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "KSPDestroy_BiCG" 
int KSPDestroy_BiCG(KSP ksp)
{
  int ierr;

  PetscFunctionBegin;
  ierr = KSPDefaultFreeWork(ksp);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "KSPCreate_BiCG"
int KSPCreate_BiCG(KSP ksp)
{
  PetscFunctionBegin;
  ksp->data                      = (void*)0;
  ksp->pc_side                   = PC_LEFT;
  ksp->ops->setup                = KSPSetUp_BiCG;
  ksp->ops->solve                = KSPSolve_BiCG;
  ksp->ops->destroy              = KSPDestroy_BiCG;
  ksp->ops->view                 = 0;
  ksp->ops->setfromoptions       = 0;
  ksp->ops->buildsolution        = KSPDefaultBuildSolution;
  ksp->ops->buildresidual        = KSPDefaultBuildResidual;

  PetscFunctionReturn(0);
}
EXTERN_C_END





