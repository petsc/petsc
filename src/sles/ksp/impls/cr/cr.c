/*$Id: cr.c,v 1.64 2001/08/07 03:03:49 balay Exp $*/

/*                       
           This implements Preconditioned Conjugate Residuals.       
*/
#include "src/sles/ksp/kspimpl.h"

#undef __FUNCT__  
#define __FUNCT__ "KSPSetUp_CR"
static int KSPSetUp_CR(KSP ksp)
{
  int ierr;

  PetscFunctionBegin;
  if (ksp->pc_side == PC_RIGHT) {SETERRQ(2,"no right preconditioning for KSPCR");}
  else if (ksp->pc_side == PC_SYMMETRIC) {SETERRQ(2,"no symmetric preconditioning for KSPCR");}
  ierr = KSPDefaultGetWork(ksp,6);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "KSPSolve_CR"
static int  KSPSolve_CR(KSP ksp,int *its)
{
  int          i = 0, maxit, cerr = 0, ierr;
  MatStructure pflag;
  PetscReal    dp;
  PetscScalar  ai, bi;
  PetscScalar  apq,btop, bbot, tmp, mone = -1.0;
  Vec          X,B,R,RT,P,AP,ART,Q;
  Mat          Amat, Pmat;

  PetscFunctionBegin;

  maxit   = ksp->max_it;
  X       = ksp->vec_sol;
  B       = ksp->vec_rhs;
  R       = ksp->work[0];
  RT      = ksp->work[1];
  P       = ksp->work[2];
  AP      = ksp->work[3];
  ART     = ksp->work[4];
  Q       = ksp->work[5];

                                      /*  we follow Rati Chandra's PhD */
  ierr = PCGetOperators(ksp->B,&Amat,&Pmat,&pflag); CHKERRQ(ierr);
                                                 /*                      */
  if (!ksp->guess_zero) {
    ierr = KSP_MatMult(ksp,Amat,X,R); CHKERRQ(ierr);     /*   r <- Ax            */
    ierr = VecAYPX(&mone,B,R); CHKERRQ(ierr);    /*   r <- b-r == b-Ax   */
  } else { 
    ierr = VecCopy(B,R); CHKERRQ(ierr);          /*   r <- b (x is 0)    */
  }
  ierr = KSP_PCApply(ksp,ksp->B,R,P); CHKERRQ(ierr);     /*   P <- Br            */
  ierr = KSP_MatMult(ksp,Amat,P,AP); CHKERRQ(ierr);      /*   AP <- A p          */
  ierr = VecCopy(P,RT); CHKERRQ(ierr);           /*   rt <- p            */
  ierr = VecCopy(AP,ART); CHKERRQ(ierr);         /*   ART <- AP          */
  ierr   = VecDot(RT,ART,&btop); CHKERRQ(ierr);  /*   (RT,ART)           */
  if (ksp->normtype == KSP_PRECONDITIONED_NORM) {
    ierr = VecNorm(P,NORM_2,&dp); CHKERRQ(ierr); /*   dp <- z'*z         */
  } else if (ksp->normtype == KSP_UNPRECONDITIONED_NORM) {
    ierr = VecNorm(R,NORM_2,&dp); CHKERRQ(ierr); /*   dp <- r'*r         */
  } else if (ksp->normtype == KSP_NATURAL_NORM) {
    dp = PetscAbsScalar(btop);                  /* dp = (R,AR) (fdi)*/
  }
  ierr = (*ksp->converged)(ksp,0,dp,&ksp->reason,ksp->cnvP);CHKERRQ(ierr);
  if (ksp->reason) {*its = 0; PetscFunctionReturn(0);}
  KSPMonitor(ksp,0,dp);
  ierr = PetscObjectTakeAccess(ksp);CHKERRQ(ierr);
  ksp->its = 0;
  ksp->rnorm              = dp;
  ierr = PetscObjectGrantAccess(ksp);CHKERRQ(ierr);
  KSPLogResidualHistory(ksp,dp);

  for ( i=0; i<maxit; i++) {
    ierr   = KSP_PCApply(ksp,ksp->B,AP,Q); CHKERRQ(ierr);/*   q <- B AP          */
                                                  /* Step 3              */

    ierr   = VecDot(AP,Q,&apq); CHKERRQ(ierr);  
    ai = btop/apq;                              /* ai = (RT,ART)/(AP,Q) */

    ierr   = VecAXPY(&ai,P,X); CHKERRQ(ierr);    /*   x <- x + ai p      */
    tmp    = -ai; 
    ierr   = VecAXPY(&tmp,Q,RT); CHKERRQ(ierr);  /*   rt <- rt - ai q    */
    ierr   = KSP_MatMult(ksp,Amat,RT,ART); CHKERRQ(ierr);/*   RT <-   ART        */
    bbot = btop;
    ierr   = VecDot(RT,ART,&btop); CHKERRQ(ierr);

    if (ksp->normtype == KSP_PRECONDITIONED_NORM) {
      ierr = VecNorm(RT,NORM_2,&dp); CHKERRQ(ierr);/*   dp <- r'*r         */
    } else if (ksp->normtype == KSP_NATURAL_NORM) {
      dp = PetscAbsScalar(btop);                /* dp = (R,AR) (fdi)*/
    } else { dp = 0.0; }

    ierr = PetscObjectTakeAccess(ksp);CHKERRQ(ierr);
    ksp->its++;
    ksp->rnorm = dp;
    ierr = PetscObjectGrantAccess(ksp);CHKERRQ(ierr);
                                                  /* Step 2              */
    KSPLogResidualHistory(ksp,dp);
    KSPMonitor(ksp,i+1,dp);
    ierr = (*ksp->converged)(ksp,i+1,dp,&ksp->reason,ksp->cnvP);CHKERRQ(ierr);
    if (ksp->reason) break;

    bi = btop/bbot;
    ierr = VecAYPX(&bi,RT,P); CHKERRQ(ierr);     /*   P <- rt + Bi P     */
    ierr = VecAYPX(&bi,ART,AP); CHKERRQ(ierr);   /*   AP <- Art + Bi AP  */
  }
  if (i == maxit) i--;
  if (cerr <= 0) *its = -(i+1);
  else           *its = i + 1;
  PetscFunctionReturn(0);
}


EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "KSPCreate_CR"
int KSPCreate_CR(KSP ksp)
{
  PetscFunctionBegin;
  ksp->pc_side                   = PC_LEFT;
  ksp->ops->setup                = KSPSetUp_CR;
  ksp->ops->solve                = KSPSolve_CR;
  ksp->ops->destroy              = KSPDefaultDestroy;
  ksp->ops->buildsolution        = KSPDefaultBuildSolution;
  ksp->ops->buildresidual        = KSPDefaultBuildResidual;
  ksp->ops->setfromoptions       = 0;
  ksp->ops->view                 = 0;
  PetscFunctionReturn(0);
}
EXTERN_C_END
