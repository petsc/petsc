/*$Id: cgs.c,v 1.64 2001/08/07 03:03:51 balay Exp $*/

/*                       
    This code implements the CGS (Conjugate Gradient Squared) method. 
    Reference: Sonneveld, 1989.

    Note that for the complex numbers version, the VecDot() arguments
    within the code MUST remain in the order given for correct computation
    of inner products.
*/
#include "src/sles/ksp/kspimpl.h"

#undef __FUNCT__  
#define __FUNCT__ "KSPSetUp_CGS"
static int KSPSetUp_CGS(KSP ksp)
{
  int ierr;

  PetscFunctionBegin;
  if (ksp->pc_side == PC_SYMMETRIC) SETERRQ(2,"no symmetric preconditioning for KSPCGS");
  ierr = KSPDefaultGetWork(ksp,8);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "KSPSolve_CGS"
static int  KSPSolve_CGS(KSP ksp,int *its)
{
  int          i,maxit,ierr;
  PetscScalar  rho,rhoold,a,s,b,tmp,one = 1.0; 
  Vec          X,B,V,P,R,RP,T,Q,U,BINVF,AUQ;
  PetscReal    dp = 0.0;
  PetscTruth   diagonalscale;

  PetscFunctionBegin;
  ierr    = PCDiagonalScale(ksp->B,&diagonalscale);CHKERRQ(ierr);
  if (diagonalscale) SETERRQ1(1,"Krylov method %s does not support diagonal scaling",ksp->type_name);

  maxit   = ksp->max_it;
  X       = ksp->vec_sol;
  B       = ksp->vec_rhs;
  R       = ksp->work[0];
  RP      = ksp->work[1];
  V       = ksp->work[2];
  T       = ksp->work[3];
  Q       = ksp->work[4];
  P       = ksp->work[5];
  BINVF   = ksp->work[6];
  U       = ksp->work[7];
  AUQ     = V;

  /* Compute initial preconditioned residual */
  ierr = KSPInitialResidual(ksp,X,V,T,R,BINVF,B);CHKERRQ(ierr);

  /* Test for nothing to do */
  if (ksp->normtype == KSP_PRECONDITIONED_NORM) {
    ierr = VecNorm(R,NORM_2,&dp);CHKERRQ(ierr);
  } else if (ksp->normtype == KSP_NATURAL_NORM) {
    ierr = VecNorm(R,NORM_2,&dp);CHKERRQ(ierr);
  }
  ierr = PetscObjectTakeAccess(ksp);CHKERRQ(ierr);
  ksp->its   = 0;
  ksp->rnorm = dp;
  ierr = PetscObjectGrantAccess(ksp);CHKERRQ(ierr);
  KSPLogResidualHistory(ksp,dp);
  KSPMonitor(ksp,0,dp);
  ierr = (*ksp->converged)(ksp,0,dp,&ksp->reason,ksp->cnvP);CHKERRQ(ierr);
  if (ksp->reason) {*its = 0; PetscFunctionReturn(0);}

  /* Make the initial Rp == R */
  ierr = VecCopy(R,RP);CHKERRQ(ierr);

  /* Set the initial conditions */
  ierr = VecDot(R,RP,&rhoold);CHKERRQ(ierr);        /* rhoold = (r,rp)      */
  ierr = VecCopy(R,U);CHKERRQ(ierr);
  ierr = VecCopy(R,P);CHKERRQ(ierr);
  ierr = KSP_PCApplyBAorAB(ksp,ksp->B,ksp->pc_side,P,V,T);CHKERRQ(ierr);

  for (i=0; i<maxit; i++) {

    ierr = VecDot(V,RP,&s);CHKERRQ(ierr);           /* s <- (v,rp)          */
    a = rhoold / s;                                  /* a <- rho / s         */
    tmp = -a; 
    ierr = VecWAXPY(&tmp,V,U,Q);CHKERRQ(ierr);      /* q <- u - a v         */
    ierr = VecWAXPY(&one,U,Q,T);CHKERRQ(ierr);      /* t <- u + q           */
    ierr = VecAXPY(&a,T,X);CHKERRQ(ierr);           /* x <- x + a (u + q)   */
    ierr = KSP_PCApplyBAorAB(ksp,ksp->B,ksp->pc_side,T,AUQ,U);CHKERRQ(ierr);
    ierr = VecAXPY(&tmp,AUQ,R);CHKERRQ(ierr);       /* r <- r - a K (u + q) */
    if (ksp->normtype == KSP_PRECONDITIONED_NORM) {
      ierr = VecNorm(R,NORM_2,&dp);CHKERRQ(ierr);
    } else if (ksp->normtype == KSP_NATURAL_NORM) {
      dp = PetscAbsInt(rho);
    }

    ierr = PetscObjectTakeAccess(ksp);CHKERRQ(ierr);
    ksp->its++;
    ksp->rnorm = dp;
    ierr = PetscObjectGrantAccess(ksp);CHKERRQ(ierr);
    KSPLogResidualHistory(ksp,dp);
    KSPMonitor(ksp,i+1,dp);
    ierr = (*ksp->converged)(ksp,i+1,dp,&ksp->reason,ksp->cnvP);CHKERRQ(ierr);
    if (ksp->reason) break;

    ierr = VecDot(R,RP,&rho);CHKERRQ(ierr);         /* rho <- (r,rp)        */
    b    = rho / rhoold;                             /* b <- rho / rhoold    */
    ierr = VecWAXPY(&b,Q,R,U);CHKERRQ(ierr);        /* u <- r + b q         */
    ierr = VecAXPY(&b,P,Q);CHKERRQ(ierr);
    ierr = VecWAXPY(&b,Q,U,P);CHKERRQ(ierr);        /* p <- u + b(q + b p)  */
    ierr = KSP_PCApplyBAorAB(ksp,ksp->B,ksp->pc_side,P,V,Q);CHKERRQ(ierr);      /* v <- K p    */
    rhoold = rho;
  }
  if (i == maxit) {
    i--;
    ksp->reason = KSP_DIVERGED_ITS;
  }
  *its = i+1;

  ierr = KSPUnwindPreconditioner(ksp,X,T);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "KSPCreate_CGS"
int KSPCreate_CGS(KSP ksp)
{
  PetscFunctionBegin;
  ksp->data                      = (void*)0;
  ksp->pc_side                   = PC_LEFT;
  ksp->ops->setup                = KSPSetUp_CGS;
  ksp->ops->solve                = KSPSolve_CGS;
  ksp->ops->destroy              = KSPDefaultDestroy;
  ksp->ops->buildsolution        = KSPDefaultBuildSolution;
  ksp->ops->buildresidual        = KSPDefaultBuildResidual;
  ksp->ops->setfromoptions       = 0;
  ksp->ops->view                 = 0;
  PetscFunctionReturn(0);
}
EXTERN_C_END
