#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: tfqmr.c,v 1.42 1999/02/09 22:52:17 bsmith Exp bsmith $";
#endif

/*                       
    This code implements the TFQMR (Transpose-free variant of Quasi-Minimal
    Residual) method.  Reference: Freund, 1993

    Note that for the complex numbers version, the VecDot() arguments
    within the code MUST remain in the order given for correct computation
    of inner products.
*/

#include "src/sles/ksp/kspimpl.h"

#undef __FUNC__  
#define __FUNC__ "KSPSetUp_TFQMR"
static int KSPSetUp_TFQMR(KSP ksp)
{
  int ierr;

  PetscFunctionBegin;
  if (ksp->pc_side == PC_SYMMETRIC){
    SETERRQ(2,0,"no symmetric preconditioning for KSPTFQMR");
  }
  ierr = KSPDefaultGetWork( ksp,  10 ); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "KSPSolve_TFQMR"
static int  KSPSolve_TFQMR(KSP ksp,int *its)
{
  int       i=0, maxit, m, conv=0, hist_len, cerr=0, ierr;
  Scalar    rho,rhoold,a,s,b,eta,etaold,psiold,cf,tmp,one = 1.0,zero = 0.0;
  double    *history,dp,dpold,w,dpest,tau,psi,cm;
  Vec       X,B,V,P,R,RP,T,T1,Q,U, D, BINVF, AUQ;

  PetscFunctionBegin;
  maxit    = ksp->max_it;
  history  = ksp->residual_history;
  hist_len = ksp->res_hist_size;
  X        = ksp->vec_sol;
  B        = ksp->vec_rhs;
  R        = ksp->work[0];
  RP       = ksp->work[1];
  V        = ksp->work[2];
  T        = ksp->work[3];
  Q        = ksp->work[4];
  P        = ksp->work[5];
  BINVF    = ksp->work[6];
  U        = ksp->work[7];
  D        = ksp->work[8];
  T1       = ksp->work[9];
  AUQ      = V;

  /* Compute initial preconditioned residual */
  ierr = KSPResidual(ksp,X,V,T, R, BINVF, B ); CHKERRQ(ierr);

  /* Test for nothing to do */
  ierr = VecNorm(R,NORM_2,&dp); CHKERRQ(ierr);
  PetscAMSTakeAccess(ksp);
  ksp->rnorm  = dp;
  ksp->its    = 0;
  PetscAMSGrantAccess(ksp);
  if ((*ksp->converged)(ksp,0,dp,ksp->cnvP)) {*its = 0; PetscFunctionReturn(0);}
  KSPMonitor(ksp,0,dp);

  /* Make the initial Rp == R */
  ierr = VecCopy(R,RP); CHKERRQ(ierr);

  /* Set the initial conditions */
  etaold = 0.0;
  psiold = 0.0;
  tau    = dp;
  dpold  = dp;

  ierr = VecDot(R,RP,&rhoold); CHKERRQ(ierr);       /* rhoold = (r,rp)     */
  ierr = VecCopy(R,U); CHKERRQ(ierr);
  ierr = VecCopy(R,P); CHKERRQ(ierr);
  ierr = PCApplyBAorAB(ksp->B,ksp->pc_side,P,V,T); CHKERRQ(ierr);
  ierr = VecSet(&zero,D); CHKERRQ(ierr);

  for (i=0; i<maxit; i++) {
    PetscAMSTakeAccess(ksp);
    ksp->its++;
    PetscAMSGrantAccess(ksp);
    ierr = VecDot(V,RP,&s); CHKERRQ(ierr);          /* s <- (v,rp)          */
    a = rhoold / s;                                 /* a <- rho / s         */
    tmp = -a; VecWAXPY(&tmp,V,U,Q); CHKERRQ(ierr);  /* q <- u - a v         */
    ierr = VecWAXPY(&one,U,Q,T); CHKERRQ(ierr);     /* t <- u + q           */
    ierr = PCApplyBAorAB(ksp->B,ksp->pc_side,T,AUQ,T1); CHKERRQ(ierr);
    ierr = VecAXPY(&tmp,AUQ,R); CHKERRQ(ierr);      /* r <- r - a K (u + q) */
    ierr = VecNorm(R,NORM_2,&dp); CHKERRQ(ierr);
    for (m=0; m<2; m++) {
      if (m == 0) {
        w = sqrt(dp*dpold);
      } else {
        w = dp;
      }
      psi = w / tau;
      cm  = 1.0 / sqrt( 1.0 + psi * psi );
      tau = tau * psi * cm;
      eta = cm * cm * a;
      cf  = psiold * psiold * etaold / a;
      if (m == 0) {
        ierr = VecAYPX(&cf,U,D); CHKERRQ(ierr);
      } else {
	ierr = VecAYPX(&cf,Q,D); CHKERRQ(ierr);
      }
      ierr = VecAXPY(&eta,D,X); CHKERRQ(ierr);

      dpest = sqrt(m + 1.0) * tau;
      PetscAMSTakeAccess(ksp);
      ksp->rnorm                                    = dpest;
      PetscAMSGrantAccess(ksp);
      if (history && hist_len > i + 1) history[i+1] = dpest;
      KSPMonitor(ksp,i+1,dpest);
      if ((conv = cerr = (*ksp->converged)(ksp,i+1,dpest,ksp->cnvP))) break;

      etaold = eta;
      psiold = psi;
    }
    if (conv) break;

    ierr = VecDot(R,RP,&rho); CHKERRQ(ierr);        /* rho <- (r,rp)       */
    b = rho / rhoold;                               /* b <- rho / rhoold   */
    ierr = VecWAXPY(&b,Q,R,U); CHKERRQ(ierr);       /* u <- r + b q        */
    ierr = VecAXPY(&b,P,Q); CHKERRQ(ierr);
    ierr = VecWAXPY(&b,Q,U,P); CHKERRQ(ierr);       /* p <- u + b(q + b p) */
    ierr = PCApplyBAorAB(ksp->B,ksp->pc_side,P,V,Q); CHKERRQ(ierr); /* v <- K p  */

    rhoold = rho;
    dpold  = dp;
  }
  if (i == maxit) i--;
  if (history) ksp->res_act_size = (hist_len < i + 1) ? hist_len : i + 1;

  ierr = KSPUnwindPreconditioner(ksp,X,T); CHKERRQ(ierr);
  if (cerr <= 0) *its = -(i+1);
  else          *its = i + 1;
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNC__  
#define __FUNC__ "KSPCreate_TFQMR"
int KSPCreate_TFQMR(KSP ksp)
{
  PetscFunctionBegin;
  ksp->data                      = (void *) 0;
  ksp->pc_side                   = PC_LEFT;
  ksp->calc_res                  = 1;
  ksp->ops->setup                = KSPSetUp_TFQMR;
  ksp->ops->solve                = KSPSolve_TFQMR;
  ksp->ops->destroy              = KSPDefaultDestroy;
  ksp->converged                 = KSPDefaultConverged;
  ksp->ops->buildsolution        = KSPDefaultBuildSolution;
  ksp->ops->buildresidual        = KSPDefaultBuildResidual;
  ksp->ops->view                 = 0;
  PetscFunctionReturn(0);
}
EXTERN_C_END
