#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: bcgs.c,v 1.63 1999/09/27 21:01:41 bsmith Exp bsmith $";
#endif

/*                       
    This code implements the BiCGStab (Stabilized version of BiConjugate
    Gradient Squared) method.  Reference: van der Vorst, SIAM J. Sci. Stat. Comput., 1992.

    Note that for the complex numbers version, the VecDot() arguments
    within the code MUST remain in the order given for correct computation
    of inner products.
*/
#include "petsc.h"
#include "src/sles/ksp/kspimpl.h"

#undef __FUNC__  
#define __FUNC__ "KSPSetUp_BCGS"
static int KSPSetUp_BCGS(KSP ksp)
{
  int ierr;

  PetscFunctionBegin;
  if (ksp->pc_side == PC_SYMMETRIC) {
    SETERRQ(PETSC_ERR_SUP,0,"no symmetric preconditioning for KSPBCGS");
  }
  ierr = KSPDefaultGetWork( ksp, 7 );CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "KSPSolve_BCGS"
static int  KSPSolve_BCGS(KSP ksp,int *its)
{
  int       i, maxit, cerr = 0,ierr;
  Scalar    rho, rhoold, alpha, beta, omega, omegaold, d1, d2,zero = 0.0, tmp;
  Vec       X,B,V,P,R,RP,T,S, BINVF;
  double    dp = 0.0;

  PetscFunctionBegin;

  maxit   = ksp->max_it;
  X       = ksp->vec_sol;
  B       = ksp->vec_rhs;
  R       = ksp->work[0];
  RP      = ksp->work[1];
  V       = ksp->work[2];
  T       = ksp->work[3];
  S       = ksp->work[4];
  P       = ksp->work[5];
  BINVF   = ksp->work[6];

  /* Compute initial preconditioned residual */
  ierr = KSPResidual(ksp,X,V,T,R,BINVF,B);CHKERRQ(ierr);

  /* Test for nothing to do */
  if (!ksp->avoidnorms) {
    ierr = VecNorm(R,NORM_2,&dp);CHKERRQ(ierr);
  }
  ierr = PetscObjectTakeAccess(ksp);CHKERRQ(ierr);
  ksp->its   = 0;
  ksp->rnorm = dp;
  ierr = PetscObjectGrantAccess(ksp);CHKERRQ(ierr);
  KSPLogResidualHistory(ksp,dp);
  KSPMonitor(ksp,0,dp);
  if ((*ksp->converged)(ksp,0,dp,ksp->cnvP)) {*its = 0; PetscFunctionReturn(0);}

  /* Make the initial Rp == R */
  ierr = VecCopy(R,RP);CHKERRQ(ierr);

  rhoold   = 1.0;
  alpha    = 1.0;
  omegaold = 1.0;
  ierr = VecSet(&zero,P);CHKERRQ(ierr);
  ierr = VecSet(&zero,V);CHKERRQ(ierr);

  for (i=0; i<maxit; i++) {

    ierr = VecDot(R,RP,&rho);CHKERRQ(ierr);       /*   rho <- (r,rp)      */
    if (rho == 0.0) SETERRQ(PETSC_ERR_KSP_BRKDWN,0,"Breakdown, rho = r . rp = 0");
    beta = (rho/rhoold) * (alpha/omegaold);
    tmp = -omegaold; VecAXPY(&tmp,V,P);            /*   p <- p - w v       */
    ierr = VecAYPX(&beta,R,P);CHKERRQ(ierr);      /*   p <- r + p beta    */
    ierr = PCApplyBAorAB(ksp->B,ksp->pc_side,
                         P,V,T);CHKERRQ(ierr);    /*   v <- K p           */
    ierr = VecDot(V,RP,&d1);CHKERRQ(ierr);
    alpha = rho / d1; tmp = -alpha;                /*   a <- rho / (v,rp)  */
    ierr = VecWAXPY(&tmp,V,R,S);CHKERRQ(ierr);    /*   s <- r - a v       */
    ierr = PCApplyBAorAB(ksp->B,ksp->pc_side,S,T,R);CHKERRQ(ierr);/*   t <- K s    */
    ierr = VecDot(S,T,&d1);CHKERRQ(ierr);
    ierr = VecDot(T,T,&d2);CHKERRQ(ierr);
    if (d2 == 0.0) {
      /* t is 0.  if s is 0, then alpha v == r, and hence alpha p
	 may be our solution.  Give it a try? */
      ierr = VecDot(S,S,&d1);CHKERRQ(ierr);
      if (d1 != 0.0) SETERRQ(PETSC_ERR_KSP_BRKDWN,0,"Breakdown, da = s . s = 0");
      ierr = VecAXPY(&alpha,P,X);CHKERRQ(ierr);   /*   x <- x + a p       */
      ierr = PetscObjectTakeAccess(ksp);CHKERRQ(ierr);
      ksp->its++;
      ksp->rnorm = 0.0;
      ierr = PetscObjectGrantAccess(ksp);CHKERRQ(ierr);
      KSPLogResidualHistory(ksp,dp);
      KSPMonitor(ksp,i+1,0.0);
      break;
    }
    omega = d1 / d2;                               /*   w <- (t's) / (t't) */
    ierr = VecAXPY(&alpha,P,X);CHKERRQ(ierr);     /*   x <- x + a p       */
    ierr = VecAXPY(&omega,S,X);CHKERRQ(ierr);     /*   x <- x + w s       */
    tmp = -omega; 
    ierr = VecWAXPY(&tmp,T,S,R);CHKERRQ(ierr);    /*   r <- s - w t       */
    if (!ksp->avoidnorms) {
      ierr = VecNorm(R,NORM_2,&dp);CHKERRQ(ierr);
    }

    rhoold   = rho;
    omegaold = omega;

    ierr = PetscObjectTakeAccess(ksp);CHKERRQ(ierr);
    ksp->its++;
    ksp->rnorm = dp;
    ierr = PetscObjectGrantAccess(ksp);CHKERRQ(ierr);
    KSPLogResidualHistory(ksp,dp);
    KSPMonitor(ksp,i+1,dp);
    cerr = (*ksp->converged)(ksp,i+1,dp,ksp->cnvP);
    if (cerr) break;    
  }
  if (i == maxit) i--;

  ierr = KSPUnwindPreconditioner(ksp,X,T);CHKERRQ(ierr);
  if (cerr <= 0) *its = -(i+1);
  else           *its = i + 1;
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNC__  
#define __FUNC__ "KSPCreate_BCGS"
int KSPCreate_BCGS(KSP ksp)
{
  PetscFunctionBegin;
  ksp->data                 = (void *) 0;
  ksp->pc_side              = PC_LEFT;
  ksp->calc_res             = 1;
  ksp->ops->setup           = KSPSetUp_BCGS;
  ksp->ops->solve           = KSPSolve_BCGS;
  ksp->ops->destroy         = KSPDefaultDestroy;
  ksp->converged            = KSPDefaultConverged;
  ksp->ops->buildsolution   = KSPDefaultBuildSolution;
  ksp->ops->buildresidual   = KSPDefaultBuildResidual;
  ksp->ops->view            = 0;
  ksp->guess_zero           = 1; 
  PetscFunctionReturn(0);
}
EXTERN_C_END
