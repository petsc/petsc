/*$Id: symmlq.c,v 1.5 2000/05/04 16:28:54 bsmith Exp $*/
/*                       
    This code implements the SYMMLQ method. 
    Reference: Paige & Saunders, 1975.
*/
#include "src/sles/ksp/kspimpl.h"

#undef __FUNC__  
#define __FUNC__ "KSPSetUp_SYMMLQ"
int KSPSetUp_SYMMLQ(KSP ksp)
{
  int ierr;

  PetscFunctionBegin;

  if (ksp->pc_side == PC_RIGHT) {
    SETERRQ(2,0,"No right preconditioning for KSPSYMMLQ");
  } else if (ksp->pc_side == PC_SYMMETRIC) {
    SETERRQ(2,0,"No symmetric preconditioning for KSPSYMMLQ");
  }

  ierr = KSPDefaultGetWork(ksp,9);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}


#undef __FUNC__  
#define __FUNC__ "KSPSolve_SYMMLQ"
int  KSPSolve_SYMMLQ(KSP ksp,int *its)
{
  int          ierr,i,maxit;
  Scalar       alpha,malpha,beta,mbeta,ibeta,betaold,eta;
  Scalar       c=1.0,ceta,cold=1.0,coold,s=0.0,sold=0.0,soold;
  Scalar       rho0,rho1,irho1,rho2,mrho2,rho3,mrho3;
  Scalar       mone = -1.0,zero = 0.0; 
  Scalar       dp = 0.0;
  PetscReal    np;
  Vec          X,B,R,Z,U,V,W,UOLD,VOLD,WOLD,WOOLD;
  Mat          Amat,Pmat;
  MatStructure pflag;

  PetscFunctionBegin;
  maxit   = ksp->max_it;
  X       = ksp->vec_sol;
  B       = ksp->vec_rhs;
  R       = ksp->work[0];
  Z       = ksp->work[1];
  U       = ksp->work[2];
  V       = ksp->work[3];
  W       = ksp->work[4];
  UOLD    = ksp->work[5];
  VOLD    = ksp->work[6];
  WOLD    = ksp->work[7];
  WOOLD   = ksp->work[8];

  ierr = PCGetOperators(ksp->B,&Amat,&Pmat,&pflag);CHKERRQ(ierr);

  ksp->its = 0;

  ierr = VecSet(&zero,UOLD);CHKERRQ(ierr);         /*     u_old  <-   0   */
  ierr = VecCopy(UOLD,VOLD);CHKERRQ(ierr);         /*     v_old  <-   0   */
  ierr = VecCopy(UOLD,W);CHKERRQ(ierr);            /*     w      <-   0   */
  ierr = VecCopy(UOLD,WOLD);CHKERRQ(ierr);         /*     w_old  <-   0   */

  if (!ksp->guess_zero) {
    ierr = KSP_MatMult(ksp,Amat,X,R);CHKERRQ(ierr); /*     r <- b - A*x    */
    ierr = VecAYPX(&mone,B,R);CHKERRQ(ierr);
  } else { 
    ierr = VecCopy(B,R);CHKERRQ(ierr);              /*     r <- b (x is 0) */
  }

  ierr = KSP_PCApply(ksp,ksp->B,R,Z);CHKERRQ(ierr); /*     z  <- B*r       */

  ierr = VecDot(R,Z,&dp);CHKERRQ(ierr);
#if !defined(PETSC_USE_COMPLEX)
  if (dp < 0.0) SETERRQ(PETSC_ERR_KSP_BRKDWN,0,"Indefinite preconditioner");
#endif
  dp = PetscSqrtScalar(dp); 
  beta = dp;                                        /*  beta <- sqrt(r'*z  */
  eta  = beta;

  ierr = VecCopy(R,V);CHKERRQ(ierr);
  ierr = VecCopy(Z,U);CHKERRQ(ierr);
  ibeta = 1.0 / beta;
  ierr = VecScale(&ibeta,V);CHKERRQ(ierr);         /*    v <- r / beta     */
  ierr = VecScale(&ibeta,U);CHKERRQ(ierr);         /*    u <- z / beta     */

  ierr = VecNorm(Z,NORM_2,&np);CHKERRQ(ierr);      /*   np <- ||z||        */

  ierr = (*ksp->converged)(ksp,0,np,&ksp->reason,ksp->cnvP);CHKERRQ(ierr);  /* test for convergence */
  if (ksp->reason) {*its =  0; PetscFunctionReturn(0);}
  KSPLogResidualHistory(ksp,np);
  KSPMonitor(ksp,0,np);            /* call any registered monitor routines */
  ksp->rnorm = np;  

  for (i=0; i<maxit; i++) {
     ksp->its = i+1;

/*   Lanczos  */

     ierr = KSP_MatMult(ksp,Amat,U,R);CHKERRQ(ierr);   /*      r <- A*u   */
     ierr = VecDot(U,R,&alpha);CHKERRQ(ierr);          /*  alpha <- r'*u  */
     ierr = KSP_PCApply(ksp,ksp->B,R,Z);CHKERRQ(ierr); /*      z <- B*r   */

     malpha = - alpha;
     ierr = VecAXPY(&malpha,V,R);CHKERRQ(ierr);     /*  r <- r - alpha v     */
     ierr = VecAXPY(&malpha,U,Z);CHKERRQ(ierr);     /*  z <- z - alpha u     */
     mbeta = - beta;
     ierr = VecAXPY(&mbeta,VOLD,R);CHKERRQ(ierr);   /*  r <- r - beta v_old  */
     ierr = VecAXPY(&mbeta,UOLD,Z);CHKERRQ(ierr);   /*  z <- z - beta u_old  */

     betaold = beta;

     ierr = VecDot(R,Z,&dp);CHKERRQ(ierr); 
#if !defined(PETSC_USE_COMPLEX)
     if (dp < 0.0) SETERRQ(PETSC_ERR_KSP_BRKDWN,0,"Indefinite preconditioner");
#endif
     beta = PetscSqrtScalar(dp);                               /*  beta <- sqrt(r'*z)   */

/*    QR factorisation    */

     coold = cold; cold = c; soold = sold; sold = s;

     rho0 = cold * alpha - coold * sold * betaold;
     rho1 = PetscSqrtScalar(rho0*rho0 + beta*beta);
     rho2 = sold * alpha + coold * cold * betaold;
     rho3 = soold * betaold;

/*     Givens rotation    */

     c = rho0 / rho1;
     s = beta / rho1;

/*    Update    */

     ierr = VecCopy(WOLD,WOOLD);CHKERRQ(ierr);     /*  w_oold <- w_old      */
     ierr = VecCopy(W,WOLD);CHKERRQ(ierr);         /*  w_old  <- w          */
     
     ierr = VecCopy(U,W);CHKERRQ(ierr);            /*  w      <- u          */
     mrho2 = - rho2;
     ierr = VecAXPY(&mrho2,WOLD,W);CHKERRQ(ierr);  /*  w <- w - rho2 w_old  */
     mrho3 = - rho3;
     ierr = VecAXPY(&mrho3,WOOLD,W);CHKERRQ(ierr); /*  w <- w - rho3 w_oold */
     irho1 = 1.0 / rho1;
     ierr = VecScale(&irho1,W);CHKERRQ(ierr);      /*  w <- w / rho1        */

     ceta = c * eta;
     ierr = VecAXPY(&ceta,W,X);CHKERRQ(ierr);      /*  x <- x + c eta w     */ 
     eta = - s * eta;

     ierr = VecCopy(V,VOLD);CHKERRQ(ierr);
     ierr = VecCopy(U,UOLD);CHKERRQ(ierr);
     ierr = VecCopy(R,V);CHKERRQ(ierr);
     ierr = VecCopy(Z,U);CHKERRQ(ierr);
     ibeta = 1.0 / beta;
     ierr = VecScale(&ibeta,V);CHKERRQ(ierr);      /*  v <- r / beta       */
     ierr = VecScale(&ibeta,U);CHKERRQ(ierr);      /*  u <- z / beta       */
     
     np = ksp->rnorm * PetscAbsScalar(s);

     ksp->rnorm = np;
     KSPLogResidualHistory(ksp,np);
     KSPMonitor(ksp,i+1,np);
     ierr = (*ksp->converged)(ksp,i+1,np,&ksp->reason,ksp->cnvP);CHKERRQ(ierr); /* test for convergence */
     if (ksp->reason) break;
  }
  if (i == maxit) {
    ksp->its--;
    ksp->reason = KSP_DIVERGED_ITS;
  }
  *its = ksp->its;
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNC__  
#define __FUNC__ "KSPCreate_SYMMLQ"
int KSPCreate_SYMMLQ(KSP ksp)
{
  PetscFunctionBegin;

  ksp->pc_side                   = PC_LEFT;
  ksp->calc_res                  = PETSC_TRUE;

  /*
       Sets the functions that are associated with this data structure 
       (in C++ this is the same as defining virtual functions)
  */
  ksp->ops->setup                = KSPSetUp_SYMMLQ;
  ksp->ops->solve                = KSPSolve_SYMMLQ;
  ksp->ops->destroy              = KSPDefaultDestroy;
  ksp->ops->setfromoptions       = 0;
  ksp->ops->buildsolution        = KSPDefaultBuildSolution;
  ksp->ops->buildresidual        = KSPDefaultBuildResidual;

  PetscFunctionReturn(0);
}
EXTERN_C_END





