#define PETSCKSP_DLL

#include "private/kspimpl.h"

typedef struct {
  PetscReal haptol;
} KSP_MINRES;

#undef __FUNCT__  
#define __FUNCT__ "KSPSetUp_MINRES"
PetscErrorCode KSPSetUp_MINRES(KSP ksp)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (ksp->pc_side == PC_RIGHT) {
    SETERRQ(PETSC_ERR_SUP,"No right preconditioning for KSPMINRES");
  } else if (ksp->pc_side == PC_SYMMETRIC) {
    SETERRQ(PETSC_ERR_SUP,"No symmetric preconditioning for KSPMINRES");
  }
  ierr = KSPDefaultGetWork(ksp,9);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "KSPSolve_MINRES"
PetscErrorCode  KSPSolve_MINRES(KSP ksp)
{
  PetscErrorCode ierr;
  PetscInt       i;
  PetscScalar    alpha,beta,ibeta,betaold,eta,c=1.0,ceta,cold=1.0,coold,s=0.0,sold=0.0,soold;
  PetscScalar    rho0,rho1,irho1,rho2,mrho2,rho3,mrho3,dp = 0.0;
  PetscReal      np;
  Vec            X,B,R,Z,U,V,W,UOLD,VOLD,WOLD,WOOLD;
  Mat            Amat,Pmat;
  MatStructure   pflag;
  KSP_MINRES     *minres = (KSP_MINRES*)ksp->data;
  PetscTruth     diagonalscale;

  PetscFunctionBegin;
  if (ksp->normtype != KSP_NORM_PRECONDITIONED) SETERRQ(PETSC_ERR_SUP,"Only supports preconditioned residual norm for KSPMINRES");

  ierr    = PCDiagonalScale(ksp->pc,&diagonalscale);CHKERRQ(ierr);
  if (diagonalscale) SETERRQ1(PETSC_ERR_SUP,"Krylov method %s does not support diagonal scaling",((PetscObject)ksp)->type_name);

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

  ierr = PCGetOperators(ksp->pc,&Amat,&Pmat,&pflag);CHKERRQ(ierr);

  ksp->its = 0;

  ierr = VecSet(UOLD,0.0);CHKERRQ(ierr);          /*     u_old  <-   0   */
  ierr = VecCopy(UOLD,VOLD);CHKERRQ(ierr);         /*     v_old  <-   0   */
  ierr = VecCopy(UOLD,W);CHKERRQ(ierr);            /*     w      <-   0   */
  ierr = VecCopy(UOLD,WOLD);CHKERRQ(ierr);         /*     w_old  <-   0   */

  if (!ksp->guess_zero) {
    ierr = KSP_MatMult(ksp,Amat,X,R);CHKERRQ(ierr); /*     r <- b - A*x    */
    ierr = VecAYPX(R,-1.0,B);CHKERRQ(ierr);
  } else { 
    ierr = VecCopy(B,R);CHKERRQ(ierr);              /*     r <- b (x is 0) */
  }

  ierr = KSP_PCApply(ksp,R,Z);CHKERRQ(ierr); /*     z  <- B*r       */

  ierr = VecDot(R,Z,&dp);CHKERRQ(ierr);
  if (PetscAbsScalar(dp) < minres->haptol) {
    ierr = PetscInfo2(ksp,"Detected happy breakdown %G tolerance %G\n",PetscAbsScalar(dp),minres->haptol);CHKERRQ(ierr);
    dp = PetscAbsScalar(dp); /* tiny number, can't use 0.0, cause divided by below */
    if (dp == 0.0) {
      ksp->reason = KSP_CONVERGED_ATOL;
      PetscFunctionReturn(0);
    }
  }

#if !defined(PETSC_USE_COMPLEX)
  if (dp < 0.0) {
    ksp->reason = KSP_DIVERGED_INDEFINITE_PC;
    PetscFunctionReturn(0);
  }
#endif
  dp   = PetscSqrtScalar(dp); 
  beta = dp;                                        /*  beta <- sqrt(r'*z  */
  eta  = beta;

  ierr = VecCopy(R,V);CHKERRQ(ierr);
  ierr = VecCopy(Z,U);CHKERRQ(ierr);
  ibeta = 1.0 / beta;
  ierr = VecScale(V,ibeta);CHKERRQ(ierr);         /*    v <- r / beta     */
  ierr = VecScale(U,ibeta);CHKERRQ(ierr);         /*    u <- z / beta     */

  ierr = VecNorm(Z,NORM_2,&np);CHKERRQ(ierr);      /*   np <- ||z||        */

  KSPLogResidualHistory(ksp,np);
  KSPMonitor(ksp,0,np);            /* call any registered monitor routines */
  ksp->rnorm = np;  
  ierr = (*ksp->converged)(ksp,0,np,&ksp->reason,ksp->cnvP);CHKERRQ(ierr);  /* test for convergence */
  if (ksp->reason) PetscFunctionReturn(0);

  i = 0;
  do {
     ksp->its = i+1;

/*   Lanczos  */

     ierr = KSP_MatMult(ksp,Amat,U,R);CHKERRQ(ierr);   /*      r <- A*u   */
     ierr = VecDot(U,R,&alpha);CHKERRQ(ierr);          /*  alpha <- r'*u  */
     ierr = KSP_PCApply(ksp,R,Z);CHKERRQ(ierr); /*      z <- B*r   */

     ierr = VecAXPY(R,-alpha,V);CHKERRQ(ierr);     /*  r <- r - alpha v     */
     ierr = VecAXPY(Z,-alpha,U);CHKERRQ(ierr);     /*  z <- z - alpha u     */
     ierr = VecAXPY(R,-beta,VOLD);CHKERRQ(ierr);   /*  r <- r - beta v_old  */
     ierr = VecAXPY(Z,-beta,UOLD);CHKERRQ(ierr);   /*  z <- z - beta u_old  */

     betaold = beta;

     ierr = VecDot(R,Z,&dp);CHKERRQ(ierr); 
     if (PetscAbsScalar(dp) < minres->haptol) {
       ierr = PetscInfo2(ksp,"Detected happy breakdown %G tolerance %G\n",PetscAbsScalar(dp),minres->haptol);CHKERRQ(ierr);
       dp = PetscAbsScalar(dp); /* tiny number, can we use 0.0? */
     }

#if !defined(PETSC_USE_COMPLEX)
     if (dp < 0.0) {
       ksp->reason = KSP_DIVERGED_INDEFINITE_PC;
       break;
     }

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
     ierr = VecAXPY(W,mrho2,WOLD);CHKERRQ(ierr);  /*  w <- w - rho2 w_old  */
     mrho3 = - rho3;
     ierr = VecAXPY(W,mrho3,WOOLD);CHKERRQ(ierr); /*  w <- w - rho3 w_oold */
     irho1 = 1.0 / rho1;
     ierr = VecScale(W,irho1);CHKERRQ(ierr);      /*  w <- w / rho1        */

     ceta = c * eta;
     ierr = VecAXPY(X,ceta,W);CHKERRQ(ierr);      /*  x <- x + c eta w     */ 
     eta = - s * eta;

     ierr = VecCopy(V,VOLD);CHKERRQ(ierr);
     ierr = VecCopy(U,UOLD);CHKERRQ(ierr);
     ierr = VecCopy(R,V);CHKERRQ(ierr);
     ierr = VecCopy(Z,U);CHKERRQ(ierr);
     ibeta = 1.0 / beta;
     ierr = VecScale(V,ibeta);CHKERRQ(ierr);      /*  v <- r / beta       */
     ierr = VecScale(U,ibeta);CHKERRQ(ierr);      /*  u <- z / beta       */
     
     np = ksp->rnorm * PetscAbsScalar(s);

     ksp->rnorm = np;
     KSPLogResidualHistory(ksp,np);
     KSPMonitor(ksp,i+1,np);
     ierr = (*ksp->converged)(ksp,i+1,np,&ksp->reason,ksp->cnvP);CHKERRQ(ierr); /* test for convergence */
     if (ksp->reason) break;
     i++;
  } while (i<ksp->max_it);
  if (i >= ksp->max_it) {
    ksp->reason = KSP_DIVERGED_ITS;
  }
  PetscFunctionReturn(0);
}

/*MC
     KSPMINRES - This code implements the MINRES (Minimum Residual) method. 

   Options Database Keys:
.   see KSPSolve()

   Level: beginner

   Notes: The operator and the preconditioner must be symmetric and the preconditioner must
          be positive definite for this method.
          Supports only left preconditioning.

   Reference: Paige & Saunders, 1975.

   Contributed by: Robert Scheichl: maprs@maths.bath.ac.uk

.seealso: KSPCreate(), KSPSetType(), KSPType (for list of available types), KSP, KSPCG, KSPCR
M*/
EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "KSPCreate_MINRES"
PetscErrorCode PETSCKSP_DLLEXPORT KSPCreate_MINRES(KSP ksp)
{
  KSP_MINRES     *minres;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  if (ksp->pc_side != PC_LEFT) {
    ierr = PetscInfo(ksp,"WARNING! Setting PC_SIDE for MINRES to left!\n");CHKERRQ(ierr);
  }
  ksp->pc_side   = PC_LEFT;
  ierr           = PetscNewLog(ksp,KSP_MINRES,&minres);CHKERRQ(ierr);
  minres->haptol = 1.e-18;
  ksp->data      = (void*)minres;

  /*
       Sets the functions that are associated with this data structure 
       (in C++ this is the same as defining virtual functions)
  */
  ksp->ops->setup                = KSPSetUp_MINRES;
  ksp->ops->solve                = KSPSolve_MINRES;
  ksp->ops->destroy              = KSPDefaultDestroy;
  ksp->ops->setfromoptions       = 0;
  ksp->ops->buildsolution        = KSPDefaultBuildSolution;
  ksp->ops->buildresidual        = KSPDefaultBuildResidual;
  PetscFunctionReturn(0);
}
EXTERN_C_END





