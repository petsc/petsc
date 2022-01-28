
#include <petsc/private/kspimpl.h>

typedef struct {
  PetscReal haptol;
} KSP_MINRES;

static PetscErrorCode KSPSetUp_MINRES(KSP ksp)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscAssertFalse(ksp->pc_side == PC_RIGHT,PetscObjectComm((PetscObject)ksp),PETSC_ERR_SUP,"No right preconditioning for KSPMINRES");
  else PetscAssertFalse(ksp->pc_side == PC_SYMMETRIC,PetscObjectComm((PetscObject)ksp),PETSC_ERR_SUP,"No symmetric preconditioning for KSPMINRES");
  ierr = KSPSetWorkVecs(ksp,9);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode  KSPSolve_MINRES(KSP ksp)
{
  PetscErrorCode    ierr;
  PetscInt          i;
  PetscScalar       alpha,beta,ibeta,betaold,eta,c=1.0,ceta,cold=1.0,coold,s=0.0,sold=0.0,soold;
  PetscScalar       rho0,rho1,irho1,rho2,rho3,dp = 0.0;
  const PetscScalar none = -1.0;
  PetscReal         np;
  Vec               X,B,R,Z,U,V,W,UOLD,VOLD,WOLD,WOOLD;
  Mat               Amat,Pmat;
  KSP_MINRES        *minres = (KSP_MINRES*)ksp->data;
  PetscBool         diagonalscale;

  PetscFunctionBegin;
  ierr = PCGetDiagonalScale(ksp->pc,&diagonalscale);CHKERRQ(ierr);
  PetscAssertFalse(diagonalscale,PetscObjectComm((PetscObject)ksp),PETSC_ERR_SUP,"Krylov method %s does not support diagonal scaling",((PetscObject)ksp)->type_name);

  X     = ksp->vec_sol;
  B     = ksp->vec_rhs;
  R     = ksp->work[0];
  Z     = ksp->work[1];
  U     = ksp->work[2];
  V     = ksp->work[3];
  W     = ksp->work[4];
  UOLD  = ksp->work[5];
  VOLD  = ksp->work[6];
  WOLD  = ksp->work[7];
  WOOLD = ksp->work[8];

  ierr = PCGetOperators(ksp->pc,&Amat,&Pmat);CHKERRQ(ierr);

  ksp->its = 0;

  ierr = VecSet(UOLD,0.0);CHKERRQ(ierr);          /*     u_old  <-   0   */
  ierr = VecSet(VOLD,0.0);CHKERRQ(ierr);         /*     v_old  <-   0   */
  ierr = VecSet(W,0.0);CHKERRQ(ierr);            /*     w      <-   0   */
  ierr = VecSet(WOLD,0.0);CHKERRQ(ierr);         /*     w_old  <-   0   */

  if (!ksp->guess_zero) {
    ierr = KSP_MatMult(ksp,Amat,X,R);CHKERRQ(ierr); /*     r <- b - A*x    */
    ierr = VecAYPX(R,-1.0,B);CHKERRQ(ierr);
  } else {
    ierr = VecCopy(B,R);CHKERRQ(ierr);              /*     r <- b (x is 0) */
  }
  ierr = KSP_PCApply(ksp,R,Z);CHKERRQ(ierr);       /*     z  <- B*r       */
  ierr = VecNorm(Z,NORM_2,&np);CHKERRQ(ierr);      /*   np <- ||z||        */
  KSPCheckNorm(ksp,np);
  ierr = VecDot(R,Z,&dp);CHKERRQ(ierr);
  KSPCheckDot(ksp,dp);

  if (PetscRealPart(dp) < minres->haptol && np > minres->haptol) {
    PetscAssertFalse(ksp->errorifnotconverged,PetscObjectComm((PetscObject)ksp),PETSC_ERR_CONV_FAILED,"Detected indefinite operator %g tolerance %g",(double)PetscRealPart(dp),(double)minres->haptol);
    ierr = PetscInfo(ksp,"Detected indefinite operator %g tolerance %g\n",(double)PetscRealPart(dp),(double)minres->haptol);CHKERRQ(ierr);
    ksp->reason = KSP_DIVERGED_INDEFINITE_MAT;
    PetscFunctionReturn(0);
  }

  ksp->rnorm = 0.0;
  if (ksp->normtype != KSP_NORM_NONE) ksp->rnorm = np;
  ierr = KSPLogResidualHistory(ksp,ksp->rnorm);CHKERRQ(ierr);
  ierr = KSPMonitor(ksp,0,ksp->rnorm);CHKERRQ(ierr);
  ierr = (*ksp->converged)(ksp,0,ksp->rnorm,&ksp->reason,ksp->cnvP);CHKERRQ(ierr); /* test for convergence */
  if (ksp->reason) PetscFunctionReturn(0);

  dp   = PetscAbsScalar(dp);
  dp   = PetscSqrtScalar(dp);
  beta = dp;                                        /*  beta <- sqrt(r'*z  */
  eta  = beta;

  ierr  = VecCopy(R,V);CHKERRQ(ierr);
  ierr  = VecCopy(Z,U);CHKERRQ(ierr);
  ibeta = 1.0 / beta;
  ierr  = VecScale(V,ibeta);CHKERRQ(ierr);        /*    v <- r / beta     */
  ierr  = VecScale(U,ibeta);CHKERRQ(ierr);        /*    u <- z / beta     */

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
    KSPCheckDot(ksp,dp);
    dp   = PetscAbsScalar(dp);
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

    ierr  = VecCopy(U,W);CHKERRQ(ierr);           /*  w      <- u          */
    ierr  = VecAXPY(W,-rho2,WOLD);CHKERRQ(ierr); /*  w <- w - rho2 w_old  */
    ierr  = VecAXPY(W,-rho3,WOOLD);CHKERRQ(ierr); /*  w <- w - rho3 w_oold */
    irho1 = 1.0 / rho1;
    ierr  = VecScale(W,irho1);CHKERRQ(ierr);     /*  w <- w / rho1        */

    ceta = c * eta;
    ierr = VecAXPY(X,ceta,W);CHKERRQ(ierr);      /*  x <- x + c eta w     */

    /*
        when dp is really small we have either convergence or an indefinite operator so compute true
        residual norm to check for convergence
    */
    if (PetscRealPart(dp) < minres->haptol) {
      ierr = PetscInfo(ksp,"Possible indefinite operator %g tolerance %g\n",(double)PetscRealPart(dp),(double)minres->haptol);CHKERRQ(ierr);
      ierr = KSP_MatMult(ksp,Amat,X,VOLD);CHKERRQ(ierr);
      ierr = VecAXPY(VOLD,none,B);CHKERRQ(ierr);
      ierr = VecNorm(VOLD,NORM_2,&np);CHKERRQ(ierr);
      KSPCheckNorm(ksp,np);
    } else {
      /* otherwise compute new residual norm via recurrence relation */
      np *= PetscAbsScalar(s);
    }

    if (ksp->normtype != KSP_NORM_NONE) ksp->rnorm = np;
    ierr = KSPLogResidualHistory(ksp,ksp->rnorm);CHKERRQ(ierr);
    ierr = KSPMonitor(ksp,i+1,ksp->rnorm);CHKERRQ(ierr);
    ierr = (*ksp->converged)(ksp,i+1,ksp->rnorm,&ksp->reason,ksp->cnvP);CHKERRQ(ierr); /* test for convergence */
    if (ksp->reason) break;

    if (PetscRealPart(dp) < minres->haptol) {
      PetscAssertFalse(ksp->errorifnotconverged,PetscObjectComm((PetscObject)ksp),PETSC_ERR_CONV_FAILED,"Detected indefinite operator %g tolerance %g",(double)PetscRealPart(dp),(double)minres->haptol);
      ierr = PetscInfo(ksp,"Detected indefinite operator %g tolerance %g\n",(double)PetscRealPart(dp),(double)minres->haptol);CHKERRQ(ierr);
      ksp->reason = KSP_DIVERGED_INDEFINITE_MAT;
      break;
    }

    eta  = -s * eta;
    ierr  = VecCopy(V,VOLD);CHKERRQ(ierr);
    ierr  = VecCopy(U,UOLD);CHKERRQ(ierr);
    ierr  = VecCopy(R,V);CHKERRQ(ierr);
    ierr  = VecCopy(Z,U);CHKERRQ(ierr);
    ibeta = 1.0 / beta;
    ierr  = VecScale(V,ibeta);CHKERRQ(ierr);     /*  v <- r / beta       */
    ierr  = VecScale(U,ibeta);CHKERRQ(ierr);     /*  u <- z / beta       */

    i++;
  } while (i<ksp->max_it);
  if (i >= ksp->max_it) ksp->reason = KSP_DIVERGED_ITS;
  PetscFunctionReturn(0);
}

/*MC
     KSPMINRES - This code implements the MINRES (Minimum Residual) method.

   Options Database Keys:
.   see KSPSolve()

   Level: beginner

   Notes:
    The operator and the preconditioner must be symmetric and the preconditioner must
          be positive definite for this method.
          Supports only left preconditioning.

   Reference: Paige & Saunders, 1975.

   Contributed by: Robert Scheichl: maprs@maths.bath.ac.uk

.seealso: KSPCreate(), KSPSetType(), KSPType (for list of available types), KSP, KSPCG, KSPCR
M*/
PETSC_EXTERN PetscErrorCode KSPCreate_MINRES(KSP ksp)
{
  KSP_MINRES     *minres;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = KSPSetSupportedNorm(ksp,KSP_NORM_PRECONDITIONED,PC_LEFT,3);CHKERRQ(ierr);
  ierr = KSPSetSupportedNorm(ksp,KSP_NORM_NONE,PC_LEFT,1);CHKERRQ(ierr);
  ierr = PetscNewLog(ksp,&minres);CHKERRQ(ierr);

  /* this parameter is arbitrary; but e-50 didn't work for __float128 in one example */
#if defined(PETSC_USE_REAL___FLOAT128)
  minres->haptol = 1.e-100;
#elif defined(PETSC_USE_REAL_SINGLE)
  minres->haptol = 1.e-25;
#else
  minres->haptol = 1.e-50;
#endif
  ksp->data      = (void*)minres;

  /*
       Sets the functions that are associated with this data structure
       (in C++ this is the same as defining virtual functions)
  */
  ksp->ops->setup          = KSPSetUp_MINRES;
  ksp->ops->solve          = KSPSolve_MINRES;
  ksp->ops->destroy        = KSPDestroyDefault;
  ksp->ops->setfromoptions = NULL;
  ksp->ops->buildsolution  = KSPBuildSolutionDefault;
  ksp->ops->buildresidual  = KSPBuildResidualDefault;
  PetscFunctionReturn(0);
}
