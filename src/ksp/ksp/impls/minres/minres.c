
#include <petsc/private/kspimpl.h>

typedef struct {
  PetscReal haptol;
} KSP_MINRES;

static PetscErrorCode KSPSetUp_MINRES(KSP ksp)
{
  PetscFunctionBegin;
  PetscCheckFalse(ksp->pc_side == PC_RIGHT,PetscObjectComm((PetscObject)ksp),PETSC_ERR_SUP,"No right preconditioning for KSPMINRES");
  else PetscCheckFalse(ksp->pc_side == PC_SYMMETRIC,PetscObjectComm((PetscObject)ksp),PETSC_ERR_SUP,"No symmetric preconditioning for KSPMINRES");
  CHKERRQ(KSPSetWorkVecs(ksp,9));
  PetscFunctionReturn(0);
}

static PetscErrorCode  KSPSolve_MINRES(KSP ksp)
{
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
  CHKERRQ(PCGetDiagonalScale(ksp->pc,&diagonalscale));
  PetscCheckFalse(diagonalscale,PetscObjectComm((PetscObject)ksp),PETSC_ERR_SUP,"Krylov method %s does not support diagonal scaling",((PetscObject)ksp)->type_name);

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

  CHKERRQ(PCGetOperators(ksp->pc,&Amat,&Pmat));

  ksp->its = 0;

  CHKERRQ(VecSet(UOLD,0.0));          /*     u_old  <-   0   */
  CHKERRQ(VecSet(VOLD,0.0));         /*     v_old  <-   0   */
  CHKERRQ(VecSet(W,0.0));            /*     w      <-   0   */
  CHKERRQ(VecSet(WOLD,0.0));         /*     w_old  <-   0   */

  if (!ksp->guess_zero) {
    CHKERRQ(KSP_MatMult(ksp,Amat,X,R)); /*     r <- b - A*x    */
    CHKERRQ(VecAYPX(R,-1.0,B));
  } else {
    CHKERRQ(VecCopy(B,R));              /*     r <- b (x is 0) */
  }
  CHKERRQ(KSP_PCApply(ksp,R,Z));       /*     z  <- B*r       */
  CHKERRQ(VecNorm(Z,NORM_2,&np));      /*   np <- ||z||        */
  KSPCheckNorm(ksp,np);
  CHKERRQ(VecDot(R,Z,&dp));
  KSPCheckDot(ksp,dp);

  if (PetscRealPart(dp) < minres->haptol && np > minres->haptol) {
    PetscCheckFalse(ksp->errorifnotconverged,PetscObjectComm((PetscObject)ksp),PETSC_ERR_CONV_FAILED,"Detected indefinite operator %g tolerance %g",(double)PetscRealPart(dp),(double)minres->haptol);
    CHKERRQ(PetscInfo(ksp,"Detected indefinite operator %g tolerance %g\n",(double)PetscRealPart(dp),(double)minres->haptol));
    ksp->reason = KSP_DIVERGED_INDEFINITE_MAT;
    PetscFunctionReturn(0);
  }

  ksp->rnorm = 0.0;
  if (ksp->normtype != KSP_NORM_NONE) ksp->rnorm = np;
  CHKERRQ(KSPLogResidualHistory(ksp,ksp->rnorm));
  CHKERRQ(KSPMonitor(ksp,0,ksp->rnorm));
  CHKERRQ((*ksp->converged)(ksp,0,ksp->rnorm,&ksp->reason,ksp->cnvP)); /* test for convergence */
  if (ksp->reason) PetscFunctionReturn(0);

  dp   = PetscAbsScalar(dp);
  dp   = PetscSqrtScalar(dp);
  beta = dp;                                        /*  beta <- sqrt(r'*z  */
  eta  = beta;

  CHKERRQ(VecCopy(R,V));
  CHKERRQ(VecCopy(Z,U));
  ibeta = 1.0 / beta;
  CHKERRQ(VecScale(V,ibeta));        /*    v <- r / beta     */
  CHKERRQ(VecScale(U,ibeta));        /*    u <- z / beta     */

  i = 0;
  do {
    ksp->its = i+1;

    /*   Lanczos  */

    CHKERRQ(KSP_MatMult(ksp,Amat,U,R));   /*      r <- A*u   */
    CHKERRQ(VecDot(U,R,&alpha));          /*  alpha <- r'*u  */
    CHKERRQ(KSP_PCApply(ksp,R,Z)); /*      z <- B*r   */

    CHKERRQ(VecAXPY(R,-alpha,V));     /*  r <- r - alpha v     */
    CHKERRQ(VecAXPY(Z,-alpha,U));     /*  z <- z - alpha u     */
    CHKERRQ(VecAXPY(R,-beta,VOLD));   /*  r <- r - beta v_old  */
    CHKERRQ(VecAXPY(Z,-beta,UOLD));   /*  z <- z - beta u_old  */

    betaold = beta;

    CHKERRQ(VecDot(R,Z,&dp));
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

    CHKERRQ(VecCopy(WOLD,WOOLD));     /*  w_oold <- w_old      */
    CHKERRQ(VecCopy(W,WOLD));         /*  w_old  <- w          */

    CHKERRQ(VecCopy(U,W));           /*  w      <- u          */
    CHKERRQ(VecAXPY(W,-rho2,WOLD)); /*  w <- w - rho2 w_old  */
    CHKERRQ(VecAXPY(W,-rho3,WOOLD)); /*  w <- w - rho3 w_oold */
    irho1 = 1.0 / rho1;
    CHKERRQ(VecScale(W,irho1));     /*  w <- w / rho1        */

    ceta = c * eta;
    CHKERRQ(VecAXPY(X,ceta,W));      /*  x <- x + c eta w     */

    /*
        when dp is really small we have either convergence or an indefinite operator so compute true
        residual norm to check for convergence
    */
    if (PetscRealPart(dp) < minres->haptol) {
      CHKERRQ(PetscInfo(ksp,"Possible indefinite operator %g tolerance %g\n",(double)PetscRealPart(dp),(double)minres->haptol));
      CHKERRQ(KSP_MatMult(ksp,Amat,X,VOLD));
      CHKERRQ(VecAXPY(VOLD,none,B));
      CHKERRQ(VecNorm(VOLD,NORM_2,&np));
      KSPCheckNorm(ksp,np);
    } else {
      /* otherwise compute new residual norm via recurrence relation */
      np *= PetscAbsScalar(s);
    }

    if (ksp->normtype != KSP_NORM_NONE) ksp->rnorm = np;
    CHKERRQ(KSPLogResidualHistory(ksp,ksp->rnorm));
    CHKERRQ(KSPMonitor(ksp,i+1,ksp->rnorm));
    CHKERRQ((*ksp->converged)(ksp,i+1,ksp->rnorm,&ksp->reason,ksp->cnvP)); /* test for convergence */
    if (ksp->reason) break;

    if (PetscRealPart(dp) < minres->haptol) {
      PetscCheckFalse(ksp->errorifnotconverged,PetscObjectComm((PetscObject)ksp),PETSC_ERR_CONV_FAILED,"Detected indefinite operator %g tolerance %g",(double)PetscRealPart(dp),(double)minres->haptol);
      CHKERRQ(PetscInfo(ksp,"Detected indefinite operator %g tolerance %g\n",(double)PetscRealPart(dp),(double)minres->haptol));
      ksp->reason = KSP_DIVERGED_INDEFINITE_MAT;
      break;
    }

    eta  = -s * eta;
    CHKERRQ(VecCopy(V,VOLD));
    CHKERRQ(VecCopy(U,UOLD));
    CHKERRQ(VecCopy(R,V));
    CHKERRQ(VecCopy(Z,U));
    ibeta = 1.0 / beta;
    CHKERRQ(VecScale(V,ibeta));     /*  v <- r / beta       */
    CHKERRQ(VecScale(U,ibeta));     /*  u <- z / beta       */

    i++;
  } while (i<ksp->max_it);
  if (i >= ksp->max_it) ksp->reason = KSP_DIVERGED_ITS;
  PetscFunctionReturn(0);
}

/*MC
     KSPMINRES - This code implements the MINRES (Minimum Residual) method.

   Options Database Keys:
    see KSPSolve()

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

  PetscFunctionBegin;
  CHKERRQ(KSPSetSupportedNorm(ksp,KSP_NORM_PRECONDITIONED,PC_LEFT,3));
  CHKERRQ(KSPSetSupportedNorm(ksp,KSP_NORM_NONE,PC_LEFT,1));
  CHKERRQ(PetscNewLog(ksp,&minres));

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
