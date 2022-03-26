
#include <petsc/private/kspimpl.h>

typedef struct {
  PetscReal haptol;
} KSP_MINRES;

static PetscErrorCode KSPSetUp_MINRES(KSP ksp)
{
  PetscFunctionBegin;
  PetscCheckFalse(ksp->pc_side == PC_RIGHT,PetscObjectComm((PetscObject)ksp),PETSC_ERR_SUP,"No right preconditioning for KSPMINRES");
  else PetscCheckFalse(ksp->pc_side == PC_SYMMETRIC,PetscObjectComm((PetscObject)ksp),PETSC_ERR_SUP,"No symmetric preconditioning for KSPMINRES");
  PetscCall(KSPSetWorkVecs(ksp,9));
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
  PetscCall(PCGetDiagonalScale(ksp->pc,&diagonalscale));
  PetscCheck(!diagonalscale,PetscObjectComm((PetscObject)ksp),PETSC_ERR_SUP,"Krylov method %s does not support diagonal scaling",((PetscObject)ksp)->type_name);

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

  PetscCall(PCGetOperators(ksp->pc,&Amat,&Pmat));

  ksp->its = 0;

  PetscCall(VecSet(UOLD,0.0));          /*     u_old  <-   0   */
  PetscCall(VecSet(VOLD,0.0));         /*     v_old  <-   0   */
  PetscCall(VecSet(W,0.0));            /*     w      <-   0   */
  PetscCall(VecSet(WOLD,0.0));         /*     w_old  <-   0   */

  if (!ksp->guess_zero) {
    PetscCall(KSP_MatMult(ksp,Amat,X,R)); /*     r <- b - A*x    */
    PetscCall(VecAYPX(R,-1.0,B));
  } else {
    PetscCall(VecCopy(B,R));              /*     r <- b (x is 0) */
  }
  PetscCall(KSP_PCApply(ksp,R,Z));       /*     z  <- B*r       */
  PetscCall(VecNorm(Z,NORM_2,&np));      /*   np <- ||z||        */
  KSPCheckNorm(ksp,np);
  PetscCall(VecDot(R,Z,&dp));
  KSPCheckDot(ksp,dp);

  if (PetscRealPart(dp) < minres->haptol && np > minres->haptol) {
    PetscCheck(!ksp->errorifnotconverged,PetscObjectComm((PetscObject)ksp),PETSC_ERR_CONV_FAILED,"Detected indefinite operator %g tolerance %g",(double)PetscRealPart(dp),(double)minres->haptol);
    PetscCall(PetscInfo(ksp,"Detected indefinite operator %g tolerance %g\n",(double)PetscRealPart(dp),(double)minres->haptol));
    ksp->reason = KSP_DIVERGED_INDEFINITE_MAT;
    PetscFunctionReturn(0);
  }

  ksp->rnorm = 0.0;
  if (ksp->normtype != KSP_NORM_NONE) ksp->rnorm = np;
  PetscCall(KSPLogResidualHistory(ksp,ksp->rnorm));
  PetscCall(KSPMonitor(ksp,0,ksp->rnorm));
  PetscCall((*ksp->converged)(ksp,0,ksp->rnorm,&ksp->reason,ksp->cnvP)); /* test for convergence */
  if (ksp->reason) PetscFunctionReturn(0);

  dp   = PetscAbsScalar(dp);
  dp   = PetscSqrtScalar(dp);
  beta = dp;                                        /*  beta <- sqrt(r'*z  */
  eta  = beta;

  PetscCall(VecCopy(R,V));
  PetscCall(VecCopy(Z,U));
  ibeta = 1.0 / beta;
  PetscCall(VecScale(V,ibeta));        /*    v <- r / beta     */
  PetscCall(VecScale(U,ibeta));        /*    u <- z / beta     */

  i = 0;
  do {
    ksp->its = i+1;

    /*   Lanczos  */

    PetscCall(KSP_MatMult(ksp,Amat,U,R));   /*      r <- A*u   */
    PetscCall(VecDot(U,R,&alpha));          /*  alpha <- r'*u  */
    PetscCall(KSP_PCApply(ksp,R,Z)); /*      z <- B*r   */

    PetscCall(VecAXPY(R,-alpha,V));     /*  r <- r - alpha v     */
    PetscCall(VecAXPY(Z,-alpha,U));     /*  z <- z - alpha u     */
    PetscCall(VecAXPY(R,-beta,VOLD));   /*  r <- r - beta v_old  */
    PetscCall(VecAXPY(Z,-beta,UOLD));   /*  z <- z - beta u_old  */

    betaold = beta;

    PetscCall(VecDot(R,Z,&dp));
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

    PetscCall(VecCopy(WOLD,WOOLD));     /*  w_oold <- w_old      */
    PetscCall(VecCopy(W,WOLD));         /*  w_old  <- w          */

    PetscCall(VecCopy(U,W));           /*  w      <- u          */
    PetscCall(VecAXPY(W,-rho2,WOLD)); /*  w <- w - rho2 w_old  */
    PetscCall(VecAXPY(W,-rho3,WOOLD)); /*  w <- w - rho3 w_oold */
    irho1 = 1.0 / rho1;
    PetscCall(VecScale(W,irho1));     /*  w <- w / rho1        */

    ceta = c * eta;
    PetscCall(VecAXPY(X,ceta,W));      /*  x <- x + c eta w     */

    /*
        when dp is really small we have either convergence or an indefinite operator so compute true
        residual norm to check for convergence
    */
    if (PetscRealPart(dp) < minres->haptol) {
      PetscCall(PetscInfo(ksp,"Possible indefinite operator %g tolerance %g\n",(double)PetscRealPart(dp),(double)minres->haptol));
      PetscCall(KSP_MatMult(ksp,Amat,X,VOLD));
      PetscCall(VecAXPY(VOLD,none,B));
      PetscCall(VecNorm(VOLD,NORM_2,&np));
      KSPCheckNorm(ksp,np);
    } else {
      /* otherwise compute new residual norm via recurrence relation */
      np *= PetscAbsScalar(s);
    }

    if (ksp->normtype != KSP_NORM_NONE) ksp->rnorm = np;
    PetscCall(KSPLogResidualHistory(ksp,ksp->rnorm));
    PetscCall(KSPMonitor(ksp,i+1,ksp->rnorm));
    PetscCall((*ksp->converged)(ksp,i+1,ksp->rnorm,&ksp->reason,ksp->cnvP)); /* test for convergence */
    if (ksp->reason) break;

    if (PetscRealPart(dp) < minres->haptol) {
      PetscCheck(!ksp->errorifnotconverged,PetscObjectComm((PetscObject)ksp),PETSC_ERR_CONV_FAILED,"Detected indefinite operator %g tolerance %g",(double)PetscRealPart(dp),(double)minres->haptol);
      PetscCall(PetscInfo(ksp,"Detected indefinite operator %g tolerance %g\n",(double)PetscRealPart(dp),(double)minres->haptol));
      ksp->reason = KSP_DIVERGED_INDEFINITE_MAT;
      break;
    }

    eta  = -s * eta;
    PetscCall(VecCopy(V,VOLD));
    PetscCall(VecCopy(U,UOLD));
    PetscCall(VecCopy(R,V));
    PetscCall(VecCopy(Z,U));
    ibeta = 1.0 / beta;
    PetscCall(VecScale(V,ibeta));     /*  v <- r / beta       */
    PetscCall(VecScale(U,ibeta));     /*  u <- z / beta       */

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
  PetscCall(KSPSetSupportedNorm(ksp,KSP_NORM_PRECONDITIONED,PC_LEFT,3));
  PetscCall(KSPSetSupportedNorm(ksp,KSP_NORM_NONE,PC_LEFT,1));
  PetscCall(PetscNewLog(ksp,&minres));

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
