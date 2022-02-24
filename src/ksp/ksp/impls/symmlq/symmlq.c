
#include <petsc/private/kspimpl.h>

typedef struct {
  PetscReal haptol;
} KSP_SYMMLQ;

PetscErrorCode KSPSetUp_SYMMLQ(KSP ksp)
{
  PetscFunctionBegin;
  CHKERRQ(KSPSetWorkVecs(ksp,9));
  PetscFunctionReturn(0);
}

PetscErrorCode  KSPSolve_SYMMLQ(KSP ksp)
{
  PetscInt       i;
  PetscScalar    alpha,beta,ibeta,betaold,beta1,ceta = 0,ceta_oold = 0.0, ceta_old = 0.0,ceta_bar;
  PetscScalar    c  = 1.0,cold=1.0,s=0.0,sold=0.0,coold,soold,rho0,rho1,rho2,rho3;
  PetscScalar    dp = 0.0;
  PetscReal      np = 0.0,s_prod;
  Vec            X,B,R,Z,U,V,W,UOLD,VOLD,Wbar;
  Mat            Amat,Pmat;
  KSP_SYMMLQ     *symmlq = (KSP_SYMMLQ*)ksp->data;
  PetscBool      diagonalscale;

  PetscFunctionBegin;
  CHKERRQ(PCGetDiagonalScale(ksp->pc,&diagonalscale));
  PetscCheckFalse(diagonalscale,PetscObjectComm((PetscObject)ksp),PETSC_ERR_SUP,"Krylov method %s does not support diagonal scaling",((PetscObject)ksp)->type_name);

  X    = ksp->vec_sol;
  B    = ksp->vec_rhs;
  R    = ksp->work[0];
  Z    = ksp->work[1];
  U    = ksp->work[2];
  V    = ksp->work[3];
  W    = ksp->work[4];
  UOLD = ksp->work[5];
  VOLD = ksp->work[6];
  Wbar = ksp->work[7];

  CHKERRQ(PCGetOperators(ksp->pc,&Amat,&Pmat));

  ksp->its = 0;

  CHKERRQ(VecSet(UOLD,0.0));           /* u_old <- zeros;  */
  CHKERRQ(VecCopy(UOLD,VOLD));          /* v_old <- u_old;  */
  CHKERRQ(VecCopy(UOLD,W));             /* w     <- u_old;  */
  CHKERRQ(VecCopy(UOLD,Wbar));          /* w_bar <- u_old;  */
  if (!ksp->guess_zero) {
    CHKERRQ(KSP_MatMult(ksp,Amat,X,R)); /*     r <- b - A*x */
    CHKERRQ(VecAYPX(R,-1.0,B));
  } else {
    CHKERRQ(VecCopy(B,R));              /*     r <- b (x is 0) */
  }

  CHKERRQ(KSP_PCApply(ksp,R,Z)); /* z  <- B*r       */
  CHKERRQ(VecDot(R,Z,&dp));             /* dp = r'*z;      */
  KSPCheckDot(ksp,dp);
  if (PetscAbsScalar(dp) < symmlq->haptol) {
    CHKERRQ(PetscInfo(ksp,"Detected happy breakdown %g tolerance %g\n",(double)PetscAbsScalar(dp),(double)symmlq->haptol));
    ksp->rnorm  = 0.0;  /* what should we really put here? */
    ksp->reason = KSP_CONVERGED_HAPPY_BREAKDOWN;  /* bugfix proposed by Lourens (lourens.vanzanen@shell.com) */
    PetscFunctionReturn(0);
  }

#if !defined(PETSC_USE_COMPLEX)
  if (dp < 0.0) {
    ksp->reason = KSP_DIVERGED_INDEFINITE_PC;
    PetscFunctionReturn(0);
  }
#endif
  dp     = PetscSqrtScalar(dp);
  beta   = dp;                         /*  beta <- sqrt(r'*z)  */
  beta1  = beta;
  s_prod = PetscAbsScalar(beta1);

  CHKERRQ(VecCopy(R,V)); /* v <- r; */
  CHKERRQ(VecCopy(Z,U)); /* u <- z; */
  ibeta = 1.0 / beta;
  CHKERRQ(VecScale(V,ibeta));    /* v <- ibeta*v; */
  CHKERRQ(VecScale(U,ibeta));    /* u <- ibeta*u; */
  CHKERRQ(VecCopy(U,Wbar));       /* w_bar <- u;   */
  if (ksp->normtype != KSP_NORM_NONE) {
    CHKERRQ(VecNorm(Z,NORM_2,&np));     /*   np <- ||z||        */
    KSPCheckNorm(ksp,np);
  }
  CHKERRQ(KSPLogResidualHistory(ksp,np));
  CHKERRQ(KSPMonitor(ksp,0,np));
  ksp->rnorm = np;
  CHKERRQ((*ksp->converged)(ksp,0,np,&ksp->reason,ksp->cnvP)); /* test for convergence */
  if (ksp->reason) PetscFunctionReturn(0);

  i = 0; ceta = 0.;
  do {
    ksp->its = i+1;

    /*    Update    */
    if (ksp->its > 1) {
      CHKERRQ(VecCopy(V,VOLD));  /* v_old <- v; */
      CHKERRQ(VecCopy(U,UOLD));  /* u_old <- u; */

      CHKERRQ(VecCopy(R,V));
      CHKERRQ(VecScale(V,1.0/beta)); /* v <- ibeta*r; */
      CHKERRQ(VecCopy(Z,U));
      CHKERRQ(VecScale(U,1.0/beta)); /* u <- ibeta*z; */

      CHKERRQ(VecCopy(Wbar,W));
      CHKERRQ(VecScale(W,c));
      CHKERRQ(VecAXPY(W,s,U));   /* w  <- c*w_bar + s*u;    (w_k) */
      CHKERRQ(VecScale(Wbar,-s));
      CHKERRQ(VecAXPY(Wbar,c,U)); /* w_bar <- -s*w_bar + c*u; (w_bar_(k+1)) */
      CHKERRQ(VecAXPY(X,ceta,W)); /* x <- x + ceta * w;       (xL_k)  */

      ceta_oold = ceta_old;
      ceta_old  = ceta;
    }

    /*   Lanczos  */
    CHKERRQ(KSP_MatMult(ksp,Amat,U,R));   /*  r     <- Amat*u; */
    CHKERRQ(VecDot(U,R,&alpha));          /*  alpha <- u'*r;   */
    CHKERRQ(KSP_PCApply(ksp,R,Z)); /*      z <- B*r;    */

    CHKERRQ(VecAXPY(R,-alpha,V));   /*  r <- r - alpha* v;  */
    CHKERRQ(VecAXPY(Z,-alpha,U));   /*  z <- z - alpha* u;  */
    CHKERRQ(VecAXPY(R,-beta,VOLD)); /*  r <- r - beta * v_old; */
    CHKERRQ(VecAXPY(Z,-beta,UOLD)); /*  z <- z - beta * u_old; */
    betaold = beta;                                /* beta_k                  */
    CHKERRQ(VecDot(R,Z,&dp));       /* dp <- r'*z;             */
    KSPCheckDot(ksp,dp);
    if (PetscAbsScalar(dp) < symmlq->haptol) {
      CHKERRQ(PetscInfo(ksp,"Detected happy breakdown %g tolerance %g\n",(double)PetscAbsScalar(dp),(double)symmlq->haptol));
      dp   = 0.0;
    }

#if !defined(PETSC_USE_COMPLEX)
    if (dp < 0.0) {
      ksp->reason = KSP_DIVERGED_INDEFINITE_PC;
      break;
    }
#endif
    beta = PetscSqrtScalar(dp);                    /*  beta = sqrt(dp); */

    /*    QR factorization    */
    coold = cold; cold = c; soold = sold; sold = s;
    rho0  = cold * alpha - coold * sold * betaold;   /* gamma_bar */
    rho1  = PetscSqrtScalar(rho0*rho0 + beta*beta);  /* gamma     */
    rho2  = sold * alpha + coold * cold * betaold;   /* delta     */
    rho3  = soold * betaold;                         /* epsilon   */

    /* Givens rotation: [c -s; s c] (different from the Reference!) */
    c = rho0 / rho1; s = beta / rho1;

    if (ksp->its==1) ceta = beta1/rho1;
    else ceta = -(rho2*ceta_old + rho3*ceta_oold)/rho1;

    s_prod = s_prod*PetscAbsScalar(s);
    if (c == 0.0) np = s_prod*1.e16;
    else np = s_prod/PetscAbsScalar(c);       /* residual norm for xc_k (CGNORM) */

    if (ksp->normtype != KSP_NORM_NONE) ksp->rnorm = np;
    else ksp->rnorm = 0.0;
    CHKERRQ(KSPLogResidualHistory(ksp,ksp->rnorm));
    CHKERRQ(KSPMonitor(ksp,i+1,ksp->rnorm));
    CHKERRQ((*ksp->converged)(ksp,i+1,ksp->rnorm,&ksp->reason,ksp->cnvP)); /* test for convergence */
    if (ksp->reason) break;
    i++;
  } while (i<ksp->max_it);

  /* move to the CG point: xc_(k+1) */
  if (c == 0.0) ceta_bar = ceta*1.e15;
  else ceta_bar = ceta/c;

  CHKERRQ(VecAXPY(X,ceta_bar,Wbar)); /* x <- x + ceta_bar*w_bar */

  if (i >= ksp->max_it) ksp->reason = KSP_DIVERGED_ITS;
  PetscFunctionReturn(0);
}

/*MC
     KSPSYMMLQ -  This code implements the SYMMLQ method.

   Options Database Keys:
    see KSPSolve()

   Level: beginner

   Notes:
    The operator and the preconditioner must be symmetric for this method. The
          preconditioner must be POSITIVE-DEFINITE.

          Supports only left preconditioning.

   Reference: Paige & Saunders, 1975.

.seealso: KSPCreate(), KSPSetType(), KSPType (for list of available types), KSP
M*/
PETSC_EXTERN PetscErrorCode KSPCreate_SYMMLQ(KSP ksp)
{
  KSP_SYMMLQ     *symmlq;

  PetscFunctionBegin;
  CHKERRQ(KSPSetSupportedNorm(ksp,KSP_NORM_PRECONDITIONED,PC_LEFT,3));
  CHKERRQ(KSPSetSupportedNorm(ksp,KSP_NORM_NONE,PC_LEFT,1));

  CHKERRQ(PetscNewLog(ksp,&symmlq));
  symmlq->haptol = 1.e-18;
  ksp->data      = (void*)symmlq;

  /*
       Sets the functions that are associated with this data structure
       (in C++ this is the same as defining virtual functions)
  */
  ksp->ops->setup          = KSPSetUp_SYMMLQ;
  ksp->ops->solve          = KSPSolve_SYMMLQ;
  ksp->ops->destroy        = KSPDestroyDefault;
  ksp->ops->setfromoptions = NULL;
  ksp->ops->buildsolution  = KSPBuildSolutionDefault;
  ksp->ops->buildresidual  = KSPBuildResidualDefault;
  PetscFunctionReturn(0);
}
