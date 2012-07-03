
#include <petsc-private/kspimpl.h>

typedef struct {
  PetscReal haptol;
} KSP_SYMMLQ;

#undef __FUNCT__  
#define __FUNCT__ "KSPSetUp_SYMMLQ"
PetscErrorCode KSPSetUp_SYMMLQ(KSP ksp)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = KSPDefaultGetWork(ksp,9);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "KSPSolve_SYMMLQ"
PetscErrorCode  KSPSolve_SYMMLQ(KSP ksp)
{
  PetscErrorCode ierr;
  PetscInt       i;
  PetscScalar    alpha,beta,ibeta,betaold,beta1,ceta = 0,ceta_oold = 0.0, ceta_old = 0.0,ceta_bar;
  PetscScalar    c=1.0,cold=1.0,s=0.0,sold=0.0,coold,soold,rho0,rho1,rho2,rho3;
  PetscScalar    dp = 0.0;
  PetscReal      np,s_prod;
  Vec            X,B,R,Z,U,V,W,UOLD,VOLD,Wbar;
  Mat            Amat,Pmat;
  MatStructure   pflag;
  KSP_SYMMLQ     *symmlq = (KSP_SYMMLQ*)ksp->data;
  PetscBool      diagonalscale;

  PetscFunctionBegin;
  ierr    = PCGetDiagonalScale(ksp->pc,&diagonalscale);CHKERRQ(ierr);
  if (diagonalscale) SETERRQ1(((PetscObject)ksp)->comm,PETSC_ERR_SUP,"Krylov method %s does not support diagonal scaling",((PetscObject)ksp)->type_name);

  X       = ksp->vec_sol;
  B       = ksp->vec_rhs;
  R       = ksp->work[0];
  Z       = ksp->work[1];
  U       = ksp->work[2];
  V       = ksp->work[3];
  W       = ksp->work[4];
  UOLD    = ksp->work[5];
  VOLD    = ksp->work[6];
  Wbar    = ksp->work[7];
  
  ierr = PCGetOperators(ksp->pc,&Amat,&Pmat,&pflag);CHKERRQ(ierr);

  ksp->its = 0;

  ierr = VecSet(UOLD,0.0);CHKERRQ(ierr);           /* u_old <- zeros;  */
  ierr = VecCopy(UOLD,VOLD);CHKERRQ(ierr);          /* v_old <- u_old;  */
  ierr = VecCopy(UOLD,W);CHKERRQ(ierr);             /* w     <- u_old;  */ 
  ierr = VecCopy(UOLD,Wbar);CHKERRQ(ierr);          /* w_bar <- u_old;  */
  if (!ksp->guess_zero) {
    ierr = KSP_MatMult(ksp,Amat,X,R);CHKERRQ(ierr); /*     r <- b - A*x */
    ierr = VecAYPX(R,-1.0,B);CHKERRQ(ierr);
  } else { 
    ierr = VecCopy(B,R);CHKERRQ(ierr);              /*     r <- b (x is 0) */
  }

  ierr = KSP_PCApply(ksp,R,Z);CHKERRQ(ierr); /* z  <- B*r       */
  ierr = VecDot(R,Z,&dp);CHKERRQ(ierr);             /* dp = r'*z;      */
  if (PetscAbsScalar(dp) < symmlq->haptol) {
    ierr = PetscInfo2(ksp,"Detected happy breakdown %G tolerance %G\n",PetscAbsScalar(dp),symmlq->haptol);CHKERRQ(ierr);
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

  ierr = VecCopy(R,V);CHKERRQ(ierr);  /* v <- r; */
  ierr = VecCopy(Z,U);CHKERRQ(ierr);  /* u <- z; */
  ibeta = 1.0 / beta;
  ierr = VecScale(V,ibeta);CHKERRQ(ierr);     /* v <- ibeta*v; */
  ierr = VecScale(U,ibeta);CHKERRQ(ierr);     /* u <- ibeta*u; */
  ierr = VecCopy(U,Wbar);CHKERRQ(ierr);        /* w_bar <- u;   */
  ierr = VecNorm(Z,NORM_2,&np);CHKERRQ(ierr);      /*   np <- ||z||        */
  KSPLogResidualHistory(ksp,np);
  ierr = KSPMonitor(ksp,0,np);CHKERRQ(ierr);
  ksp->rnorm = np;  
  ierr = (*ksp->converged)(ksp,0,np,&ksp->reason,ksp->cnvP);CHKERRQ(ierr);  /* test for convergence */
  if (ksp->reason) PetscFunctionReturn(0);

  i = 0; ceta = 0.;
  do {
    ksp->its = i+1;

    /*    Update    */
    if (ksp->its > 1){
      ierr = VecCopy(V,VOLD);CHKERRQ(ierr);  /* v_old <- v; */     
      ierr = VecCopy(U,UOLD);CHKERRQ(ierr);  /* u_old <- u; */
     
      ierr = VecCopy(R,V);CHKERRQ(ierr);
      ierr = VecScale(V,1.0/beta);CHKERRQ(ierr); /* v <- ibeta*r; */
      ierr = VecCopy(Z,U);CHKERRQ(ierr);
      ierr = VecScale(U,1.0/beta);CHKERRQ(ierr); /* u <- ibeta*z; */

      ierr = VecCopy(Wbar,W);CHKERRQ(ierr);
      ierr = VecScale(W,c);CHKERRQ(ierr);
      ierr = VecAXPY(W,s,U);CHKERRQ(ierr);   /* w  <- c*w_bar + s*u;    (w_k) */
      ierr = VecScale(Wbar,-s);CHKERRQ(ierr);
      ierr = VecAXPY(Wbar,c,U);CHKERRQ(ierr); /* w_bar <- -s*w_bar + c*u; (w_bar_(k+1)) */
      ierr = VecAXPY(X,ceta,W);CHKERRQ(ierr); /* x <- x + ceta * w;       (xL_k)  */

      ceta_oold = ceta_old;
      ceta_old  = ceta;
    }

    /*   Lanczos  */
    ierr = KSP_MatMult(ksp,Amat,U,R);CHKERRQ(ierr);   /*  r     <- Amat*u; */  
    ierr = VecDot(U,R,&alpha);CHKERRQ(ierr);          /*  alpha <- u'*r;   */
    ierr = KSP_PCApply(ksp,R,Z);CHKERRQ(ierr); /*      z <- B*r;    */

    ierr = VecAXPY(R,-alpha,V);CHKERRQ(ierr);     /*  r <- r - alpha* v;  */
    ierr = VecAXPY(Z,-alpha,U);CHKERRQ(ierr);     /*  z <- z - alpha* u;  */
    ierr = VecAXPY(R,-beta,VOLD);CHKERRQ(ierr);   /*  r <- r - beta * v_old; */
    ierr = VecAXPY(Z,-beta,UOLD);CHKERRQ(ierr);   /*  z <- z - beta * u_old; */
    betaold = beta;                                /* beta_k                  */
    ierr = VecDot(R,Z,&dp);CHKERRQ(ierr);          /* dp <- r'*z;             */
    if (PetscAbsScalar(dp) < symmlq->haptol) {
      ierr = PetscInfo2(ksp,"Detected happy breakdown %G tolerance %G\n",PetscAbsScalar(dp),symmlq->haptol);CHKERRQ(ierr);
      dp = 0.0;
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
    rho0 = cold * alpha - coold * sold * betaold;    /* gamma_bar */ 
    rho1 = PetscSqrtScalar(rho0*rho0 + beta*beta);   /* gamma     */
    rho2 = sold * alpha + coold * cold * betaold;    /* delta     */
    rho3 = soold * betaold;                          /* epsilon   */

    /* Givens rotation: [c -s; s c] (different from the Reference!) */
    c = rho0 / rho1; s = beta / rho1;

    if (ksp->its==1){
      ceta = beta1/rho1;
    } else {
      ceta = -(rho2*ceta_old + rho3*ceta_oold)/rho1;
    }
          
    s_prod = s_prod*PetscAbsScalar(s);
    if (c == 0.0){
      np = s_prod*1.e16;
    } else {
      np = s_prod/PetscAbsScalar(c);       /* residual norm for xc_k (CGNORM) */
    }
    ksp->rnorm = np;
    KSPLogResidualHistory(ksp,np);
    ierr = KSPMonitor(ksp,i+1,np);CHKERRQ(ierr);
    ierr = (*ksp->converged)(ksp,i+1,np,&ksp->reason,ksp->cnvP);CHKERRQ(ierr); /* test for convergence */
    if (ksp->reason) break;
    i++;
  } while (i<ksp->max_it);

  /* move to the CG point: xc_(k+1) */
  if (c == 0.0){
    ceta_bar = ceta*1.e15;
  } else {
    ceta_bar = ceta/c;
  }
  ierr = VecAXPY(X,ceta_bar,Wbar);CHKERRQ(ierr); /* x <- x + ceta_bar*w_bar */

  if (i >= ksp->max_it) {
    ksp->reason = KSP_DIVERGED_ITS;
  }
  PetscFunctionReturn(0);
}

/*MC
     KSPSYMMLQ -  This code implements the SYMMLQ method. 

   Options Database Keys:
.   see KSPSolve()

   Level: beginner

   Notes: The operator and the preconditioner must be symmetric for this method. The 
          preconditioner must be POSITIVE-DEFINITE.

          Supports only left preconditioning.

   Reference: Paige & Saunders, 1975.

.seealso: KSPCreate(), KSPSetType(), KSPType (for list of available types), KSP
M*/
EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "KSPCreate_SYMMLQ"
PetscErrorCode  KSPCreate_SYMMLQ(KSP ksp)
{
  KSP_SYMMLQ     *symmlq;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = KSPSetSupportedNorm(ksp,KSP_NORM_PRECONDITIONED,PC_LEFT,2);CHKERRQ(ierr);
  ierr           = PetscNewLog(ksp,KSP_SYMMLQ,&symmlq);CHKERRQ(ierr);
  symmlq->haptol = 1.e-18;
  ksp->data      = (void*)symmlq;

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





