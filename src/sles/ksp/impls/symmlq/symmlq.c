/*$Id: symmlq.c,v 1.6 2000/09/08 17:48:28 bsmith Exp bsmith $*/
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
  Scalar       alpha,malpha,beta,mbeta,ibeta,betaold,beta1;
  Scalar       ceta,ceta_oold = 0.0, ceta_old = 0.0,ceta_bar;
  Scalar       c=1.0,cold=1.0,s=0.0,sold=0.0,coold,soold,ms;
  Scalar       rho0,rho1,rho2,rho3;
  Scalar       mone = -1.0,zero = 0.0,dp = 0.0;
  PetscReal    np,s_prod;
  Vec          X,B,R,Z,U,V,W,UOLD,VOLD,Wbar;
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
  Wbar    = ksp->work[7];
  
  ierr = PCGetOperators(ksp->B,&Amat,&Pmat,&pflag);CHKERRQ(ierr);

  ksp->its = 0;

  ierr = VecSet(&zero,UOLD);CHKERRQ(ierr);          /* u_old <- zeros;  */
  ierr = VecCopy(UOLD,VOLD);CHKERRQ(ierr);          /* v_old <- u_old;  */
  ierr = VecCopy(UOLD,W);CHKERRQ(ierr);             /* w     <- u_old;  */ 
  ierr = VecCopy(UOLD,Wbar);CHKERRQ(ierr);          /* w_bar <- u_old;  */
  if (!ksp->guess_zero) {
    ierr = KSP_MatMult(ksp,Amat,X,R);CHKERRQ(ierr); /*     r <- b - A*x */
    ierr = VecAYPX(&mone,B,R);CHKERRQ(ierr);
  } else { 
    ierr = VecCopy(B,R);CHKERRQ(ierr);              /*     r <- b (x is 0) */
  }

  ierr = KSP_PCApply(ksp,ksp->B,R,Z);CHKERRQ(ierr); /* z  <- B*r       */
  ierr = VecDot(R,Z,&dp);CHKERRQ(ierr);             /* dp = r'*z;      */
#if !defined(PETSC_USE_COMPLEX)
  if (dp < 0.0) SETERRQ(PETSC_ERR_KSP_BRKDWN,0,"Indefinite preconditioner");
#endif
  dp = PetscSqrtScalar(dp); 
  beta = dp;                         /*  beta <- sqrt(r'*z)  */
  beta1 = beta;
  s_prod = PetscAbsScalar(beta1); 

  ierr = VecCopy(R,V);CHKERRQ(ierr);  /* v <- r; */
  ierr = VecCopy(Z,U);CHKERRQ(ierr);  /* u <- z; */
  ibeta = 1.0 / beta;
  ierr = VecScale(&ibeta,V);CHKERRQ(ierr);     /* v <- ibeta*v; */
  ierr = VecScale(&ibeta,U);CHKERRQ(ierr);     /* u <- ibeta*u; */
  ierr = VecCopy(U,Wbar);CHKERRQ(ierr);        /* w_bar <- u;   */
  ierr = VecNorm(Z,NORM_2,&np);CHKERRQ(ierr);      /*   np <- ||z||        */
  ierr = (*ksp->converged)(ksp,0,np,&ksp->reason,ksp->cnvP);CHKERRQ(ierr);  /* test for convergence */
  if (ksp->reason) {*its =  0; PetscFunctionReturn(0);}
  KSPLogResidualHistory(ksp,np);
  KSPMonitor(ksp,0,np);            /* call any registered monitor routines */
  ksp->rnorm = np;  

  for (i=0; i<maxit; i++){
    ksp->its = i+1;

    /*    Update    */
    if (ksp->its > 1){
      ierr = VecCopy(V,VOLD);CHKERRQ(ierr);  /* v_old <- v; */     
      ierr = VecCopy(U,UOLD);CHKERRQ(ierr);  /* u_old <- u; */
     
      ibeta = 1.0 / beta;
      ierr = VecCopy(R,V);CHKERRQ(ierr);
      ierr = VecScale(&ibeta,V);CHKERRQ(ierr); /* v <- ibeta*r; */
      ierr = VecCopy(Z,U);CHKERRQ(ierr);
      ierr = VecScale(&ibeta,U);CHKERRQ(ierr); /* u <- ibeta*z; */

      ierr = VecCopy(Wbar,W);CHKERRQ(ierr);
      ierr = VecScale(&c,W);CHKERRQ(ierr);
      ierr = VecAXPY(&s,U,W);CHKERRQ(ierr);   /* w  <- c*w_bar + s*u;    (w_k) */
      ms = -s;
      ierr = VecScale(&ms,Wbar);CHKERRQ(ierr);
      ierr = VecAXPY(&c,U,Wbar);CHKERRQ(ierr); /* w_bar <- -s*w_bar + c*u; (w_bar_(k+1)) */
      ierr = VecAXPY(&ceta,W,X);CHKERRQ(ierr); /* x <- x + ceta * w;       (xL_k)  */

      ceta_oold = ceta_old;
      ceta_old  = ceta;
    }

    /*   Lanczos  */
    ierr = KSP_MatMult(ksp,Amat,U,R);CHKERRQ(ierr);   /*  r     <- Amat*u; */  
    ierr = VecDot(U,R,&alpha);CHKERRQ(ierr);          /*  alpha <- u'*r;   */
    ierr = KSP_PCApply(ksp,ksp->B,R,Z);CHKERRQ(ierr); /*      z <- B*r;    */

    malpha = - alpha;
    ierr = VecAXPY(&malpha,V,R);CHKERRQ(ierr);     /*  r <- r - alpha* v;  */
    ierr = VecAXPY(&malpha,U,Z);CHKERRQ(ierr);     /*  z <- z - alpha* u;  */
    mbeta = - beta;
    ierr = VecAXPY(&mbeta,VOLD,R);CHKERRQ(ierr);   /*  r <- r - beta * v_old; */
    ierr = VecAXPY(&mbeta,UOLD,Z);CHKERRQ(ierr);   /*  z <- z - beta * u_old; */
    betaold = beta;                                /* beta_k                  */
    ierr = VecDot(R,Z,&dp);CHKERRQ(ierr);          /* dp <- r'*z;             */
#if !defined(PETSC_USE_COMPLEX)
     if (dp < 0.0) SETERRQ(PETSC_ERR_KSP_BRKDWN,0,"Indefinite preconditioner");
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
     KSPMonitor(ksp,i+1,np);
     ierr = (*ksp->converged)(ksp,i+1,np,&ksp->reason,ksp->cnvP);CHKERRQ(ierr); /* test for convergence */
     if (ksp->reason) break;
  }

  /* move to the CG point: xc_(k+1) */
  if (c == 0.0){
    ceta_bar = ceta*1.e15;
  } else {
    ceta_bar = ceta/c;
  }
  ierr = VecAXPY(&ceta_bar,Wbar,X);CHKERRQ(ierr); /* x <- x + ceta_bar*w_bar */

  if (i == maxit) {
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





