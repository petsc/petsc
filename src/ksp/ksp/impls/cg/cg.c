
/*
    This file implements the conjugate gradient method in PETSc as part of
    KSP. You can use this as a starting point for implementing your own
    Krylov method that is not provided with PETSc.

    The following basic routines are required for each Krylov method.
        KSPCreate_XXX()          - Creates the Krylov context
        KSPSetFromOptions_XXX()  - Sets runtime options
        KSPSolve_XXX()           - Runs the Krylov method
        KSPDestroy_XXX()         - Destroys the Krylov context, freeing all
                                   memory it needed
    Here the "_XXX" denotes a particular implementation, in this case
    we use _CG (e.g. KSPCreate_CG, KSPDestroy_CG). These routines
    are actually called via the common user interface routines
    KSPSetType(), KSPSetFromOptions(), KSPSolve(), and KSPDestroy() so the
    application code interface remains identical for all preconditioners.

    Other basic routines for the KSP objects include
        KSPSetUp_XXX()
        KSPView_XXX()            - Prints details of solver being used.

    Detailed Notes:
    By default, this code implements the CG (Conjugate Gradient) method,
    which is valid for real symmetric (and complex Hermitian) positive
    definite matrices. Note that for the complex Hermitian case, the
    VecDot() arguments within the code MUST remain in the order given
    for correct computation of inner products.

    Reference: Hestenes and Steifel, 1952.

    By switching to the indefinite vector inner product, VecTDot(), the
    same code is used for the complex symmetric case as well.  The user
    must call KSPCGSetType(ksp,KSP_CG_SYMMETRIC) or use the option
    -ksp_cg_type symmetric to invoke this variant for the complex case.
    Note, however, that the complex symmetric code is NOT valid for
    all such matrices ... and thus we don't recommend using this method.
*/
/*
    cgimpl.h defines the simple data structured used to store information
    related to the type of matrix (e.g. complex symmetric) being solved and
    data used during the optional Lanczo process used to compute eigenvalues
*/
#include <../src/ksp/ksp/impls/cg/cgimpl.h>       /*I "petscksp.h" I*/
extern PetscErrorCode KSPComputeExtremeSingularValues_CG(KSP,PetscReal*,PetscReal*);
extern PetscErrorCode KSPComputeEigenvalues_CG(KSP,PetscInt,PetscReal*,PetscReal*,PetscInt*);

/*
     KSPSetUp_CG - Sets up the workspace needed by the CG method.

      This is called once, usually automatically by KSPSolve() or KSPSetUp()
     but can be called directly by KSPSetUp()
*/
static PetscErrorCode KSPSetUp_CG(KSP ksp)
{
  KSP_CG         *cgP = (KSP_CG*)ksp->data;
  PetscErrorCode ierr;
  PetscInt       maxit = ksp->max_it,nwork = 3;

  PetscFunctionBegin;
  /* get work vectors needed by CG */
  if (cgP->singlereduction) nwork += 2;
  ierr = KSPSetWorkVecs(ksp,nwork);CHKERRQ(ierr);

  /*
     If user requested computations of eigenvalues then allocate
     work space needed
  */
  if (ksp->calc_sings) {
    ierr = PetscFree4(cgP->e,cgP->d,cgP->ee,cgP->dd);CHKERRQ(ierr);
    ierr = PetscMalloc4(maxit+1,&cgP->e,maxit+1,&cgP->d,maxit+1,&cgP->ee,maxit+1,&cgP->dd);CHKERRQ(ierr);
    ierr = PetscLogObjectMemory((PetscObject)ksp,2*(maxit+1)*(sizeof(PetscScalar)+sizeof(PetscReal)));CHKERRQ(ierr);

    ksp->ops->computeextremesingularvalues = KSPComputeExtremeSingularValues_CG;
    ksp->ops->computeeigenvalues           = KSPComputeEigenvalues_CG;
  }
  PetscFunctionReturn(0);
}

/*
     A macro used in the following KSPSolve_CG and KSPSolve_CG_SingleReduction routines
*/
#define VecXDot(x,y,a) (((cg->type) == (KSP_CG_HERMITIAN)) ? VecDot(x,y,a) : VecTDot(x,y,a))

/*
     KSPSolve_CG - This routine actually applies the conjugate gradient method

     Note : this routine can be replaced with another one (see below) which implements
            another variant of CG.

   Input Parameter:
.     ksp - the Krylov space object that was set to use conjugate gradient, by, for
            example, KSPCreate(MPI_Comm,KSP *ksp); KSPSetType(ksp,KSPCG);
*/
static PetscErrorCode KSPSolve_CG(KSP ksp)
{
  PetscErrorCode ierr;
  PetscInt       i,stored_max_it,eigs;
  PetscScalar    dpi = 0.0,a = 1.0,beta,betaold = 1.0,b = 0,*e = NULL,*d = NULL,dpiold;
  PetscReal      dp  = 0.0;
  Vec            X,B,Z,R,P,W;
  KSP_CG         *cg;
  Mat            Amat,Pmat;
  PetscBool      diagonalscale;

  PetscFunctionBegin;
  ierr = PCGetDiagonalScale(ksp->pc,&diagonalscale);CHKERRQ(ierr);
  PetscAssertFalse(diagonalscale,PetscObjectComm((PetscObject)ksp),PETSC_ERR_SUP,"Krylov method %s does not support diagonal scaling",((PetscObject)ksp)->type_name);

  cg            = (KSP_CG*)ksp->data;
  eigs          = ksp->calc_sings;
  stored_max_it = ksp->max_it;
  X             = ksp->vec_sol;
  B             = ksp->vec_rhs;
  R             = ksp->work[0];
  Z             = ksp->work[1];
  P             = ksp->work[2];
  W             = Z;

  if (eigs) {e = cg->e; d = cg->d; e[0] = 0.0; }
  ierr = PCGetOperators(ksp->pc,&Amat,&Pmat);CHKERRQ(ierr);

  ksp->its = 0;
  if (!ksp->guess_zero) {
    ierr = KSP_MatMult(ksp,Amat,X,R);CHKERRQ(ierr);            /*    r <- b - Ax                       */
    ierr = VecAYPX(R,-1.0,B);CHKERRQ(ierr);
  } else {
    ierr = VecCopy(B,R);CHKERRQ(ierr);                         /*    r <- b (x is 0)                   */
  }

  switch (ksp->normtype) {
    case KSP_NORM_PRECONDITIONED:
      ierr = KSP_PCApply(ksp,R,Z);CHKERRQ(ierr);               /*    z <- Br                           */
      ierr = VecNorm(Z,NORM_2,&dp);CHKERRQ(ierr);              /*    dp <- z'*z = e'*A'*B'*B*A*e       */
      KSPCheckNorm(ksp,dp);
      break;
    case KSP_NORM_UNPRECONDITIONED:
      ierr = VecNorm(R,NORM_2,&dp);CHKERRQ(ierr);              /*    dp <- r'*r = e'*A'*A*e            */
      KSPCheckNorm(ksp,dp);
      break;
    case KSP_NORM_NATURAL:
      ierr = KSP_PCApply(ksp,R,Z);CHKERRQ(ierr);               /*    z <- Br                           */
      ierr = VecXDot(Z,R,&beta);CHKERRQ(ierr);                 /*    beta <- z'*r                      */
      KSPCheckDot(ksp,beta);
      dp = PetscSqrtReal(PetscAbsScalar(beta));                /*    dp <- r'*z = r'*B*r = e'*A'*B*A*e */
      break;
    case KSP_NORM_NONE:
      dp = 0.0;
      break;
    default: SETERRQ(PetscObjectComm((PetscObject)ksp),PETSC_ERR_SUP,"%s",KSPNormTypes[ksp->normtype]);
  }
  ierr       = KSPLogResidualHistory(ksp,dp);CHKERRQ(ierr);
  ierr       = KSPMonitor(ksp,0,dp);CHKERRQ(ierr);
  ksp->rnorm = dp;

  ierr = (*ksp->converged)(ksp,0,dp,&ksp->reason,ksp->cnvP);CHKERRQ(ierr);     /* test for convergence */
  if (ksp->reason) PetscFunctionReturn(0);

  if (ksp->normtype != KSP_NORM_PRECONDITIONED && (ksp->normtype != KSP_NORM_NATURAL)) {
    ierr = KSP_PCApply(ksp,R,Z);CHKERRQ(ierr);                /*     z <- Br                           */
  }
  if (ksp->normtype != KSP_NORM_NATURAL) {
    ierr = VecXDot(Z,R,&beta);CHKERRQ(ierr);                  /*     beta <- z'*r                      */
    KSPCheckDot(ksp,beta);
  }

  i = 0;
  do {
    ksp->its = i+1;
    if (beta == 0.0) {
      ksp->reason = KSP_CONVERGED_ATOL;
      ierr        = PetscInfo(ksp,"converged due to beta = 0\n");CHKERRQ(ierr);
      break;
#if !defined(PETSC_USE_COMPLEX)
    } else if ((i > 0) && (beta*betaold < 0.0)) {
      PetscAssertFalse(ksp->errorifnotconverged,PetscObjectComm((PetscObject)ksp),PETSC_ERR_NOT_CONVERGED,"Diverged due to indefinite preconditioner, beta %g, betaold %g",(double)beta,(double)betaold);
      ksp->reason = KSP_DIVERGED_INDEFINITE_PC;
      ierr        = PetscInfo(ksp,"diverging due to indefinite preconditioner\n");CHKERRQ(ierr);
      break;
#endif
    }
    if (!i) {
      ierr = VecCopy(Z,P);CHKERRQ(ierr);                       /*     p <- z                           */
      b    = 0.0;
    } else {
      b = beta/betaold;
      if (eigs) {
        PetscAssertFalse(ksp->max_it != stored_max_it,PetscObjectComm((PetscObject)ksp),PETSC_ERR_SUP,"Can not change maxit AND calculate eigenvalues");
        e[i] = PetscSqrtReal(PetscAbsScalar(b))/a;
      }
      ierr = VecAYPX(P,b,Z);CHKERRQ(ierr);                     /*     p <- z + b* p                    */
    }
    dpiold = dpi;
    ierr = KSP_MatMult(ksp,Amat,P,W);CHKERRQ(ierr);            /*     w <- Ap                          */
    ierr = VecXDot(P,W,&dpi);CHKERRQ(ierr);                    /*     dpi <- p'w                       */
    KSPCheckDot(ksp,dpi);
    betaold = beta;

    if ((dpi == 0.0) || ((i > 0) && ((PetscSign(PetscRealPart(dpi))*PetscSign(PetscRealPart(dpiold))) < 0.0))) {
      PetscAssertFalse(ksp->errorifnotconverged,PetscObjectComm((PetscObject)ksp),PETSC_ERR_NOT_CONVERGED,"Diverged due to indefinite matrix, dpi %g, dpiold %g",(double)PetscRealPart(dpi),(double)PetscRealPart(dpiold));
      ksp->reason = KSP_DIVERGED_INDEFINITE_MAT;
      ierr        = PetscInfo(ksp,"diverging due to indefinite or negative definite matrix\n");CHKERRQ(ierr);
      break;
    }
    a = beta/dpi;                                              /*     a = beta/p'w                     */
    if (eigs) d[i] = PetscSqrtReal(PetscAbsScalar(b))*e[i] + 1.0/a;
    ierr = VecAXPY(X,a,P);CHKERRQ(ierr);                       /*     x <- x + ap                      */
    ierr = VecAXPY(R,-a,W);CHKERRQ(ierr);                      /*     r <- r - aw                      */
    if (ksp->normtype == KSP_NORM_PRECONDITIONED && ksp->chknorm < i+2) {
      ierr = KSP_PCApply(ksp,R,Z);CHKERRQ(ierr);               /*     z <- Br                          */
      ierr = VecNorm(Z,NORM_2,&dp);CHKERRQ(ierr);              /*     dp <- z'*z                       */
      KSPCheckNorm(ksp,dp);
    } else if (ksp->normtype == KSP_NORM_UNPRECONDITIONED && ksp->chknorm < i+2) {
      ierr = VecNorm(R,NORM_2,&dp);CHKERRQ(ierr);              /*     dp <- r'*r                       */
      KSPCheckNorm(ksp,dp);
    } else if (ksp->normtype == KSP_NORM_NATURAL) {
      ierr = KSP_PCApply(ksp,R,Z);CHKERRQ(ierr);               /*     z <- Br                          */
      ierr = VecXDot(Z,R,&beta);CHKERRQ(ierr);                 /*     beta <- r'*z                     */
      KSPCheckDot(ksp,beta);
      dp = PetscSqrtReal(PetscAbsScalar(beta));
    } else {
      dp = 0.0;
    }
    ksp->rnorm = dp;
    ierr = KSPLogResidualHistory(ksp,dp);CHKERRQ(ierr);
    if (eigs) cg->ned = ksp->its;
    ierr = KSPMonitor(ksp,i+1,dp);CHKERRQ(ierr);
    ierr = (*ksp->converged)(ksp,i+1,dp,&ksp->reason,ksp->cnvP);CHKERRQ(ierr);
    if (ksp->reason) break;

    if ((ksp->normtype != KSP_NORM_PRECONDITIONED && (ksp->normtype != KSP_NORM_NATURAL)) || (ksp->chknorm >= i+2)) {
      ierr = KSP_PCApply(ksp,R,Z);CHKERRQ(ierr);               /*     z <- Br                          */
    }
    if ((ksp->normtype != KSP_NORM_NATURAL) || (ksp->chknorm >= i+2)) {
      ierr = VecXDot(Z,R,&beta);CHKERRQ(ierr);                 /*     beta <- z'*r                     */
      KSPCheckDot(ksp,beta);
    }

    i++;
  } while (i<ksp->max_it);
  if (i >= ksp->max_it) ksp->reason = KSP_DIVERGED_ITS;
  PetscFunctionReturn(0);
}

/*
       KSPSolve_CG_SingleReduction

       This variant of CG is identical in exact arithmetic to the standard algorithm,
       but is rearranged to use only a single reduction stage per iteration, using additional
       intermediate vectors.

       See KSPCGUseSingleReduction_CG()

*/
static PetscErrorCode KSPSolve_CG_SingleReduction(KSP ksp)
{
  PetscErrorCode ierr;
  PetscInt       i,stored_max_it,eigs;
  PetscScalar    dpi = 0.0,a = 1.0,beta,betaold = 1.0,b = 0,*e = NULL,*d = NULL,delta,dpiold,tmp[2];
  PetscReal      dp  = 0.0;
  Vec            X,B,Z,R,P,S,W,tmpvecs[2];
  KSP_CG         *cg;
  Mat            Amat,Pmat;
  PetscBool      diagonalscale;

  PetscFunctionBegin;
  ierr = PCGetDiagonalScale(ksp->pc,&diagonalscale);CHKERRQ(ierr);
  PetscAssertFalse(diagonalscale,PetscObjectComm((PetscObject)ksp),PETSC_ERR_SUP,"Krylov method %s does not support diagonal scaling",((PetscObject)ksp)->type_name);

  cg            = (KSP_CG*)ksp->data;
  eigs          = ksp->calc_sings;
  stored_max_it = ksp->max_it;
  X             = ksp->vec_sol;
  B             = ksp->vec_rhs;
  R             = ksp->work[0];
  Z             = ksp->work[1];
  P             = ksp->work[2];
  S             = ksp->work[3];
  W             = ksp->work[4];

  if (eigs) {e = cg->e; d = cg->d; e[0] = 0.0; }
  ierr = PCGetOperators(ksp->pc,&Amat,&Pmat);CHKERRQ(ierr);

  ksp->its = 0;
  if (!ksp->guess_zero) {
    ierr = KSP_MatMult(ksp,Amat,X,R);CHKERRQ(ierr);            /*    r <- b - Ax                       */
    ierr = VecAYPX(R,-1.0,B);CHKERRQ(ierr);
  } else {
    ierr = VecCopy(B,R);CHKERRQ(ierr);                         /*    r <- b (x is 0)                   */
  }

  switch (ksp->normtype) {
    case KSP_NORM_PRECONDITIONED:
      ierr = KSP_PCApply(ksp,R,Z);CHKERRQ(ierr);               /*    z <- Br                           */
      ierr = VecNorm(Z,NORM_2,&dp);CHKERRQ(ierr);              /*    dp <- z'*z = e'*A'*B'*B*A'*e'     */
      KSPCheckNorm(ksp,dp);
      break;
    case KSP_NORM_UNPRECONDITIONED:
      ierr = VecNorm(R,NORM_2,&dp);CHKERRQ(ierr);              /*    dp <- r'*r = e'*A'*A*e            */
      KSPCheckNorm(ksp,dp);
      break;
    case KSP_NORM_NATURAL:
      ierr = KSP_PCApply(ksp,R,Z);CHKERRQ(ierr);               /*    z <- Br                           */
      ierr = KSP_MatMult(ksp,Amat,Z,S);CHKERRQ(ierr);
      ierr = VecXDot(Z,S,&delta);CHKERRQ(ierr);                /*    delta <- z'*A*z = r'*B*A*B*r      */
      ierr = VecXDot(Z,R,&beta);CHKERRQ(ierr);                 /*    beta <- z'*r                      */
      KSPCheckDot(ksp,beta);
      dp = PetscSqrtReal(PetscAbsScalar(beta));                /*    dp <- r'*z = r'*B*r = e'*A'*B*A*e */
      break;
    case KSP_NORM_NONE:
      dp = 0.0;
      break;
    default: SETERRQ(PetscObjectComm((PetscObject)ksp),PETSC_ERR_SUP,"%s",KSPNormTypes[ksp->normtype]);
  }
  ierr       = KSPLogResidualHistory(ksp,dp);CHKERRQ(ierr);
  ierr       = KSPMonitor(ksp,0,dp);CHKERRQ(ierr);
  ksp->rnorm = dp;

  ierr = (*ksp->converged)(ksp,0,dp,&ksp->reason,ksp->cnvP);CHKERRQ(ierr);     /* test for convergence */
  if (ksp->reason) PetscFunctionReturn(0);

  if (ksp->normtype != KSP_NORM_PRECONDITIONED && (ksp->normtype != KSP_NORM_NATURAL)) {
    ierr = KSP_PCApply(ksp,R,Z);CHKERRQ(ierr);                 /*    z <- Br                           */
  }
  if (ksp->normtype != KSP_NORM_NATURAL) {
    ierr = KSP_MatMult(ksp,Amat,Z,S);CHKERRQ(ierr);
    ierr = VecXDot(Z,S,&delta);CHKERRQ(ierr);                  /*    delta <- z'*A*z = r'*B*A*B*r      */
    ierr = VecXDot(Z,R,&beta);CHKERRQ(ierr);                   /*    beta <- z'*r                      */
    KSPCheckDot(ksp,beta);
  }

  i = 0;
  do {
    ksp->its = i+1;
    if (beta == 0.0) {
      ksp->reason = KSP_CONVERGED_ATOL;
      ierr        = PetscInfo(ksp,"converged due to beta = 0\n");CHKERRQ(ierr);
      break;
#if !defined(PETSC_USE_COMPLEX)
    } else if ((i > 0) && (beta*betaold < 0.0)) {
      PetscAssertFalse(ksp->errorifnotconverged,PetscObjectComm((PetscObject)ksp),PETSC_ERR_NOT_CONVERGED,"Diverged due to indefinite preconditioner");
      ksp->reason = KSP_DIVERGED_INDEFINITE_PC;
      ierr        = PetscInfo(ksp,"diverging due to indefinite preconditioner\n");CHKERRQ(ierr);
      break;
#endif
    }
    if (!i) {
      ierr = VecCopy(Z,P);CHKERRQ(ierr);                       /*    p <- z                           */
      b    = 0.0;
    } else {
      b = beta/betaold;
      if (eigs) {
        PetscAssertFalse(ksp->max_it != stored_max_it,PetscObjectComm((PetscObject)ksp),PETSC_ERR_SUP,"Can not change maxit AND calculate eigenvalues");
        e[i] = PetscSqrtReal(PetscAbsScalar(b))/a;
      }
      ierr = VecAYPX(P,b,Z);CHKERRQ(ierr);                     /*    p <- z + b* p                     */
    }
    dpiold = dpi;
    if (!i) {
      ierr = KSP_MatMult(ksp,Amat,P,W);CHKERRQ(ierr);          /*    w <- Ap                           */
      ierr = VecXDot(P,W,&dpi);CHKERRQ(ierr);                  /*    dpi <- p'w                        */
    } else {
      ierr = VecAYPX(W,beta/betaold,S);CHKERRQ(ierr);          /*    w <- Ap                           */
      dpi  = delta - beta*beta*dpiold/(betaold*betaold);       /*    dpi <- p'w                        */
    }
    betaold = beta;
    KSPCheckDot(ksp,beta);

    if ((dpi == 0.0) || ((i > 0) && (PetscRealPart(dpi*dpiold) <= 0.0))) {
      PetscAssertFalse(ksp->errorifnotconverged,PetscObjectComm((PetscObject)ksp),PETSC_ERR_NOT_CONVERGED,"Diverged due to indefinite matrix");
      ksp->reason = KSP_DIVERGED_INDEFINITE_MAT;
      ierr        = PetscInfo(ksp,"diverging due to indefinite or negative definite matrix\n");CHKERRQ(ierr);
      break;
    }
    a = beta/dpi;                                              /*    a = beta/p'w                      */
    if (eigs) d[i] = PetscSqrtReal(PetscAbsScalar(b))*e[i] + 1.0/a;
    ierr = VecAXPY(X,a,P);CHKERRQ(ierr);                       /*    x <- x + ap                       */
    ierr = VecAXPY(R,-a,W);CHKERRQ(ierr);                      /*    r <- r - aw                       */
    if (ksp->normtype == KSP_NORM_PRECONDITIONED && ksp->chknorm < i+2) {
      ierr = KSP_PCApply(ksp,R,Z);CHKERRQ(ierr);               /*    z <- Br                           */
      ierr = KSP_MatMult(ksp,Amat,Z,S);CHKERRQ(ierr);
      ierr = VecNorm(Z,NORM_2,&dp);CHKERRQ(ierr);              /*    dp <- z'*z                        */
      KSPCheckNorm(ksp,dp);
    } else if (ksp->normtype == KSP_NORM_UNPRECONDITIONED && ksp->chknorm < i+2) {
      ierr = VecNorm(R,NORM_2,&dp);CHKERRQ(ierr);              /*    dp <- r'*r                        */
      KSPCheckNorm(ksp,dp);
    } else if (ksp->normtype == KSP_NORM_NATURAL) {
      ierr = KSP_PCApply(ksp,R,Z);CHKERRQ(ierr);               /*    z <- Br                           */
      tmpvecs[0] = S; tmpvecs[1] = R;
      ierr  = KSP_MatMult(ksp,Amat,Z,S);CHKERRQ(ierr);
      ierr  = VecMDot(Z,2,tmpvecs,tmp);CHKERRQ(ierr);          /*    delta <- z'*A*z = r'*B*A*B*r      */
      delta = tmp[0]; beta = tmp[1];                           /*    beta <- z'*r                      */
      KSPCheckDot(ksp,beta);
      dp = PetscSqrtReal(PetscAbsScalar(beta));                /*    dp <- r'*z = r'*B*r = e'*A'*B*A*e */
    } else {
      dp = 0.0;
    }
    ksp->rnorm = dp;
    ierr = KSPLogResidualHistory(ksp,dp);CHKERRQ(ierr);
    if (eigs) cg->ned = ksp->its;
    ierr = KSPMonitor(ksp,i+1,dp);CHKERRQ(ierr);
    ierr = (*ksp->converged)(ksp,i+1,dp,&ksp->reason,ksp->cnvP);CHKERRQ(ierr);
    if (ksp->reason) break;

    if ((ksp->normtype != KSP_NORM_PRECONDITIONED && (ksp->normtype != KSP_NORM_NATURAL)) || (ksp->chknorm >= i+2)) {
      ierr = KSP_PCApply(ksp,R,Z);CHKERRQ(ierr);               /*    z <- Br                           */
      ierr = KSP_MatMult(ksp,Amat,Z,S);CHKERRQ(ierr);
    }
    if ((ksp->normtype != KSP_NORM_NATURAL) || (ksp->chknorm >= i+2)) {
      tmpvecs[0] = S; tmpvecs[1] = R;
      ierr  = VecMDot(Z,2,tmpvecs,tmp);CHKERRQ(ierr);
      delta = tmp[0]; beta = tmp[1];                           /*    delta <- z'*A*z = r'*B'*A*B*r     */
      KSPCheckDot(ksp,beta);                                   /*    beta <- z'*r                      */
    }

    i++;
  } while (i<ksp->max_it);
  if (i >= ksp->max_it) ksp->reason = KSP_DIVERGED_ITS;
  PetscFunctionReturn(0);
}

/*
     KSPDestroy_CG - Frees resources allocated in KSPSetup_CG and clears function
                     compositions from KSPCreate_CG. If adding your own KSP implementation,
                     you must be sure to free all allocated resources here to prevent
                     leaks.
*/
PetscErrorCode KSPDestroy_CG(KSP ksp)
{
  KSP_CG         *cg = (KSP_CG*)ksp->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFree4(cg->e,cg->d,cg->ee,cg->dd);CHKERRQ(ierr);
  ierr = KSPDestroyDefault(ksp);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)ksp,"KSPCGSetType_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)ksp,"KSPCGUseSingleReduction_C",NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
     KSPView_CG - Prints information about the current Krylov method being used.
                  If your Krylov method has special options or flags that information
                  should be printed here.
*/
PetscErrorCode KSPView_CG(KSP ksp,PetscViewer viewer)
{
  KSP_CG         *cg = (KSP_CG*)ksp->data;
  PetscErrorCode ierr;
  PetscBool      iascii;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
#if defined(PETSC_USE_COMPLEX)
    ierr = PetscViewerASCIIPrintf(viewer,"  variant %s\n",KSPCGTypes[cg->type]);CHKERRQ(ierr);
#endif
    if (cg->singlereduction) {
      ierr = PetscViewerASCIIPrintf(viewer,"  using single-reduction variant\n");CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

/*
    KSPSetFromOptions_CG - Checks the options database for options related to the
                           conjugate gradient method.
*/
PetscErrorCode KSPSetFromOptions_CG(PetscOptionItems *PetscOptionsObject,KSP ksp)
{
  PetscErrorCode ierr;
  KSP_CG         *cg = (KSP_CG*)ksp->data;
  PetscBool      flg;

  PetscFunctionBegin;
  ierr = PetscOptionsHead(PetscOptionsObject,"KSP CG and CGNE options");CHKERRQ(ierr);
#if defined(PETSC_USE_COMPLEX)
  ierr = PetscOptionsEnum("-ksp_cg_type","Matrix is Hermitian or complex symmetric","KSPCGSetType",KSPCGTypes,(PetscEnum)cg->type,
                          (PetscEnum*)&cg->type,NULL);CHKERRQ(ierr);
#endif
  ierr = PetscOptionsBool("-ksp_cg_single_reduction","Merge inner products into single MPI_Allreduce()","KSPCGUseSingleReduction",cg->singlereduction,&cg->singlereduction,&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = KSPCGUseSingleReduction(ksp,cg->singlereduction);CHKERRQ(ierr);
  }
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
    KSPCGSetType_CG - This is an option that is SPECIFIC to this particular Krylov method.
                      This routine is registered below in KSPCreate_CG() and called from the
                      routine KSPCGSetType() (see the file cgtype.c).
*/
PetscErrorCode  KSPCGSetType_CG(KSP ksp,KSPCGType type)
{
  KSP_CG *cg = (KSP_CG*)ksp->data;

  PetscFunctionBegin;
  cg->type = type;
  PetscFunctionReturn(0);
}

/*
    KSPCGUseSingleReduction_CG

    This routine sets a flag to use a variant of CG. Note that (in somewhat
    atypical fashion) it also swaps out the routine called when KSPSolve()
    is invoked.
*/
static PetscErrorCode  KSPCGUseSingleReduction_CG(KSP ksp,PetscBool flg)
{
  KSP_CG *cg = (KSP_CG*)ksp->data;

  PetscFunctionBegin;
  cg->singlereduction = flg;
  if (cg->singlereduction) {
    ksp->ops->solve = KSPSolve_CG_SingleReduction;
  } else {
    ksp->ops->solve = KSPSolve_CG;
  }
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode KSPBuildResidual_CG(KSP ksp,Vec t,Vec v,Vec *V)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecCopy(ksp->work[0],v);CHKERRQ(ierr);
  *V   = v;
  PetscFunctionReturn(0);
}

/*
    KSPCreate_CG - Creates the data structure for the Krylov method CG and sets the
       function pointers for all the routines it needs to call (KSPSolve_CG() etc)

    It must be labeled as PETSC_EXTERN to be dynamically linkable in C++
*/
/*MC
     KSPCG - The Preconditioned Conjugate Gradient (PCG) iterative method

   Options Database Keys:
+   -ksp_cg_type Hermitian - (for complex matrices only) indicates the matrix is Hermitian, see KSPCGSetType()
.   -ksp_cg_type symmetric - (for complex matrices only) indicates the matrix is symmetric
-   -ksp_cg_single_reduction - performs both inner products needed in the algorithm with a single MPI_Allreduce() call, see KSPCGUseSingleReduction()

   Level: beginner

   Notes:
    The PCG method requires both the matrix and preconditioner to be symmetric positive (or negative) (semi) definite.

   Only left preconditioning is supported; there are several ways to motivate preconditioned CG, but they all produce the same algorithm.
   One can interpret preconditioning A with B to mean any of the following\:
.n  (1) Solve a left-preconditioned system BAx = Bb, using inv(B) to define an inner product in the algorithm.
.n  (2) Solve a right-preconditioned system ABy = b, x = By, using B to define an inner product in the algorithm.
.n  (3) Solve a symmetrically-preconditioned system, E^TAEy = E^Tb, x = Ey, where B = EE^T.
.n  (4) Solve Ax=b with CG, but use the inner product defined by B to define the method [2].
.n  In all cases, the resulting algorithm only requires application of B to vectors.

   For complex numbers there are two different CG methods, one for Hermitian symmetric matrices and one for non-Hermitian symmetric matrices. Use
   KSPCGSetType() to indicate which type you are using.

   Developer Notes:
    KSPSolve_CG() should actually query the matrix to determine if it is Hermitian symmetric or not and NOT require the user to
   indicate it to the KSP object.

   References:
+   1. - Magnus R. Hestenes and Eduard Stiefel, Methods of Conjugate Gradients for Solving Linear Systems,
   Journal of Research of the National Bureau of Standards Vol. 49, No. 6, December 1952 Research Paper 2379
-   2. - Josef Malek and Zdenek Strakos, Preconditioning and the Conjugate Gradient Method in the Context of Solving PDEs,
    SIAM, 2014.

.seealso:  KSPCreate(), KSPSetType(), KSPType (for list of available types), KSP,
           KSPCGSetType(), KSPCGUseSingleReduction(), KSPPIPECG, KSPGROPPCG

M*/
PETSC_EXTERN PetscErrorCode KSPCreate_CG(KSP ksp)
{
  PetscErrorCode ierr;
  KSP_CG         *cg;

  PetscFunctionBegin;
  ierr = PetscNewLog(ksp,&cg);CHKERRQ(ierr);
#if !defined(PETSC_USE_COMPLEX)
  cg->type = KSP_CG_SYMMETRIC;
#else
  cg->type = KSP_CG_HERMITIAN;
#endif
  ksp->data = (void*)cg;

  ierr = KSPSetSupportedNorm(ksp,KSP_NORM_PRECONDITIONED,PC_LEFT,3);CHKERRQ(ierr);
  ierr = KSPSetSupportedNorm(ksp,KSP_NORM_UNPRECONDITIONED,PC_LEFT,2);CHKERRQ(ierr);
  ierr = KSPSetSupportedNorm(ksp,KSP_NORM_NATURAL,PC_LEFT,2);CHKERRQ(ierr);
  ierr = KSPSetSupportedNorm(ksp,KSP_NORM_NONE,PC_LEFT,1);CHKERRQ(ierr);

  /*
       Sets the functions that are associated with this data structure
       (in C++ this is the same as defining virtual functions)
  */
  ksp->ops->setup          = KSPSetUp_CG;
  ksp->ops->solve          = KSPSolve_CG;
  ksp->ops->destroy        = KSPDestroy_CG;
  ksp->ops->view           = KSPView_CG;
  ksp->ops->setfromoptions = KSPSetFromOptions_CG;
  ksp->ops->buildsolution  = KSPBuildSolutionDefault;
  ksp->ops->buildresidual  = KSPBuildResidual_CG;

  /*
      Attach the function KSPCGSetType_CG() to this object. The routine
      KSPCGSetType() checks for this attached function and calls it if it finds
      it. (Sort of like a dynamic member function that can be added at run time
  */
  ierr = PetscObjectComposeFunction((PetscObject)ksp,"KSPCGSetType_C",KSPCGSetType_CG);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)ksp,"KSPCGUseSingleReduction_C",KSPCGUseSingleReduction_CG);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
