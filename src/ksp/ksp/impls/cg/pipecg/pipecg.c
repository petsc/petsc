
#include <petsc/private/kspimpl.h>

/*
     KSPSetUp_PIPECG - Sets up the workspace needed by the PIPECG method.

      This is called once, usually automatically by KSPSolve() or KSPSetUp()
     but can be called directly by KSPSetUp()
*/
static PetscErrorCode KSPSetUp_PIPECG(KSP ksp)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* get work vectors needed by PIPECG */
  ierr = KSPSetWorkVecs(ksp,9);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
 KSPSolve_PIPECG - This routine actually applies the pipelined conjugate gradient method
*/
static PetscErrorCode  KSPSolve_PIPECG(KSP ksp)
{
  PetscErrorCode ierr;
  PetscInt       i;
  PetscScalar    alpha = 0.0,beta = 0.0,gamma = 0.0,gammaold = 0.0,delta = 0.0;
  PetscReal      dp    = 0.0;
  Vec            X,B,Z,P,W,Q,U,M,N,R,S;
  Mat            Amat,Pmat;
  PetscBool      diagonalscale;

  PetscFunctionBegin;
  ierr = PCGetDiagonalScale(ksp->pc,&diagonalscale);CHKERRQ(ierr);
  if (diagonalscale) SETERRQ1(PetscObjectComm((PetscObject)ksp),PETSC_ERR_SUP,"Krylov method %s does not support diagonal scaling",((PetscObject)ksp)->type_name);

  X = ksp->vec_sol;
  B = ksp->vec_rhs;
  R = ksp->work[0];
  Z = ksp->work[1];
  P = ksp->work[2];
  N = ksp->work[3];
  W = ksp->work[4];
  Q = ksp->work[5];
  U = ksp->work[6];
  M = ksp->work[7];
  S = ksp->work[8];

  ierr = PCGetOperators(ksp->pc,&Amat,&Pmat);CHKERRQ(ierr);

  ksp->its = 0;
  if (!ksp->guess_zero) {
    ierr = KSP_MatMult(ksp,Amat,X,R);CHKERRQ(ierr);            /*     r <- b - Ax     */
    ierr = VecAYPX(R,-1.0,B);CHKERRQ(ierr);
  } else {
    ierr = VecCopy(B,R);CHKERRQ(ierr);                         /*     r <- b (x is 0) */
  }

  ierr = KSP_PCApply(ksp,R,U);CHKERRQ(ierr);                   /*     u <- Br   */

  switch (ksp->normtype) {
  case KSP_NORM_PRECONDITIONED:
    ierr = VecNormBegin(U,NORM_2,&dp);CHKERRQ(ierr);                /*     dp <- u'*u = e'*A'*B'*B*A'*e'     */
    ierr = PetscCommSplitReductionBegin(PetscObjectComm((PetscObject)U));CHKERRQ(ierr);
    ierr = KSP_MatMult(ksp,Amat,U,W);CHKERRQ(ierr);              /*     w <- Au   */
    ierr = VecNormEnd(U,NORM_2,&dp);CHKERRQ(ierr);
    break;
  case KSP_NORM_UNPRECONDITIONED:
    ierr = VecNormBegin(R,NORM_2,&dp);CHKERRQ(ierr);                /*     dp <- r'*r = e'*A'*A*e            */
    ierr = PetscCommSplitReductionBegin(PetscObjectComm((PetscObject)R));CHKERRQ(ierr);
    ierr = KSP_MatMult(ksp,Amat,U,W);CHKERRQ(ierr);              /*     w <- Au   */
    ierr = VecNormEnd(R,NORM_2,&dp);CHKERRQ(ierr);
    break;
  case KSP_NORM_NATURAL:
    ierr = VecDotBegin(R,U,&gamma);CHKERRQ(ierr);                  /*     gamma <- u'*r       */
    ierr = PetscCommSplitReductionBegin(PetscObjectComm((PetscObject)R));CHKERRQ(ierr);
    ierr = KSP_MatMult(ksp,Amat,U,W);CHKERRQ(ierr);              /*     w <- Au   */
    ierr = VecDotEnd(R,U,&gamma);CHKERRQ(ierr);
    KSPCheckDot(ksp,gamma);
    dp = PetscSqrtReal(PetscAbsScalar(gamma));                  /*     dp <- r'*u = r'*B*r = e'*A'*B*A*e */
    break;
  case KSP_NORM_NONE:
    ierr = KSP_MatMult(ksp,Amat,U,W);CHKERRQ(ierr);
    dp   = 0.0;
    break;
  default: SETERRQ1(PetscObjectComm((PetscObject)ksp),PETSC_ERR_SUP,"%s",KSPNormTypes[ksp->normtype]);
  }
  ierr       = KSPLogResidualHistory(ksp,dp);CHKERRQ(ierr);
  ierr       = KSPMonitor(ksp,0,dp);CHKERRQ(ierr);
  ksp->rnorm = dp;
  ierr       = (*ksp->converged)(ksp,0,dp,&ksp->reason,ksp->cnvP);CHKERRQ(ierr); /* test for convergence */
  if (ksp->reason) PetscFunctionReturn(0);

  i = 0;
  do {
    if (i > 0 && ksp->normtype == KSP_NORM_UNPRECONDITIONED) {
      ierr = VecNormBegin(R,NORM_2,&dp);CHKERRQ(ierr);
    } else if (i > 0 && ksp->normtype == KSP_NORM_PRECONDITIONED) {
      ierr = VecNormBegin(U,NORM_2,&dp);CHKERRQ(ierr);
    }
    if (!(i == 0 && ksp->normtype == KSP_NORM_NATURAL)) {
      ierr = VecDotBegin(R,U,&gamma);CHKERRQ(ierr);
    }
    ierr = VecDotBegin(W,U,&delta);CHKERRQ(ierr);
    ierr = PetscCommSplitReductionBegin(PetscObjectComm((PetscObject)R));CHKERRQ(ierr);

    ierr = KSP_PCApply(ksp,W,M);CHKERRQ(ierr);           /*   m <- Bw       */
    ierr = KSP_MatMult(ksp,Amat,M,N);CHKERRQ(ierr);      /*   n <- Am       */

    if (i > 0 && ksp->normtype == KSP_NORM_UNPRECONDITIONED) {
      ierr = VecNormEnd(R,NORM_2,&dp);CHKERRQ(ierr);
    } else if (i > 0 && ksp->normtype == KSP_NORM_PRECONDITIONED) {
      ierr = VecNormEnd(U,NORM_2,&dp);CHKERRQ(ierr);
    }
    if (!(i == 0 && ksp->normtype == KSP_NORM_NATURAL)) {
      ierr = VecDotEnd(R,U,&gamma);CHKERRQ(ierr);
    }
    ierr = VecDotEnd(W,U,&delta);CHKERRQ(ierr);

    if (i > 0) {
      if (ksp->normtype == KSP_NORM_NATURAL) dp = PetscSqrtReal(PetscAbsScalar(gamma));
      else if (ksp->normtype == KSP_NORM_NONE) dp = 0.0;

      ksp->rnorm = dp;
      ierr = KSPLogResidualHistory(ksp,dp);CHKERRQ(ierr);
      ierr = KSPMonitor(ksp,i,dp);CHKERRQ(ierr);
      ierr = (*ksp->converged)(ksp,i,dp,&ksp->reason,ksp->cnvP);CHKERRQ(ierr);
      if (ksp->reason) PetscFunctionReturn(0);
    }

    if (i == 0) {
      alpha = gamma / delta;
      ierr  = VecCopy(N,Z);CHKERRQ(ierr);        /*     z <- n          */
      ierr  = VecCopy(M,Q);CHKERRQ(ierr);        /*     q <- m          */
      ierr  = VecCopy(U,P);CHKERRQ(ierr);        /*     p <- u          */
      ierr  = VecCopy(W,S);CHKERRQ(ierr);        /*     s <- w          */
    } else {
      beta  = gamma / gammaold;
      alpha = gamma / (delta - beta / alpha * gamma);
      ierr  = VecAYPX(Z,beta,N);CHKERRQ(ierr);   /*     z <- n + beta * z   */
      ierr  = VecAYPX(Q,beta,M);CHKERRQ(ierr);   /*     q <- m + beta * q   */
      ierr  = VecAYPX(P,beta,U);CHKERRQ(ierr);   /*     p <- u + beta * p   */
      ierr  = VecAYPX(S,beta,W);CHKERRQ(ierr);   /*     s <- w + beta * s   */
    }
    ierr     = VecAXPY(X, alpha,P);CHKERRQ(ierr); /*     x <- x + alpha * p   */
    ierr     = VecAXPY(U,-alpha,Q);CHKERRQ(ierr); /*     u <- u - alpha * q   */
    ierr     = VecAXPY(W,-alpha,Z);CHKERRQ(ierr); /*     w <- w - alpha * z   */
    ierr     = VecAXPY(R,-alpha,S);CHKERRQ(ierr); /*     r <- r - alpha * s   */
    gammaold = gamma;
    i++;
    ksp->its = i;

    /* if (i%50 == 0) { */
    /*   ierr = KSP_MatMult(ksp,Amat,X,R);CHKERRQ(ierr);            /\*     w <- b - Ax     *\/ */
    /*   ierr = VecAYPX(R,-1.0,B);CHKERRQ(ierr); */
    /*   ierr = KSP_PCApply(ksp,R,U);CHKERRQ(ierr); */
    /*   ierr = KSP_MatMult(ksp,Amat,U,W);CHKERRQ(ierr); */
    /* } */

  } while (i<=ksp->max_it);
  if (!ksp->reason) ksp->reason = KSP_DIVERGED_ITS;
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode KSPBuildResidual_CG(KSP,Vec,Vec,Vec*);

/*MC
   KSPPIPECG - Pipelined conjugate gradient method.

   This method has only a single non-blocking reduction per iteration, compared to 2 blocking for standard CG.  The
   non-blocking reduction is overlapped by the matrix-vector product and preconditioner application.

   See also KSPPIPECR, where the reduction is only overlapped with the matrix-vector product.

   Level: intermediate

   Notes:
   MPI configuration may be necessary for reductions to make asynchronous progress, which is important for performance of pipelined methods.
   See the FAQ on the PETSc website for details.

   Contributed by:
   Pieter Ghysels, Universiteit Antwerpen, Intel Exascience lab Flanders

   Reference:
   P. Ghysels and W. Vanroose, "Hiding global synchronization latency in the preconditioned Conjugate Gradient algorithm",
   Submitted to Parallel Computing, 2012.

.seealso: KSPCreate(), KSPSetType(), KSPPIPECR, KSPGROPPCG, KSPPGMRES, KSPCG, KSPCGUseSingleReduction()
M*/
PETSC_EXTERN PetscErrorCode KSPCreate_PIPECG(KSP ksp)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = KSPSetSupportedNorm(ksp,KSP_NORM_UNPRECONDITIONED,PC_LEFT,2);CHKERRQ(ierr);
  ierr = KSPSetSupportedNorm(ksp,KSP_NORM_PRECONDITIONED,PC_LEFT,2);CHKERRQ(ierr);
  ierr = KSPSetSupportedNorm(ksp,KSP_NORM_NATURAL,PC_LEFT,2);CHKERRQ(ierr);
  ierr = KSPSetSupportedNorm(ksp,KSP_NORM_NONE,PC_LEFT,1);CHKERRQ(ierr);

  ksp->ops->setup          = KSPSetUp_PIPECG;
  ksp->ops->solve          = KSPSolve_PIPECG;
  ksp->ops->destroy        = KSPDestroyDefault;
  ksp->ops->view           = NULL;
  ksp->ops->setfromoptions = NULL;
  ksp->ops->buildsolution  = KSPBuildSolutionDefault;
  ksp->ops->buildresidual  = KSPBuildResidual_CG;
  PetscFunctionReturn(0);
}
