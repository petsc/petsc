#include <petsc/private/kspimpl.h>

/*
 KSPSetUp_GROPPCG - Sets up the workspace needed by the GROPPCG method.

 This is called once, usually automatically by KSPSolve() or KSPSetUp()
 but can be called directly by KSPSetUp()
*/
static PetscErrorCode KSPSetUp_GROPPCG(KSP ksp)
{
  PetscFunctionBegin;
  CHKERRQ(KSPSetWorkVecs(ksp,6));
  PetscFunctionReturn(0);
}

/*
 KSPSolve_GROPPCG

 Input Parameter:
 .     ksp - the Krylov space object that was set to use conjugate gradient, by, for
             example, KSPCreate(MPI_Comm,KSP *ksp); KSPSetType(ksp,KSPCG);
*/
static PetscErrorCode  KSPSolve_GROPPCG(KSP ksp)
{
  PetscInt       i;
  PetscScalar    alpha,beta = 0.0,gamma,gammaNew,t;
  PetscReal      dp = 0.0;
  Vec            x,b,r,p,s,S,z,Z;
  Mat            Amat,Pmat;
  PetscBool      diagonalscale;

  PetscFunctionBegin;
  CHKERRQ(PCGetDiagonalScale(ksp->pc,&diagonalscale));
  PetscCheckFalse(diagonalscale,PetscObjectComm((PetscObject)ksp),PETSC_ERR_SUP,"Krylov method %s does not support diagonal scaling",((PetscObject)ksp)->type_name);

  x = ksp->vec_sol;
  b = ksp->vec_rhs;
  r = ksp->work[0];
  p = ksp->work[1];
  s = ksp->work[2];
  S = ksp->work[3];
  z = ksp->work[4];
  Z = ksp->work[5];

  CHKERRQ(PCGetOperators(ksp->pc,&Amat,&Pmat));

  ksp->its = 0;
  if (!ksp->guess_zero) {
    CHKERRQ(KSP_MatMult(ksp,Amat,x,r));            /*     r <- b - Ax     */
    CHKERRQ(VecAYPX(r,-1.0,b));
  } else {
    CHKERRQ(VecCopy(b,r));                         /*     r <- b (x is 0) */
  }

  CHKERRQ(KSP_PCApply(ksp,r,z));                   /*     z <- Br   */
  CHKERRQ(VecCopy(z,p));                           /*     p <- z    */
  CHKERRQ(VecDotBegin(r,z,&gamma));                  /*     gamma <- z'*r       */
  CHKERRQ(PetscCommSplitReductionBegin(PetscObjectComm((PetscObject)r)));
  CHKERRQ(KSP_MatMult(ksp,Amat,p,s));              /*     s <- Ap   */
  CHKERRQ(VecDotEnd(r,z,&gamma));                  /*     gamma <- z'*r       */

  switch (ksp->normtype) {
  case KSP_NORM_PRECONDITIONED:
    /* This could be merged with the computation of gamma above */
    CHKERRQ(VecNorm(z,NORM_2,&dp));                /*     dp <- z'*z = e'*A'*B'*B*A'*e'     */
    break;
  case KSP_NORM_UNPRECONDITIONED:
    /* This could be merged with the computation of gamma above */
    CHKERRQ(VecNorm(r,NORM_2,&dp));                /*     dp <- r'*r = e'*A'*A*e            */
    break;
  case KSP_NORM_NATURAL:
    KSPCheckDot(ksp,gamma);
    dp = PetscSqrtReal(PetscAbsScalar(gamma));                  /*     dp <- r'*z = r'*B*r = e'*A'*B*A*e */
    break;
  case KSP_NORM_NONE:
    dp = 0.0;
    break;
  default: SETERRQ(PetscObjectComm((PetscObject)ksp),PETSC_ERR_SUP,"%s",KSPNormTypes[ksp->normtype]);
  }
  CHKERRQ(KSPLogResidualHistory(ksp,dp));
  CHKERRQ(KSPMonitor(ksp,0,dp));
  ksp->rnorm = dp;
  CHKERRQ((*ksp->converged)(ksp,0,dp,&ksp->reason,ksp->cnvP)); /* test for convergence */
  if (ksp->reason) PetscFunctionReturn(0);

  i = 0;
  do {
    ksp->its = i+1;
    i++;

    CHKERRQ(VecDotBegin(p,s,&t));
    CHKERRQ(PetscCommSplitReductionBegin(PetscObjectComm((PetscObject)p)));

    CHKERRQ(KSP_PCApply(ksp,s,S));         /*   S <- Bs       */

    CHKERRQ(VecDotEnd(p,s,&t));

    alpha = gamma / t;
    CHKERRQ(VecAXPY(x, alpha,p));   /*     x <- x + alpha * p   */
    CHKERRQ(VecAXPY(r,-alpha,s));   /*     r <- r - alpha * s   */
    CHKERRQ(VecAXPY(z,-alpha,S));   /*     z <- z - alpha * S   */

    if (ksp->normtype == KSP_NORM_UNPRECONDITIONED) {
      CHKERRQ(VecNormBegin(r,NORM_2,&dp));
    } else if (ksp->normtype == KSP_NORM_PRECONDITIONED) {
      CHKERRQ(VecNormBegin(z,NORM_2,&dp));
    }
    CHKERRQ(VecDotBegin(r,z,&gammaNew));
    CHKERRQ(PetscCommSplitReductionBegin(PetscObjectComm((PetscObject)r)));

    CHKERRQ(KSP_MatMult(ksp,Amat,z,Z));      /*   Z <- Az       */

    if (ksp->normtype == KSP_NORM_UNPRECONDITIONED) {
      CHKERRQ(VecNormEnd(r,NORM_2,&dp));
    } else if (ksp->normtype == KSP_NORM_PRECONDITIONED) {
      CHKERRQ(VecNormEnd(z,NORM_2,&dp));
    }
    CHKERRQ(VecDotEnd(r,z,&gammaNew));

    if (ksp->normtype == KSP_NORM_NATURAL) {
      KSPCheckDot(ksp,gammaNew);
      dp = PetscSqrtReal(PetscAbsScalar(gammaNew));                  /*     dp <- r'*z = r'*B*r = e'*A'*B*A*e */
    } else if (ksp->normtype == KSP_NORM_NONE) {
      dp = 0.0;
    }
    ksp->rnorm = dp;
    CHKERRQ(KSPLogResidualHistory(ksp,dp));
    CHKERRQ(KSPMonitor(ksp,i,dp));
    CHKERRQ((*ksp->converged)(ksp,i,dp,&ksp->reason,ksp->cnvP));
    if (ksp->reason) PetscFunctionReturn(0);

    beta  = gammaNew / gamma;
    gamma = gammaNew;
    CHKERRQ(VecAYPX(p,beta,z));   /*     p <- z + beta * p   */
    CHKERRQ(VecAYPX(s,beta,Z));   /*     s <- Z + beta * s   */

  } while (i<ksp->max_it);

  if (i >= ksp->max_it) ksp->reason = KSP_DIVERGED_ITS;
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode KSPBuildResidual_CG(KSP,Vec,Vec,Vec*);

/*MC
   KSPGROPPCG - A pipelined conjugate gradient method from Bill Gropp

   This method has two reductions, one of which is overlapped with the matrix-vector product and one of which is
   overlapped with the preconditioner.

   See also KSPPIPECG, which has only a single reduction that overlaps both the matrix-vector product and the preconditioner.

   Level: intermediate

   Notes:
   MPI configuration may be necessary for reductions to make asynchronous progress, which is important for performance of pipelined methods.
   See the FAQ on the PETSc website for details.

   Contributed by:
   Pieter Ghysels, Universiteit Antwerpen, Intel Exascience lab Flanders

   Reference:
   http://www.cs.uiuc.edu/~wgropp/bib/talks/tdata/2012/icerm.pdf

.seealso: KSPCreate(), KSPSetType(), KSPPIPECG, KSPPIPECR, KSPPGMRES, KSPCG, KSPCGUseSingleReduction()
M*/

PETSC_EXTERN PetscErrorCode KSPCreate_GROPPCG(KSP ksp)
{
  PetscFunctionBegin;
  CHKERRQ(KSPSetSupportedNorm(ksp,KSP_NORM_UNPRECONDITIONED,PC_LEFT,2));
  CHKERRQ(KSPSetSupportedNorm(ksp,KSP_NORM_PRECONDITIONED,PC_LEFT,2));
  CHKERRQ(KSPSetSupportedNorm(ksp,KSP_NORM_NATURAL,PC_LEFT,2));
  CHKERRQ(KSPSetSupportedNorm(ksp,KSP_NORM_NONE,PC_LEFT,1));

  ksp->ops->setup          = KSPSetUp_GROPPCG;
  ksp->ops->solve          = KSPSolve_GROPPCG;
  ksp->ops->destroy        = KSPDestroyDefault;
  ksp->ops->view           = NULL;
  ksp->ops->setfromoptions = NULL;
  ksp->ops->buildsolution  = KSPBuildSolutionDefault;
  ksp->ops->buildresidual  = KSPBuildResidual_CG;
  PetscFunctionReturn(0);
}
