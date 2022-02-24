
#include <petsc/private/kspimpl.h>

static PetscErrorCode KSPSetUp_CR(KSP ksp)
{
  PetscFunctionBegin;
  PetscCheckFalse(ksp->pc_side == PC_RIGHT,PetscObjectComm((PetscObject)ksp),PETSC_ERR_SUP,"no right preconditioning for KSPCR");
  else PetscCheckFalse(ksp->pc_side == PC_SYMMETRIC,PETSC_COMM_SELF,PETSC_ERR_SUP,"no symmetric preconditioning for KSPCR");
  CHKERRQ(KSPSetWorkVecs(ksp,6));
  PetscFunctionReturn(0);
}

static PetscErrorCode  KSPSolve_CR(KSP ksp)
{
  PetscInt       i = 0;
  PetscReal      dp;
  PetscScalar    ai, bi;
  PetscScalar    apq,btop, bbot;
  Vec            X,B,R,RT,P,AP,ART,Q;
  Mat            Amat, Pmat;

  PetscFunctionBegin;
  X   = ksp->vec_sol;
  B   = ksp->vec_rhs;
  R   = ksp->work[0];
  RT  = ksp->work[1];
  P   = ksp->work[2];
  AP  = ksp->work[3];
  ART = ksp->work[4];
  Q   = ksp->work[5];

  /* R is the true residual norm, RT is the preconditioned residual norm */
  CHKERRQ(PCGetOperators(ksp->pc,&Amat,&Pmat));
  if (!ksp->guess_zero) {
    CHKERRQ(KSP_MatMult(ksp,Amat,X,R));     /*   R <- A*X           */
    CHKERRQ(VecAYPX(R,-1.0,B));            /*   R <- B-R == B-A*X  */
  } else {
    CHKERRQ(VecCopy(B,R));                  /*   R <- B (X is 0)    */
  }
  /* This may be true only on a subset of MPI ranks; setting it here so it will be detected by the first norm computation below */
  if (ksp->reason == KSP_DIVERGED_PC_FAILED) {
    CHKERRQ(VecSetInf(R));
  }
  CHKERRQ(KSP_PCApply(ksp,R,P));     /*   P   <- B*R         */
  CHKERRQ(KSP_MatMult(ksp,Amat,P,AP));      /*   AP  <- A*P         */
  CHKERRQ(VecCopy(P,RT));                   /*   RT  <- P           */
  CHKERRQ(VecCopy(AP,ART));                 /*   ART <- AP          */
  CHKERRQ(VecDotBegin(RT,ART,&btop));          /*   (RT,ART)           */

  if (ksp->normtype == KSP_NORM_PRECONDITIONED) {
    CHKERRQ(VecNormBegin(RT,NORM_2,&dp));        /*   dp <- RT'*RT       */
    CHKERRQ(VecDotEnd   (RT,ART,&btop));           /*   (RT,ART)           */
    CHKERRQ(VecNormEnd  (RT,NORM_2,&dp));        /*   dp <- RT'*RT       */
    KSPCheckNorm(ksp,dp);
  } else if (ksp->normtype == KSP_NORM_NONE) {
    dp   = 0.0; /* meaningless value that is passed to monitor and convergence test */
    CHKERRQ(VecDotEnd   (RT,ART,&btop));           /*   (RT,ART)           */
  } else if (ksp->normtype == KSP_NORM_UNPRECONDITIONED) {
    CHKERRQ(VecNormBegin(R,NORM_2,&dp));         /*   dp <- R'*R         */
    CHKERRQ(VecDotEnd   (RT,ART,&btop));          /*   (RT,ART)           */
    CHKERRQ(VecNormEnd  (R,NORM_2,&dp));        /*   dp <- RT'*RT       */
    KSPCheckNorm(ksp,dp);
  } else if (ksp->normtype == KSP_NORM_NATURAL) {
    CHKERRQ(VecDotEnd   (RT,ART,&btop));           /*   (RT,ART)           */
    dp   = PetscSqrtReal(PetscAbsScalar(btop));                  /* dp = sqrt(R,AR)      */
  } else SETERRQ(PetscObjectComm((PetscObject)ksp),PETSC_ERR_SUP,"KSPNormType of %d not supported",(int)ksp->normtype);
  if (PetscAbsScalar(btop) < 0.0) {
    ksp->reason = KSP_DIVERGED_INDEFINITE_MAT;
    CHKERRQ(PetscInfo(ksp,"diverging due to indefinite or negative definite matrix\n"));
    PetscFunctionReturn(0);
  }

  ksp->its   = 0;
  CHKERRQ(KSPMonitor(ksp,0,dp));
  CHKERRQ(PetscObjectSAWsTakeAccess((PetscObject)ksp));
  ksp->rnorm = dp;
  CHKERRQ(PetscObjectSAWsGrantAccess((PetscObject)ksp));
  CHKERRQ(KSPLogResidualHistory(ksp,dp));
  CHKERRQ((*ksp->converged)(ksp,0,dp,&ksp->reason,ksp->cnvP));
  if (ksp->reason) PetscFunctionReturn(0);

  i = 0;
  do {
    CHKERRQ(KSP_PCApply(ksp,AP,Q));  /*   Q <- B* AP          */

    CHKERRQ(VecDot(AP,Q,&apq));
    KSPCheckDot(ksp,apq);
    if (PetscRealPart(apq) <= 0.0) {
      ksp->reason = KSP_DIVERGED_INDEFINITE_PC;
      CHKERRQ(PetscInfo(ksp,"KSPSolve_CR:diverging due to indefinite or negative definite PC\n"));
      break;
    }
    ai = btop/apq;                                      /* ai = (RT,ART)/(AP,Q)  */

    CHKERRQ(VecAXPY(X,ai,P));              /*   X   <- X + ai*P     */
    CHKERRQ(VecAXPY(RT,-ai,Q));             /*   RT  <- RT - ai*Q    */
    CHKERRQ(KSP_MatMult(ksp,Amat,RT,ART));  /*   ART <-   A*RT       */
    bbot = btop;
    CHKERRQ(VecDotBegin(RT,ART,&btop));

    if (ksp->normtype == KSP_NORM_PRECONDITIONED) {
      CHKERRQ(VecNormBegin(RT,NORM_2,&dp));      /*   dp <- || RT ||      */
      CHKERRQ(VecDotEnd   (RT,ART,&btop));
      CHKERRQ(VecNormEnd  (RT,NORM_2,&dp));      /*   dp <- || RT ||      */
      KSPCheckNorm(ksp,dp);
    } else if (ksp->normtype == KSP_NORM_NATURAL) {
      CHKERRQ(VecDotEnd(RT,ART,&btop));
      dp   = PetscSqrtReal(PetscAbsScalar(btop));                /* dp = sqrt(R,AR)       */
    } else if (ksp->normtype == KSP_NORM_NONE) {
      CHKERRQ(VecDotEnd(RT,ART,&btop));
      dp   = 0.0; /* meaningless value that is passed to monitor and convergence test */
    } else if (ksp->normtype == KSP_NORM_UNPRECONDITIONED) {
      CHKERRQ(VecAXPY(R,ai,AP));           /*   R   <- R - ai*AP    */
      CHKERRQ(VecNormBegin(R,NORM_2,&dp));       /*   dp <- R'*R          */
      CHKERRQ(VecDotEnd   (RT,ART,&btop));
      CHKERRQ(VecNormEnd  (R,NORM_2,&dp));       /*   dp <- R'*R          */
      KSPCheckNorm(ksp,dp);
    } else SETERRQ(PetscObjectComm((PetscObject)ksp),PETSC_ERR_SUP,"KSPNormType of %d not supported",(int)ksp->normtype);
    if (PetscAbsScalar(btop) < 0.0) {
      ksp->reason = KSP_DIVERGED_INDEFINITE_MAT;
      CHKERRQ(PetscInfo(ksp,"diverging due to indefinite or negative definite PC\n"));
      break;
    }

    CHKERRQ(PetscObjectSAWsTakeAccess((PetscObject)ksp));
    ksp->its++;
    ksp->rnorm = dp;
    CHKERRQ(PetscObjectSAWsGrantAccess((PetscObject)ksp));

    CHKERRQ(KSPLogResidualHistory(ksp,dp));
    CHKERRQ(KSPMonitor(ksp,i+1,dp));
    CHKERRQ((*ksp->converged)(ksp,i+1,dp,&ksp->reason,ksp->cnvP));
    if (ksp->reason) break;

    bi   = btop/bbot;
    CHKERRQ(VecAYPX(P,bi,RT));              /*   P <- RT + Bi P     */
    CHKERRQ(VecAYPX(AP,bi,ART));            /*   AP <- ART + Bi AP  */
    i++;
  } while (i<ksp->max_it);
  if (i >= ksp->max_it) ksp->reason =  KSP_DIVERGED_ITS;
  PetscFunctionReturn(0);
}

/*MC
     KSPCR - This code implements the (preconditioned) conjugate residuals method

   Options Database Keys:
    see KSPSolve()

   Level: beginner

   Notes:
    The operator and the preconditioner must be symmetric for this method. The
          preconditioner must be POSITIVE-DEFINITE and the operator POSITIVE-SEMIDEFINITE.
          Support only for left preconditioning.

   References:
.  * - Magnus R. Hestenes and Eduard Stiefel, Methods of Conjugate Gradients for Solving Linear Systems,
   Journal of Research of the National Bureau of Standards Vol. 49, No. 6, December 1952 Research Paper 2379

.seealso: KSPCreate(), KSPSetType(), KSPType (for list of available types), KSP, KSPCG
M*/
PETSC_EXTERN PetscErrorCode KSPCreate_CR(KSP ksp)
{
  PetscFunctionBegin;
  CHKERRQ(KSPSetSupportedNorm(ksp,KSP_NORM_PRECONDITIONED,PC_LEFT,3));
  CHKERRQ(KSPSetSupportedNorm(ksp,KSP_NORM_UNPRECONDITIONED,PC_LEFT,2));
  CHKERRQ(KSPSetSupportedNorm(ksp,KSP_NORM_NATURAL,PC_LEFT,2));
  CHKERRQ(KSPSetSupportedNorm(ksp,KSP_NORM_NONE,PC_LEFT,1));

  ksp->ops->setup          = KSPSetUp_CR;
  ksp->ops->solve          = KSPSolve_CR;
  ksp->ops->destroy        = KSPDestroyDefault;
  ksp->ops->buildsolution  = KSPBuildSolutionDefault;
  ksp->ops->buildresidual  = KSPBuildResidualDefault;
  ksp->ops->setfromoptions = NULL;
  ksp->ops->view           = NULL;
  PetscFunctionReturn(0);
}
