
#include <petsc/private/kspimpl.h>

static PetscErrorCode KSPSetUp_CR(KSP ksp)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (ksp->pc_side == PC_RIGHT) SETERRQ(PetscObjectComm((PetscObject)ksp),PETSC_ERR_SUP,"no right preconditioning for KSPCR");
  else if (ksp->pc_side == PC_SYMMETRIC) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"no symmetric preconditioning for KSPCR");
  ierr = KSPSetWorkVecs(ksp,6);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode  KSPSolve_CR(KSP ksp)
{
  PetscErrorCode ierr;
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
  ierr = PCGetOperators(ksp->pc,&Amat,&Pmat);CHKERRQ(ierr);
  if (!ksp->guess_zero) {
    ierr = KSP_MatMult(ksp,Amat,X,R);CHKERRQ(ierr);     /*   R <- A*X           */
    ierr = VecAYPX(R,-1.0,B);CHKERRQ(ierr);            /*   R <- B-R == B-A*X  */
  } else {
    ierr = VecCopy(B,R);CHKERRQ(ierr);                  /*   R <- B (X is 0)    */
  }
  ierr = KSP_PCApply(ksp,R,P);CHKERRQ(ierr);     /*   P   <- B*R         */
  ierr = KSP_MatMult(ksp,Amat,P,AP);CHKERRQ(ierr);      /*   AP  <- A*P         */
  ierr = VecCopy(P,RT);CHKERRQ(ierr);                   /*   RT  <- P           */
  ierr = VecCopy(AP,ART);CHKERRQ(ierr);                 /*   ART <- AP          */
  ierr = VecDotBegin(RT,ART,&btop);CHKERRQ(ierr);          /*   (RT,ART)           */

  if (ksp->normtype == KSP_NORM_PRECONDITIONED) {
    ierr = VecNormBegin(RT,NORM_2,&dp);CHKERRQ(ierr);        /*   dp <- RT'*RT       */
    ierr = VecDotEnd   (RT,ART,&btop);CHKERRQ(ierr);           /*   (RT,ART)           */
    ierr = VecNormEnd  (RT,NORM_2,&dp);CHKERRQ(ierr);        /*   dp <- RT'*RT       */
    KSPCheckNorm(ksp,dp);
  } else if (ksp->normtype == KSP_NORM_NONE) {
    dp   = 0.0; /* meaningless value that is passed to monitor and convergence test */
    ierr = VecDotEnd   (RT,ART,&btop);CHKERRQ(ierr);           /*   (RT,ART)           */
  } else if (ksp->normtype == KSP_NORM_UNPRECONDITIONED) {
    ierr = VecNormBegin(R,NORM_2,&dp);CHKERRQ(ierr);         /*   dp <- R'*R         */
    ierr = VecDotEnd   (RT,ART,&btop);CHKERRQ(ierr);          /*   (RT,ART)           */
    ierr = VecNormEnd  (R,NORM_2,&dp);CHKERRQ(ierr);        /*   dp <- RT'*RT       */
    KSPCheckNorm(ksp,dp);
  } else if (ksp->normtype == KSP_NORM_NATURAL) {
    ierr = VecDotEnd   (RT,ART,&btop);CHKERRQ(ierr);           /*   (RT,ART)           */
    dp   = PetscSqrtReal(PetscAbsScalar(btop));                  /* dp = sqrt(R,AR)      */
  } else SETERRQ1(PetscObjectComm((PetscObject)ksp),PETSC_ERR_SUP,"KSPNormType of %d not supported",(int)ksp->normtype);
  if (PetscAbsScalar(btop) < 0.0) {
    ksp->reason = KSP_DIVERGED_INDEFINITE_MAT;
    ierr        = PetscInfo(ksp,"diverging due to indefinite or negative definite matrix\n");CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }

  ksp->its   = 0;
  ierr       = KSPMonitor(ksp,0,dp);CHKERRQ(ierr);
  ierr       = PetscObjectSAWsTakeAccess((PetscObject)ksp);CHKERRQ(ierr);
  ksp->rnorm = dp;
  ierr = PetscObjectSAWsGrantAccess((PetscObject)ksp);CHKERRQ(ierr);
  ierr = KSPLogResidualHistory(ksp,dp);CHKERRQ(ierr);
  ierr = (*ksp->converged)(ksp,0,dp,&ksp->reason,ksp->cnvP);CHKERRQ(ierr);
  if (ksp->reason) PetscFunctionReturn(0);

  i = 0;
  do {
    ierr = KSP_PCApply(ksp,AP,Q);CHKERRQ(ierr);  /*   Q <- B* AP          */

    ierr = VecDot(AP,Q,&apq);CHKERRQ(ierr);
    KSPCheckDot(ksp,apq);
    if (PetscRealPart(apq) <= 0.0) {
      ksp->reason = KSP_DIVERGED_INDEFINITE_PC;
      ierr        = PetscInfo(ksp,"KSPSolve_CR:diverging due to indefinite or negative definite PC\n");CHKERRQ(ierr);
      break;
    }
    ai = btop/apq;                                      /* ai = (RT,ART)/(AP,Q)  */

    ierr = VecAXPY(X,ai,P);CHKERRQ(ierr);              /*   X   <- X + ai*P     */
    ierr = VecAXPY(RT,-ai,Q);CHKERRQ(ierr);             /*   RT  <- RT - ai*Q    */
    ierr = KSP_MatMult(ksp,Amat,RT,ART);CHKERRQ(ierr);  /*   ART <-   A*RT       */
    bbot = btop;
    ierr = VecDotBegin(RT,ART,&btop);CHKERRQ(ierr);

    if (ksp->normtype == KSP_NORM_PRECONDITIONED) {
      ierr = VecNormBegin(RT,NORM_2,&dp);CHKERRQ(ierr);      /*   dp <- || RT ||      */
      ierr = VecDotEnd   (RT,ART,&btop);CHKERRQ(ierr);
      ierr = VecNormEnd  (RT,NORM_2,&dp);CHKERRQ(ierr);      /*   dp <- || RT ||      */
      KSPCheckNorm(ksp,dp);
    } else if (ksp->normtype == KSP_NORM_NATURAL) {
      ierr = VecDotEnd(RT,ART,&btop);CHKERRQ(ierr);
      dp   = PetscSqrtReal(PetscAbsScalar(btop));                /* dp = sqrt(R,AR)       */
    } else if (ksp->normtype == KSP_NORM_NONE) {
      ierr = VecDotEnd(RT,ART,&btop);CHKERRQ(ierr);
      dp   = 0.0; /* meaningless value that is passed to monitor and convergence test */
    } else if (ksp->normtype == KSP_NORM_UNPRECONDITIONED) {
      ierr = VecAXPY(R,ai,AP);CHKERRQ(ierr);           /*   R   <- R - ai*AP    */
      ierr = VecNormBegin(R,NORM_2,&dp);CHKERRQ(ierr);       /*   dp <- R'*R          */
      ierr = VecDotEnd   (RT,ART,&btop);CHKERRQ(ierr);
      ierr = VecNormEnd  (R,NORM_2,&dp);CHKERRQ(ierr);       /*   dp <- R'*R          */
      KSPCheckNorm(ksp,dp);
    } else SETERRQ1(PetscObjectComm((PetscObject)ksp),PETSC_ERR_SUP,"KSPNormType of %d not supported",(int)ksp->normtype);
    if (PetscAbsScalar(btop) < 0.0) {
      ksp->reason = KSP_DIVERGED_INDEFINITE_MAT;
      ierr        = PetscInfo(ksp,"diverging due to indefinite or negative definite PC\n");CHKERRQ(ierr);
      break;
    }

    ierr = PetscObjectSAWsTakeAccess((PetscObject)ksp);CHKERRQ(ierr);
    ksp->its++;
    ksp->rnorm = dp;
    ierr       = PetscObjectSAWsGrantAccess((PetscObject)ksp);CHKERRQ(ierr);

    ierr = KSPLogResidualHistory(ksp,dp);CHKERRQ(ierr);
    ierr = KSPMonitor(ksp,i+1,dp);CHKERRQ(ierr);
    ierr = (*ksp->converged)(ksp,i+1,dp,&ksp->reason,ksp->cnvP);CHKERRQ(ierr);
    if (ksp->reason) break;

    bi   = btop/bbot;
    ierr = VecAYPX(P,bi,RT);CHKERRQ(ierr);              /*   P <- RT + Bi P     */
    ierr = VecAYPX(AP,bi,ART);CHKERRQ(ierr);            /*   AP <- ART + Bi AP  */
    i++;
  } while (i<ksp->max_it);
  if (i >= ksp->max_it) ksp->reason =  KSP_DIVERGED_ITS;
  PetscFunctionReturn(0);
}


/*MC
     KSPCR - This code implements the (preconditioned) conjugate residuals method

   Options Database Keys:
.   see KSPSolve()

   Level: beginner

   Notes:
    The operator and the preconditioner must be symmetric for this method. The
          preconditioner must be POSITIVE-DEFINITE and the operator POSITIVE-SEMIDEFINITE.
          Support only for left preconditioning.

   References:
.   1. - Magnus R. Hestenes and Eduard Stiefel, Methods of Conjugate Gradients for Solving Linear Systems,
   Journal of Research of the National Bureau of Standards Vol. 49, No. 6, December 1952 Research Paper 2379

.seealso: KSPCreate(), KSPSetType(), KSPType (for list of available types), KSP, KSPCG
M*/
PETSC_EXTERN PetscErrorCode KSPCreate_CR(KSP ksp)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = KSPSetSupportedNorm(ksp,KSP_NORM_PRECONDITIONED,PC_LEFT,3);CHKERRQ(ierr);
  ierr = KSPSetSupportedNorm(ksp,KSP_NORM_UNPRECONDITIONED,PC_LEFT,2);CHKERRQ(ierr);
  ierr = KSPSetSupportedNorm(ksp,KSP_NORM_NATURAL,PC_LEFT,2);CHKERRQ(ierr);
  ierr = KSPSetSupportedNorm(ksp,KSP_NORM_NONE,PC_LEFT,1);CHKERRQ(ierr);

  ksp->ops->setup          = KSPSetUp_CR;
  ksp->ops->solve          = KSPSolve_CR;
  ksp->ops->destroy        = KSPDestroyDefault;
  ksp->ops->buildsolution  = KSPBuildSolutionDefault;
  ksp->ops->buildresidual  = KSPBuildResidualDefault;
  ksp->ops->setfromoptions = NULL;
  ksp->ops->view           = NULL;
  PetscFunctionReturn(0);
}
